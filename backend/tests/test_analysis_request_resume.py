"""An interrupted book retains its model and voice-selection configuration."""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config, database
from backend.database import Base, Book
from backend.routes.book_analysis import router
from backend.services import book_analysis, task_queue


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'request.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)

    def get_db():
        with sessions() as db:
            yield db

    monkeypatch.setattr(database, "get_db", get_db)
    pending = []
    monkeypatch.setattr(task_queue, "create_background_task", lambda coro: pending.append(coro))
    analysis = AsyncMock()
    monkeypatch.setattr(book_analysis, "run_analysis_task", analysis)
    with sessions() as db:
        book = Book(id=str(uuid.uuid4()), title="长篇小说", source_format="txt", status="imported")
        db.add(book)
        db.commit()
        book_id = book.id
    yield book_id, sessions, pending, analysis, tmp_path
    for coro in pending:
        coro.close()
    engine.dispose()


def test_retry_after_restart_recovers_voice_mode_narrator_and_model(runtime):
    book_id, sessions, pending, analysis, directory = runtime
    book_analysis.enqueue_analysis(book_id, "4B", "narrator-profile", voice_mode="designed")
    pending.pop().close()
    path = directory / "analysis" / book_id / "request.json"
    original = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    with sessions() as db:
        assert book_analysis.recover_interrupted_analyses(db) == 1
    book_analysis.enqueue_analysis(book_id, None, None)
    asyncio.run(pending.pop())
    kwargs = analysis.await_args.kwargs
    assert kwargs["resume"] is True
    assert kwargs["model_size"] == "4B"
    assert kwargs["narrator_voice_id"] == "narrator-profile"
    assert kwargs["voice_mode"] == "designed"
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == original_mtime
    assert not path.with_suffix(".json.tmp").exists()


def test_explicit_retry_options_override_manifest_without_resetting_start_time(runtime):
    book_id, sessions, pending, analysis, _ = runtime
    first = book_analysis.resolve_analysis_request(book_id, "4B", "narrator-profile", voice_mode="designed")
    with sessions() as db:
        db.get(Book, book_id).status = "error"
        db.commit()
    book_analysis.enqueue_analysis(book_id, "1.7B", "auto", voice_mode="preset")
    asyncio.run(pending.pop())
    kwargs = analysis.await_args.kwargs
    assert (kwargs["model_size"], kwargs["narrator_voice_id"], kwargs["voice_mode"]) == ("1.7B", "auto", "preset")
    stored = book_analysis.load_analysis_request(book_id)
    assert stored["started_at"] == first["started_at"]
    assert stored["voice_mode"] == "preset"


def test_new_analysis_uses_normal_qwen_defaults_instead_of_old_request(runtime):
    book_id, _, pending, analysis, _ = runtime
    book_analysis.resolve_analysis_request(book_id, "4B", "narrator-profile", voice_mode="designed")
    book_analysis.enqueue_analysis(book_id, None, None)
    asyncio.run(pending.pop())
    kwargs = analysis.await_args.kwargs
    assert kwargs["resume"] is False
    assert (kwargs["model_size"], kwargs["narrator_voice_id"], kwargs["voice_mode"]) == ("1.7B", "auto", "preset")


def test_legacy_retry_preserves_v2_checkpoint_bytes_and_uses_defaults(runtime):
    book_id, sessions, pending, analysis, directory = runtime
    folder = directory / "analysis" / book_id
    folder.mkdir(parents=True)
    checkpoint = folder / "chapter-id.json"
    payload = {"version": 2, "model_size": "1.7B", "analysis": {"segments": [{"text": "已完成的原文"}]}}
    checkpoint.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    original = checkpoint.read_bytes()
    with sessions() as db:
        db.get(Book, book_id).status = "error"
        db.commit()
    book_analysis.enqueue_analysis(book_id, None, None)
    asyncio.run(pending.pop())
    assert analysis.await_args.kwargs["resume"] is True
    assert analysis.await_args.kwargs["model_size"] == "1.7B"
    assert analysis.await_args.kwargs["voice_mode"] == "preset"
    assert checkpoint.read_bytes() == original


def test_corrupt_request_does_not_prevent_retry(runtime):
    book_id, sessions, pending, analysis, directory = runtime
    folder = directory / "analysis" / book_id
    folder.mkdir(parents=True)
    (folder / "request.json").write_text('{"model_size":', encoding="utf-8")
    with sessions() as db:
        db.get(Book, book_id).status = "error"
        db.commit()
    book_analysis.enqueue_analysis(book_id, None, None)
    asyncio.run(pending.pop())
    assert analysis.await_args.kwargs["voice_mode"] == "preset"
    assert book_analysis.load_analysis_request(book_id)["model_size"] == "1.7B"


@pytest.mark.parametrize(("body", "expected"), [({}, None), ({"voice_mode": "preset"}, "preset"), ({"voice_mode": "designed"}, "designed")])
def test_route_preserves_omitted_voice_mode_for_resume(runtime, monkeypatch, body, expected):
    book_id, sessions, _, _, _ = runtime
    captured = {}

    def enqueue(*args, **kwargs):
        captured.update(kwargs)
        return "task-id"

    def get_db():
        with sessions() as db:
            yield db

    monkeypatch.setattr(book_analysis, "enqueue_analysis", enqueue)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[database.get_db] = get_db
    # The route's imported dependency predates the fixture's monkeypatch.
    from backend.routes import book_analysis as route_module
    app.dependency_overrides[route_module.get_db] = get_db
    with TestClient(app) as client:
        response = client.post(f"/books/{book_id}/analyze", json=body)
    assert response.status_code == 202
    assert captured["voice_mode"] == expected
