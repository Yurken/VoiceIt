"""Cancellation, queue restart, and partial materialization must release books."""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import database
from backend.database import Base, Book, BookCharacter, BookSegment, Chapter, Generation, VoiceProfile
from backend.services import book_generation as books, task_queue


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'lifecycle.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    def get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(database, "get_db", get_db)
    db = factory()
    book = Book(title="全书生命周期", source_format="txt", status="analyzed")
    voice = VoiceProfile(name="已有Qwen", voice_type="preset", preset_engine="qwen_custom_voice", preset_voice_id="Serena", default_engine="qwen_custom_voice")
    db.add_all([book, voice])
    db.flush()
    char = BookCharacter(book_id=book.id, name="林青", profile_id=voice.id)
    db.add(char)
    db.flush()
    segments = []
    for number in (1, 2):
        chapter = Chapter(book_id=book.id, number=number, raw_text="你好。")
        db.add(chapter)
        db.flush()
        segment = BookSegment(chapter_id=chapter.id, character_id=char.id, type="dialogue", text="你好。", order=0, audio_status="none")
        db.add(segment)
        segments.append(segment)
    db.commit()
    yield db, book, segments
    db.close()
    engine.dispose()


def attach_pending_generation(db, book, segment):
    generation = Generation(profile_id=db.get(BookCharacter, segment.character_id).profile_id, text=segment.text,
                            language="zh", audio_path="", duration=0, source="book_import", status="pending")
    db.add(generation)
    db.flush()
    segment.generation_id = generation.id
    segment.audio_status = "pending"
    book.status = "generating"
    db.commit()
    return generation


@pytest.mark.asyncio
async def test_running_cancellation_settles_generation_segment_and_book(runtime):
    db, book, segments = runtime
    generation = attach_pending_generation(db, book, segments[0])
    started = asyncio.Event()

    async def inner():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(books._per_generation_completion_hook(segments[0].id, book.id, inner()))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    db.expire_all()
    assert db.get(Generation, generation.id).status == "failed"
    assert db.get(BookSegment, segments[0].id).audio_status == "error"
    assert db.get(Book, book.id).status == "analyzed"


@pytest.mark.asyncio
async def test_generic_cancel_accepts_pending_book_job_and_settles_book(runtime, monkeypatch):
    from backend.routes import generations

    db, book, segments = runtime
    generation = attach_pending_generation(db, book, segments[0])
    monkeypatch.setattr(generations, "cancel_generation_job", lambda generation_id: "queued")
    monkeypatch.setattr(generations, "get_task_manager", lambda: MagicMock())
    response = await generations.cancel_generation(generation.id, db)
    assert "Queued" in response["message"]
    db.expire_all()
    assert db.get(Generation, generation.id).status == "failed"
    assert db.get(BookSegment, segments[0].id).audio_status == "error"
    assert db.get(Book, book.id).status == "analyzed"


def test_partial_book_materialization_failure_releases_previously_committed_chapter(runtime, monkeypatch):
    db, book, segments = runtime
    original = books.generate_chapter
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second chapter could not materialize")
        return original(*args, **kwargs)

    monkeypatch.setattr(books, "generate_chapter", fail_second)
    with pytest.raises(RuntimeError, match="second chapter"):
        books.enqueue_book_generation(book.id, db)
    db.expire_all()
    assert db.get(Book, book.id).status == "analyzed"
    assert db.get(BookSegment, segments[0].id).audio_status == "error"
    assert db.get(BookSegment, segments[1].id).audio_status == "none"
    assert db.query(Generation).filter_by(status="pending").count() == 0


def test_single_chapter_materialization_failure_releases_book(runtime, monkeypatch):
    db, book, segments = runtime
    monkeypatch.setattr(books, "generate_chapter", MagicMock(side_effect=RuntimeError("database unavailable")))
    with pytest.raises(RuntimeError, match="database unavailable"):
        books.enqueue_chapter_generation(book.id, segments[0].chapter_id, db)
    db.expire_all()
    assert db.get(Book, book.id).status == "analyzed"


@pytest.mark.asyncio
@pytest.mark.parametrize("swallow_cancel", [False, True])
async def test_force_restart_closes_queued_inner_coroutines_and_uses_original_queue(monkeypatch, swallow_cancel):
    monkeypatch.setattr(task_queue, "_force_fail_if_active", AsyncMock())
    task_queue.init_queue(force=True)
    old_queue = task_queue._generation_queue
    old_workers = list(task_queue._generation_worker_tasks)
    started = asyncio.Event()
    inners = []

    async def block():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not swallow_cancel:
                raise

    async def owned(inner):
        await inner

    for number in range(8):
        inner = block()
        inners.append(inner)
        task_queue.enqueue_generation(f"old-{number}", task_queue.with_coro_cleanup(owned(inner), inner))
    await asyncio.wait_for(started.wait(), 1)
    task_queue.init_queue(force=True)
    new_workers = list(task_queue._generation_worker_tasks)
    try:
        await asyncio.gather(*old_workers, return_exceptions=True)
        await asyncio.wait_for(old_queue.join(), 1)
        assert all(inspect.getcoroutinestate(inner) == inspect.CORO_CLOSED for inner in inners)
        ran = asyncio.Event()

        async def new_job():
            ran.set()

        task_queue.enqueue_generation("new", new_job())
        await asyncio.wait_for(ran.wait(), 1)
        await asyncio.wait_for(task_queue._generation_queue.join(), 1)
        assert all(not worker.done() for worker in new_workers)
    finally:
        for worker in new_workers:
            worker.cancel()
        await asyncio.gather(*new_workers, return_exceptions=True)
