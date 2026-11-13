"""CustomVoice must reuse cached MLX weights and preserve speaker selection."""

import asyncio
import os
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from huggingface_hub import constants as hf_constants

from backend import backends
from backend.backends.qwen_custom_voice_backend import MLXQwenCustomVoiceBackend, QwenCustomVoiceBackend


@pytest.mark.parametrize("backend_type", ["mlx", "pytorch"])
def test_custom_voice_registry_and_factory_agree(monkeypatch, backend_type):
    monkeypatch.setattr(backends, "get_backend_type", lambda: backend_type)
    monkeypatch.setattr(backends, "_tts_backends", {})
    backend = backends.get_tts_backend_for_engine("qwen_custom_voice")
    expected = MLXQwenCustomVoiceBackend if backend_type == "mlx" else QwenCustomVoiceBackend
    assert type(backend) is expected
    try:
        for size in ("1.7B", "0.6B"):
            config = backends.get_model_config(f"qwen-custom-voice-{size}")
            assert config.hf_repo_id == backend._get_model_path(size)
            assert ("mlx-community/" in config.hf_repo_id) == (backend_type == "mlx")
    finally:
        if backend_type == "mlx":
            backend._executor.shutdown()


@pytest.fixture
def cached_mlx(monkeypatch, tmp_path):
    cache = tmp_path / "hub"
    repo = cache / "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs/main").write_text(snapshot.name)
    for filename in backends.QWEN_CUSTOM_VOICE_MLX_REQUIRED_FILES:
        path = snapshot / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if filename.endswith(".json") else b"cached-test-weights")
    (repo / "blobs").mkdir()
    (repo / "blobs/abandoned-download.incomplete").write_bytes(b"partial-other-revision")
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))

    def refuse_network(*args, **kwargs):
        raise AssertionError("Cached CustomVoice must not access the network")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse_network)
    events = []
    chunks = [np.array([0.1, 0.2], dtype=np.float64), np.array([0.3])]

    def generate_custom_voice(**kwargs):
        assert hf_constants.HF_HUB_OFFLINE
        events.append(("generate", threading.get_ident(), kwargs))
        for chunk in chunks:
            yield SimpleNamespace(audio=chunk, sample_rate=24000)

    def load(path):
        assert hf_constants.HF_HUB_OFFLINE
        assert isinstance(path, Path)
        assert path == snapshot
        events.append(("load", threading.get_ident(), path))
        return SimpleNamespace(generate_custom_voice=generate_custom_voice)

    package = ModuleType("mlx_audio")
    tts_module = ModuleType("mlx_audio.tts")
    tts_module.load = load
    package.tts = tts_module
    monkeypatch.setitem(sys.modules, "mlx_audio", package)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", tts_module)
    backend = MLXQwenCustomVoiceBackend()
    yield backend, events, chunks
    backend._executor.shutdown()


async def test_cached_generation_is_offline_and_keeps_voice_and_style(cached_mlx):
    backend, events, _ = cached_mlx
    original_offline = os.environ.get("HF_HUB_OFFLINE")
    await asyncio.gather(backend.load_model("1.7B"), backend.load_model("1.7B"))
    audio, rate = await backend.generate(
        "雨停的时候。", {"preset_voice_id": "Ono_Anna"}, language="zh", instruct="低声讲述"
    )
    assert [event[0] for event in events] == ["load", "generate"]
    assert events[0][1] == events[1][1] != threading.get_ident()
    assert events[1][2] == {
        "text": "雨停的时候。", "speaker": "Ono_Anna", "language": "chinese", "instruct": "低声讲述"
    }
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0.1, 0.2, 0.3])
    assert rate == 24000
    assert os.environ.get("HF_HUB_OFFLINE") == original_offline


async def test_empty_audio_is_reported_as_failure(cached_mlx):
    backend, _, chunks = cached_mlx
    chunks.clear()
    with pytest.raises(RuntimeError, match="no audio"):
        await backend.generate("测试", {"preset_voice_id": "Vivian"}, language="zh")


async def test_incomplete_new_revision_does_not_hide_complete_cached_model(cached_mlx):
    backend, events, _ = cached_mlx
    repo = Path(hf_constants.HF_HUB_CACHE) / "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    partial = repo / "snapshots" / ("b" * 40)
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    (repo / "refs/main").write_text(partial.name)
    await backend.load_model("1.7B")
    assert events[0][2].name == "a" * 40


def test_missing_speech_tokenizer_is_not_a_complete_model(cached_mlx):
    backend, _, _ = cached_mlx
    repo = Path(hf_constants.HF_HUB_CACHE) / "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    next((repo / "snapshots").glob("*/speech_tokenizer/model.safetensors")).unlink()
    assert not backend._is_model_cached("1.7B")


def test_model_status_recognizes_the_same_cached_checkpoint(monkeypatch, cached_mlx):
    from backend.routes.models import router
    from backend.utils import tasks

    monkeypatch.setattr(backends, "get_backend_type", lambda: "mlx")
    monkeypatch.setattr(backends, "check_model_loaded", lambda config: False)
    monkeypatch.setattr(tasks, "_task_manager", None)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/models/status")
    assert response.status_code == 200
    models = {item["model_name"]: item for item in response.json()["models"]}
    model = models["qwen-custom-voice-1.7B"]
    assert model["downloaded"]
    assert not model["downloading"]
    assert model["hf_repo_id"] == backends.QWEN_CUSTOM_VOICE_MLX_REPOS["1.7B"]
    assert not models["qwen-custom-voice-0.6B"]["downloaded"]
