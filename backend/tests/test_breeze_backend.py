"""Exercise the isolated-worker lifecycle and real JSON-lines protocol."""

import asyncio
import json
import sys

import numpy as np
import pytest

from backend.backends.breeze_backend import BREEZE_REQUIRED_FILES, BreezeTTSBackend


@pytest.fixture
def breeze_runtime(tmp_path, monkeypatch):
    import backend.config as config

    monkeypatch.setattr(config, "_data_dir", tmp_path / "data")
    source = tmp_path / "runtime"
    package = source / "mlx_audio/tts"
    package.mkdir(parents=True)
    (source / "mlx_audio/__init__.py").write_text("")
    (package / "__init__.py").write_text('''
import json, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def record(data):
    with Path(os.environ['BREEZE_TEST_RECORD']).open('a') as output:
        output.write(json.dumps(data) + '\\n')

def load(model_path):
    record({'loaded':model_path})
    print('noisy library startup redirected to stderr')
    class Model:
        sample_rate = 24000
        def generate(self, **kwargs):
            record(kwargs)
            print('noisy library inference redirected to stderr')
            if kwargs['text'] == 'fail':
                raise ValueError('reference synthesis failed')
            yield SimpleNamespace(audio=np.array([kwargs.get('seed') or 0, 0.25], dtype=np.float32), sample_rate=24000)
    return Model()
''')
    model = tmp_path / "model"
    for name in BREEZE_REQUIRED_FILES:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    record = tmp_path / "requests.jsonl"
    monkeypatch.setenv("VOICEIT_BREEZE_PYTHON", sys.executable)
    monkeypatch.setenv("VOICEIT_BREEZE_SOURCE_DIR", str(source))
    monkeypatch.setenv("VOICEIT_BREEZE_MODEL_DIR", str(model))
    monkeypatch.setenv("BREEZE_TEST_RECORD", str(record))
    backend = BreezeTTSBackend()
    yield backend, record, tmp_path
    backend.unload_model()
    backend._executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_worker_loads_once_and_preserves_design_reference_direction_seed(breeze_runtime):
    backend, record, directory = breeze_runtime
    reference = directory / "reference.wav"
    reference.write_bytes(b"test-audio")
    prompt, _ = await backend.create_voice_prompt(str(reference), "参考音频准确文本")
    prompt["design_prompt"] = "温柔年轻女性声音"
    first, second = await asyncio.gather(
        backend.generate("第一章", prompt, language="zh", seed=123, instruct="少年时期，语气惊讶"),
        backend.generate("第二章", prompt, language="zh", seed=123, instruct="成年时期，平静叙述"),
    )
    assert backend.is_loaded()
    assert first[1] == second[1] == 24000
    np.testing.assert_array_equal(first[0], second[0])
    events = [json.loads(line) for line in record.read_text().splitlines()]
    assert sum("loaded" in event for event in events) == 1
    generated = [event for event in events if "text" in event]
    assert generated[0]["voice"] == "S0"
    assert generated[0]["ref_audio"] == generated[1]["ref_audio"] == str(reference)
    assert generated[0]["ref_text"] == "参考音频准确文本"
    assert "温柔年轻女性声音" in generated[0]["instruct"]
    assert "少年时期" in generated[0]["instruct"]
    assert "成年时期" in generated[1]["instruct"]
    assert generated[0]["seed"] == generated[1]["seed"] == 123
    process = backend._process
    backend.unload_model()
    assert not backend.is_loaded()
    assert process.poll() is not None
    assert not list((directory / "data/tmp").glob("breeze-*"))


@pytest.mark.asyncio
async def test_design_without_reference_and_worker_error_are_explicit(breeze_runtime):
    backend, _record, _directory = breeze_runtime
    audio, rate = await backend.generate("新设计声音", {"voice_type": "designed", "design_prompt": "沉稳旁白"}, seed=9)
    assert audio.size
    assert rate == 24000
    with pytest.raises(RuntimeError, match="reference synthesis failed"):
        await backend.generate("fail", {"design_prompt": "沉稳旁白"})
    assert backend.is_loaded()  # a failed request does not silently change voices
    audio, rate = await backend.generate("下一句", {"design_prompt": "沉稳旁白"})
    assert audio.size
    assert rate == 24000


@pytest.mark.asyncio
async def test_missing_reference_is_not_silently_replaced_by_random_voice(breeze_runtime):
    backend, _record, directory = breeze_runtime
    with pytest.raises(FileNotFoundError, match="参考音频不存在"):
        await backend.create_voice_prompt(str(directory / "missing.wav"), "test")
    with pytest.raises(RuntimeError, match="参考音频不存在"):
        await backend.generate("你好", {"ref_audio": str(directory / "missing.wav"), "ref_text": "test"})


@pytest.mark.asyncio
async def test_missing_local_weights_never_download(breeze_runtime, monkeypatch, tmp_path):
    backend, _record, _directory = breeze_runtime
    monkeypatch.setenv("VOICEIT_BREEZE_MODEL_DIR", str(tmp_path / "unavailable-model"))
    assert not backend._is_model_cached()
    with pytest.raises(RuntimeError, match="不会自动重新下载"):
        await backend.load_model()
    assert backend._process is None


def test_partial_worker_response_times_out_and_reaps_process(breeze_runtime):
    import subprocess
    import time

    backend, _record, _directory = breeze_runtime
    process = subprocess.Popen([sys.executable, "-u", "-c", "import os,time;os.write(1,b'{');time.sleep(10)"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    backend._process = process
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="超时"):
        backend._read_response(0.1)
    assert time.monotonic() - start < 2
    assert process.poll() is not None


def test_worker_response_size_is_bounded_and_reaps_process(breeze_runtime):
    import subprocess

    backend, _record, _directory = breeze_runtime
    process = subprocess.Popen([sys.executable, "-u", "-c", "import os,time;os.write(1,b'x'*70000);time.sleep(10)"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    backend._process = process
    with pytest.raises(RuntimeError, match="超出大小限制"):
        backend._read_response(5)
    assert process.poll() is not None
