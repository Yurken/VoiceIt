"""Breeze TTS 2 using the user's isolated, already-installed MLX runtime.

The newer Breeze mlx-audio/transformers stack must not be imported into the
VoiceIt process. One persistent worker loads the local model once and handles
serialized JSON-lines requests on its main thread, including reference cloning.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import selectors
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .base import combine_voice_prompts as _combine_voice_prompts, model_load_progress, run_pinned

logger = logging.getLogger(__name__)
BREEZE_REQUIRED_FILES = [
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors",
]


def breeze_source_dir() -> Path:
    configured = os.environ.get("VOICEIT_BREEZE_SOURCE_DIR")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2].parent / "mlx-audio"


def breeze_model_dir() -> Path:
    configured = os.environ.get("VOICEIT_BREEZE_MODEL_DIR")
    return Path(configured).expanduser().resolve() if configured else breeze_source_dir() / "output/models/Breeze-TTS-2-mlx-8bit"


class BreezeTTSBackend:
    """Deferred-file voice prompts and an isolated, persistent MLX worker."""

    def __init__(self):
        self.model_size = "default"
        self._current_model_size = None
        self._process: subprocess.Popen | None = None
        self._worker_log = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="breeze-bridge")
        atexit.register(self.unload_model)

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(breeze_model_dir())

    def _is_model_cached(self, model_size: str = "default") -> bool:
        directory = breeze_model_dir()
        return all((directory / filename).is_file() for filename in BREEZE_REQUIRED_FILES)

    def is_loaded(self) -> bool:
        return bool(self._process and self._process.poll() is None and self._current_model_size)

    def _read_response(self, timeout: float) -> dict:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Breeze 推理进程未启动")
        # Read bounded bytes ourselves: readline() can block past the timeout
        # when a broken worker writes an incomplete line.
        deadline = time.monotonic() + timeout
        payload = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in payload:
                if not selector.select(max(0, deadline - time.monotonic())):
                    self.unload_model()
                    raise TimeoutError("Breeze 推理超时，请查看 data/logs/breeze-worker.log")
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    self.unload_model()
                    raise RuntimeError("Breeze 推理进程已退出，请查看 data/logs/breeze-worker.log")
                payload.extend(chunk)
                if len(payload) > 65536:
                    self.unload_model()
                    raise RuntimeError("Breeze 推理进程响应超出大小限制")
        try:
            response = json.loads(payload.decode("utf-8"))
            if not isinstance(response, dict):
                raise ValueError("Expected a JSON object")
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            self.unload_model()
            raise RuntimeError("Breeze 推理进程返回了无效数据") from exc
        if response.get("status") == "error":
            raise RuntimeError(response.get("error") or "Breeze 生成失败")
        return response

    async def load_model(self, model_size: str = "default") -> None:
        await run_pinned(self._executor, self._load_model_sync)

    load_model_async = load_model

    def _load_model_sync(self) -> None:
        with self._lock:
            if self.is_loaded():
                return
            self.unload_model()
            source = breeze_source_dir()
            python_path = Path(os.environ.get("VOICEIT_BREEZE_PYTHON") or source / ".conda/bin/python").expanduser()
            if not python_path.is_file():
                raise RuntimeError("未找到 Breeze 的独立 Python 环境，请设置 VOICEIT_BREEZE_PYTHON")
            if not self._is_model_cached():
                raise RuntimeError("未找到完整的 Breeze 本地模型，请设置 VOICEIT_BREEZE_MODEL_DIR；不会自动重新下载")
            worker_script = Path(__file__).with_name("breeze_worker.py")
            if not worker_script.is_file():
                raise RuntimeError("Breeze worker 脚本缺失，请重新安装 VoiceIt")
            from .. import config

            logs_dir = config.get_data_dir() / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._worker_log = (logs_dir / "breeze-worker.log").open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            with model_load_progress("breeze-tts", True):
                try:
                    self._process = subprocess.Popen(
                        [str(python_path), "-u", str(worker_script), str(breeze_model_dir()), str(source)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._worker_log,
                        bufsize=0, env=env,
                    )
                    ready = self._read_response(120)
                    if ready.get("status") != "ready":
                        raise RuntimeError("Breeze 模型未就绪")
                    self._current_model_size = "default"
                except BaseException:
                    self.unload_model()
                    raise
            logger.info("Breeze local worker ready (pid=%s)", self._process.pid)

    def unload_model(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            self._current_model_size = None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                for pipe in (process.stdin, process.stdout):
                    if pipe is not None:
                        pipe.close()
            if self._worker_log is not None:
                self._worker_log.close()
                self._worker_log = None

    async def create_voice_prompt(self, audio_path: str, reference_text: str, use_cache: bool = True) -> tuple[dict, bool]:
        if not Path(audio_path).is_file():
            raise FileNotFoundError("Breeze 角色参考音频不存在：" + str(audio_path))
        if not reference_text.strip():
            raise ValueError("Breeze 声音克隆需要参考音频的准确文本")
        return {"ref_audio": str(audio_path), "ref_text": reference_text}, False

    async def combine_voice_prompts(self, audio_paths: list[str], reference_texts: list[str]) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=24000)

    async def generate(
        self, text: str, voice_prompt: dict, language: str = "zh",
        seed: int | None = None, instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        if not text.strip():
            raise ValueError("朗读文本不能为空")
        return await run_pinned(self._executor, self._generate_sync, text, voice_prompt, seed, instruct)

    def _generate_sync(self, text: str, voice_prompt: dict, seed: int | None, instruct: str | None) -> tuple[np.ndarray, int]:
        with self._lock:
            self._load_model_sync()
            from .. import config

            work_dir = config.get_data_dir() / "tmp"
            work_dir.mkdir(parents=True, exist_ok=True)
            request_id = uuid.uuid4().hex
            with tempfile.TemporaryDirectory(prefix="breeze-", dir=work_dir) as temporary:
                output_path = Path(temporary) / "audio.npy"
                request = {
                    "id": request_id, "method": "generate", "text": text,
                    "voice_prompt": {key: voice_prompt.get(key) for key in ("ref_audio", "ref_audio_path", "ref_text", "design_prompt")},
                    "seed": seed, "instruct": instruct, "output_path": str(output_path),
                }
                self._process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
                self._process.stdin.flush()
                response = self._read_response(600)
                if response.get("id") != request_id:
                    self.unload_model()
                    raise RuntimeError("Breeze 推理响应与请求不匹配")
                audio = np.asarray(np.load(output_path, allow_pickle=False), dtype=np.float32).reshape(-1)
                sample_rate = int(response["sample_rate"])
                if not audio.size or not np.isfinite(audio).all() or sample_rate <= 0:
                    raise RuntimeError("Breeze 返回了无效音频")
                return audio, sample_rate
