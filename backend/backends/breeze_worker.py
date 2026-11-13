"""Standalone JSON-lines bridge, executed by the existing Breeze Python runtime.

Keep this file free of VoiceIt imports: it is also shipped as source data for
frozen builds, whose Python environment differs from the MLX runtime.
"""

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    protocol_output = sys.stdout
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_path, source_path = sys.argv[1:3]
    if source_path:
        sys.path.insert(0, source_path)

    def respond(payload):
        protocol_output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        protocol_output.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            from mlx_audio.tts import load

            model = load(model_path)
        respond({"status": "ready", "sample_rate": model.sample_rate})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        respond({"status": "error", "error": str(exc)})
        return 1

    # Loading and generation stay on this main thread so MLX owns one stream.
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("method") != "generate":
                raise ValueError("Unknown Breeze worker method")
            prompt = request.get("voice_prompt") or {}
            reference = prompt.get("ref_audio") or prompt.get("ref_audio_path")
            if reference and not Path(reference).is_file():
                raise FileNotFoundError("角色参考音频不存在：" + reference)
            reference_text = prompt.get("ref_text")
            if reference and not reference_text:
                raise ValueError("Breeze 声音克隆需要参考音频的准确文本")
            direction = "\n".join(filter(None, [prompt.get("design_prompt"), request.get("instruct")]))
            with contextlib.redirect_stdout(sys.stderr):
                chunks = list(model.generate(
                    text=request["text"], voice="S0", instruct=direction or None,
                    ref_audio=reference, ref_text=reference_text,
                    seed=request.get("seed"), cfg_scale=4,
                ))
                if not chunks:
                    raise RuntimeError("Breeze 未生成音频")
                sample_rate = chunks[0].sample_rate
                if any(chunk.sample_rate != sample_rate for chunk in chunks):
                    raise RuntimeError("Breeze 返回了不一致的采样率")
                audio = np.concatenate([np.asarray(chunk.audio, dtype=np.float32).reshape(-1) for chunk in chunks])
                if not audio.size or not np.isfinite(audio).all():
                    raise RuntimeError("Breeze 返回了空或无效音频")
                np.save(request["output_path"], audio, allow_pickle=False)
            respond({"id": request["id"], "status": "ok", "sample_rate": sample_rate})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            respond({"id": request.get("id"), "status": "error", "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
