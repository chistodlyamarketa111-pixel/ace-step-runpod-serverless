#!/usr/bin/env python3
"""
ACE-Step v1.5 — HTTP Test Server for RunPod Pod (hourly)
Uses ACEStepPipeline from ace-step package + Python stdlib http.server.
"""

import base64
import json
import os
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch

CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/workspace/checkpoints")
CPU_OFFLOAD = os.environ.get("ACESTEP_CPU_OFFLOAD", "false").lower() == "true"

VALID_MODELS = {
    "acestep-v15-turbo",
    "acestep-v15-sft",
    "acestep-v15-base",
    "acestep-v15-turbo-shift3",
}

DEFAULT_STEPS = {
    "acestep-v15-turbo": 8,
    "acestep-v15-sft": 32,
    "acestep-v15-base": 50,
    "acestep-v15-turbo-shift3": 8,
}

DEFAULT_MODEL = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-turbo")

pipeline = None
current_model = None


def get_pipeline(model_name: str):
    global pipeline, current_model

    if pipeline is not None and current_model == model_name:
        return pipeline

    from acestep.pipeline_ace_step import ACEStepPipeline

    model_path = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model '{model_name}' not found at {model_path}")

    if pipeline is None:
        print(f"[ACE-Step] Creating pipeline with checkpoint_dir={CHECKPOINT_DIR}")
        start = time.time()
        pipeline = ACEStepPipeline(
            checkpoint_dir=CHECKPOINT_DIR,
            cpu_offload=CPU_OFFLOAD,
        )
        elapsed = time.time() - start
        print(f"[ACE-Step] Pipeline created in {elapsed:.1f}s")

    print(f"[ACE-Step] Loading checkpoint: {model_name} from {model_path}")
    start = time.time()
    pipeline.load_checkpoint(checkpoint_dir=model_path)
    elapsed = time.time() - start
    print(f"[ACE-Step] Checkpoint '{model_name}' loaded in {elapsed:.1f}s, loaded={pipeline.loaded}")
    current_model = model_name

    return pipeline


def _json_response(handler, data, status=200):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def handle_health():
    available = [m for m in VALID_MODELS if os.path.exists(os.path.join(CHECKPOINT_DIR, m))]
    return {
        "status": "ok",
        "available_models": available,
        "current_model": current_model,
        "pipeline_loaded": pipeline.loaded if pipeline else False,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else 0,
    }


def handle_generate(data):
    model_name = data.get("model", DEFAULT_MODEL)
    if model_name not in VALID_MODELS:
        return {"error": f"Invalid model '{model_name}'. Available: {list(VALID_MODELS)}"}, 400

    pipe = get_pipeline(model_name)

    if not pipe.loaded:
        return {"error": "Pipeline not loaded properly"}, 500

    prompt = data.get("prompt", "")
    lyrics = data.get("lyrics", "")
    duration = float(data.get("duration", 30))
    audio_format = data.get("audio_format", "wav")
    seed = int(data.get("seed", -1))
    default_steps = DEFAULT_STEPS.get(model_name, 8)
    infer_step = int(data.get("inference_steps", default_steps))
    guidance_scale = float(data.get("guidance_scale", 15.0))
    task = data.get("task_type", "text2music")
    batch_size = int(data.get("batch_size", 1))

    manual_seeds = None
    if seed >= 0:
        manual_seeds = [seed] * batch_size

    save_dir = "/tmp/ace_output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"gen_{int(time.time())}")

    print(f"[ACE-Step] Generate: model={model_name}, prompt='{prompt[:80]}', "
          f"duration={duration}s, steps={infer_step}, guidance={guidance_scale}")

    gen_method = getattr(pipe, "calc_v", None) or getattr(pipe, "calc", None)
    if gen_method is None:
        return {"error": "Pipeline has no calc/calc_v method"}, 500

    start = time.time()
    result = gen_method(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=duration,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        task=task,
        format=audio_format if audio_format in ("wav", "mp3", "flac") else "wav",
        manual_seeds=manual_seeds,
        batch_size=batch_size,
        save_path=save_path,
    )
    gen_time = time.time() - start

    print(f"[ACE-Step] Done in {gen_time:.1f}s, result type: {type(result).__name__}")

    audio_b64 = None

    if isinstance(result, dict):
        print(f"[ACE-Step] Result keys: {list(result.keys())}")
        for key in ("audio", "audio_path", "path", "file"):
            if key in result:
                val = result[key]
                if isinstance(val, bytes):
                    audio_b64 = base64.b64encode(val).decode("utf-8")
                    break
                elif isinstance(val, str) and os.path.exists(val):
                    with open(val, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                    break

    elif isinstance(result, (list, tuple)):
        print(f"[ACE-Step] Result is list with {len(result)} items")
        for item in result:
            if isinstance(item, str) and os.path.exists(item):
                with open(item, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                break
            elif isinstance(item, dict):
                for key in ("audio_path", "path", "file", "audio"):
                    if key in item:
                        val = item[key]
                        if isinstance(val, str) and os.path.exists(val):
                            with open(val, "rb") as f:
                                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                            break
                if audio_b64:
                    break

    elif isinstance(result, str):
        if os.path.exists(result):
            with open(result, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    if audio_b64 is None:
        saved = [f for f in os.listdir(save_dir) if f.startswith(f"gen_")]
        print(f"[ACE-Step] Checking save_dir: {saved}")
        for fname in sorted(saved, reverse=True):
            fpath = os.path.join(save_dir, fname)
            if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                with open(fpath, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                audio_format = fname.rsplit(".", 1)[-1] if "." in fname else audio_format
                break

    if audio_b64 is None:
        result_info = f"type={type(result).__name__}"
        if isinstance(result, dict):
            result_info += f" keys={list(result.keys())}"
        return {"error": f"No audio produced. {result_info}"}, 500

    return {
        "model": model_name,
        "audio_base64": audio_b64,
        "audio_format": audio_format,
        "generation_time": round(gen_time, 1),
        "duration": duration,
        "inference_steps": infer_step,
        "seed": seed,
    }, 200


class AceStepHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            _json_response(self, handle_health())
        else:
            _json_response(self, {"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/generate":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                data = json.loads(body)
                result, status = handle_generate(data)
                _json_response(self, result, status)
            except Exception as e:
                traceback.print_exc()
                _json_response(self, {"error": str(e)}, 500)
        else:
            _json_response(self, {"error": "Not found"}, 404)

    def log_message(self, format, *args):
        print(f"[HTTP] {args[0]}")


if __name__ == "__main__":
    print(f"[ACE-Step] Checkpoint dir: {CHECKPOINT_DIR}")
    available = [m for m in VALID_MODELS if os.path.exists(os.path.join(CHECKPOINT_DIR, m))]
    print(f"[ACE-Step] Available models: {available}")

    print(f"[ACE-Step] Pre-loading default model: {DEFAULT_MODEL}...")
    get_pipeline(DEFAULT_MODEL)

    port = int(os.environ.get("PORT", "8888"))
    server = HTTPServer(("0.0.0.0", port), AceStepHandler)
    print(f"[ACE-Step] Server ready on :{port}")
    server.serve_forever()
