#!/usr/bin/env python3
"""
ACE-Step v1.5 — HTTP Test Server for RunPod Pod (hourly)
Uses ACEStepPipeline from ace-step package + Python stdlib http.server.
"""

import base64
import io
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

pipelines = {}


def get_pipeline(model_name: str):
    global pipelines
    if model_name in pipelines:
        return pipelines[model_name]

    from acestep.pipeline_ace_step import ACEStepPipeline

    model_path = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model '{model_name}' not found at {model_path}")

    print(f"[ACE-Step] Loading pipeline for: {model_name} from {model_path}")
    start = time.time()
    pipe = ACEStepPipeline(
        checkpoint_dir=model_path,
        cpu_offload=CPU_OFFLOAD,
    )
    elapsed = time.time() - start
    print(f"[ACE-Step] Pipeline '{model_name}' loaded in {elapsed:.1f}s")

    pipelines[model_name] = pipe
    return pipe


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
        "loaded_models": list(pipelines.keys()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else 0,
    }


def handle_generate(data):
    model_name = data.get("model", DEFAULT_MODEL)
    if model_name not in VALID_MODELS:
        return {"error": f"Invalid model '{model_name}'. Available: {list(VALID_MODELS)}"}, 400

    pipe = get_pipeline(model_name)

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

    print(f"[ACE-Step] Generate: model={model_name}, prompt='{prompt[:80]}', "
          f"duration={duration}s, steps={infer_step}, guidance={guidance_scale}")

    start = time.time()
    result = pipe.calc(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=duration,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        task=task,
        format=audio_format if audio_format in ("wav", "mp3", "flac") else "wav",
        manual_seeds=manual_seeds,
        batch_size=batch_size,
        save_path=None,
    )
    gen_time = time.time() - start

    print(f"[ACE-Step] Done in {gen_time:.1f}s, result type: {type(result)}")

    audio_b64 = None
    if isinstance(result, dict):
        if "audio" in result:
            audio_data = result["audio"]
            if isinstance(audio_data, bytes):
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
            elif isinstance(audio_data, str):
                if os.path.exists(audio_data):
                    with open(audio_data, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                else:
                    audio_b64 = audio_data
        elif "audio_path" in result or "path" in result:
            path = result.get("audio_path", result.get("path", ""))
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    elif isinstance(result, (list, tuple)):
        if len(result) > 0:
            item = result[0]
            if isinstance(item, str) and os.path.exists(item):
                with open(item, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            elif isinstance(item, dict):
                path = item.get("audio_path", item.get("path", ""))
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            elif isinstance(item, bytes):
                audio_b64 = base64.b64encode(item).decode("utf-8")

    elif isinstance(result, str) and os.path.exists(result):
        with open(result, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    if audio_b64 is None:
        result_info = str(type(result))
        if isinstance(result, dict):
            result_info = str(list(result.keys()))
        elif isinstance(result, (list, tuple)):
            result_info = f"list[{len(result)}] first={type(result[0]) if result else 'empty'}"
        return {"error": f"Could not extract audio from result. Type: {result_info}", "raw_keys": result_info}, 500

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

    server = HTTPServer(("0.0.0.0", 8888), AceStepHandler)
    print("[ACE-Step] Server ready on :8888")
    server.serve_forever()
