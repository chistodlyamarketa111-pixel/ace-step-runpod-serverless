#!/usr/bin/env python3
"""
ACE-Step v1.5 — HTTP Test Server for RunPod Pod (hourly)
Uses ACEStepPipeline from ace-step package + Python stdlib http.server.
"""

import base64
import glob
import json
import os
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

import gc
import torch

CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/workspace/checkpoints")
CPU_OFFLOAD = os.environ.get("ACESTEP_CPU_OFFLOAD", "false").lower() == "true"

VALID_MODELS = {
    "acestep-v15-turbo",
    "acestep-v15-sft",
    "acestep-v15-base",
}

DEFAULT_STEPS = {
    "acestep-v15-turbo": 8,
    "acestep-v15-sft": 32,
    "acestep-v15-base": 50,
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

    if current_model != model_name:
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
    if audio_format not in ("wav", "mp3", "flac"):
        audio_format = "wav"
    seed = int(data.get("seed", -1))
    default_steps = DEFAULT_STEPS.get(model_name, 8)
    infer_step = int(data.get("inference_steps", default_steps))
    guidance_scale = float(data.get("guidance_scale", 15.0))
    task = data.get("task_type", "text2music")
    batch_size = int(data.get("batch_size", 1))

    manual_seeds = None
    if seed >= 0:
        manual_seeds = [seed] * batch_size

    save_dir = f"/tmp/ace_output/{int(time.time()*1000)}"
    os.makedirs(save_dir, exist_ok=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        total_mem = torch.cuda.mem_get_info()[1] / 1e9
        print(f"[ACE-Step] GPU memory: {free_mem:.1f}GB free / {total_mem:.1f}GB total")

    print(f"[ACE-Step] Generate: model={model_name}, prompt='{prompt[:80]}', "
          f"duration={duration}s, steps={infer_step}, guidance={guidance_scale}, "
          f"save_path={save_dir}")

    start = time.time()
    try:
        result = pipe(
            prompt=prompt,
            lyrics=lyrics,
            audio_duration=duration,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            task=task,
            format=audio_format,
            manual_seeds=manual_seeds,
            batch_size=batch_size,
            save_path=save_dir,
        )
    except TypeError as e:
        print(f"[ACE-Step] __call__ TypeError: {e}, trying with fewer params...")
        try:
            result = pipe(
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=duration,
                infer_step=infer_step,
                guidance_scale=guidance_scale,
                task=task,
                manual_seeds=manual_seeds,
                batch_size=batch_size,
                save_path=save_dir,
            )
        except TypeError as e2:
            print(f"[ACE-Step] __call__ TypeError again: {e2}")
            import inspect
            sig = inspect.signature(pipe.__call__)
            return {"error": f"TypeError: {e2}", "call_signature": str(sig)}, 500
    gen_time = time.time() - start

    print(f"[ACE-Step] Done in {gen_time:.1f}s")

    audio_b64 = None
    found_path = None

    audio_files = glob.glob(os.path.join(save_dir, f"*.{audio_format}"))
    if not audio_files:
        audio_files = glob.glob(os.path.join(save_dir, "*.wav")) + \
                      glob.glob(os.path.join(save_dir, "*.mp3")) + \
                      glob.glob(os.path.join(save_dir, "*.flac"))
    if not audio_files:
        audio_files = glob.glob(os.path.join(save_dir, "**/*"), recursive=True)
        audio_files = [f for f in audio_files if os.path.isfile(f) and os.path.getsize(f) > 1000]

    if audio_files:
        found_path = audio_files[0]
        with open(found_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = found_path.rsplit(".", 1)[-1] if "." in found_path else audio_format
        if ext in ("wav", "mp3", "flac"):
            audio_format = ext
        print(f"[ACE-Step] Found audio file: {found_path} ({os.path.getsize(found_path)} bytes)")

    if audio_b64 is None and result is not None:
        if isinstance(result, (list, tuple)) and len(result) > 0:
            item = result[0]
            if isinstance(item, str) and os.path.exists(item):
                with open(item, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                found_path = item
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                sub = item[0]
                if isinstance(sub, str) and os.path.exists(sub):
                    with open(sub, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                    found_path = sub

    if audio_b64 is None:
        all_files = []
        for root, dirs, files in os.walk(save_dir):
            for fname in files:
                fp = os.path.join(root, fname)
                all_files.append(f"{fp} ({os.path.getsize(fp)}b)")
        result_info = f"type={type(result).__name__}"
        if isinstance(result, (list, tuple)):
            result_info += f" len={len(result)}"
            if result:
                result_info += f" first_type={type(result[0]).__name__}"
                if isinstance(result[0], (list, tuple)) and result[0]:
                    result_info += f" inner={type(result[0][0]).__name__}"
        return {
            "error": f"No audio produced. {result_info}",
            "save_dir_contents": all_files,
        }, 500

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

    import socket
    port = int(os.environ.get("PORT", "8888"))

    class ReusableHTTPServer(HTTPServer):
        allow_reuse_address = True
        allow_reuse_port = True
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            super().server_bind()

    server = ReusableHTTPServer(("0.0.0.0", port), AceStepHandler)
    print(f"[ACE-Step] Server ready on :{port}")
    server.serve_forever()
