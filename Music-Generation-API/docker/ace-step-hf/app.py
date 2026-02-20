#!/usr/bin/env python3
"""
ACE-Step v1.5 — FastAPI server for Hugging Face Inference Endpoints.
Exposes /generate (sync), /generate-async + /job/{id} (async), /health.
Port 80 (HF default).
"""

import base64
import gc
import glob
import json
import os
import time
import traceback
import threading
import uuid

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="ACE-Step v1.5")

CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/app/checkpoints")
CPU_OFFLOAD = os.environ.get("ACESTEP_CPU_OFFLOAD", "true").lower() == "true"

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
async_jobs: dict = {}
generation_lock = threading.Lock()


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
        print(f"[ACE-Step] Checkpoint '{model_name}' loaded in {elapsed:.1f}s")
        current_model = model_name

    return pipeline


def handle_generate(data: dict) -> tuple[dict, int]:
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

    manual_seeds = [seed] * batch_size if seed >= 0 else None

    save_dir = f"/tmp/ace_output/{int(time.time()*1000)}"
    os.makedirs(save_dir, exist_ok=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"[ACE-Step] Generate: model={model_name}, prompt='{prompt[:80]}', "
          f"duration={duration}s, steps={infer_step}")

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
    except TypeError:
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
    gen_time = time.time() - start
    print(f"[ACE-Step] Done in {gen_time:.1f}s")

    audio_b64 = None
    audio_files = glob.glob(os.path.join(save_dir, f"*.{audio_format}"))
    if not audio_files:
        audio_files = (glob.glob(os.path.join(save_dir, "*.wav")) +
                       glob.glob(os.path.join(save_dir, "*.mp3")) +
                       glob.glob(os.path.join(save_dir, "*.flac")))
    if not audio_files:
        audio_files = [f for f in glob.glob(os.path.join(save_dir, "**/*"), recursive=True)
                       if os.path.isfile(f) and os.path.getsize(f) > 1000]

    if audio_files:
        found_path = audio_files[0]
        with open(found_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = found_path.rsplit(".", 1)[-1] if "." in found_path else audio_format
        if ext in ("wav", "mp3", "flac"):
            audio_format = ext

    if audio_b64 is None:
        return {"error": "No audio produced"}, 500

    return {
        "model": model_name,
        "audio_base64": audio_b64,
        "audio_format": audio_format,
        "generation_time": round(gen_time, 1),
        "duration": duration,
        "inference_steps": infer_step,
        "seed": seed,
    }, 200


def run_async_generation(job_id: str, data: dict):
    with generation_lock:
        try:
            async_jobs[job_id]["status"] = "IN_PROGRESS"
            result, status_code = handle_generate(data)
            if status_code == 200:
                async_jobs[job_id] = {"status": "COMPLETED", "result": result}
            else:
                async_jobs[job_id] = {"status": "FAILED", "error": result.get("error", "Unknown")}
        except Exception as e:
            traceback.print_exc()
            async_jobs[job_id] = {"status": "FAILED", "error": str(e)}


@app.get("/health")
def health():
    available = [m for m in VALID_MODELS if os.path.exists(os.path.join(CHECKPOINT_DIR, m))]
    try:
        loaded = getattr(pipeline, "loaded", False) if pipeline else False
    except Exception:
        loaded = False
    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        gpu = "unknown"
    try:
        vram = round(torch.cuda.mem_get_info()[0] / 1e9, 1) if torch.cuda.is_available() else 0
    except Exception:
        vram = 0
    return {
        "status": "ok",
        "available_models": available,
        "current_model": current_model,
        "pipeline_loaded": loaded,
        "gpu": gpu,
        "vram_gb": vram,
    }


@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    with generation_lock:
        result, status = handle_generate(data)
    if status != 200:
        raise HTTPException(status_code=status, detail=result.get("error", "Error"))
    return result


@app.post("/generate-async")
async def generate_async(request: Request):
    data = await request.json()
    job_id = str(uuid.uuid4())[:8]
    async_jobs[job_id] = {"status": "QUEUED"}
    thread = threading.Thread(target=run_async_generation, args=(job_id, data), daemon=True)
    thread.start()
    return {"id": job_id, "status": "QUEUED"}


@app.get("/job/{job_id}")
def get_job(job_id: str):
    if job_id not in async_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = async_jobs[job_id]
    if job["status"] == "COMPLETED":
        return {"status": "COMPLETED", "id": job_id, **job.get("result", {})}
    elif job["status"] == "FAILED":
        return {"status": "FAILED", "id": job_id, "error": job.get("error", "Unknown")}
    return {"status": job["status"], "id": job_id}


if __name__ == "__main__":
    print(f"[ACE-Step] Checkpoint dir: {CHECKPOINT_DIR}")

    available = [m for m in VALID_MODELS if os.path.exists(os.path.join(CHECKPOINT_DIR, m))]
    print(f"[ACE-Step] Available models: {available}")

    print(f"[ACE-Step] Pre-loading default model: {DEFAULT_MODEL}...")
    try:
        get_pipeline(DEFAULT_MODEL)
        print("[ACE-Step] Pipeline ready")
    except Exception as e:
        print(f"[ACE-Step] Warning: failed to preload model: {e}")

    port = int(os.environ.get("PORT", "80"))
    print(f"[ACE-Step] Starting server on :{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
