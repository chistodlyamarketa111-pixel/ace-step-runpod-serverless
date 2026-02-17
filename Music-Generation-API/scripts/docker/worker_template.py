#!/usr/bin/env python3
"""
RunPod Serverless Worker Template
=================================
Copy this file and modify it to add a new music generation engine.

Steps:
  1. Copy this file: cp worker_template.py my_engine_worker.py
  2. Implement generate_audio() with your model's inference logic
  3. (Optional) Implement the pipeline mode if your engine supports post-processing
  4. Build Docker image: see Dockerfile.example
  5. Push to Docker Hub and create RunPod Serverless Endpoint
  6. Add TypeScript client + engine class in the main project

Modes supported:
  - "generate": Generate audio from prompt/lyrics
  - "pipeline": Generate + Demucs + remix + mastering (uses shared_utils)
"""

import os
import sys

sys.path.insert(0, "/workspace")

import runpod
from shared_utils import (
    OUTPUTS_DIR,
    file_to_base64,
    base64_to_file,
    get_content_type,
    run_demucs,
    run_mastering,
    remix_stems,
    build_response,
)

# ============================================================
# MODEL LOADING — runs once when worker starts (outside handler)
# Put expensive model loading here to avoid reloading on each request
# ============================================================

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/MyModel")

# Example:
# import torch
# from my_model import MyMusicModel
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = MyMusicModel.from_pretrained(MODEL_DIR).to(device)
# print(f"[MyEngine] Model loaded on {device}")


# ============================================================
# GENERATION — implement your model's inference here
# ============================================================

def generate_audio(job_id: str, params: dict) -> tuple[str, str]:
    """
    Generate audio using your model.

    Args:
        job_id: Unique job identifier
        params: Dict with keys like prompt, lyrics, duration, seed, style, etc.

    Returns:
        tuple of (output_file_path, logs_string)

    Raises:
        Exception if generation fails
    """
    prompt = params.get("prompt", "instrumental music")
    lyrics = params.get("lyrics", "")
    duration = params.get("duration", 120)
    seed = params.get("seed", -1)

    output_path = os.path.join(OUTPUTS_DIR, f"output_{job_id}.wav")

    # TODO: Replace with your model's inference code
    # Example:
    # audio = model.generate(prompt=prompt, lyrics=lyrics, duration=duration, seed=seed)
    # torchaudio.save(output_path, audio.cpu(), 44100)

    raise NotImplementedError(
        "Implement generate_audio() with your model's inference logic. "
        "See diffrhythm_serverless_worker.py for a working example."
    )

    logs = f"Generated {duration}s of audio for prompt: {prompt[:80]}"
    return output_path, logs


# ============================================================
# HANDLER — routes requests to the right function
# Usually no changes needed here
# ============================================================

def handler(job):
    job_input = job["input"]
    mode = job_input.get("mode", "generate")
    job_id = job["id"][:12]

    if mode == "generate":
        output_path, logs = generate_audio(job_id, job_input)
        return build_response(output_path, logs)

    elif mode == "pipeline":
        runpod.serverless.progress_update(job, "Step 1/3: Generating audio...")
        output_path, logs = generate_audio(job_id, job_input)

        runpod.serverless.progress_update(job, "Step 2/3: Demucs stem separation...")
        try:
            stems = run_demucs(output_path, job_input.get("demucs_model", "htdemucs"))
            volumes = {
                "vocals": job_input.get("vol_vocals", 1.0),
                "drums": job_input.get("vol_drums", 1.0),
                "bass": job_input.get("vol_bass", 1.0),
                "other": job_input.get("vol_other", 1.0),
            }
            source_for_master = remix_stems(stems, volumes)
        except Exception as e:
            logs += f"\n[pipeline] Demucs/remix failed: {e}, using raw audio"
            source_for_master = output_path

        runpod.serverless.progress_update(job, "Step 3/3: Mastering...")
        try:
            final_path = run_mastering(source_for_master)
        except Exception as e:
            logs += f"\n[pipeline] Mastering failed: {e}, returning raw"
            final_path = output_path

        return build_response(final_path, logs)

    else:
        return {"error": f"Unknown mode: {mode}"}


runpod.serverless.start({"handler": handler})
