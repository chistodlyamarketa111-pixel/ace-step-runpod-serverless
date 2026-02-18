#!/usr/bin/env python3
"""
ACE-Step 1.5 RunPod Serverless Worker
Handles music generation with ACE-Step model.

Setup Docker image:
  FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

  RUN pip install ace-step runpod

  COPY ace_step_serverless_worker.py /workspace/handler.py
  CMD ["python", "/workspace/handler.py"]

Or build with uv:
  FROM python:3.11-slim
  RUN pip install uv
  RUN uv pip install --system ace-step runpod torch torchaudio
  COPY ace_step_serverless_worker.py /workspace/handler.py
  CMD ["python", "/workspace/handler.py"]
"""

import base64
import io
import os
import time
import traceback

import runpod
import torch
import torchaudio

model = None


def get_model():
    global model
    if model is None:
        print("[ACE-Step] Loading model...")
        start = time.time()
        from acestep.api import ACEStep
        model = ACEStep()
        elapsed = time.time() - start
        print(f"[ACE-Step] Model loaded in {elapsed:.1f}s")
    return model


def handler(job):
    try:
        job_input = job["input"]

        prompt = job_input.get("prompt", "pop music")
        lyrics = job_input.get("lyrics", "")
        duration = job_input.get("audio_duration", job_input.get("duration", 30))
        task_type = job_input.get("task_type", "text2music")
        audio_format = job_input.get("audio_format", "mp3")
        seed = job_input.get("seed", -1)
        inference_steps = job_input.get("inference_steps", job_input.get("infer_step", 27))
        guidance_scale = job_input.get("guidance_scale", 3.0)
        thinking = job_input.get("thinking", True)
        batch_size = job_input.get("batch_size", 1)
        bpm = job_input.get("bpm", None)
        key_scale = job_input.get("key_scale", None)
        time_signature = job_input.get("time_signature", None)
        vocal_language = job_input.get("vocal_language", None)
        model_name = job_input.get("model", None)

        print(f"[ACE-Step] Generating: prompt='{prompt[:80]}', duration={duration}s, steps={inference_steps}")

        ace = get_model()

        if seed >= 0:
            torch.manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "audio_duration": float(duration),
            "infer_step": inference_steps,
            "guidance_scale": guidance_scale,
        }

        if lyrics and lyrics.strip():
            kwargs["lyrics"] = lyrics
        if thinking is not None:
            kwargs["thinking"] = thinking
        if batch_size > 1:
            kwargs["batch_size"] = batch_size

        start = time.time()
        audio_out = ace.infer(**kwargs)
        gen_time = time.time() - start
        print(f"[ACE-Step] Generation complete in {gen_time:.1f}s")

        if hasattr(audio_out, 'audio') and audio_out.audio is not None:
            audio_tensor = audio_out.audio
            sample_rate = getattr(audio_out, 'sample_rate', 44100)
        elif isinstance(audio_out, torch.Tensor):
            audio_tensor = audio_out
            sample_rate = 44100
        elif isinstance(audio_out, tuple) and len(audio_out) == 2:
            audio_tensor, sample_rate = audio_out
        else:
            audio_tensor = audio_out
            sample_rate = 44100

        if isinstance(audio_tensor, list):
            audio_tensor = audio_tensor[0]

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        buf = io.BytesIO()
        if audio_format == "wav":
            torchaudio.save(buf, audio_tensor.cpu(), sample_rate, format="wav")
            content_type = "audio/wav"
            filename = f"ace_step_{job['id'][:12]}.wav"
        else:
            torchaudio.save(buf, audio_tensor.cpu(), sample_rate, format="mp3")
            content_type = "audio/mpeg"
            filename = f"ace_step_{job['id'][:12]}.mp3"

        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "audio_base64": audio_b64,
            "content_type": content_type,
            "filename": filename,
            "generation_time": round(gen_time, 1),
            "duration": duration,
            "sample_rate": sample_rate,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()[-1000:]}


runpod.serverless.start({"handler": handler})
