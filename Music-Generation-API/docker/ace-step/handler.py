#!/usr/bin/env python3
"""
ACE-Step v1.5 — RunPod Serverless Handler

Loads the ACE-Step v1.5 model (DiT + LM) and processes music generation
requests via RunPod's serverless infrastructure.

Supports:
  - text2music: Generate music from text prompt + lyrics
  - cover: Generate a cover of a reference audio
  - repaint: Repaint a section of existing audio
"""

import base64
import io
import os
import time
import traceback

import runpod
import torch
import soundfile as sf

CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/app/checkpoints")
DIT_MODEL = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")
CPU_OFFLOAD = os.environ.get("ACESTEP_CPU_OFFLOAD", "false").lower() == "true"

dit_handler = None
llm_handler = None


def load_models():
    global dit_handler, llm_handler

    if dit_handler is not None:
        return

    print(f"[ACE-Step] Loading models from {CHECKPOINT_DIR}")
    print(f"[ACE-Step] DiT model: {DIT_MODEL}, LM model: {LM_MODEL}")
    start = time.time()

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.gpu_config import get_gpu_config, set_global_gpu_config

    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    print(f"[ACE-Step] GPU config: {gpu_config}")

    dit_path = os.path.join(CHECKPOINT_DIR, DIT_MODEL)
    dit_handler = AceStepHandler(
        checkpoint_dir=CHECKPOINT_DIR,
        dit_model_path=dit_path,
        cpu_offload=CPU_OFFLOAD,
    )

    lm_path = os.path.join(CHECKPOINT_DIR, LM_MODEL)
    if os.path.exists(lm_path):
        llm_handler = LLMHandler(
            model_path=lm_path,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        print(f"[ACE-Step] LLM handler loaded from {lm_path}")
    else:
        llm_handler = None
        print(f"[ACE-Step] LM model not found at {lm_path}, CoT disabled")

    elapsed = time.time() - start
    print(f"[ACE-Step] Models loaded in {elapsed:.1f}s")


def handler(job):
    try:
        load_models()

        job_input = job["input"]

        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        prompt = job_input.get("prompt", "")
        lyrics = job_input.get("lyrics", "")
        duration = float(job_input.get("audio_duration", job_input.get("duration", -1)))
        task_type = job_input.get("task_type", "text2music")
        audio_format = job_input.get("audio_format", "mp3")
        seed = int(job_input.get("seed", -1))
        inference_steps = int(job_input.get("inference_steps", job_input.get("infer_step", 8)))
        guidance_scale = float(job_input.get("guidance_scale", 7.0))
        thinking = job_input.get("thinking", True)
        batch_size = int(job_input.get("batch_size", 1))

        bpm = job_input.get("bpm", None)
        if bpm is not None:
            bpm = int(bpm)
        key_scale = job_input.get("key_scale", job_input.get("keyscale", ""))
        time_signature = job_input.get("time_signature", job_input.get("timesignature", ""))
        vocal_language = job_input.get("vocal_language", "unknown")
        instrumental = job_input.get("instrumental", False)

        lm_temperature = float(job_input.get("lm_temperature", 0.85))
        lm_cfg_scale = float(job_input.get("lm_cfg_scale", 2.0))

        print(f"[ACE-Step] Job {job['id'][:12]}: prompt='{prompt[:80]}', "
              f"duration={duration}s, steps={inference_steps}, task={task_type}, "
              f"thinking={thinking}, batch={batch_size}")

        params = GenerationParams(
            caption=prompt,
            lyrics=lyrics,
            duration=duration,
            task_type=task_type,
            seed=seed,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            thinking=thinking if llm_handler is not None else False,
            bpm=bpm,
            keyscale=key_scale,
            timesignature=time_signature,
            vocal_language=vocal_language,
            instrumental=instrumental,
            lm_temperature=lm_temperature,
            lm_cfg_scale=lm_cfg_scale,
        )

        config = GenerationConfig(
            batch_size=batch_size,
            use_random_seed=(seed < 0),
            seeds=[seed] if seed >= 0 else None,
            audio_format=audio_format if audio_format in ("mp3", "wav", "flac") else "mp3",
        )

        start = time.time()
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
        )
        gen_time = time.time() - start

        if not result.success:
            return {"error": result.error or "Generation failed", "status_message": result.status_message}

        print(f"[ACE-Step] Generation complete in {gen_time:.1f}s, {len(result.audios)} audio(s)")

        content_type_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "flac": "audio/flac",
            "opus": "audio/opus",
            "aac": "audio/aac",
        }

        audios_output = []
        for i, audio_info in enumerate(result.audios):
            audio_path = audio_info.get("path", audio_info.get("audio_path", ""))
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
            elif "audio" in audio_info and audio_info["audio"] is not None:
                import torchaudio
                tensor = audio_info["audio"]
                sr = audio_info.get("sample_rate", 44100)
                buf = io.BytesIO()
                torchaudio.save(buf, tensor.cpu(), sr, format=audio_format)
                audio_bytes = buf.getvalue()
            else:
                print(f"[ACE-Step] Skipping audio {i}: no path or tensor. Keys: {list(audio_info.keys())}")
                continue

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            ext = os.path.splitext(audio_path)[1].lstrip(".") if audio_path else audio_format

            audios_output.append({
                "audio_base64": audio_b64,
                "content_type": content_type_map.get(ext, "audio/mpeg"),
                "filename": f"ace_step_{job['id'][:12]}_{i}.{ext}",
                "index": i,
            })

        if not audios_output:
            return {"error": "No audio files generated"}

        response = {
            "audio_base64": audios_output[0]["audio_base64"],
            "content_type": audios_output[0]["content_type"],
            "filename": audios_output[0]["filename"],
            "generation_time": round(gen_time, 1),
            "duration": duration,
            "sample_rate": 44100,
            "status_message": result.status_message,
        }

        if len(audios_output) > 1:
            response["batch_audios"] = audios_output

        extra = result.extra_outputs or {}
        if extra:
            response["extra"] = {
                k: v for k, v in extra.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }

        return response

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()[-2000:]}


print("[ACE-Step] Starting RunPod Serverless worker...")
load_models()
print("[ACE-Step] Worker ready, waiting for jobs...")
runpod.serverless.start({"handler": handler})
