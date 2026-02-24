#!/usr/bin/env python3
"""
ACE-Step v1.5 — RunPod Serverless Handler
Minimal version with detailed error logging for debugging.
"""

import base64
import io
import os
import sys
import time
import traceback
import tempfile

print("[ACE-Step] Handler starting...", flush=True)
print(f"[ACE-Step] Python: {sys.version}", flush=True)
print(f"[ACE-Step] CWD: {os.getcwd()}", flush=True)

try:
    import runpod
    print(f"[ACE-Step] runpod imported OK: {runpod.__version__}", flush=True)
except Exception as e:
    print(f"[ACE-Step] FATAL: Cannot import runpod: {e}", flush=True)
    sys.exit(1)

try:
    import torch
    print(f"[ACE-Step] torch imported OK: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[ACE-Step] GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", flush=True)
except Exception as e:
    print(f"[ACE-Step] FATAL: Cannot import torch: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/app/checkpoints")
PROJECT_ROOT = os.environ.get("ACESTEP_PROJECT_ROOT", "/app")
DEFAULT_MODEL = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")

print(f"[ACE-Step] CHECKPOINT_DIR: {CHECKPOINT_DIR}", flush=True)
print(f"[ACE-Step] PROJECT_ROOT: {PROJECT_ROOT}", flush=True)

if os.path.exists(CHECKPOINT_DIR):
    print(f"[ACE-Step] Checkpoint contents: {os.listdir(CHECKPOINT_DIR)}", flush=True)
else:
    print(f"[ACE-Step] WARNING: CHECKPOINT_DIR does not exist!", flush=True)

ace_step_path = "/app/ace-step"
if os.path.exists(ace_step_path):
    print(f"[ACE-Step] ace-step dir contents: {os.listdir(ace_step_path)}", flush=True)
else:
    print(f"[ACE-Step] WARNING: /app/ace-step does not exist!", flush=True)

try:
    import acestep
    print(f"[ACE-Step] acestep package imported OK from: {acestep.__file__}", flush=True)
except Exception as e:
    print(f"[ACE-Step] FATAL: Cannot import acestep: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

dit_handler = None
llm_handler = None

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


def init_models():
    global dit_handler, llm_handler

    print(f"[ACE-Step] Initializing DiT handler...", flush=True)
    try:
        from acestep.handler import AceStepHandler
        print(f"[ACE-Step] AceStepHandler imported OK", flush=True)

        dit_handler = AceStepHandler()
        print(f"[ACE-Step] AceStepHandler() created OK", flush=True)

        status, success = dit_handler.initialize_service(
            project_root=PROJECT_ROOT,
            config_path=DEFAULT_MODEL,
            device="cuda",
        )
        print(f"[ACE-Step] initialize_service result: success={success}, status={status[:200]}", flush=True)
        if not success:
            print(f"[ACE-Step] WARNING: DiT init failed, but continuing...", flush=True)
    except Exception as e:
        print(f"[ACE-Step] ERROR initializing DiT: {e}", flush=True)
        traceback.print_exc()

    print(f"[ACE-Step] Initializing LLM handler...", flush=True)
    try:
        lm_path = os.path.join(CHECKPOINT_DIR, LM_MODEL)
        if not os.path.exists(lm_path):
            print(f"[ACE-Step] LM model not found at {lm_path}, skipping LLM", flush=True)
            return

        from acestep.llm_inference import LLMHandler
        print(f"[ACE-Step] LLMHandler imported OK", flush=True)

        llm_handler = LLMHandler()
        print(f"[ACE-Step] LLMHandler() created OK", flush=True)

        llm_handler.initialize(
            checkpoint_dir=CHECKPOINT_DIR,
            lm_model_path=LM_MODEL,
            backend="pt",
            device="cuda",
        )
        print(f"[ACE-Step] LLM initialized OK", flush=True)
    except Exception as e:
        print(f"[ACE-Step] ERROR initializing LLM: {e}", flush=True)
        traceback.print_exc()
        llm_handler = None


def handler(job):
    global dit_handler, llm_handler

    try:
        if dit_handler is None:
            return {"error": "DiT handler not initialized"}

        job_input = job["input"]
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        model_name = job_input.get("model", DEFAULT_MODEL)
        prompt = job_input.get("prompt", "")
        lyrics = job_input.get("lyrics", "")
        duration = float(job_input.get("audio_duration", job_input.get("duration", -1)))
        task_type = job_input.get("task_type", "text2music")
        audio_format = job_input.get("audio_format", "mp3")
        seed = int(job_input.get("seed", -1))
        default_steps = DEFAULT_STEPS.get(model_name, 8)
        inference_steps = int(job_input.get("inference_steps", job_input.get("infer_step", default_steps)))
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

        print(f"[ACE-Step] Job {job['id'][:12]}: model={model_name}, prompt='{prompt[:80]}', "
              f"duration={duration}s, steps={inference_steps}", flush=True)

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
        )

        config = GenerationConfig(
            batch_size=batch_size,
            use_random_seed=(seed < 0),
            seeds=[seed] if seed >= 0 else None,
            audio_format=audio_format if audio_format in ("mp3", "wav", "flac") else "mp3",
        )

        save_dir = tempfile.mkdtemp(prefix="ace_step_")

        start = time.time()
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=save_dir,
        )
        gen_time = time.time() - start

        if not result.success:
            return {"error": result.error or "Generation failed", "status_message": result.status_message}

        print(f"[ACE-Step] Done in {gen_time:.1f}s, {len(result.audios)} audio(s)", flush=True)

        for i, audio_info in enumerate(result.audios):
            audio_path = audio_info.get("path", "")
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                ext = os.path.splitext(audio_path)[1].lstrip(".") or audio_format
                content_type_map = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}
                return {
                    "audio_base64": audio_b64,
                    "content_type": content_type_map.get(ext, "audio/mpeg"),
                    "filename": f"ace_step_{job['id'][:12]}_{i}.{ext}",
                    "generation_time": round(gen_time, 1),
                    "duration": duration,
                    "sample_rate": 48000,
                    "model": model_name,
                }
            elif "tensor" in audio_info and audio_info["tensor"] is not None:
                import torchaudio
                tensor = audio_info["tensor"]
                sr = audio_info.get("sample_rate", 48000)
                buf = io.BytesIO()
                torchaudio.save(buf, tensor.cpu(), sr, format=audio_format)
                audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content_type_map = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}
                return {
                    "audio_base64": audio_b64,
                    "content_type": content_type_map.get(audio_format, "audio/mpeg"),
                    "filename": f"ace_step_{job['id'][:12]}_{i}.{audio_format}",
                    "generation_time": round(gen_time, 1),
                    "duration": duration,
                    "sample_rate": sr,
                    "model": model_name,
                }

        return {"error": "No audio files generated"}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()[-2000:]}


print("[ACE-Step] Loading models...", flush=True)
init_models()
print("[ACE-Step] Worker ready!", flush=True)
runpod.serverless.start({"handler": handler})
