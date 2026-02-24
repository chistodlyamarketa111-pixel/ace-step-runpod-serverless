#!/usr/bin/env python3
"""
ACE-Step v1.5 — RunPod Serverless Handler

Uses the v1.5 API with initialize_service() for proper model loading.
"""

import base64
import io
import os
import sys
import time
import traceback
import tempfile

import runpod

PROJECT_ROOT = os.environ.get("ACESTEP_PROJECT_ROOT", "/app/ace-step")
CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/app/checkpoints")
DEFAULT_DIT_MODEL = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")

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

dit_handlers = {}
llm_handler = None


def get_dit_handler(model_name: str):
    global dit_handlers
    if model_name in dit_handlers:
        return dit_handlers[model_name]

    from acestep.handler import AceStepHandler

    dit_path = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.exists(dit_path):
        raise ValueError(f"Model '{model_name}' not found at {dit_path}. Available: {list(VALID_MODELS)}")

    print(f"[ACE-Step] Loading DiT model: {model_name}")
    start = time.time()

    handler = AceStepHandler()
    handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path=model_name,
        device="cuda",
    )

    elapsed = time.time() - start
    print(f"[ACE-Step] DiT model '{model_name}' loaded in {elapsed:.1f}s")

    dit_handlers[model_name] = handler
    return handler


def get_llm_handler():
    global llm_handler
    if llm_handler is not None:
        return llm_handler

    lm_path = os.path.join(CHECKPOINT_DIR, LM_MODEL)
    if not os.path.exists(lm_path):
        print(f"[ACE-Step] LM model not found at {lm_path}, CoT disabled")
        return None

    from acestep.llm_inference import LLMHandler

    print(f"[ACE-Step] Loading LLM: {LM_MODEL}")
    start = time.time()

    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=CHECKPOINT_DIR,
        lm_model_path=LM_MODEL,
        backend="pt",
        device="cuda",
    )

    elapsed = time.time() - start
    print(f"[ACE-Step] LLM loaded in {elapsed:.1f}s")
    return llm_handler


def load_default_models():
    print(f"[ACE-Step] Project root: {PROJECT_ROOT}")
    print(f"[ACE-Step] Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"[ACE-Step] Contents of checkpoint dir: {os.listdir(CHECKPOINT_DIR) if os.path.exists(CHECKPOINT_DIR) else 'NOT FOUND'}")

    get_dit_handler(DEFAULT_DIT_MODEL)
    get_llm_handler()

    available = [m for m in VALID_MODELS if os.path.exists(os.path.join(CHECKPOINT_DIR, m))]
    print(f"[ACE-Step] Available models: {available}")


def handler(job):
    try:
        job_input = job["input"]

        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        model_name = job_input.get("model", DEFAULT_DIT_MODEL)
        if model_name not in VALID_MODELS:
            return {"error": f"Invalid model '{model_name}'. Available: {list(VALID_MODELS)}"}

        current_dit = get_dit_handler(model_name)
        current_llm = get_llm_handler()

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

        lm_temperature = float(job_input.get("lm_temperature", 0.85))
        lm_cfg_scale = float(job_input.get("lm_cfg_scale", 2.0))

        print(f"[ACE-Step] Job {job['id'][:12]}: model={model_name}, prompt='{prompt[:80]}', "
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
            thinking=thinking if current_llm is not None else False,
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

        save_dir = tempfile.mkdtemp(prefix="ace_step_")

        start = time.time()
        result = generate_music(
            dit_handler=current_dit,
            llm_handler=current_llm,
            params=params,
            config=config,
            save_dir=save_dir,
        )
        gen_time = time.time() - start

        if not result.success:
            return {"error": result.error or "Generation failed", "status_message
