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
        task_
