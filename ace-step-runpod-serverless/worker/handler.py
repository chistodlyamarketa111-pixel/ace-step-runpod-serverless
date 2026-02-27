#!/usr/bin/env python3
"""
ACE-Step v1.5 — RunPod Serverless Handler
LAZY LOADING: Models load on first request, not at startup.
Supports LoRA adapters for style customization.
"""

import base64
import io
import os
import sys
import time
import traceback
import tempfile

print("[ACE-Step] Handler starting (lazy loading mode)...", flush=True)

import runpod
import torch
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/app/checkpoints")
PROJECT_ROOT = os.environ.get("ACESTEP_PROJECT_ROOT", "/app")
DEFAULT_MODEL = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")
LORA_DIR = os.environ.get("ACESTEP_LORA_DIR", "/app/loras")
NETWORK_VOLUME_LORA_DIR = os.environ.get("NETWORK_VOLUME_LORA_DIR", "/runpod-volume/loras")

dit_handler = None
llm_handler = None
models_loaded = False
current_lora = None
last_init_error = None

DEFAULT_STEPS = {
    "acestep-v15-turbo": 8,
    "acestep-v15-sft": 32,
    "acestep-v15-base": 50,
    "acestep-v15-turbo-shift3": 8,
}


def _scan_lora_dir(directory, loras):
    if not os.path.exists(directory):
        return
    for name in os.listdir(directory):
        lora_path = os.path.join(directory, name)
        if os.path.isdir(lora_path):
            has_adapter = (
                os.path.exists(os.path.join(lora_path, "adapter_config.json"))
                or os.path.exists(os.path.join(lora_path, "lokr_weights.safetensors"))
            )
            has_weights = (
                os.path.exists(os.path.join(lora_path, "adapter_model.safetensors"))
                or os.path.exists(os.path.join(lora_path, "adapter_model.bin"))
                or os.path.exists(os.path.join(lora_path, "lokr_weights.safetensors"))
            )
            if has_adapter and has_weights:
                source = "volume" if directory == NETWORK_VOLUME_LORA_DIR else "builtin"
                loras[name] = {"path": lora_path, "source": source}
                files = os.listdir(lora_path)
                print(f"[ACE-Step] Found LoRA: {name} -> {lora_path} ({source}), files: {files}", flush=True)


def scan_available_loras():
    loras = {}
    _scan_lora_dir(LORA_DIR, loras)
    _scan_lora_dir(NETWORK_VOLUME_LORA_DIR, loras)
    return loras


def apply_lora(lora_name, lora_scale=1.0):
    """Load/unload LoRA adapter. Returns (True, None) on success, (False, error_msg) on failure."""
    global dit_handler, current_lora

    if not lora_name or lora_name == "none":
        if current_lora:
            try:
                result = dit_handler.unload_lora()
                print(f"[ACE-Step] Unloaded LoRA: {current_lora}, result: {result}", flush=True)
                current_lora = None
            except Exception as e:
                print(f"[ACE-Step] Error unloading LoRA: {e}", flush=True)
                traceback.print_exc()
        return True, None

    if current_lora == lora_name:
        print(f"[ACE-Step] LoRA already loaded: {lora_name}", flush=True)
        return True, None

    available = scan_available_loras()
    if lora_name not in available:
        msg = f"LoRA not found: {lora_name}. Available: {list(available.keys())}"
        print(f"[ACE-Step] {msg}", flush=True)
        return False, msg

    try:
        if current_lora:
            try:
                dit_handler.unload_lora()
                print(f"[ACE-Step] Unloaded previous LoRA: {current_lora}", flush=True)
            except Exception as e:
                print(f"[ACE-Step] Warning: unload_lora error: {e}", flush=True)

        lora_info = available[lora_name]
        lora_path = lora_info["path"]
        print(f"[ACE-Step] Loading LoRA: {lora_name} (scale={lora_scale}) from {lora_path} ({lora_info['source']})", flush=True)
        print(f"[ACE-Step] LoRA dir contents: {os.listdir(lora_path)}", flush=True)

        result = dit_handler.load_lora(lora_path)
        print(f"[ACE-Step] load_lora result: {result}", flush=True)

        if isinstance(result, str) and "❌" in result:
            return False, f"load_lora returned: {result}"

        if lora_scale != 1.0:
            adapter_name = getattr(dit_handler, '_lora_active_adapter', None) or lora_name
            if hasattr(dit_handler, 'set_lora_scale'):
                try:
                    dit_handler.set_lora_scale(adapter_name, lora_scale)
                    print(f"[ACE-Step] Set LoRA scale to {lora_scale} for adapter '{adapter_name}'", flush=True)
                except Exception as e:
                    print(f"[ACE-Step] Warning: set_lora_scale failed: {e}", flush=True)

        current_lora = lora_name
        print(f"[ACE-Step] LoRA loaded successfully: {lora_name}", flush=True)
        return True, None
    except Exception as e:
        msg = f"Error loading LoRA {lora_name}: {e}\n{traceback.format_exc()[-1000:]}"
        print(f"[ACE-Step] {msg}", flush=True)
        current_lora = None
        return False, msg


def ensure_models_loaded():
    global dit_handler, llm_handler, models_loaded, last_init_error

    if models_loaded:
        return True

    print(f"[ACE-Step] Loading models (first request)...", flush=True)
    print(f"[ACE-Step] Config: PROJECT_ROOT={PROJECT_ROOT}, CHECKPOINT_DIR={CHECKPOINT_DIR}, DEFAULT_MODEL={DEFAULT_MODEL}", flush=True)
    start = time.time()

    checkpoint_model_dir = os.path.join(CHECKPOINT_DIR, DEFAULT_MODEL)
    print(f"[ACE-Step] Expected model dir: {checkpoint_model_dir}, exists: {os.path.exists(checkpoint_model_dir)}", flush=True)
    if os.path.exists(CHECKPOINT_DIR):
        print(f"[ACE-Step] Checkpoint dir contents: {os.listdir(CHECKPOINT_DIR)}", flush=True)
    if os.path.exists(checkpoint_model_dir):
        print(f"[ACE-Step] Model dir contents: {os.listdir(checkpoint_model_dir)[:20]}", flush=True)

    try:
        print(f"[ACE-Step] Creating AceStepHandler...", flush=True)
        dit_handler = AceStepHandler()
        print(f"[ACE-Step] AceStepHandler created OK", flush=True)

        print(f"[ACE-Step] Calling initialize_service(project_root={PROJECT_ROOT}, config_path={DEFAULT_MODEL}, device=cuda)...", flush=True)
        result = dit_handler.initialize_service(
            project_root=PROJECT_ROOT,
            config_path=DEFAULT_MODEL,
            device="cuda",
        )

        if isinstance(result, tuple) and len(result) == 2:
            status_msg, success = result
            print(f"[ACE-Step] initialize_service: success={success}", flush=True)
            print(f"[ACE-Step] initialize_service status: {str(status_msg)[:500]}", flush=True)
            if not success:
                last_init_error = f"initialize_service failed: {str(status_msg)[:500]}"
                print(f"[ACE-Step] FATAL: {last_init_error}", flush=True)
                dit_handler = None
                return False
        else:
            print(f"[ACE-Step] initialize_service returned unexpected type: {type(result)}: {str(result)[:300]}", flush=True)
    except Exception as e:
        last_init_error = f"DiT init ERROR: {str(e)}"
        print(f"[ACE-Step] {last_init_error}", flush=True)
        traceback.print_exc()
        dit_handler = None
        return False

    try:
        lm_path = os.path.join(CHECKPOINT_DIR, LM_MODEL)
        print(f"[ACE-Step] LM model path: {lm_path}, exists: {os.path.exists(lm_path)}", flush=True)
        if os.path.exists(lm_path):
            print(f"[ACE-Step] Loading LLM from {lm_path}...", flush=True)
            from acestep.llm_inference import LLMHandler
            llm_handler = LLMHandler()
            llm_result = llm_handler.initialize(
                checkpoint_dir=CHECKPOINT_DIR,
                lm_model_path=LM_MODEL,
                backend="pt",
                device="cuda",
            )
            if isinstance(llm_result, tuple) and len(llm_result) == 2:
                llm_status, llm_success = llm_result
                print(f"[ACE-Step] LLM initialize: success={llm_success}, status={str(llm_status)[:200]}", flush=True)
                if not llm_success:
                    print(f"[ACE-Step] LLM init failed (non-fatal): {llm_status}", flush=True)
                    llm_handler = None
            else:
                print(f"[ACE-Step] LLM initialize returned: {str(llm_result)[:200]}", flush=True)
        else:
            print(f"[ACE-Step] LM model not found at {lm_path}, skipping", flush=True)
            llm_handler = None
    except Exception as e:
        print(f"[ACE-Step] LLM init ERROR (non-fatal): {e}", flush=True)
        traceback.print_exc()
        llm_handler = None

    available_loras = scan_available_loras()
    print(f"[ACE-Step] Available LoRAs: {list(available_loras.keys()) if available_loras else 'none'}", flush=True)

    if os.path.exists(LORA_DIR):
        print(f"[ACE-Step] LoRA dir contents: {os.listdir(LORA_DIR)}", flush=True)

    elapsed = time.time() - start
    print(f"[ACE-Step] Models loaded in {elapsed:.1f}s (LLM: {'yes' if llm_handler else 'no'})", flush=True)
    models_loaded = True
    last_init_error = None
    return True


def handler(job):
    global dit_handler, llm_handler

    try:
        if not ensure_models_loaded():
            return {"error": f"Failed to load models: {last_init_error or 'unknown error'}"}

        job_input = job["input"]

        if job_input.get("action") == "list_loras":
            available = scan_available_loras()
            return {
                "loras": [
                    {"name": name, "source": info["source"], "path": info["path"]}
                    for name, info in available.items()
                ],
                "current_lora": current_lora,
                "lora_dirs": {
                    "builtin": LORA_DIR,
                    "network_volume": NETWORK_VOLUME_LORA_DIR,
                },
            }

        if job_input.get("action") == "debug_model":
            decoder = dit_handler.model.decoder if dit_handler.model else None
            module_names = []
            if decoder:
                for name, _ in decoder.named_modules():
                    module_names.append(name)
            attn_modules = [n for n in module_names if any(k in n for k in ["to_k", "to_q", "to_v", "to_out", "attn", "qkv", "proj"])]
            return {
                "decoder_type": str(type(decoder)) if decoder else None,
                "total_modules": len(module_names),
                "attn_modules_sample": attn_modules[:50],
                "all_leaf_names": list(set(n.split(".")[-1] for n in module_names if n))[:100],
                "pytorch_version": torch.__version__,
                "lora_methods": [m for m in dir(dit_handler) if "lora" in m.lower()],
            }

        model_name = job_input.get("model", DEFAULT_MODEL)
        prompt = job_input.get("prompt", "")
        lyrics = job_input.get("lyrics", "")
        duration = float(job_input.get("audio_duration", job_input.get("duration", -1)))
        task_type = job_input.get("task_type", "text2music")
        audio_format = job_input.get("audio_format", "mp3")
        seed = int(job_input.get("seed", -1))
        default_steps = DEFAULT_STEPS.get(model_name, 8)
        inference_steps = int(job_input.get("inference_steps", job_input.get("infer_step", job_input.get("num_inference_steps", default_steps))))
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

        lora_name = job_input.get("lora_name", None)
        lora_scale = float(job_input.get("lora_scale", 1.0))

        if model_name != DEFAULT_MODEL:
            if current_lora:
                apply_lora(None)
            model_dir = os.path.join(CHECKPOINT_DIR, model_name)
            if os.path.exists(model_dir):
                print(f"[ACE-Step] Switching model to {model_name}...", flush=True)
                try:
                    result = dit_handler.initialize_service(
                        project_root=PROJECT_ROOT,
                        config_path=model_name,
                        device="cuda",
                    )
                    if isinstance(result, tuple) and len(result) == 2:
                        _, success = result
                        if not success:
                            print(f"[ACE-Step] Model switch failed, using {DEFAULT_MODEL}", flush=True)
                            model_name = DEFAULT_MODEL
                except Exception as e:
                    print(f"[ACE-Step] Model switch error: {e}", flush=True)
                    model_name = DEFAULT_MODEL

        if lora_name and lora_name != "none":
            ok, err = apply_lora(lora_name, lora_scale)
            if not ok:
                return {"error": f"Failed to load LoRA: {err}"}
        elif current_lora and (not lora_name or lora_name == "none"):
            apply_lora(None)

        lora_info = f", lora={lora_name}(x{lora_scale})" if lora_name and lora_name != "none" else ""
        print(f"[ACE-Step] Job {job['id'][:12]}: model={model_name}, prompt='{prompt[:80]}', "
              f"duration={duration}s, steps={inference_steps}{lora_info}", flush=True)

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
                    "sample_rate": audio_info.get("sample_rate", 48000),
                    "model": model_name,
                    "lora": lora_name if lora_name and lora_name != "none" else None,
                }
            elif audio_info.get("tensor") is not None:
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
                    "lora": lora_name if lora_name and lora_name != "none" else None,
                }

        return {"error": "No audio files generated"}

    except Exception as e:
        print(f"[ACE-Step] Job error: {e}", flush=True)
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()[-2000:]}


print("[ACE-Step] Worker starting (models will load on first request)...", flush=True)
runpod.serverless.start({"handler": handler})
