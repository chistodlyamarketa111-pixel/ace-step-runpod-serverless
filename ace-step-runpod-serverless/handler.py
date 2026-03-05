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

HANDLER_VERSION = "2026-03-05-v4"
print(f"[ACE-Step] Handler starting (lazy loading mode) version={HANDLER_VERSION}...", flush=True)

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
            config_file = os.path.join(lora_path, "adapter_config.json")
            safetensors = os.path.join(lora_path, "adapter_model.safetensors")
            bin_file = os.path.join(lora_path, "adapter_model.bin")
            lokr_file = os.path.join(lora_path, "lokr_weights.safetensors")
            if os.path.exists(config_file) and (os.path.exists(safetensors) or os.path.exists(bin_file)):
                source = "volume" if directory == NETWORK_VOLUME_LORA_DIR else "builtin"
                loras[name] = {"path": lora_path, "source": source}
                print(f"[ACE-Step] Found LoRA: {name} -> {lora_path} ({source})", flush=True)
            elif os.path.exists(lokr_file):
                source = "volume" if directory == NETWORK_VOLUME_LORA_DIR else "builtin"
                loras[name] = {"path": lora_path, "source": source}
                print(f"[ACE-Step] Found LoKr: {name} -> {lora_path} ({source})", flush=True)


def scan_available_loras():
    loras = {}
    _scan_lora_dir(LORA_DIR, loras)
    _scan_lora_dir(NETWORK_VOLUME_LORA_DIR, loras)
    return loras


HF_LORA_REPO_PREFIX = os.environ.get("HF_LORA_REPO_PREFIX", "ruslanmusinrusmus")


def download_lora_from_hf(lora_name):
    try:
        from huggingface_hub import snapshot_download
        repo_id = f"{HF_LORA_REPO_PREFIX}/{lora_name}"
        target_dir = os.path.join(NETWORK_VOLUME_LORA_DIR, lora_name)
        os.makedirs(NETWORK_VOLUME_LORA_DIR, exist_ok=True)
        print(f"[ACE-Step] Downloading LoRA from HuggingFace: {repo_id} -> {target_dir}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            ignore_patterns=["*.md", ".gitattributes"],
        )
        config_file = os.path.join(target_dir, "adapter_config.json")
        safetensors = os.path.join(target_dir, "adapter_model.safetensors")
        bin_file = os.path.join(target_dir, "adapter_model.bin")
        if os.path.exists(config_file) and (os.path.exists(safetensors) or os.path.exists(bin_file)):
            print(f"[ACE-Step] LoRA downloaded successfully: {lora_name}", flush=True)
            return True
        print(f"[ACE-Step] Downloaded repo missing adapter files: {lora_name}", flush=True)
        return False
    except Exception as e:
        print(f"[ACE-Step] Failed to download LoRA {lora_name} from HF: {e}", flush=True)
        return False


def apply_lora(lora_name, lora_scale=1.0):
    """Load/unload LoRA using ACE-Step's built-in add_lora/unload_lora methods.
    Returns True on success, or error string on failure."""
    global dit_handler, current_lora

    if not lora_name or lora_name == "none":
        if current_lora:
            try:
                result = dit_handler.unload_lora()
                print(f"[ACE-Step] Unloaded LoRA: {current_lora} -> {result}", flush=True)
                current_lora = None
            except Exception as e:
                print(f"[ACE-Step] Error unloading LoRA: {e}", flush=True)
        return True

    if current_lora == lora_name:
        print(f"[ACE-Step] LoRA already loaded: {lora_name}", flush=True)
        return True

    available = scan_available_loras()
    if lora_name not in available:
        print(f"[ACE-Step] LoRA not found locally: {lora_name}. Trying HuggingFace download...", flush=True)
        if download_lora_from_hf(lora_name):
            available = scan_available_loras()
        if lora_name not in available:
            return f"LoRA '{lora_name}' not found. Available: {list(available.keys())}"

    lora_info = available[lora_name]
    lora_path = lora_info["path"]

    try:
        if current_lora:
            try:
                result = dit_handler.unload_lora()
                print(f"[ACE-Step] Unloaded previous LoRA: {current_lora} -> {result}", flush=True)
            except Exception as ue:
                print(f"[ACE-Step] Warning during unload: {ue}", flush=True)

        print(f"[ACE-Step] Loading LoRA via add_lora: {lora_name} from {lora_path} ({lora_info['source']})", flush=True)

        result = dit_handler.add_lora(lora_path, adapter_name=lora_name)
        print(f"[ACE-Step] add_lora result: {result}", flush=True)

        if isinstance(result, str) and result.startswith("❌"):
            current_lora = None
            return f"add_lora failed: {result}"

        dit_handler.use_lora = True
        dit_handler.lora_scale = lora_scale

        if lora_scale != 1.0:
            try:
                dit_handler.set_lora_scale(lora_name, lora_scale)
                print(f"[ACE-Step] Set LoRA scale: {lora_scale}", flush=True)
            except Exception as se:
                print(f"[ACE-Step] Warning setting scale: {se}", flush=True)

        current_lora = lora_name
        print(f"[ACE-Step] LoRA loaded successfully: {lora_name}", flush=True)
        return True
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ACE-Step] Error loading LoRA {lora_name}: {err_msg}", flush=True)
        traceback.print_exc()
        current_lora = None
        return f"Exception: {err_msg}"


def ensure_models_loaded():
    global dit_handler, llm_handler, models_loaded
    if models_loaded:
        return True

    print(f"[ACE-Step] Loading models (first request)...", flush=True)
    start = time.time()

    try:
        print(f"[ACE-Step] Creating AceStepHandler...", flush=True)
        dit_handler = AceStepHandler()
        print(f"[ACE-Step] AceStepHandler created OK", flush=True)

        print(f"[ACE-Step] Calling initialize_service(project_root={PROJECT_ROOT}, config_path={DEFAULT_MODEL})...", flush=True)
        status, success = dit_handler.initialize_service(
            project_root=PROJECT_ROOT,
            config_path=DEFAULT_MODEL,
            device="cuda",
        )
        print(f"[ACE-Step] initialize_service: success={success}, status={status[:300]}", flush=True)
    except Exception as e:
        print(f"[ACE-Step] DiT init ERROR: {e}", flush=True)
        traceback.print_exc()
        dit_handler = None
        return False

    try:
        lm_path = os.path.join(CHECKPOINT_DIR, LM_MODEL)
        if os.path.exists(lm_path):
            print(f"[ACE-Step] Loading LLM from {lm_path}...", flush=True)
            from acestep.llm_inference import LLMHandler
            llm_handler = LLMHandler()
            llm_handler.initialize(
                checkpoint_dir=CHECKPOINT_DIR,
                lm_model_path=LM_MODEL,
                backend="pt",
                device="cuda",
            )
            print(f"[ACE-Step] LLM loaded OK", flush=True)
        else:
            print(f"[ACE-Step] LM model not found at {lm_path}, skipping", flush=True)
    except Exception as e:
        print(f"[ACE-Step] LLM init ERROR (non-fatal): {e}", flush=True)
        traceback.print_exc()
        llm_handler = None

    elapsed = time.time() - start
    print(f"[ACE-Step] Models loaded in {elapsed:.1f}s", flush=True)
    models_loaded = True
    return True


def handler_worker(job):
    try:
        job_input = job.get("input", {})

        action = job_input.get("action")
        if action == "health":
            return {
                "status": "ok",
                "handler_version": HANDLER_VERSION,
                "models_loaded": models_loaded,
                "current_lora": current_lora,
                "available_loras": list(scan_available_loras().keys()),
            }

        if action == "list_loras":
            available = scan_available_loras()
            result = {}
            for name, info in available.items():
                config_path = os.path.join(info["path"], "adapter_config.json")
                peft_type = "unknown"
                if os.path.exists(config_path):
                    import json as _json
                    with open(config_path) as f:
                        cfg = _json.load(f)
                        peft_type = cfg.get("peft_type", "unknown")
                result[name] = {"path": info["path"], "source": info["source"], "peft_type": peft_type}
            return {"loras": result, "current_lora": current_lora, "handler_version": HANDLER_VERSION}

        if action == "diagnose_lora":
            diag_name = job_input.get("lora_name", "russianpop")
            if not ensure_models_loaded():
                return {"error": "Model init failed"}
            available = scan_available_loras()
            diag = {
                "handler_version": HANDLER_VERSION,
                "available_loras": {},
                "model_type": type(dit_handler.model).__name__ if dit_handler.model else "None",
                "decoder_type": type(dit_handler.model.decoder).__name__ if dit_handler.model and hasattr(dit_handler.model, 'decoder') else "None",
                "has_add_lora": hasattr(dit_handler, 'add_lora'),
                "has_load_lora": hasattr(dit_handler, 'load_lora'),
                "current_lora": current_lora,
                "quantization": dit_handler.quantization if hasattr(dit_handler, 'quantization') else "N/A",
            }
            for name, info in available.items():
                files = os.listdir(info["path"])
                diag["available_loras"][name] = {"path": info["path"], "files": files, "source": info["source"]}
            if diag_name in available:
                lora_result = apply_lora(diag_name)
                diag["load_result"] = str(lora_result)
                diag["load_success"] = lora_result is True
            else:
                diag["load_result"] = f"LoRA '{diag_name}' not found"
                diag["load_success"] = False
            return diag

        if not ensure_models_loaded():
            return {"error": "Failed to initialize models. Check worker logs."}

        lora_name = job_input.get("lora_name")
        lora_scale = float(job_input.get("lora_scale", 1.0))

        if lora_name and lora_name != "none":
            lora_result = apply_lora(lora_name, lora_scale)
            if lora_result is not True:
                return {"error": f"Failed to load LoRA '{lora_name}': {lora_result}"}
        elif current_lora and (not lora_name or lora_name == "none"):
            apply_lora(None)

        prompt = job_input.get("prompt", "")
        lyrics = job_input.get("lyrics", "")
        duration = float(job_input.get("audio_duration", job_input.get("duration", -1)))
        model_name = job_input.get("model", DEFAULT_MODEL)
        audio_format = job_input.get("audio_format", "wav")
        inference_steps = int(job_input.get("infer_step", DEFAULT_STEPS.get(model_name, 8)))
        guidance_scale = float(job_input.get("guidance_scale", 15.0))
        guidance_scale_text = float(job_input.get("guidance_scale_text", 0.0))
        guidance_scale_lyric = float(job_input.get("guidance_scale_lyric", 0.0))
        scheduler_type = job_input.get("scheduler_type", "euler")
        cfg_type = job_input.get("cfg_type", "apg")
        omega = float(job_input.get("omega", 10.0))
        granularity = float(job_input.get("granularity", 100.0))
        manual_seeds = job_input.get("manual_seeds", [-1])
        instrumental = job_input.get("instrumental", False)

        lora_info = f", lora={lora_name}(x{lora_scale})" if lora_name and lora_name != "none" else ""
        print(f"[ACE-Step] Job {job['id'][:12]}: model={model_name}, prompt='{prompt[:80]}', "
              f"duration={duration}s, steps={inference_steps}{lora_info}", flush=True)

        params = GenerationParams(
            audio_duration=duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega=omega,
            granularity=granularity,
            manual_seeds=manual_seeds if isinstance(manual_seeds, list) else [manual_seeds],
            guidance_interval=0.5,
            guidance_interval_decay=0.0,
            min_guidance_scale=3.0,
            use_erg_tag=True,
            use_erg_lyric=True,
            use_erg_diffusion=True,
            oss_steps=None,
            instrumental=instrumental,
        )

        config = GenerationConfig(
            project_root=PROJECT_ROOT,
            config_path=model_name,
            seed=-1,
            batch_size=1,
        )

        gen_start = time.time()
        results = generate_music(dit_handler, llm_handler, params, config)
        gen_time = time.time() - gen_start
        print(f"[ACE-Step] Generation done in {gen_time:.1f}s", flush=True)

        if not results:
            return {"error": "No results from generate_music"}

        for audio_info in results:
            if "audio_path" in audio_info and audio_info["audio_path"]:
                audio_path = audio_info["audio_path"]
                if os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        audio_data = f.read()
                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                    return {
                        "audio_base64": audio_b64,
                        "format": audio_format,
                        "duration": duration,
                        "generation_time": round(gen_time, 2),
                        "sample_rate": 48000,
                        "model": model_name,
                        "lora": lora_name if lora_name and lora_name != "none" else None,
                        "handler_version": HANDLER_VERSION,
                    }
            elif "tensor" in audio_info and audio_info["tensor"] is not None:
                import torchaudio
                tensor = audio_info["tensor"]
                sr = audio_info.get("sample_rate", 48000)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                buf = io.BytesIO()
                torchaudio.save(buf, tensor.cpu(), sr, format="wav")
                audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return {
                    "audio_base64": audio_b64,
                    "format": "wav",
                    "duration": duration,
                    "generation_time": round(gen_time, 2),
                    "sample_rate": sr,
                    "model": model_name,
                    "lora": lora_name if lora_name and lora_name != "none" else None,
                    "handler_version": HANDLER_VERSION,
                }

        return {"error": "No audio files generated"}

    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {str(e)}", "handler_version": HANDLER_VERSION}


print(f"[ACE-Step] Handler ready (version={HANDLER_VERSION}), waiting for jobs...", flush=True)
runpod.serverless.start({"handler": handler_worker})
