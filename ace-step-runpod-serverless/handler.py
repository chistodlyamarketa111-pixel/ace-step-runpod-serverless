#!/usr/bin/env python3
"""
ACE-Step v1.5 — RunPod Serverless Handler
LAZY LOADING: Models load on first request, not at startup.
Supports LoRA adapters for style customization.
Supports hybrid mode: instrumental + vocal separation + mixing.
"""

import base64
import io
import os
import sys
import time
import traceback
import tempfile
import subprocess

HANDLER_VERSION = "2026-03-05-v6"
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
            if os.path.exists(config_file) and (os.path.exists(safetensors) or os.path.exists(bin_file)):
                source = "volume" if directory == NETWORK_VOLUME_LORA_DIR else "builtin"
                loras[name] = {"path": lora_path, "source": source}
                print(f"[ACE-Step] Found LoRA: {name} -> {lora_path} ({source})", flush=True)


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


def validate_lora_compatibility(lora_path):
    import json as _json
    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return False, "adapter_config.json not found"

    with open(config_path) as f:
        config = _json.load(f)

    target_modules = config.get("target_modules", [])
    old_modules = {"to_q", "to_k", "to_v", "to_out.0"}
    new_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}

    if old_modules & set(target_modules):
        return False, f"LoRA has old target_modules {target_modules} (ACE-Step v1 format). Need {list(new_modules)} for v1.5."

    if not (new_modules & set(target_modules)):
        return False, f"LoRA target_modules {target_modules} don't match model attention projections {list(new_modules)}"

    st_path = os.path.join(lora_path, "adapter_model.safetensors")
    if os.path.exists(st_path):
        try:
            from safetensors import safe_open
            with safe_open(st_path, framework="pt") as f:
                keys = f.keys()
                for key in keys:
                    tensor = f.get_tensor(key)
                    shape = list(tensor.shape)
                    if any(d == 2560 for d in shape):
                        return False, f"LoRA tensor {key} has dimension 2560 (ACE-Step v1). Model v1.5 uses hidden_size=2048."
                    break
        except Exception as e:
            print(f"[ACE-Step] Warning checking safetensors: {e}", flush=True)

    return True, "OK"


def apply_lora(lora_name, lora_scale=1.0):
    global dit_handler, current_lora

    if not lora_name or lora_name == "none":
        if current_lora:
            try:
                dit_handler.unload_lora()
                print(f"[ACE-Step] Unloaded LoRA: {current_lora}", flush=True)
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
            print(f"[ACE-Step] LoRA not found: {lora_name}. Available: {list(available.keys())}", flush=True)
            return False

    lora_info = available[lora_name]
    lora_path = lora_info["path"]

    compatible, reason = validate_lora_compatibility(lora_path)
    if not compatible:
        print(f"[ACE-Step] LoRA '{lora_name}' incompatible: {reason}", flush=True)
        return f"LoRA incompatible with ACE-Step v1.5: {reason}"

    try:
        if current_lora:
            try:
                result = dit_handler.unload_lora()
                print(f"[ACE-Step] Unloaded previous LoRA: {current_lora} -> {result}", flush=True)
            except Exception as ue:
                print(f"[ACE-Step] Warning during unload: {ue}", flush=True)

        print(f"[ACE-Step] Loading LoRA: {lora_name} (scale={lora_scale}) from {lora_path} ({lora_info['source']})", flush=True)

        result = dit_handler.add_lora(lora_path, adapter_name=lora_name)
        print(f"[ACE-Step] add_lora result: {result}", flush=True)

        if isinstance(result, str) and result.startswith("❌"):
            return f"add_lora error: {result}"

        if lora_scale != 1.0 and hasattr(dit_handler, 'set_lora_scale'):
            dit_handler.set_lora_scale(lora_name, lora_scale)
            print(f"[ACE-Step] Set LoRA scale: {lora_scale}", flush=True)

        current_lora = lora_name
        print(f"[ACE-Step] LoRA loaded successfully: {lora_name}", flush=True)
        return True
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)[:1000]}"
        print(f"[ACE-Step] Error loading LoRA {lora_name}: {err_msg}", flush=True)
        traceback.print_exc()
        current_lora = None
        return err_msg


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

    available_loras = scan_available_loras()
    print(f"[ACE-Step] Available LoRAs: {list(available_loras.keys()) if available_loras else 'none'}", flush=True)

    elapsed = time.time() - start
    print(f"[ACE-Step] Models loaded in {elapsed:.1f}s", flush=True)
    models_loaded = True
    return True


demucs_model = None

def ensure_demucs_loaded():
    global demucs_model
    if demucs_model is not None:
        return True
    try:
        import demucs.pretrained
        print("[ACE-Step] Loading Demucs model (htdemucs)...", flush=True)
        demucs_model = demucs.pretrained.get_model("htdemucs")
        demucs_model.to("cuda")
        demucs_model.eval()
        print("[ACE-Step] Demucs loaded OK", flush=True)
        return True
    except Exception as e:
        print(f"[ACE-Step] Demucs load error: {e}", flush=True)
        traceback.print_exc()
        return False


def separate_vocals(audio_path):
    if not ensure_demucs_loaded():
        raise RuntimeError("Demucs not available")

    import torchaudio
    import demucs.apply

    waveform, sr = torchaudio.load(audio_path)
    if sr != demucs_model.samplerate:
        waveform = torchaudio.functional.resample(waveform, sr, demucs_model.samplerate)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()
    sources = demucs.apply.apply_model(demucs_model, waveform.unsqueeze(0).to("cuda"), progress=False)
    sources = sources * ref.std() + ref.mean()

    source_names = demucs_model.sources
    vocals_idx = source_names.index("vocals")
    vocals = sources[0, vocals_idx].cpu()

    return vocals, demucs_model.samplerate


def mix_audio(vocals_tensor, vocals_sr, instrumental_path, output_path, vocal_volume=1.0, instrumental_volume=0.85):
    import torchaudio

    inst_waveform, inst_sr = torchaudio.load(instrumental_path)
    if inst_sr != vocals_sr:
        inst_waveform = torchaudio.functional.resample(inst_waveform, inst_sr, vocals_sr)

    if inst_waveform.shape[0] == 1:
        inst_waveform = inst_waveform.repeat(2, 1)
    if vocals_tensor.shape[0] == 1:
        vocals_tensor = vocals_tensor.repeat(2, 1)

    min_len = min(vocals_tensor.shape[1], inst_waveform.shape[1])
    vocals_tensor = vocals_tensor[:, :min_len]
    inst_waveform = inst_waveform[:, :min_len]

    mixed = vocals_tensor * vocal_volume + inst_waveform * instrumental_volume
    peak = mixed.abs().max()
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    torchaudio.save(output_path, mixed, vocals_sr, format="mp3")
    return output_path


def generate_single(job_input, job_id, override_params=None):
    params_dict = {
        "caption": job_input.get("prompt", ""),
        "lyrics": job_input.get("lyrics", ""),
        "duration": float(job_input.get("audio_duration", job_input.get("duration", -1))),
        "task_type": job_input.get("task_type", "text2music"),
        "seed": int(job_input.get("seed", -1)),
        "inference_steps": int(job_input.get("inference_steps", job_input.get("infer_step",
            DEFAULT_STEPS.get(job_input.get("model", DEFAULT_MODEL), 8)))),
        "guidance_scale": float(job_input.get("guidance_scale", 7.0)),
        "thinking": job_input.get("thinking", True) if llm_handler is not None else False,
        "bpm": int(job_input.get("bpm")) if job_input.get("bpm") is not None else None,
        "keyscale": job_input.get("key_scale", job_input.get("keyscale", "")),
        "timesignature": job_input.get("time_signature", job_input.get("timesignature", "")),
        "vocal_language": job_input.get("vocal_language", "unknown"),
        "instrumental": job_input.get("instrumental", False),
    }
    if override_params:
        params_dict.update(override_params)

    params = GenerationParams(**params_dict)
    audio_format = job_input.get("audio_format", "mp3")
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=(params_dict["seed"] < 0),
        seeds=[params_dict["seed"]] if params_dict["seed"] >= 0 else None,
        audio_format=audio_format if audio_format in ("mp3", "wav", "flac") else "mp3",
    )

    save_dir = tempfile.mkdtemp(prefix="ace_step_")
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=save_dir,
    )

    if not result.success:
        return None, result.error or "Generation failed"

    for audio_info in result.audios:
        audio_path = audio_info.get("path", "")
        if audio_path and os.path.exists(audio_path):
            return audio_path, None
        elif "tensor" in audio_info and audio_info["tensor"] is not None:
            import torchaudio
            tensor = audio_info["tensor"]
            sr = audio_info.get("sample_rate", 48000)
            out_path = os.path.join(save_dir, f"output.{audio_format}")
            torchaudio.save(out_path, tensor.cpu(), sr, format=audio_format)
            return out_path, None

    return None, "No audio generated"


def handle_hybrid(job, job_input, model_name, lora_name, lora_scale, audio_format):
    start = time.time()
    vocal_volume = float(job_input.get("vocal_volume", 1.0))
    instrumental_volume = float(job_input.get("instrumental_volume", 0.85))

    lora_info = f", lora={lora_name}(x{lora_scale})" if lora_name and lora_name != "none" else ""
    print(f"[ACE-Step] HYBRID mode: generating full track + instrumental{lora_info}", flush=True)

    print("[ACE-Step] Step 1/4: Generating full track with vocals...", flush=True)
    full_path, err = generate_single(job_input, job["id"])
    if err:
        return {"error": f"Hybrid step 1 (full track) failed: {err}"}
    print(f"[ACE-Step] Full track generated: {full_path}", flush=True)

    print("[ACE-Step] Step 2/4: Generating clean instrumental...", flush=True)
    inst_path, err = generate_single(job_input, job["id"], override_params={"instrumental": True, "lyrics": ""})
    if err:
        return {"error": f"Hybrid step 2 (instrumental) failed: {err}"}
    print(f"[ACE-Step] Instrumental generated: {inst_path}", flush=True)

    print("[ACE-Step] Step 3/4: Separating vocals with Demucs...", flush=True)
    try:
        vocals_tensor, vocals_sr = separate_vocals(full_path)
        print(f"[ACE-Step] Vocals separated: shape={vocals_tensor.shape}, sr={vocals_sr}", flush=True)
    except Exception as e:
        print(f"[ACE-Step] Vocal separation failed: {e}. Returning full track instead.", flush=True)
        with open(full_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        gen_time = time.time() - start
        return {
            "audio_base64": audio_b64,
            "content_type": "audio/mpeg",
            "filename": f"ace_step_hybrid_{job['id'][:12]}.mp3",
            "generation_time": round(gen_time, 1),
            "duration": float(job_input.get("audio_duration", job_input.get("duration", -1))),
            "sample_rate": 48000,
            "model": model_name,
            "lora": lora_name if lora_name and lora_name != "none" else None,
            "mode": "hybrid",
            "hybrid_status": "fallback_no_demucs",
        }

    print("[ACE-Step] Step 4/4: Mixing vocals + clean instrumental...", flush=True)
    mix_dir = tempfile.mkdtemp(prefix="ace_hybrid_")
    mix_path = os.path.join(mix_dir, f"hybrid_mix.mp3")
    try:
        mix_audio(vocals_tensor, vocals_sr, inst_path, mix_path, vocal_volume, instrumental_volume)
    except Exception as e:
        print(f"[ACE-Step] Mix failed: {e}. Returning full track.", flush=True)
        with open(full_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        gen_time = time.time() - start
        return {
            "audio_base64": audio_b64,
            "content_type": "audio/mpeg",
            "filename": f"ace_step_hybrid_{job['id'][:12]}.mp3",
            "generation_time": round(gen_time, 1),
            "duration": float(job_input.get("audio_duration", job_input.get("duration", -1))),
            "sample_rate": 48000,
            "model": model_name,
            "lora": lora_name if lora_name and lora_name != "none" else None,
            "mode": "hybrid",
            "hybrid_status": "fallback_mix_failed",
        }

    gen_time = time.time() - start
    with open(mix_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    print(f"[ACE-Step] HYBRID complete in {gen_time:.1f}s", flush=True)
    return {
        "audio_base64": audio_b64,
        "content_type": "audio/mpeg",
        "filename": f"ace_step_hybrid_{job['id'][:12]}.mp3",
        "generation_time": round(gen_time, 1),
        "duration": float(job_input.get("audio_duration", job_input.get("duration", -1))),
        "sample_rate": 48000,
        "model": model_name,
        "lora": lora_name if lora_name and lora_name != "none" else None,
        "mode": "hybrid",
        "hybrid_status": "success",
    }


def handler(job):
    global dit_handler, llm_handler

    try:
        if not ensure_models_loaded():
            return {"error": "Failed to load models. Check worker logs."}

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

        mode = job_input.get("mode", "normal")
        model_name = job_input.get("model", DEFAULT_MODEL)
        audio_format = job_input.get("audio_format", "mp3")

        lora_name = job_input.get("lora_name", None)
        lora_scale = float(job_input.get("lora_scale", 1.0))

        if job_input.get("action") == "exec_python":
            code = job_input.get("code", "")
            if not code:
                return {"error": "No code provided"}
            import io as _io
            import contextlib
            stdout_buf = _io.StringIO()
            stderr_buf = _io.StringIO()
            local_vars = {
                "dit_handler": dit_handler, "llm_handler": llm_handler,
                "torch": torch, "os": os, "sys": sys,
                "LORA_DIR": LORA_DIR, "NETWORK_VOLUME_LORA_DIR": NETWORK_VOLUME_LORA_DIR,
                "scan_available_loras": scan_available_loras,
            }
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(code, local_vars)
                return {
                    "stdout": stdout_buf.getvalue()[-4000:],
                    "stderr": stderr_buf.getvalue()[-2000:],
                    "success": True,
                }
            except Exception as _e:
                return {
                    "stdout": stdout_buf.getvalue()[-2000:],
                    "stderr": stderr_buf.getvalue()[-2000:],
                    "error": f"{type(_e).__name__}: {str(_e)[:2000]}",
                    "success": False,
                }

        if job_input.get("action") == "diagnose_lora":
            diag = {
                "handler_version": HANDLER_VERSION,
                "dit_type": str(type(dit_handler.dit)),
                "dit_class": dit_handler.dit.__class__.__name__,
            }
            diag["lora_methods"] = [m for m in dir(dit_handler) if "lora" in m.lower()]
            for mname in ["load_lora", "add_lora", "unload_lora"]:
                diag[f"has_{mname}"] = hasattr(dit_handler, mname)

            model_obj = dit_handler.dit
            sd = model_obj.state_dict()
            attn_shapes = {}
            for k, v in sd.items():
                if any(proj in k for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    if "weight" in k and "lora" not in k:
                        attn_shapes[k] = list(v.shape)
            sample_keys = sorted(attn_shapes.keys())[:16]
            diag["model_attn_shapes"] = {k: attn_shapes[k] for k in sample_keys}
            diag["model_total_params"] = sum(p.numel() for p in model_obj.parameters())

            available = scan_available_loras()
            diag["available_loras"] = {}
            for n, info in available.items():
                lp = info["path"]
                files = os.listdir(lp) if os.path.isdir(lp) else []
                cfg_path = os.path.join(lp, "adapter_config.json")
                cfg_data = {}
                if os.path.exists(cfg_path):
                    import json as _j
                    with open(cfg_path) as _f:
                        cfg_data = _j.load(_f)
                lora_shapes = {}
                st_path = os.path.join(lp, "adapter_model.safetensors")
                if os.path.exists(st_path):
                    from safetensors import safe_open
                    with safe_open(st_path, framework="pt") as f:
                        for key in sorted(f.keys())[:16]:
                            lora_shapes[key] = list(f.get_tensor(key).shape)
                diag["available_loras"][n] = {
                    "path": lp, "source": info["source"], "files": files,
                    "config": cfg_data, "sample_shapes": lora_shapes,
                }

            test_lora = job_input.get("test_lora", list(available.keys())[0] if available else None)
            if test_lora and test_lora in available:
                test_path = available[test_lora]["path"]
                diag["test_results"] = {}
                try:
                    result = dit_handler.add_lora(test_path, adapter_name="test_diag")
                    diag["test_results"]["add_lora"] = f"Result: {str(result)[:500]}"
                    try:
                        dit_handler.unload_lora()
                    except:
                        pass
                except Exception as _e:
                    diag["test_results"]["add_lora"] = f"FAILED: {str(_e)[:800]}"

                try:
                    from safetensors import safe_open
                    st_path = os.path.join(test_path, "adapter_model.safetensors")
                    mismatches = []
                    with safe_open(st_path, framework="pt") as f:
                        for key in f.keys():
                            clean = key.replace("base_model.model.", "")
                            parts = clean.rsplit(".", 2)
                            module_path = parts[0] if len(parts) > 1 else clean
                            parent_key = module_path + ".weight"
                            lora_shape = list(f.get_tensor(key).shape)
                            if parent_key in sd:
                                model_shape = list(sd[parent_key].shape)
                                if any(ls not in model_shape for ls in lora_shape if ls > 64):
                                    mismatches.append({
                                        "lora_key": key, "lora_shape": lora_shape,
                                        "model_key": parent_key, "model_shape": model_shape,
                                    })
                    diag["dimension_mismatches"] = mismatches[:10]
                    diag["total_mismatches"] = len(mismatches)
                except Exception as _e:
                    diag["dimension_check_error"] = str(_e)[:500]

            return diag

        if lora_name and lora_name != "none":
            lora_result = apply_lora(lora_name, lora_scale)
            if lora_result is not True:
                err_detail = lora_result if isinstance(lora_result, str) else "unknown"
                return {"error": f"Failed to load LoRA: {lora_name}. Detail: {err_detail}. Available: {list(scan_available_loras().keys())}"}
        elif current_lora and (not lora_name or lora_name == "none"):
            apply_lora(None)

        if mode == "hybrid":
            return handle_hybrid(job, job_input, model_name, lora_name, lora_scale, audio_format)

        prompt = job_input.get("prompt", "")
        lyrics = job_input.get("lyrics", "")
        duration = float(job_input.get("audio_duration", job_input.get("duration", -1)))
        task_type = job_input.get("task_type", "text2music")
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
                    "sample_rate": 48000,
                    "model": model_name,
                    "lora": lora_name if lora_name and lora_name != "none" else None,
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
                    "lora": lora_name if lora_name and lora_name != "none" else None,
                }

        return {"error": "No audio files generated"}

    except Exception as e:
        print(f"[ACE-Step] Job error: {e}", flush=True)
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()[-2000:]}


print("[ACE-Step] Worker starting (models will load on first request)...", flush=True)
runpod.serverless.start({"handler": handler})
