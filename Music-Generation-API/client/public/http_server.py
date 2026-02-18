#!/usr/bin/env python3
"""
ACE-Step v1.5 — HTTP Test Server for RunPod Pod (hourly)
Simple Flask server for testing models interactively.
"""

import base64
import io
import json
import os
import time
import traceback

import torch
import soundfile as sf
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/workspace/checkpoints")
DEFAULT_DIT_MODEL = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")
CPU_OFFLOAD = os.environ.get("ACESTEP_CPU_OFFLOAD", "false").lower() == "true"

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
_gpu_initialized = False


def _init_gpu():
    global _gpu_initialized
    if _gpu_initialized:
        return
    from acestep.gpu_config import get_gpu_config, set_global_gpu_config
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    print(f"[ACE-Step] GPU config: {gpu_config}")
    _gpu_initialized = True


def get_dit_handler(model_name: str):
    global dit_handlers
    if model_name in dit_handlers:
        return dit_handlers[model_name]

    _init_gpu()

    from acestep.handler import AceStepHandler

    dit_path = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.exists(dit_path):
        raise ValueError(f"Model '{model_name}' not found at {dit_path}")

    print(f"[ACE-Step] Loading DiT model: {model_name} from {dit_path}")
    start = time.time()
    handler = AceStepHandler(
        checkpoint_dir=CHECKPOINT_DIR,
        dit_model_path=dit_path,
        cpu_offload=CPU_OFFLOAD,
    )
    elapsed = time.time() - start
    print(f"[ACE-Step] DiT model '{model_name}' loaded in {elapsed:.1f}s")

    dit_handlers[model_name] = handler
    return handler


def get_llm_handler():
    global llm_handler
    if llm_handler is not None:
        return llm_handler

    _init_gpu()

    lm_path = os.path.join(CHECKPOINT_DIR, LM_MODEL)
    if os.path.exists(lm_path):
        from acestep.llm_inference import LLMHandler
        print(f"[ACE-Step] Loading LLM: {LM_MODEL}")
        start = time.time()
        llm_handler = LLMHandler(
            model_path=lm_path,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        elapsed = time.time() - start
        print(f"[ACE-Step] LLM loaded in {elapsed:.1f}s")
    else:
        llm_handler = None
        print(f"[ACE-Step] LM model not found at {lm_path}, CoT disabled")

    return llm_handler


@app.route("/health", methods=["GET"])
def health():
    available = [m for m in VALID_MODELS if os.path.exists(os.path.join(CHECKPOINT_DIR, m))]
    return jsonify({
        "status": "ok",
        "available_models": available,
        "loaded_models": list(dit_handlers.keys()),
        "llm_loaded": llm_handler is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else 0,
    })


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json or {}

        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        model_name = data.get("model", DEFAULT_DIT_MODEL)
        if model_name not in VALID_MODELS:
            return jsonify({"error": f"Invalid model '{model_name}'. Available: {list(VALID_MODELS)}"}), 400

        current_dit = get_dit_handler(model_name)
        current_llm = get_llm_handler()

        prompt = data.get("prompt", "")
        lyrics = data.get("lyrics", "")
        duration = float(data.get("duration", 30))
        task_type = data.get("task_type", "text2music")
        audio_format = data.get("audio_format", "mp3")
        seed = int(data.get("seed", -1))
        default_steps = DEFAULT_STEPS.get(model_name, 8)
        inference_steps = int(data.get("inference_steps", default_steps))
        guidance_scale = float(data.get("guidance_scale", 7.0))
        thinking = data.get("thinking", True)
        batch_size = int(data.get("batch_size", 1))

        bpm = data.get("bpm", None)
        if bpm is not None:
            bpm = int(bpm)
        key_scale = data.get("key_scale", "")
        time_signature = data.get("time_signature", "")
        vocal_language = data.get("vocal_language", "unknown")
        instrumental = data.get("instrumental", False)

        print(f"[ACE-Step] Generate: model={model_name}, prompt='{prompt[:80]}', "
              f"duration={duration}s, steps={inference_steps}")

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
        )

        config = GenerationConfig(
            batch_size=batch_size,
            use_random_seed=(seed < 0),
            seeds=[seed] if seed >= 0 else None,
            audio_format=audio_format if audio_format in ("mp3", "wav", "flac") else "mp3",
        )

        start = time.time()
        result = generate_music(
            dit_handler=current_dit,
            llm_handler=current_llm,
            params=params,
            config=config,
        )
        gen_time = time.time() - start

        if not result.success:
            return jsonify({"error": result.error or "Generation failed"}), 500

        print(f"[ACE-Step] Done in {gen_time:.1f}s, {len(result.audios)} audio(s)")

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
                continue

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            audios_output.append({
                "audio_base64": audio_b64,
                "filename": f"ace_step_{model_name}_{i}.{audio_format}",
                "index": i,
            })

        if not audios_output:
            return jsonify({"error": "No audio files generated"}), 500

        return jsonify({
            "model": model_name,
            "audio_base64": audios_output[0]["audio_base64"],
            "audio_format": audio_format,
            "filename": audios_output[0]["filename"],
            "generation_time": round(gen_time, 1),
            "duration": duration,
            "inference_steps": inference_steps,
            "seed": seed,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/download", methods=["POST"])
def download():
    resp = generate()
    if isinstance(resp, tuple):
        return resp
    data = resp.get_json()
    if "error" in data:
        return resp

    audio_bytes = base64.b64decode(data["audio_base64"])
    buf = io.BytesIO(audio_bytes)
    buf.seek(0)
    return send_file(buf, mimetype="audio/mpeg", as_attachment=True,
                     download_name=data["filename"])


if __name__ == "__main__":
    print(f"[ACE-Step] Checkpoint dir: {CHECKPOINT_DIR}")
    available = [m for m in VALID_MODELS if os.path.exists(os.path.join(CHECKPOINT_DIR, m))]
    print(f"[ACE-Step] Available models: {available}")

    print("[ACE-Step] Pre-loading default model...")
    get_dit_handler(DEFAULT_DIT_MODEL)
    get_llm_handler()

    print("[ACE-Step] Starting HTTP server on port 8888...")
    app.run(host="0.0.0.0", port=8888, threaded=False)
