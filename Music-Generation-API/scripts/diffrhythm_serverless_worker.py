#!/usr/bin/env python3
"""
DiffRhythm RunPod Serverless Worker
Handles music generation with DiffRhythm + Demucs stem separation + Matchering mastering.

Supports modes:
  - "generate": DiffRhythm generation only
  - "pipeline": Full pipeline (generate → demucs → remix → master)
  - "demucs": Demucs stem separation only (requires source_base64)
  - "master": Mastering only (requires source_base64)

Setup Docker image:
  FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
  RUN git clone https://github.com/ASLP-lab/DiffRhythm.git /workspace/DiffRhythm
  WORKDIR /workspace/DiffRhythm
  RUN pip install -r requirements.txt
  RUN pip install demucs matchering runpod
  RUN apt-get update && apt-get install -y espeak-ng ffmpeg
  COPY diffrhythm_serverless_worker.py /workspace/handler.py
  CMD ["python", "/workspace/handler.py"]
"""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

import runpod

DIFFRHYTHM_DIR = os.environ.get("DIFFRHYTHM_DIR", "/workspace/DiffRhythm")
OUTPUTS_DIR = "/tmp/outputs"
DEMUCS_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "demucs")
MASTER_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "mastered")

for d in [OUTPUTS_DIR, DEMUCS_OUTPUTS_DIR, MASTER_OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)


def file_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def base64_to_file(b64_data, file_path):
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(b64_data))


def get_content_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac"}.get(ext, "application/octet-stream")


def run_diffrhythm(job_id, params):
    prompt = params.get("prompt", "pop music")
    lyrics = params.get("lyrics", "")
    duration = params.get("duration", 95)
    seed = params.get("seed", -1)
    use_fp16 = params.get("fp16", True)
    chunked = params.get("chunked", True)

    output_filename = f"diffrhythm_{job_id}.wav"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    custom_infer_path = f"/tmp/diffrhythm_infer_{job_id}.py"
    with open(custom_infer_path, "w") as f:
        f.write(f'''
import sys
sys.path.insert(0, "{DIFFRHYTHM_DIR}")

import torch
import torchaudio

try:
    from infer import prepare_model, inference
except ImportError:
    try:
        from diffrhythm.infer import prepare_model, inference
    except ImportError:
        from scripts.infer import prepare_model, inference

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DiffRhythm] Device: {{device}}")

use_fp16 = {use_fp16} and device == "cuda"

cfm, tokenizer, muq, vae = prepare_model(device)

if use_fp16:
    cfm = cfm.half()
    if hasattr(muq, "half"):
        muq = muq.half()
    if hasattr(vae, "half"):
        vae = vae.half()

torch.cuda.empty_cache()

prompt = """{prompt.replace('"', '\\"')}"""
lyrics_text = """{lyrics.replace('"', '\\"')}"""
seed = {seed}

if seed >= 0:
    torch.manual_seed(seed)

audio = inference(
    cfm=cfm, tokenizer=tokenizer, muq=muq, vae=vae,
    prompt=prompt,
    lyrics=lyrics_text if lyrics_text else None,
    device=device, chunked={chunked},
)

torchaudio.save("{output_path}", audio.cpu(), 44100)
print("DIFFRHYTHM_DONE")
''')

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DIFFRHYTHM_DIR}:{env.get('PYTHONPATH', '')}"

    proc = subprocess.run(
        ["python", "-u", custom_infer_path],
        capture_output=True, text=True,
        cwd=DIFFRHYTHM_DIR, env=env,
        timeout=600,
    )

    try:
        os.remove(custom_infer_path)
    except:
        pass

    logs = proc.stdout + proc.stderr

    if proc.returncode != 0 or not os.path.exists(output_path):
        raise Exception(f"DiffRhythm generation failed (code {proc.returncode}): {logs[-500:]}")

    return output_path, logs


def run_demucs(source_path, model="htdemucs"):
    job_id = str(uuid.uuid4())[:12]
    output_dir = os.path.join(DEMUCS_OUTPUTS_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    proc = subprocess.run(
        ["python", "-m", "demucs", "-n", model, "-o", output_dir, "--mp3", source_path],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        raise Exception(f"Demucs failed (code {proc.returncode}): {proc.stderr[-300:]}")

    source_basename = os.path.splitext(os.path.basename(source_path))[0]
    stems_dir = os.path.join(output_dir, model, source_basename)

    if not os.path.isdir(stems_dir):
        for root, dirs, files in os.walk(output_dir):
            if any(f.endswith(".mp3") or f.endswith(".wav") for f in files):
                stems_dir = root
                break

    stems = {}
    if os.path.isdir(stems_dir):
        for f in os.listdir(stems_dir):
            if f.endswith(".mp3") or f.endswith(".wav"):
                full_path = os.path.join(stems_dir, f)
                stem_name = os.path.splitext(f)[0]
                stems[stem_name] = full_path

    return stems, proc.stdout


def run_mastering(source_path, reference_path=None):
    job_id = str(uuid.uuid4())[:12]
    output_path = os.path.join(MASTER_OUTPUTS_DIR, f"mastered_{job_id}.mp3")
    logs = ""

    if reference_path and os.path.exists(reference_path):
        matchering_out = output_path.replace(".mp3", "_matchered.wav")
        mg_script = f"/tmp/matchering_{job_id}.py"
        with open(mg_script, "w") as f:
            f.write(f'''
import matchering as mg
mg.process(
    target="{source_path}",
    reference="{reference_path}",
    results=[mg.pcm16("{matchering_out}")],
)
print("MATCHERING_DONE")
''')
        proc = subprocess.run(["python", mg_script], capture_output=True, text=True, timeout=300)
        logs += proc.stdout + proc.stderr
        try:
            os.remove(mg_script)
        except:
            pass

        if proc.returncode == 0 and os.path.exists(matchering_out):
            source_path = matchering_out
            logs += "[master] Matchering complete\n"

    cmd_pass1 = [
        "ffmpeg", "-y", "-i", source_path,
        "-af", "highpass=f=30,acompressor=threshold=-18dB:ratio=3:attack=10:release=200:makeup=2dB,loudnorm=I=-14:TP=-1:LRA=11:print_format=json",
        "-f", "null", "-"
    ]
    r1 = subprocess.run(cmd_pass1, capture_output=True, text=True, timeout=120)

    m_i = re.search(r'"input_i"\s*:\s*"([^"]+)"', r1.stderr)
    m_tp = re.search(r'"input_tp"\s*:\s*"([^"]+)"', r1.stderr)
    m_lra = re.search(r'"input_lra"\s*:\s*"([^"]+)"', r1.stderr)
    m_thresh = re.search(r'"input_thresh"\s*:\s*"([^"]+)"', r1.stderr)

    if m_i and m_tp and m_lra and m_thresh:
        loudnorm = (
            f"loudnorm=I=-14:TP=-1:LRA=11:"
            f"measured_I={m_i.group(1)}:measured_TP={m_tp.group(1)}:"
            f"measured_LRA={m_lra.group(1)}:measured_thresh={m_thresh.group(1)}:linear=true"
        )
    else:
        loudnorm = "loudnorm=I=-14:TP=-1:LRA=11"

    cmd_pass2 = [
        "ffmpeg", "-y", "-i", source_path,
        "-af", f"highpass=f=30,acompressor=threshold=-18dB:ratio=3:attack=10:release=200:makeup=2dB,{loudnorm},alimiter=limit=-1dB:level=false",
        "-ar", "44100", "-b:a", "320k", output_path,
    ]
    r2 = subprocess.run(cmd_pass2, capture_output=True, text=True, timeout=120)

    if r2.returncode != 0:
        raise Exception(f"ffmpeg mastering failed: {r2.stderr[-300:]}")

    return output_path, logs


def handler(job):
    job_input = job["input"]
    mode = job_input.get("mode", "generate")

    if mode == "generate":
        output_path, logs = run_diffrhythm(job["id"][:12], job_input)
        audio_b64 = file_to_base64(output_path)
        content_type = get_content_type(output_path)
        return {
            "audio_base64": audio_b64,
            "content_type": content_type,
            "filename": os.path.basename(output_path),
            "logs": logs[-2000:],
        }

    elif mode == "pipeline":
        runpod.serverless.progress_update(job, "Step 1/3: Generating with DiffRhythm...")
        output_path, gen_logs = run_diffrhythm(job["id"][:12], job_input)
        all_logs = gen_logs

        runpod.serverless.progress_update(job, "Step 2/3: Demucs stem separation...")
        try:
            stems, demucs_logs = run_demucs(output_path, job_input.get("demucs_model", "htdemucs"))
            all_logs += demucs_logs

            vocals = stems.get("vocals")
            other = stems.get("other")
            drums = stems.get("drums")
            bass = stems.get("bass")

            remix_path = os.path.join(OUTPUTS_DIR, f"remix_{job['id'][:12]}.wav")
            inputs = []
            filters = []
            idx = 0
            for stem_name, stem_path in [("vocals", vocals), ("other", other), ("drums", drums), ("bass", bass)]:
                if stem_path and os.path.exists(stem_path):
                    inputs.extend(["-i", stem_path])
                    vol = job_input.get(f"vol_{stem_name}", 1.0)
                    filters.append(f"[{idx}:a]volume={vol}[s{idx}]")
                    idx += 1

            if idx > 1:
                mix_inputs = "".join(f"[s{i}]" for i in range(idx))
                filter_complex = ";".join(filters) + f";{mix_inputs}amix=inputs={idx}:duration=longest:normalize=0"
                remix_cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex, "-ar", "44100", remix_path]
                r = subprocess.run(remix_cmd, capture_output=True, text=True, timeout=120)
                if r.returncode == 0 and os.path.exists(remix_path):
                    source_for_master = remix_path
                else:
                    source_for_master = output_path
            else:
                source_for_master = output_path
        except Exception as e:
            all_logs += f"[pipeline] Demucs failed: {e}, using raw audio\n"
            source_for_master = output_path

        runpod.serverless.progress_update(job, "Step 3/3: Mastering...")
        try:
            mastered_path, master_logs = run_mastering(source_for_master)
            all_logs += master_logs
            final_path = mastered_path
        except Exception as e:
            all_logs += f"[pipeline] Mastering failed: {e}, returning raw\n"
            final_path = output_path

        audio_b64 = file_to_base64(final_path)
        content_type = get_content_type(final_path)

        return {
            "audio_base64": audio_b64,
            "content_type": content_type,
            "filename": os.path.basename(final_path),
            "logs": all_logs[-3000:],
        }

    elif mode == "demucs":
        source_b64 = job_input.get("source_base64")
        if not source_b64:
            return {"error": "source_base64 required for demucs mode"}

        source_path = os.path.join(OUTPUTS_DIR, f"demucs_input_{job['id'][:12]}.wav")
        base64_to_file(source_b64, source_path)

        stems, logs = run_demucs(source_path, job_input.get("model", "htdemucs"))

        stems_b64 = {}
        for name, path in stems.items():
            stems_b64[name] = {
                "audio_base64": file_to_base64(path),
                "content_type": get_content_type(path),
                "filename": os.path.basename(path),
            }

        return {"stems": stems_b64, "logs": logs[-2000:]}

    elif mode == "master":
        source_b64 = job_input.get("source_base64")
        if not source_b64:
            return {"error": "source_base64 required for master mode"}

        source_path = os.path.join(OUTPUTS_DIR, f"master_input_{job['id'][:12]}.wav")
        base64_to_file(source_b64, source_path)

        mastered_path, logs = run_mastering(source_path)
        audio_b64 = file_to_base64(mastered_path)

        return {
            "audio_base64": audio_b64,
            "content_type": get_content_type(mastered_path),
            "filename": os.path.basename(mastered_path),
            "logs": logs[-2000:],
        }

    else:
        return {"error": f"Unknown mode: {mode}"}


runpod.serverless.start({"handler": handler})
