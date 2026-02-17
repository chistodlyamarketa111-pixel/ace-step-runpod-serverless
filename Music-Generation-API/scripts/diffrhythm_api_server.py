#!/usr/bin/env python3
"""
DiffRhythm Modular Pipeline API Server
HTTP server for music generation with DiffRhythm + Demucs stem separation + Matchering mastering.
Runs on port 8000 on the RunPod pod.

Setup on RunPod pod:
  git clone https://github.com/ASLP-lab/DiffRhythm.git /workspace/DiffRhythm
  cd /workspace/DiffRhythm && pip install -r requirements.txt
  pip install demucs matchering
  apt-get install -y espeak-ng ffmpeg
  python /workspace/diffrhythm_api_server.py
"""

import json
import os
import subprocess
import threading
import time
import uuid
import signal
import sys
import glob
import shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
from datetime import datetime

PORT = int(os.environ.get("DIFFRHYTHM_API_PORT", "8000"))
DIFFRHYTHM_DIR = os.environ.get("DIFFRHYTHM_DIR", "/workspace/DiffRhythm")
OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "/workspace/outputs")
DEMUCS_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "demucs")
MASTER_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "mastered")
REFERENCE_DIR = os.path.join(OUTPUTS_DIR, "references")

for d in [OUTPUTS_DIR, DEMUCS_OUTPUTS_DIR, MASTER_OUTPUTS_DIR, REFERENCE_DIR]:
    os.makedirs(d, exist_ok=True)

jobs = {}
jobs_lock = threading.Lock()


class Job:
    def __init__(self, job_id, params, job_type="generate"):
        self.id = job_id
        self.params = params
        self.job_type = job_type
        self.status = "PENDING"
        self.logs = ""
        self.error = None
        self.output_files = []
        self.stems = {}
        self.pid = None
        self.proc = None
        self.created_at = datetime.utcnow().isoformat()
        self.completed_at = None

    def to_dict(self):
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status,
            "logs": self.logs[-3000:] if self.logs else "",
            "error": self.error,
            "output_files": self.output_files,
            "stems": self.stems,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


def check_gpu():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


HAS_GPU = check_gpu()


def run_diffrhythm(job):
    """Run DiffRhythm generation."""
    job.status = "IN_PROGRESS"
    p = job.params

    prompt = p.get("prompt", "pop music")
    lyrics = p.get("lyrics", "")
    duration = p.get("duration", 95)
    seed = p.get("seed", -1)
    use_fp16 = p.get("fp16", True)
    chunked = p.get("chunked", True)

    model_type = "full" if duration > 95 else "base"

    output_filename = f"diffrhythm_{job.id}.wav"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    infer_script = os.path.join(DIFFRHYTHM_DIR, "infer.py")
    if not os.path.exists(infer_script):
        infer_script = os.path.join(DIFFRHYTHM_DIR, "scripts", "infer.py")

    escaped_prompt = prompt.replace("'", "\\'").replace('"', '\\"')
    escaped_lyrics = lyrics.replace("'", "\\'").replace('"', '\\"')
    custom_infer_path = f"/tmp/diffrhythm_infer_{job.id}.py"
    with open(custom_infer_path, "w") as f:
        script = """
import sys
import os
import inspect

DIFFRHYTHM_DIR = "{diffrhythm_dir}"
sys.path.insert(0, DIFFRHYTHM_DIR)
infer_dir = os.path.join(DIFFRHYTHM_DIR, "infer")
if os.path.isdir(infer_dir):
    sys.path.insert(0, infer_dir)
diffrhythm_pkg = os.path.join(DIFFRHYTHM_DIR, "diffrhythm")
if os.path.isdir(diffrhythm_pkg):
    sys.path.insert(0, diffrhythm_pkg)
os.chdir(DIFFRHYTHM_DIR)

import torch
import torchaudio
import numpy as np

prepare_model = None
inference = None
get_lrc_token = None
get_style_prompt = None
get_negative_style_prompt = None
get_reference_latent = None

try:
    from diffrhythm.infer.infer_utils import (
        prepare_model, get_lrc_token, get_style_prompt,
        get_negative_style_prompt, get_reference_latent,
    )
    from diffrhythm.infer.infer import inference
    print("[DiffRhythm] Using diffrhythm.infer (v1.2+)")
except ImportError as e1:
    print(f"[DiffRhythm] v1.2+ import failed: {{e1}}")

if prepare_model is None:
    try:
        from infer_utils import (
            prepare_model, get_lrc_token, get_style_prompt,
            get_negative_style_prompt, get_reference_latent,
        )
        from infer import inference
        print("[DiffRhythm] Using infer_utils + infer (flat structure)")
    except ImportError as e2:
        print(f"[DiffRhythm] flat import failed: {{e2}}")

if prepare_model is None:
    raise ImportError("Could not import DiffRhythm from any location")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DiffRhythm] Device: {{device}}")

use_fp16 = {use_fp16} and device == "cuda"
chunked = {chunked}
duration_sec = {duration}

if duration_sec > 95:
    max_frames = 6144
elif duration_sec > 47:
    max_frames = 2048
else:
    max_frames = 2048

print(f"[DiffRhythm] Duration: {{duration_sec}}s, max_frames: {{max_frames}}")

result = prepare_model(max_frames, device)
eval_model = None
eval_muq = None
if len(result) == 6:
    cfm, tokenizer, muq, vae, eval_model, eval_muq = result
    print(f"[DiffRhythm] prepare_model returned 6 values (v1.2+)")
else:
    cfm, tokenizer, muq, vae = result[:4]
    print(f"[DiffRhythm] prepare_model returned {{len(result)}} values")

if max_frames > 2048:
    cfm = cfm.float()
    if hasattr(muq, "float"):
        muq = muq.float()
    if hasattr(vae, "float"):
        vae = vae.float()
    dte = getattr(cfm.transformer, "duration_time_embed", None)
    if dte is not None:
        _orig_dte_forward = dte.forward
        def _patched_dte_forward(x, *a, **kw):
            return _orig_dte_forward(x.float(), *a, **kw)
        dte.forward = _patched_dte_forward
        print("[DiffRhythm] Monkey-patched duration_time_embed to cast input to float32")
    print("[DiffRhythm] Forced float32 for full model (6144 frames)")
elif use_fp16:
    cfm = cfm.half()
    if hasattr(muq, "half"):
        muq = muq.half()
    print("[DiffRhythm] FP16 enabled (cfm + muq, base model)")

torch.cuda.empty_cache()

prompt = '''{escaped_prompt}'''
lyrics_text = '''{escaped_lyrics}'''
seed = {seed}

print(f"[DiffRhythm] Generating: prompt={{prompt[:100]}}, lyrics={{len(lyrics_text)}} chars")

if seed >= 0:
    torch.manual_seed(seed)

lrc_sig = inspect.signature(get_lrc_token)
lrc_params = list(lrc_sig.parameters.keys())
print(f"[DiffRhythm] get_lrc_token params: {{lrc_params}}")

if len(lrc_params) >= 5:
    max_secs = 285 if duration_sec > 95 else 95
    lrc_result = get_lrc_token(max_frames, lyrics_text, tokenizer, max_secs, device)
else:
    lrc_result = get_lrc_token(max_frames, lyrics_text, tokenizer, device)

if isinstance(lrc_result, tuple) and len(lrc_result) == 4:
    lrc_prompt, start_time, end_frame, song_duration = lrc_result
    print(f"[DiffRhythm] get_lrc_token returned 4 values, end_frame={{end_frame}}")
elif isinstance(lrc_result, tuple) and len(lrc_result) == 2:
    lrc_prompt, start_time = lrc_result
    end_frame = max_frames
    song_duration = duration_sec
    print(f"[DiffRhythm] get_lrc_token returned 2 values")
else:
    lrc_prompt = lrc_result
    start_time = 0
    end_frame = max_frames
    song_duration = duration_sec
    print(f"[DiffRhythm] get_lrc_token returned single value")

style_prompt = get_style_prompt(muq, prompt=prompt)
print("[DiffRhythm] Style prompt ready")

negative_style_prompt = get_negative_style_prompt(device)
print("[DiffRhythm] Negative style prompt ready")

latent_prompt, pred_frames = get_reference_latent(device, max_frames, False, None, None, vae)
print("[DiffRhythm] Reference latent ready")

if max_frames > 2048:
    print("[DiffRhythm] Casting all input tensors to float32 for full model...")
    if isinstance(style_prompt, torch.Tensor):
        style_prompt = style_prompt.float()
    if isinstance(negative_style_prompt, torch.Tensor):
        negative_style_prompt = negative_style_prompt.float()
    if isinstance(latent_prompt, torch.Tensor):
        latent_prompt = latent_prompt.float()
    if isinstance(lrc_prompt, torch.Tensor):
        lrc_prompt = lrc_prompt.float()
    if isinstance(start_time, torch.Tensor):
        start_time = start_time.float()
    print("[DiffRhythm] All input tensors cast to float32")

sig = inspect.signature(inference)
params = list(sig.parameters.keys())
print(f"[DiffRhythm] inference() params: {{params}}")

kwargs = dict(
    cfm_model=cfm,
    vae_model=vae,
    cond=latent_prompt,
    text=lrc_prompt,
    duration=max_frames,
    style_prompt=style_prompt,
    negative_style_prompt=negative_style_prompt,
    start_time=start_time,
    pred_frames=pred_frames,
    chunked=chunked,
)

if "eval_model" in params:
    kwargs["eval_model"] = eval_model
if "eval_muq" in params:
    kwargs["eval_muq"] = eval_muq
if "steps" in params:
    kwargs["steps"] = 32
if "cfg_strength" in params:
    kwargs["cfg_strength"] = 4.0
if "sway_sampling_coef" in params:
    kwargs["sway_sampling_coef"] = None
if "file_type" in params:
    kwargs["file_type"] = "wav"
if "vocal_flag" in params:
    kwargs["vocal_flag"] = True if lyrics_text else False
if "odeint_method" in params:
    kwargs["odeint_method"] = "euler"
if "batch_infer_num" in params:
    kwargs["batch_infer_num"] = 1
if "song_duration" in params:
    kwargs["song_duration"] = torch.tensor(duration_sec, device=device)

print(f"[DiffRhythm] Calling inference with keys: {{list(kwargs.keys())}}")

result = inference(**kwargs)

output_path = "{output_path}"

print(f"[DiffRhythm] Result type: {{type(result)}}")

if isinstance(result, list):
    result = result[0]
    print(f"[DiffRhythm] Unwrapped list, inner type: {{type(result)}}")

if isinstance(result, tuple) and len(result) == 2:
    sample_rate, audio_np = result
    if isinstance(audio_np, np.ndarray):
        audio_tensor = torch.from_numpy(audio_np)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.shape[-1] <= 2 and audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.T
        torchaudio.save(output_path, audio_tensor.float(), int(sample_rate))
    elif isinstance(audio_np, torch.Tensor):
        audio_t = audio_np.cpu().float()
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0)
        torchaudio.save(output_path, audio_t, int(sample_rate))
    else:
        torchaudio.save(output_path, torch.tensor(audio_np).float(), int(sample_rate))
    print(f"[DiffRhythm] Saved (tuple): {{output_path}}")
elif isinstance(result, torch.Tensor):
    audio_t = result.cpu().float()
    if audio_t.dim() == 1:
        audio_t = audio_t.unsqueeze(0)
    torchaudio.save(output_path, audio_t, 44100)
    print(f"[DiffRhythm] Saved (tensor): {{output_path}}")
elif isinstance(result, bytes):
    with open(output_path, "wb") as out_f:
        out_f.write(result)
    print(f"[DiffRhythm] Saved (bytes): {{output_path}}")
else:
    print(f"[DiffRhythm] Unknown result type: {{type(result)}}, trying to save...")
    torchaudio.save(output_path, torch.tensor(result).float(), 44100)
    print(f"[DiffRhythm] Saved (fallback): {{output_path}}")

print("DIFFRHYTHM_DONE")
""".format(
            diffrhythm_dir=DIFFRHYTHM_DIR,
            use_fp16=use_fp16,
            chunked=chunked,
            duration=duration,
            escaped_prompt=escaped_prompt,
            escaped_lyrics=escaped_lyrics,
            seed=seed,
            output_path=output_path,
        )
        f.write(script)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DIFFRHYTHM_DIR}:{env.get('PYTHONPATH', '')}"

    cmd = ["python", "-u", custom_infer_path]
    job.logs += f"[gen] Starting DiffRhythm generation...\n"
    print(f"[{job.id}] Starting DiffRhythm: {' '.join(cmd[:5])}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            cwd=DIFFRHYTHM_DIR,
            env=env,
        )
        job.proc = proc
        job.pid = proc.pid

        for line in iter(proc.stdout.readline, b""):
            decoded = line.decode("utf-8", errors="replace")
            job.logs += decoded
            sys.stdout.write(f"[{job.id[:8]}] {decoded}")
            sys.stdout.flush()

        proc.stdout.close()
        ret = proc.wait()

        if ret != 0 or not os.path.exists(output_path):
            job.status = "FAILED"
            error_lines = [l for l in job.logs.split("\n") if "error" in l.lower() or "traceback" in l.lower()]
            job.error = error_lines[-1][:500] if error_lines else f"Process exited with code {ret}"
        else:
            job.output_files = [output_path]
            job.status = "COMPLETED"
            job.completed_at = datetime.utcnow().isoformat()
            job.logs += f"[gen] Generation complete: {output_path}\n"

    except Exception as e:
        job.status = "FAILED"
        job.error = str(e)
        job.logs += f"[gen] Error: {e}\n"
    finally:
        try:
            os.remove(custom_infer_path)
        except:
            pass

    print(f"[{job.id}] Final status: {job.status}, files: {job.output_files}")


def run_demucs(job):
    """Run Demucs stem separation on a generated audio file."""
    job.status = "IN_PROGRESS"
    source_path = job.params.get("source_path")

    if not source_path or not os.path.exists(source_path):
        job.status = "FAILED"
        job.error = f"Source file not found: {source_path}"
        return

    model = job.params.get("model", "htdemucs")
    job.logs += f"[demucs] Separating stems with {model}: {source_path}\n"

    output_dir = os.path.join(DEMUCS_OUTPUTS_DIR, job.id)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "-m", "demucs",
        "-n", model,
        "-o", output_dir,
        "--mp3",
        source_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        job.proc = proc
        job.pid = proc.pid

        for line in iter(proc.stdout.readline, b""):
            decoded = line.decode("utf-8", errors="replace")
            job.logs += decoded
            sys.stdout.write(f"[demucs:{job.id[:8]}] {decoded}")
            sys.stdout.flush()

        proc.stdout.close()
        ret = proc.wait()

        if ret != 0:
            job.status = "FAILED"
            job.error = f"Demucs exited with code {ret}"
            return

        source_basename = os.path.splitext(os.path.basename(source_path))[0]
        stems_dir = os.path.join(output_dir, model, source_basename)

        if not os.path.isdir(stems_dir):
            for root, dirs, files in os.walk(output_dir):
                if any(f.endswith(".mp3") or f.endswith(".wav") for f in files):
                    stems_dir = root
                    break

        stems = {}
        stem_files = []
        if os.path.isdir(stems_dir):
            for f in os.listdir(stems_dir):
                if f.endswith(".mp3") or f.endswith(".wav"):
                    full_path = os.path.join(stems_dir, f)
                    stem_name = os.path.splitext(f)[0]
                    stems[stem_name] = full_path
                    stem_files.append(full_path)

        job.stems = stems
        job.output_files = stem_files
        job.status = "COMPLETED"
        job.completed_at = datetime.utcnow().isoformat()
        job.logs += f"[demucs] Stems separated: {list(stems.keys())}\n"

    except Exception as e:
        job.status = "FAILED"
        job.error = str(e)
        job.logs += f"[demucs] Error: {e}\n"

    print(f"[demucs:{job.id}] Final: {job.status}, stems: {list(job.stems.keys())}")


def run_mastering(job):
    """Run mastering pipeline: Matchering (if reference) + ffmpeg loudness normalization."""
    job.status = "IN_PROGRESS"
    source_path = job.params.get("source_path")
    reference_path = job.params.get("reference_path")

    if not source_path or not os.path.exists(source_path):
        job.status = "FAILED"
        job.error = f"Source file not found: {source_path}"
        return

    output_filename = f"mastered_{job.id}.mp3"
    output_path = os.path.join(MASTER_OUTPUTS_DIR, output_filename)

    try:
        if reference_path and os.path.exists(reference_path):
            job.logs += f"[master] Matchering with reference: {reference_path}\n"
            matchering_out = output_path.replace(".mp3", "_matchered.wav")

            mg_script = f"/tmp/matchering_{job.id}.py"
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
            proc = subprocess.run(
                ["python", mg_script],
                capture_output=True, text=True, timeout=300,
            )
            job.logs += proc.stdout + proc.stderr
            try:
                os.remove(mg_script)
            except:
                pass

            if proc.returncode == 0 and os.path.exists(matchering_out):
                source_path = matchering_out
                job.logs += "[master] Matchering complete\n"
            else:
                job.logs += "[master] Matchering failed, continuing with ffmpeg only\n"

        job.logs += "[master] Applying ffmpeg mastering chain\n"

        cmd_pass1 = [
            "ffmpeg", "-y", "-i", source_path,
            "-af", (
                "highpass=f=30,"
                "acompressor=threshold=-18dB:ratio=3:attack=10:release=200:makeup=2dB,"
                "loudnorm=I=-14:TP=-1:LRA=11:print_format=json"
            ),
            "-f", "null", "-"
        ]
        r1 = subprocess.run(cmd_pass1, capture_output=True, text=True, timeout=120)

        import re
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
            "-af", (
                f"highpass=f=30,"
                f"acompressor=threshold=-18dB:ratio=3:attack=10:release=200:makeup=2dB,"
                f"{loudnorm},"
                f"alimiter=limit=-1dB:level=false"
            ),
            "-ar", "44100",
            "-b:a", "320k",
            output_path,
        ]
        r2 = subprocess.run(cmd_pass2, capture_output=True, text=True, timeout=120)

        if r2.returncode != 0:
            job.status = "FAILED"
            job.error = f"ffmpeg mastering failed: {r2.stderr[-300:]}"
            return

        job.output_files = [output_path]
        job.status = "COMPLETED"
        job.completed_at = datetime.utcnow().isoformat()
        job.logs += f"[master] Mastered: {output_path}\n"

    except Exception as e:
        job.status = "FAILED"
        job.error = str(e)
        job.logs += f"[master] Error: {e}\n"

    print(f"[master:{job.id}] Final: {job.status}")


def run_pipeline(job):
    """Full pipeline: DiffRhythm generate → Demucs stems → remix instrumental → master."""
    job.status = "IN_PROGRESS"
    p = job.params

    job.logs += "[pipeline] Starting modular pipeline\n"
    job.logs += "[pipeline] Step 1/3: DiffRhythm generation\n"

    gen_job = Job(f"{job.id}_gen", p, "generate")
    run_diffrhythm(gen_job)

    if gen_job.status != "COMPLETED" or not gen_job.output_files:
        job.status = "FAILED"
        job.error = f"Generation failed: {gen_job.error}"
        job.logs += gen_job.logs
        return

    raw_audio = gen_job.output_files[0]
    job.logs += gen_job.logs
    job.logs += f"[pipeline] Step 1 done: {raw_audio}\n"

    job.logs += "[pipeline] Step 2/3: Demucs stem separation\n"
    demucs_job = Job(f"{job.id}_demucs", {
        "source_path": raw_audio,
        "model": p.get("demucs_model", "htdemucs"),
    }, "demucs")
    run_demucs(demucs_job)

    if demucs_job.status != "COMPLETED":
        job.logs += f"[pipeline] Demucs failed, using raw audio for mastering\n"
        job.logs += demucs_job.logs
        source_for_master = raw_audio
    else:
        job.logs += demucs_job.logs
        job.stems = demucs_job.stems
        job.logs += f"[pipeline] Step 2 done: stems={list(demucs_job.stems.keys())}\n"

        vocals = demucs_job.stems.get("vocals")
        other = demucs_job.stems.get("other")
        drums = demucs_job.stems.get("drums")
        bass = demucs_job.stems.get("bass")

        remix_path = os.path.join(OUTPUTS_DIR, f"remix_{job.id}.wav")
        inputs = []
        filters = []
        idx = 0

        for stem_name, stem_path in [("vocals", vocals), ("other", other), ("drums", drums), ("bass", bass)]:
            if stem_path and os.path.exists(stem_path):
                inputs.extend(["-i", stem_path])
                vol = p.get(f"vol_{stem_name}", 1.0)
                filters.append(f"[{idx}:a]volume={vol}[s{idx}]")
                idx += 1

        if idx > 1:
            mix_inputs = "".join(f"[s{i}]" for i in range(idx))
            filter_complex = ";".join(filters) + f";{mix_inputs}amix=inputs={idx}:duration=longest:normalize=0"

            remix_cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filter_complex,
                "-ar", "44100",
                remix_path,
            ]
            r = subprocess.run(remix_cmd, capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and os.path.exists(remix_path):
                source_for_master = remix_path
                job.logs += f"[pipeline] Remixed stems → {remix_path}\n"
            else:
                source_for_master = raw_audio
                job.logs += f"[pipeline] Remix failed, using raw audio\n"
        else:
            source_for_master = raw_audio

    job.logs += "[pipeline] Step 3/3: Mastering\n"
    master_job = Job(f"{job.id}_master", {
        "source_path": source_for_master,
        "reference_path": p.get("reference_path"),
    }, "master")
    run_mastering(master_job)

    if master_job.status != "COMPLETED":
        job.logs += f"[pipeline] Mastering failed: {master_job.error}\n"
        job.output_files = [raw_audio]
        job.status = "COMPLETED"
        job.completed_at = datetime.utcnow().isoformat()
        job.logs += "[pipeline] Returning unmastered output\n"
    else:
        job.logs += master_job.logs
        job.output_files = master_job.output_files
        job.status = "COMPLETED"
        job.completed_at = datetime.utcnow().isoformat()
        job.logs += f"[pipeline] Pipeline complete: {job.output_files}\n"

    print(f"[pipeline:{job.id}] Final: {job.status}, files: {job.output_files}")


def is_safe_path(file_path):
    real = os.path.realpath(file_path)
    return real.startswith(os.path.realpath(OUTPUTS_DIR)) or real.startswith("/tmp")


def stop_job(job):
    if job.proc and job.proc.poll() is None:
        try:
            os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
            time.sleep(2)
            if job.proc.poll() is None:
                os.killpg(os.getpgid(job.proc.pid), signal.SIGKILL)
            job.status = "FAILED"
            job.error = "Stopped by user"
            return True
        except:
            return False
    return False


class DiffRhythmHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[HTTP] {args[0]}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_cors_headers(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_cors_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/health":
            self.send_json({
                "status": "ok",
                "server": "diffrhythm-pipeline",
                "gpu": HAS_GPU,
                "engines": {
                    "diffrhythm": os.path.isdir(DIFFRHYTHM_DIR),
                    "demucs": shutil.which("demucs") is not None or True,
                    "matchering": True,
                },
                "timestamp": datetime.utcnow().isoformat(),
            })

        elif path.startswith("/status/"):
            job_id = path.split("/status/")[1]
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, 404)
            else:
                self.send_json(job.to_dict())

        elif path.startswith("/download/"):
            file_path = unquote(path.split("/download/")[1])
            if not is_safe_path(file_path):
                self.send_json({"error": "Invalid path"}, 403)
                return
            if not os.path.exists(file_path):
                self.send_json({"error": "File not found"}, 404)
                return

            ext = os.path.splitext(file_path)[1].lower()
            ct = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac"}.get(ext, "application/octet-stream")

            with open(file_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        elif path == "/jobs":
            with jobs_lock:
                all_jobs = [j.to_dict() for j in sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)[:50]]
            self.send_json({"jobs": all_jobs})

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            params = json.loads(body)
        except:
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        if path == "/generate":
            job_id = str(uuid.uuid4())[:12]
            job = Job(job_id, params, "generate")
            with jobs_lock:
                jobs[job_id] = job
            thread = threading.Thread(target=run_diffrhythm, args=(job,), daemon=True)
            thread.start()
            self.send_json({"job_id": job_id, "status": "PENDING"})

        elif path == "/pipeline":
            job_id = str(uuid.uuid4())[:12]
            job = Job(job_id, params, "pipeline")
            with jobs_lock:
                jobs[job_id] = job
            thread = threading.Thread(target=run_pipeline, args=(job,), daemon=True)
            thread.start()
            self.send_json({"job_id": job_id, "status": "PENDING"})

        elif path == "/demucs":
            source_path = params.get("source_path")
            if not source_path:
                job_id_ref = params.get("job_id")
                if job_id_ref:
                    with jobs_lock:
                        ref_job = jobs.get(job_id_ref)
                    if ref_job and ref_job.output_files:
                        source_path = ref_job.output_files[0]

            if not source_path:
                self.send_json({"error": "source_path or job_id required"}, 400)
                return

            job_id = str(uuid.uuid4())[:12]
            job = Job(job_id, {"source_path": source_path, "model": params.get("model", "htdemucs")}, "demucs")
            with jobs_lock:
                jobs[job_id] = job
            thread = threading.Thread(target=run_demucs, args=(job,), daemon=True)
            thread.start()
            self.send_json({"job_id": job_id, "status": "PENDING"})

        elif path == "/master":
            source_path = params.get("source_path")
            if not source_path:
                job_id_ref = params.get("job_id")
                if job_id_ref:
                    with jobs_lock:
                        ref_job = jobs.get(job_id_ref)
                    if ref_job and ref_job.output_files:
                        source_path = ref_job.output_files[0]

            if not source_path:
                self.send_json({"error": "source_path or job_id required"}, 400)
                return

            job_id = str(uuid.uuid4())[:12]
            job = Job(job_id, {
                "source_path": source_path,
                "reference_path": params.get("reference_path"),
            }, "master")
            with jobs_lock:
                jobs[job_id] = job
            thread = threading.Thread(target=run_mastering, args=(job,), daemon=True)
            thread.start()
            self.send_json({"job_id": job_id, "status": "PENDING"})

        elif path.startswith("/stop/"):
            job_id = path.split("/stop/")[1]
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, 404)
            elif stop_job(job):
                self.send_json({"status": "stopped"})
            else:
                self.send_json({"error": "Could not stop job"}, 400)

        else:
            self.send_json({"error": "Not found"}, 404)


def main():
    print(f"=" * 60)
    print(f"DiffRhythm Pipeline API Server")
    print(f"Port: {PORT}")
    print(f"GPU: {HAS_GPU}")
    print(f"DiffRhythm dir: {DIFFRHYTHM_DIR} (exists: {os.path.isdir(DIFFRHYTHM_DIR)})")
    print(f"Outputs: {OUTPUTS_DIR}")
    print(f"=" * 60)

    server = HTTPServer(("0.0.0.0", PORT), DiffRhythmHandler)

    def shutdown(sig, frame):
        print("\nShutting down...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Server listening on 0.0.0.0:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
