"""
Shared utilities for RunPod Serverless workers.
Included in the base Docker image and available to all engine workers.
"""

import base64
import os
import re
import subprocess
import uuid


OUTPUTS_DIR = "/tmp/outputs"
DEMUCS_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "demucs")
MASTER_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "mastered")

for d in [OUTPUTS_DIR, DEMUCS_OUTPUTS_DIR, MASTER_OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)


def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def base64_to_file(b64_data: str, file_path: str) -> str:
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return file_path


def get_content_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(ext, "application/octet-stream")


def run_demucs(source_path: str, model: str = "htdemucs") -> dict:
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

    return stems


def run_mastering(source_path: str, reference_path: str = None) -> str:
    job_id = str(uuid.uuid4())[:12]
    output_path = os.path.join(MASTER_OUTPUTS_DIR, f"mastered_{job_id}.mp3")

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
''')
        proc = subprocess.run(["python", mg_script], capture_output=True, text=True, timeout=300)
        try:
            os.remove(mg_script)
        except:
            pass
        if proc.returncode == 0 and os.path.exists(matchering_out):
            source_path = matchering_out

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

    return output_path


def remix_stems(stems: dict, volumes: dict = None, output_dir: str = None) -> str:
    if not output_dir:
        output_dir = OUTPUTS_DIR

    job_id = str(uuid.uuid4())[:12]
    remix_path = os.path.join(output_dir, f"remix_{job_id}.wav")

    if not volumes:
        volumes = {}

    inputs = []
    filters = []
    idx = 0

    for stem_name in ["vocals", "other", "drums", "bass"]:
        stem_path = stems.get(stem_name)
        if stem_path and os.path.exists(stem_path):
            inputs.extend(["-i", stem_path])
            vol = volumes.get(stem_name, 1.0)
            filters.append(f"[{idx}:a]volume={vol}[s{idx}]")
            idx += 1

    if idx == 0:
        raise Exception("No stems found to remix")

    if idx == 1:
        return list(stems.values())[0]

    mix_inputs = "".join(f"[s{i}]" for i in range(idx))
    filter_complex = ";".join(filters) + f";{mix_inputs}amix=inputs={idx}:duration=longest:normalize=0"

    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex, "-ar", "44100", remix_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if r.returncode != 0 or not os.path.exists(remix_path):
        raise Exception(f"Remix failed: {r.stderr[-200:]}")

    return remix_path


def build_response(file_path: str, logs: str = "", extra: dict = None) -> dict:
    result = {
        "audio_base64": file_to_base64(file_path),
        "content_type": get_content_type(file_path),
        "filename": os.path.basename(file_path),
        "logs": logs[-3000:] if logs else "",
    }
    if extra:
        result.update(extra)
    return result
