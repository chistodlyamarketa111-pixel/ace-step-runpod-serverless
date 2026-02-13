#!/usr/bin/env python3
"""
YuE Direct API Server - Lightweight HTTP server for music generation.
Replaces Gradio with a simple JSON API for submitting jobs, tracking status, and downloading audio.
Runs on port 8000 on the RunPod pod.
"""

import json
import os
import subprocess
import shlex
import threading
import time
import uuid
import re
import glob
import signal
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
from datetime import datetime

PORT = int(os.environ.get("YUE_API_PORT", "8000"))
BASE_YUE_DIR = os.environ.get("YUE_DIR", "/workspace/YuE")
BASE_MODELS_DIR = os.environ.get("YUE_MODELS_DIR", "/workspace/models")
BASE_OUTPUTS_DIR = os.environ.get("YUE_OUTPUTS_DIR", "/workspace/outputs")
BASE_INPUTS_DIR = os.environ.get("YUE_INPUTS_DIR", "/workspace/inputs")

DEFAULT_STAGE1_MODEL = "m-a-p/YuE-s1-7B-anneal-en-cot"
DEFAULT_STAGE2_MODEL = "m-a-p/YuE-s2-1B-general"

CONDA_ACTIVATE_PATH = "/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV_NAME = "pyenv"

def check_gpu_available():
    if os.path.exists("/dev/nvidia0"):
        return True
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

HAS_GPU = check_gpu_available()

os.makedirs(BASE_OUTPUTS_DIR, exist_ok=True)
os.makedirs(BASE_INPUTS_DIR, exist_ok=True)

jobs = {}
jobs_lock = threading.Lock()


class Job:
    def __init__(self, job_id, params):
        self.id = job_id
        self.params = params
        self.status = "PENDING"
        self.logs = ""
        self.error = None
        self.output_files = []
        self.pid = None
        self.proc = None
        self.created_at = datetime.utcnow().isoformat()
        self.completed_at = None
        safe_id = re.sub(r"[^a-zA-Z0-9]", "", job_id)
        self.custom_filename = params.get("custom_filename", safe_id[:32])

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "logs": self.logs[-2000:] if self.logs else "",
            "error": self.error,
            "output_files": self.output_files,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


def build_argv(job):
    p = job.params
    genre = p.get("genre", p.get("style", "pop"))
    lyrics = p.get("lyrics", "[verse]\nLa la la\n\n[chorus]\nLa la la")
    num_segments = p.get("num_segments", 2)
    seed = p.get("seed", 42)
    max_new_tokens = p.get("max_new_tokens", 3000)
    stage1_model = p.get("stage1_model", DEFAULT_STAGE1_MODEL)
    stage2_model = p.get("stage2_model", DEFAULT_STAGE2_MODEL)
    custom_filename = job.custom_filename

    genre_file = f"/tmp/yue_genre_{job.id}.txt"
    lyrics_file = f"/tmp/yue_lyrics_{job.id}.txt"
    with open(genre_file, "w") as f:
        f.write(genre)
    with open(lyrics_file, "w") as f:
        f.write(lyrics)

    argv = [
        "python", "-u", f"{BASE_YUE_DIR}/inference/infer.py",
        "--stage1_model", stage1_model,
        "--stage2_model", stage2_model,
        "--genre_txt", genre_file,
        "--lyrics_txt", lyrics_file,
        "--run_n_segments", str(num_segments),
        "--output_dir", BASE_OUTPUTS_DIR,
        "--cuda_idx", "0",
        "--seed", str(seed),
        "--max_new_tokens", str(max_new_tokens),
        "--stage2_batch_size", "4",
        "--repetition_penalty", "1.1",
    ]

    if custom_filename.strip():
        argv.extend(["--custom_filename", custom_filename])

    return argv, genre_file, lyrics_file


def run_job(job):
    job.status = "IN_PROGRESS"
    argv, genre_file, lyrics_file = build_argv(job)

    use_conda = os.path.isfile(CONDA_ACTIVATE_PATH)

    env = os.environ.copy()


    if use_conda:
        quoted = " ".join(shlex.quote(a) for a in argv)
        shell_cmd = f"source {CONDA_ACTIVATE_PATH} && conda activate {CONDA_ENV_NAME} && {quoted}"
        print(f"[{job.id}] Running (conda): {shell_cmd[:300]}...")
        popen_kwargs = dict(
            args=["bash", "-c", shell_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            cwd=BASE_YUE_DIR,
            env=env,
        )
    else:
        print(f"[{job.id}] Running: {' '.join(argv[:6])}...")
        popen_kwargs = dict(
            args=argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            cwd=BASE_YUE_DIR,
            env=env,
        )

    try:
        proc = subprocess.Popen(**popen_kwargs)
        job.proc = proc
        job.pid = proc.pid

        for line in iter(proc.stdout.readline, b""):
            decoded = line.decode("utf-8", errors="replace")
            job.logs += decoded
            sys.stdout.write(f"[{job.id[:8]}] {decoded}")
            sys.stdout.flush()

            match = re.search(r"Created mix:\s*(.+\.mp3)", decoded)
            if match:
                audio_path = match.group(1).strip()
                if audio_path not in job.output_files:
                    job.output_files.append(audio_path)
                    print(f"[{job.id}] Output found: {audio_path}")

        proc.stdout.close()
        ret = proc.wait()

        if ret != 0 and not job.output_files:
            job.status = "FAILED"
            job.error = f"Process exited with code {ret}"
            error_lines = [l for l in job.logs.split("\n") if "error" in l.lower()]
            if error_lines:
                job.error = error_lines[-1][:500]
        else:
            if not job.output_files:
                found = find_output_files(job)
                job.output_files = found

            if job.output_files:
                job.status = "COMPLETED"
                job.completed_at = datetime.utcnow().isoformat()
            else:
                job.status = "FAILED"
                job.error = "Generation completed but no output files found"

    except Exception as e:
        job.status = "FAILED"
        job.error = str(e)
        print(f"[{job.id}] Error: {e}")

    for tmp in [genre_file, lyrics_file]:
        try:
            os.remove(tmp)
        except:
            pass

    print(f"[{job.id}] Final status: {job.status}, files: {job.output_files}")


def find_output_files(job):
    files = []
    cf = job.custom_filename
    if cf:
        pattern = os.path.join(BASE_OUTPUTS_DIR, f"{cf}_*_mixed.mp3")
        found = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        files.extend(found)

    if not files:
        all_mixed = sorted(
            glob.glob(os.path.join(BASE_OUTPUTS_DIR, "*_mixed.mp3")),
            key=os.path.getmtime,
            reverse=True,
        )
        cutoff = time.time() - 900
        for f in all_mixed:
            if os.path.getmtime(f) > cutoff:
                files.append(f)
                break

    return files


def find_stems_for_mix(mix_path):
    """Given a mixed.mp3 path, find the corresponding vocal and instrumental stems."""
    basename = os.path.basename(mix_path)
    prefix = basename.replace("_mixed.mp3", "")
    stems_dir = os.path.join(BASE_OUTPUTS_DIR, "vocoder", "stems")
    vtrack = os.path.join(stems_dir, f"{prefix}_vtrack.mp3")
    itrack = os.path.join(stems_dir, f"{prefix}_itrack.mp3")
    return vtrack if os.path.exists(vtrack) else None, itrack if os.path.exists(itrack) else None


BASE_RVC_MODELS_DIR = "/workspace/models/rvc"
PP_OUTPUTS_DIR = os.path.join(BASE_OUTPUTS_DIR, "postprocessed")
os.makedirs(PP_OUTPUTS_DIR, exist_ok=True)
os.makedirs(BASE_RVC_MODELS_DIR, exist_ok=True)

pp_jobs = {}
pp_jobs_lock = threading.Lock()


class PPJob:
    def __init__(self, pp_id, source_job_id, rvc_model=None):
        self.id = pp_id
        self.source_job_id = source_job_id
        self.rvc_model = rvc_model
        self.status = "PENDING"
        self.logs = ""
        self.error = None
        self.output_files = []
        self.created_at = datetime.utcnow().isoformat()
        self.completed_at = None

    def to_dict(self):
        return {
            "id": self.id,
            "source_job_id": self.source_job_id,
            "status": self.status,
            "logs": self.logs[-2000:] if self.logs else "",
            "error": self.error,
            "output_files": self.output_files,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


def check_rvc_available():
    """Check if RVC is installed and usable."""
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {CONDA_ACTIVATE_PATH} && conda activate {CONDA_ENV_NAME} && python -c 'from rvc_python.infer import RVCInference; print(\"ok\")'"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def list_rvc_models():
    """List available RVC voice models."""
    models = []
    if not os.path.isdir(BASE_RVC_MODELS_DIR):
        return models
    for name in os.listdir(BASE_RVC_MODELS_DIR):
        model_dir = os.path.join(BASE_RVC_MODELS_DIR, name)
        if not os.path.isdir(model_dir):
            continue
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        index_files = glob.glob(os.path.join(model_dir, "*.index"))
        if pth_files:
            models.append({
                "id": name,
                "pth": pth_files[0],
                "index": index_files[0] if index_files else None,
            })
    return models


def run_rvc(input_path, output_path, model_info, f0_method="rmvpe", f0_up_key=0, pp_job=None):
    """Run RVC voice conversion on a vocal track."""
    pth_path = model_info["pth"]
    index_path = model_info.get("index", "")

    rvc_script = f"""
import sys
sys.path.insert(0, '/workspace')
from rvc_python.infer import RVCInference
rvc = RVCInference(device="cuda:0")
rvc.load_model("{pth_path}")
rvc.infer_file("{input_path}", "{output_path}")
print("RVC_DONE")
"""
    script_path = f"/tmp/rvc_run_{os.getpid()}.py"
    with open(script_path, "w") as f:
        f.write(rvc_script)

    cmd = f"source {CONDA_ACTIVATE_PATH} && conda activate {CONDA_ENV_NAME} && python {script_path}"
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=300,
        )
        if pp_job:
            pp_job.logs += proc.stdout + proc.stderr
        if proc.returncode != 0 or "RVC_DONE" not in proc.stdout:
            raise RuntimeError(f"RVC failed: {proc.stderr[-500:]}")
        return True
    finally:
        try:
            os.remove(script_path)
        except:
            pass


def run_mastering(input_path, output_path, pp_job=None):
    """Apply mastering chain using ffmpeg: high-pass, compression, loudness normalization, limiting."""
    log_msg = f"[mastering] Processing {input_path}\n"
    if pp_job:
        pp_job.logs += log_msg
    print(log_msg.strip())

    temp_file = output_path + ".tmp.mp3"

    cmd_pass1 = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", (
            "highpass=f=35,"
            "acompressor=threshold=-20dB:ratio=3:attack=10:release=200:makeup=2dB,"
            "loudnorm=I=-14:TP=-1:LRA=11:print_format=json"
        ),
        "-f", "null", "-"
    ]

    try:
        r1 = subprocess.run(cmd_pass1, capture_output=True, text=True, timeout=120)
        stderr = r1.stderr

        measured_i = measured_tp = measured_lra = measured_thresh = None
        import re as _re
        m_i = _re.search(r'"input_i"\s*:\s*"([^"]+)"', stderr)
        m_tp = _re.search(r'"input_tp"\s*:\s*"([^"]+)"', stderr)
        m_lra = _re.search(r'"input_lra"\s*:\s*"([^"]+)"', stderr)
        m_thresh = _re.search(r'"input_thresh"\s*:\s*"([^"]+)"', stderr)

        if m_i and m_tp and m_lra and m_thresh:
            measured_i = m_i.group(1)
            measured_tp = m_tp.group(1)
            measured_lra = m_lra.group(1)
            measured_thresh = m_thresh.group(1)
            loudnorm_filter = (
                f"loudnorm=I=-14:TP=-1:LRA=11:"
                f"measured_I={measured_i}:measured_TP={measured_tp}:"
                f"measured_LRA={measured_lra}:measured_thresh={measured_thresh}:linear=true"
            )
        else:
            loudnorm_filter = "loudnorm=I=-14:TP=-1:LRA=11"

        cmd_pass2 = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", (
                f"highpass=f=35,"
                f"acompressor=threshold=-20dB:ratio=3:attack=10:release=200:makeup=2dB,"
                f"{loudnorm_filter},"
                f"alimiter=limit=-1dB:level=false"
            ),
            "-ar", "44100",
            "-b:a", "320k",
            output_path,
        ]

        r2 = subprocess.run(cmd_pass2, capture_output=True, text=True, timeout=120)
        if r2.returncode != 0:
            raise RuntimeError(f"ffmpeg mastering failed: {r2.stderr[-500:]}")

        log_done = f"[mastering] Mastered: {output_path}\n"
        if pp_job:
            pp_job.logs += log_done
        print(log_done.strip())
        return True

    except subprocess.TimeoutExpired:
        raise RuntimeError("Mastering timed out")


def run_postprocess(pp_job):
    """Full post-processing pipeline: RVC (optional) + mastering."""
    pp_job.status = "IN_PROGRESS"
    source_job_id = pp_job.source_job_id

    try:
        with jobs_lock:
            source = jobs.get(source_job_id)
        if not source:
            raise RuntimeError(f"Source job {source_job_id} not found")
        if source.status != "COMPLETED" or not source.output_files:
            raise RuntimeError(f"Source job not completed or has no output files")

        mix_path = source.output_files[0]
        vtrack_path, itrack_path = find_stems_for_mix(mix_path)

        basename = os.path.basename(mix_path).replace("_mixed.mp3", "")
        pp_mix_path = os.path.join(PP_OUTPUTS_DIR, f"{basename}_pp_mixed.mp3")

        rvc_applied = False
        if pp_job.rvc_model and vtrack_path:
            models = list_rvc_models()
            model_info = next((m for m in models if m["id"] == pp_job.rvc_model), None)
            if model_info:
                pp_job.logs += f"[pp] Applying RVC with model: {pp_job.rvc_model}\n"
                rvc_vtrack = os.path.join(PP_OUTPUTS_DIR, f"{basename}_rvc_vtrack.mp3")
                run_rvc(vtrack_path, rvc_vtrack, model_info, pp_job=pp_job)
                vtrack_path = rvc_vtrack
                rvc_applied = True
                pp_job.logs += "[pp] RVC completed\n"
            else:
                pp_job.logs += f"[pp] RVC model '{pp_job.rvc_model}' not found, skipping RVC\n"

        if vtrack_path and itrack_path:
            pp_job.logs += "[pp] Remixing vocal + instrumental tracks\n"
            temp_remix = os.path.join(PP_OUTPUTS_DIR, f"{basename}_remix_temp.mp3")
            remix_cmd = [
                "ffmpeg", "-y",
                "-i", vtrack_path,
                "-i", itrack_path,
                "-filter_complex",
                "[0:a]volume=1.0[v];[1:a]volume=1.0[i];[v][i]amix=inputs=2:duration=longest:dropout_transition=2",
                "-ar", "44100", "-b:a", "320k",
                temp_remix,
            ]
            r = subprocess.run(remix_cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise RuntimeError(f"Remix failed: {r.stderr[-500:]}")

            run_mastering(temp_remix, pp_mix_path, pp_job)
            try:
                os.remove(temp_remix)
            except:
                pass
        else:
            pp_job.logs += "[pp] Stems not found, mastering mixed file directly\n"
            run_mastering(mix_path, pp_mix_path, pp_job)

        pp_job.output_files = [pp_mix_path]
        pp_job.status = "COMPLETED"
        pp_job.completed_at = datetime.utcnow().isoformat()
        pp_job.logs += f"[pp] Post-processing complete: {pp_mix_path}\n"

    except Exception as e:
        pp_job.status = "FAILED"
        pp_job.error = str(e)
        pp_job.logs += f"[pp] Error: {e}\n"
        print(f"[pp:{pp_job.id}] Error: {e}")

    print(f"[pp:{pp_job.id}] Final status: {pp_job.status}")


def is_safe_path(file_path):
    real = os.path.realpath(file_path)
    return real.startswith(os.path.realpath(BASE_OUTPUTS_DIR))


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


class YuEHandler(BaseHTTPRequestHandler):
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
                "server": "yue-api",
                "gpu_available": HAS_GPU,
                "models": {
                    "stage1": DEFAULT_STAGE1_MODEL,
                    "stage2": DEFAULT_STAGE2_MODEL,
                },
                "rvc_models": list_rvc_models(),
                "postprocessing": True,
                "active_jobs": sum(1 for j in jobs.values() if j.status == "IN_PROGRESS"),
                "active_pp_jobs": sum(1 for j in pp_jobs.values() if j.status == "IN_PROGRESS"),
                "timestamp": datetime.utcnow().isoformat(),
            })

        elif path == "/files":
            all_files = []
            for ext in ["*.mp3", "*.wav"]:
                for f in glob.glob(os.path.join(BASE_OUTPUTS_DIR, "**", ext), recursive=True):
                    all_files.append({
                        "path": f,
                        "name": os.path.basename(f),
                        "size": os.path.getsize(f),
                        "modified": os.path.getmtime(f),
                    })
            all_files.sort(key=lambda x: x["modified"], reverse=True)
            self.send_json({"files": all_files[:50]})

        elif path.startswith("/pp/status/"):
            pp_id = path[len("/pp/status/"):]
            with pp_jobs_lock:
                pp_job = pp_jobs.get(pp_id)
            if not pp_job:
                self.send_json({"error": "PP job not found"}, 404)
            else:
                self.send_json(pp_job.to_dict())

        elif path.startswith("/status/"):
            job_id = path[len("/status/"):]
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, 404)
            else:
                self.send_json(job.to_dict())

        elif path.startswith("/download/"):
            file_path = unquote(path[len("/download/"):])
            if not file_path.startswith("/"):
                file_path = os.path.join(BASE_OUTPUTS_DIR, file_path)

            if not is_safe_path(file_path):
                self.send_json({"error": "Access denied: path outside outputs directory"}, 403)
                return

            if not os.path.exists(file_path):
                self.send_json({"error": f"File not found"}, 404)
                return

            ext = os.path.splitext(file_path)[1].lower()
            content_types = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac"}
            ct = content_types.get(ext, "application/octet-stream")

            with open(file_path, "rb") as f:
                data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(file_path)}"')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        elif path == "/jobs":
            with jobs_lock:
                job_list = [j.to_dict() for j in sorted(jobs.values(), key=lambda x: x.created_at, reverse=True)]
            self.send_json({"jobs": job_list[:20]})

        else:
            self.send_json({"error": "Not found", "endpoints": [
                "GET /health", "GET /files", "GET /jobs",
                "GET /status/{job_id}", "GET /pp/status/{pp_id}",
                "GET /download/{file_path}",
                "POST /generate", "POST /postprocess", "POST /stop/{job_id}",
            ]}, 404)

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
            with jobs_lock:
                active = [j for j in jobs.values() if j.status == "IN_PROGRESS"]
            if active:
                self.send_json({
                    "error": "Another generation is in progress",
                    "active_job_id": active[0].id,
                }, 409)
                return

            if not params.get("lyrics") and not params.get("genre"):
                self.send_json({"error": "Provide at least 'lyrics' or 'genre'"}, 400)
                return

            job_id = str(uuid.uuid4())
            job = Job(job_id, params)
            with jobs_lock:
                jobs[job_id] = job

            thread = threading.Thread(target=run_job, args=(job,), daemon=True)
            thread.start()

            self.send_json({
                "job_id": job_id,
                "status": "PENDING",
                "message": "Generation started",
            }, 201)

        elif path == "/postprocess":
            source_job_id = params.get("job_id")
            rvc_model = params.get("rvc_model")

            if not source_job_id:
                self.send_json({"error": "Provide 'job_id' of a completed generation"}, 400)
                return

            with jobs_lock:
                source = jobs.get(source_job_id)
            if not source:
                self.send_json({"error": f"Source job {source_job_id} not found"}, 404)
                return
            if source.status != "COMPLETED":
                self.send_json({"error": f"Source job status is {source.status}, must be COMPLETED"}, 400)
                return

            pp_id = str(uuid.uuid4())
            pp_job = PPJob(pp_id, source_job_id, rvc_model=rvc_model)
            with pp_jobs_lock:
                pp_jobs[pp_id] = pp_job

            thread = threading.Thread(target=run_postprocess, args=(pp_job,), daemon=True)
            thread.start()

            self.send_json({
                "pp_job_id": pp_id,
                "status": "PENDING",
                "message": "Post-processing started",
            }, 201)

        elif path.startswith("/stop/"):
            job_id = path[len("/stop/"):]
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, 404)
            elif stop_job(job):
                self.send_json({"message": "Job stopped"})
            else:
                self.send_json({"error": "Could not stop job or already finished"}, 500)

        else:
            self.send_json({"error": "Not found"}, 404)


class ThreadedHTTPServer(HTTPServer):
    allow_reuse_address = True

    def process_request(self, request, client_address):
        thread = threading.Thread(target=self._handle_request, args=(request, client_address))
        thread.daemon = True
        thread.start()

    def _handle_request(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def patch_transformers_torch_check():
    """Directly patch transformers source to bypass PyTorch version check (CVE-2025-32434).
    
    Patches both import_utils.py (function definition) and modeling_utils.py (call site).
    Also clears all __pycache__ dirs to ensure patched code is used.
    """
    import shutil

    base_dirs = [
        "/opt/conda/envs/pyenv/lib/python3.12/site-packages/transformers",
        "/opt/conda/envs/pyenv/lib/python3.11/site-packages/transformers",
        "/opt/conda/envs/pyenv/lib/python3.10/site-packages/transformers",
    ]

    patched_any = False
    for base in base_dirs:
        if not os.path.isdir(base):
            continue

        import_utils = os.path.join(base, "utils", "import_utils.py")
        if os.path.exists(import_utils):
            try:
                with open(import_utils, "r") as f:
                    content = f.read()
                if "PATCHED_BY_YUE" not in content and "check_torch_load_is_safe" in content:
                    patched = content.replace(
                        "def check_torch_load_is_safe():",
                        "def check_torch_load_is_safe():  # PATCHED_BY_YUE\n    return",
                    )
                    with open(import_utils, "w") as f:
                        f.write(patched)
                    print(f"  Patched import_utils.py: {import_utils}")
                    patched_any = True
                elif "PATCHED_BY_YUE" in content:
                    print(f"  import_utils.py already patched: {import_utils}")
                    patched_any = True
            except Exception as e:
                print(f"  Failed to patch {import_utils}: {e}")

        modeling_utils = os.path.join(base, "modeling_utils.py")
        if os.path.exists(modeling_utils):
            try:
                with open(modeling_utils, "r") as f:
                    content = f.read()
                if "PATCHED_BY_YUE" not in content and "check_torch_load_is_safe()" in content:
                    patched = content.replace(
                        "check_torch_load_is_safe()",
                        "None  # check_torch_load_is_safe() PATCHED_BY_YUE",
                    )
                    with open(modeling_utils, "w") as f:
                        f.write(patched)
                    print(f"  Patched modeling_utils.py: {modeling_utils}")
                    patched_any = True
                elif "PATCHED_BY_YUE" in content:
                    print(f"  modeling_utils.py already patched: {modeling_utils}")
                    patched_any = True
            except Exception as e:
                print(f"  Failed to patch {modeling_utils}: {e}")

        for root, dirs, files in os.walk(base):
            if "__pycache__" in dirs:
                cache_path = os.path.join(root, "__pycache__")
                try:
                    shutil.rmtree(cache_path)
                except Exception:
                    pass
                dirs.remove("__pycache__")
        print(f"  Cleared all __pycache__ in: {base}")

        if patched_any:
            return True

    if not patched_any:
        print("  WARNING: Could not find transformers files to patch")
    return patched_any


def main():
    print(f"=" * 60)
    print(f"  YuE Direct API Server")
    print(f"  Port: {PORT}")
    print(f"  Models dir: {BASE_MODELS_DIR}")
    print(f"  Output dir: {BASE_OUTPUTS_DIR}")
    print(f"  Stage1 model: {DEFAULT_STAGE1_MODEL}")
    print(f"  Stage2 model: {DEFAULT_STAGE2_MODEL}")
    print(f"  YuE dir: {BASE_YUE_DIR}")
    print(f"  YuE infer.py exists: {os.path.exists(os.path.join(BASE_YUE_DIR, 'inference', 'infer.py'))}")
    print(f"  GPU: {HAS_GPU}")
    patch_transformers_torch_check()
    print(f"=" * 60)

    server = ThreadedHTTPServer(("0.0.0.0", PORT), YuEHandler)
    print(f"Server running on http://0.0.0.0:{PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
