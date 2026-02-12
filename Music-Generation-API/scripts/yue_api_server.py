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

PORT = int(os.environ.get("YUE_API_PORT", "7860"))
BASE_YUE_DIR = "/workspace/YuE-exllamav2-UI/src/yue"
BASE_MODELS_DIR = "/workspace/models"
BASE_OUTPUTS_DIR = "/workspace/outputs"
BASE_INPUTS_DIR = "/workspace/inputs"

DEFAULT_STAGE1_MODEL = f"{BASE_MODELS_DIR}/YuE-s1-7B-anneal-en-cot"
DEFAULT_STAGE2_MODEL = f"{BASE_MODELS_DIR}/YuE-s2-1B-general"

CONDA_ACTIVATE_PATH = "/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV_NAME = "pyenv"

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
        "python", "-u", f"{BASE_YUE_DIR}/infer.py",
        "--stage1_use_exl2",
        "--stage1_model", stage1_model,
        "--stage1_cache_size", "16384",
        "--stage1_cache_mode", "FP16",
        "--stage2_use_exl2",
        "--stage2_model", stage2_model,
        "--stage2_cache_size", "8192",
        "--stage2_cache_mode", "FP16",
        "--genre_txt", genre_file,
        "--lyrics_txt", lyrics_file,
        "--run_n_segments", str(num_segments),
        "--output_dir", BASE_OUTPUTS_DIR,
        "--cuda_idx", "0",
        "--seed", str(seed),
        "--max_new_tokens", str(max_new_tokens),
        "--basic_model_config", f"{BASE_YUE_DIR}/xcodec_mini_infer/final_ckpt/config.yaml",
        "--resume_path", f"{BASE_YUE_DIR}/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth",
        "--config_path", f"{BASE_YUE_DIR}/xcodec_mini_infer/decoders/config.yaml",
        "--vocal_decoder_path", f"{BASE_YUE_DIR}/xcodec_mini_infer/decoders/decoder_131000.pth",
        "--inst_decoder_path", f"{BASE_YUE_DIR}/xcodec_mini_infer/decoders/decoder_151000.pth",
    ]

    if custom_filename.strip():
        argv.extend(["--custom_filename", custom_filename])

    return argv, genre_file, lyrics_file


def run_job(job):
    job.status = "IN_PROGRESS"
    argv, genre_file, lyrics_file = build_argv(job)

    use_conda = os.path.isfile(CONDA_ACTIVATE_PATH)

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

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
                "gpu_available": os.path.exists("/dev/nvidia0"),
                "models": {
                    "stage1": os.path.exists(DEFAULT_STAGE1_MODEL),
                    "stage2": os.path.exists(DEFAULT_STAGE2_MODEL),
                },
                "active_jobs": sum(1 for j in jobs.values() if j.status == "IN_PROGRESS"),
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
                "GET /status/{job_id}", "GET /download/{file_path}",
                "POST /generate", "POST /stop/{job_id}",
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


def main():
    print(f"=" * 60)
    print(f"  YuE Direct API Server")
    print(f"  Port: {PORT}")
    print(f"  Models dir: {BASE_MODELS_DIR}")
    print(f"  Output dir: {BASE_OUTPUTS_DIR}")
    print(f"  Stage1: {os.path.exists(DEFAULT_STAGE1_MODEL)}")
    print(f"  Stage2: {os.path.exists(DEFAULT_STAGE2_MODEL)}")
    print(f"  GPU: {os.path.exists('/dev/nvidia0')}")
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
