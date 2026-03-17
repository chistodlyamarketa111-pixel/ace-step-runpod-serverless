#!/usr/bin/env python3
"""Batch generate ACE-Step tracks via exec_python + HuggingFace upload.

Usage:
    python3 scripts/gen_batch_v2.py                    # Generate all remaining
    python3 scripts/gen_batch_v2.py --start 6 --end 50 # Range
    python3 scripts/gen_batch_v2.py --ids rap_006,rap_007  # Specific IDs
"""
import json, os, sys, time, urllib.request, argparse, hashlib, shutil

ENDPOINT_ID = os.environ.get("ACESTEP_ENDPOINT_ID", "u07qbeocmy4479")
API_KEY = os.environ["RUNPOD_API_KEY"]
HF_TOKEN = os.environ.get("HF_API_TOKEN", "")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_FILE = os.path.join(SCRIPT_DIR, "..", "data", "comparison_cases.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "public", "audio", "ace-step")
HF_REPO = "ruslanmusinrusmus/russianrap-v2"

DURATION = 60
INFERENCE_STEPS = 27
POLL_INTERVAL = 10
JOB_TIMEOUT = 300


def stable_seed(case_id):
    return int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 2**31


def api_call(method, path, data=None, timeout=60):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def generate_one(case):
    seed = stable_seed(case["id"])
    hf_path = f"generated/{case['id']}.mp3"

    lyrics = case["lyrics"]
    caption = case["style_prompt"]

    code = f'''import time, traceback, os, subprocess
t0 = time.time()
try:
    import torch as th
    import torchaudio
    from huggingface_hub import HfApi

    HF_TOKEN = "{HF_TOKEN}"

    result = dit_handler.generate_music(
        captions="""{caption}""",
        lyrics="""{lyrics}""",
        audio_duration={DURATION},
        inference_steps={INFERENCE_STEPS},
        seed={seed},
        use_random_seed=False,
    )
    gen_time = time.time() - t0

    if not result.get("success"):
        print(f"FAILED: {{result.get('error', 'unknown')}}")
        raise ValueError("Generation failed")

    audio = result["audios"][0]
    wav = audio["tensor"] if isinstance(audio, dict) else audio
    sr = audio.get("sample_rate", 44100) if isinstance(audio, dict) else 44100

    if wav.dim() == 1: wav = wav.unsqueeze(0)
    elif wav.dim() == 3: wav = wav.squeeze(0)

    dur = wav.shape[-1] / sr

    tmp_wav = "/tmp/gen_{case['id']}.wav"
    tmp_mp3 = "/tmp/gen_{case['id']}.mp3"
    torchaudio.save(tmp_wav, wav.cpu().float(), sr, format="wav")
    subprocess.run(["ffmpeg", "-y", "-i", tmp_wav, "-b:a", "192k", tmp_mp3], capture_output=True)

    sz = os.path.getsize(tmp_mp3)

    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=tmp_mp3,
        path_in_repo="{hf_path}",
        repo_id="{HF_REPO}",
        repo_type="model",
        token=HF_TOKEN,
    )
    print(f"OK gen={{gen_time:.1f}}s dur={{dur:.0f}}s size={{sz}} sr={{sr}}")

except Exception as e:
    traceback.print_exc()
    print(f"ERROR: {{e}}")
'''

    payload = {"input": {"action": "exec_python", "code": code}}
    result = api_call("POST", "/run", payload)
    job_id = result["id"]

    start = time.time()
    while time.time() - start < JOB_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        try:
            st = api_call("GET", f"/status/{job_id}")
        except Exception:
            continue

        status = st.get("status", "?")
        if status == "COMPLETED":
            output = st.get("output", {})
            stdout = output.get("stdout", "")
            if stdout.startswith("OK"):
                return True, stdout.strip()
            else:
                return False, stdout.strip()[-500:]
        elif status == "FAILED":
            err = st.get("output", {}).get("error", "Unknown error")
            return False, f"FAILED: {err[:300]}"

    return False, "TIMEOUT"


def download_from_hf(case_id):
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            HF_REPO,
            f"generated/{case_id}.mp3",
            repo_type="model",
            token=HF_TOKEN,
        )
        out = os.path.join(OUTPUT_DIR, f"{case_id}.mp3")
        shutil.copy2(path, out)
        return os.path.getsize(out)
    except Exception as e:
        print(f"  Download error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--ids", type=str, default="")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    with open(CASES_FILE) as f:
        cases = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.ids:
        target_ids = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in target_ids]
    else:
        cases = cases[args.start - 1:args.end]

    existing = set()
    if args.skip_existing:
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".mp3"):
                existing.add(f.replace(".mp3", ""))

    todo = [c for c in cases if c["id"] not in existing]
    print(f"Total: {len(cases)}, existing: {len(existing)}, todo: {len(todo)}")

    success = 0
    failed = 0
    for i, case in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] {case['id']}: {case['title']}")
        ok, msg = generate_one(case)
        if ok:
            sz = download_from_hf(case["id"])
            print(f"  ✓ {msg} -> {sz/1024:.0f}KB local")
            success += 1
        else:
            print(f"  ✗ {msg}")
            failed += 1

        if i < len(todo) - 1:
            time.sleep(2)

    print(f"\nDone: {success} success, {failed} failed out of {len(todo)}")


if __name__ == "__main__":
    main()
