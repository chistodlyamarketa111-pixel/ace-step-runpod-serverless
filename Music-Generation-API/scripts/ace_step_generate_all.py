#!/usr/bin/env python3
"""Generate ACE-Step tracks for all comparison cases.

Uses RunPod Serverless generate action with LoRA support.
Skips already-generated tracks (idempotent).
Waits for GPU if throttled.

Usage:
    python3 scripts/ace_step_generate_all.py           # Generate all
    python3 scripts/ace_step_generate_all.py --start 6  # Start from case 6
"""
import json, os, sys, time, base64, hashlib, urllib.request, argparse

ENDPOINT_ID = os.environ.get("ACESTEP_ENDPOINT_ID", "u07qbeocmy4479")
API_KEY = os.environ["RUNPOD_API_KEY"]
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

CASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "comparison_cases.json")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "audio", "ace-step")

LORA_NAME = "russianrap-v2"
LORA_SCALE = 0.7
LORA_REVISION = "epoch_10"
DURATION = 60
INFERENCE_STEPS = 27

POLL_INTERVAL = 10
JOB_TIMEOUT = 600
GPU_WAIT_INTERVAL = 60
MAX_GPU_WAIT = 3600


def api(method, path, data=None, timeout=60):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_for_gpu():
    start = time.time()
    while time.time() - start < MAX_GPU_WAIT:
        try:
            h = api("GET", "/health")
            w = h["workers"]
            active = w["idle"] + w["running"] + w["initializing"]
            if active > 0:
                return True
        except:
            pass
        elapsed = int(time.time() - start)
        if elapsed % 300 == 0:
            print(f"  GPU wait: {elapsed}s...", flush=True)
        time.sleep(GPU_WAIT_INTERVAL)
    return False


def stable_seed(case_id):
    return int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 2**31


def generate_and_save(case):
    seed = stable_seed(case["id"])
    out_path = os.path.join(OUTPUT_DIR, f"{case['id']}.mp3")

    result = api("POST", "/run", {
        "input": {
            "action": "generate",
            "prompt": case["style_prompt"],
            "lyrics": case["lyrics"],
            "audio_duration": DURATION,
            "inference_steps": INFERENCE_STEPS,
            "seed": seed,
            "lora_name": LORA_NAME,
            "lora_scale": LORA_SCALE,
            "lora_revision": LORA_REVISION,
        }
    })
    job_id = result["id"]

    start = time.time()
    while time.time() - start < JOB_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        try:
            st = api("GET", f"/status/{job_id}")
        except Exception:
            continue

        status = st["status"]
        if status == "COMPLETED":
            ab = st.get("output", {}).get("audio_base64", "")
            if ab and len(ab) > 1000:
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(ab))
                gt = st.get("output", {}).get("generation_time", "?")
                sz = os.path.getsize(out_path)
                return True, f"{sz // 1024}KB gen={gt}s"
            return False, "no audio in output"
        elif status == "FAILED":
            err = st.get("error", "unknown")
            return False, f"FAILED: {err[:150]}"

    return False, "timeout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()

    with open(CASES_FILE) as f:
        cases = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = set()
    for c in cases:
        p = os.path.join(OUTPUT_DIR, f"{c['id']}.mp3")
        if os.path.exists(p) and os.path.getsize(p) > 10000:
            existing.add(c["id"])

    todo = [c for c in cases if c["id"] not in existing]
    if args.start > 1:
        todo = [c for c in cases[args.start - 1:] if c["id"] not in existing]

    completed = len(existing)
    failed = 0
    print(f"Total: {len(cases)}, Done: {completed}, Todo: {len(todo)}", flush=True)

    if not todo:
        print("All tracks generated!", flush=True)
        return

    for i, case in enumerate(todo):
        print(f"\n[{i + 1}/{len(todo)}] {case['id']} '{case['title']}'", flush=True)

        if not wait_for_gpu():
            print("  GPU wait timeout, exiting", flush=True)
            break

        try:
            ok, info = generate_and_save(case)
            if ok:
                completed += 1
                print(f"  OK: {info} ({completed}/50)", flush=True)
            else:
                failed += 1
                print(f"  FAIL: {info}", flush=True)
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}", flush=True)

        time.sleep(2)

    print(f"\n=== Done: {completed}/50, Failed: {failed} ===", flush=True)


if __name__ == "__main__":
    main()
