#!/usr/bin/env python3
"""Test RunPod Serverless endpoint.

Usage:
    RUNPOD_API_KEY=rpa_... python3 test_endpoint.py --endpoint-id ENDPOINT_ID
"""

import os
import sys
import json
import time
import base64
import argparse
import urllib.request

API_KEY = os.environ.get("RUNPOD_API_KEY", "")


def run_sync(endpoint_id, payload, timeout=600):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    data = json.dumps({"input": payload, "timeout": timeout}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
        return json.loads(resp.read())


def run_async(endpoint_id, payload):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    data = json.dumps({"input": payload}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def check_status(endpoint_id, job_id):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Test RunPod endpoint")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--duration", type=float, default=60, help="Track duration (seconds)")
    parser.add_argument("--output", default="test_output.wav", help="Output file")
    parser.add_argument("--async-mode", action="store_true", help="Use async mode")
    parser.add_argument("--caption", default="voice_macan, Russian hip-hop/rap track with emotional male vocals, atmospheric synthesizer pads, deep 808 bass, crisp hi-hats, mid-tempo trap beat 85 BPM minor key")
    parser.add_argument("--lyrics", default="")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        sys.exit(1)

    payload = {
        "caption": args.caption,
        "lyrics": args.lyrics,
        "duration": args.duration,
        "seed": 42,
        "infer_steps": 60,
        "guidance": 7.0,
        "mastering": True,
    }

    print(f"Sending request to endpoint {args.endpoint_id}...")
    print(f"  Duration: {args.duration}s")
    t0 = time.time()

    if args.async_mode:
        result = run_async(args.endpoint_id, payload)
        job_id = result["id"]
        print(f"  Job ID: {job_id}")
        print(f"  Status: {result['status']}")

        while True:
            time.sleep(10)
            status = check_status(args.endpoint_id, job_id)
            print(f"  Status: {status['status']} ({time.time()-t0:.0f}s)")
            if status["status"] in ("COMPLETED", "FAILED"):
                result = status
                break
    else:
        result = run_sync(args.endpoint_id, payload)

    elapsed = time.time() - t0

    if result.get("status") == "COMPLETED":
        output = result["output"]
        print(f"\nGeneration successful!")
        print(f"  Duration: {output['duration']:.1f}s")
        print(f"  Gen time: {output['generation_time']:.1f}s")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Sample rate: {output['sample_rate']}")

        audio_bytes = base64.b64decode(output["audio_base64"])
        with open(args.output, "wb") as f:
            f.write(audio_bytes)
        print(f"  Saved to: {args.output} ({len(audio_bytes)/1e6:.1f}MB)")
    else:
        print(f"\nFailed: {result.get('status')}")
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
