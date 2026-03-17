import json, os, subprocess, sys, time

SUNO_API_KEY = os.environ.get("SUNO_API_KEY")
BASE_URL = "https://api.sunoapi.org"

def curl_post(url, data):
    result = subprocess.run(
        ["curl", "-s", "--max-time", "30",
         "-X", "POST",
         "-H", f"Authorization: Bearer {SUNO_API_KEY}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(data),
         url],
        capture_output=True, text=True, timeout=35
    )
    return json.loads(result.stdout)

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "submit"

    with open("data/comparison_cases.json") as f:
        cases = json.load(f)

    tasks_file = "/tmp/suno_tasks.json"
    audio_dir = "public/audio/suno"
    os.makedirs(audio_dir, exist_ok=True)

    if action == "submit":
        tasks = {}
        if os.path.exists(tasks_file):
            tasks = json.load(open(tasks_file))

        for i, c in enumerate(cases):
            cid = c["id"]
            outfile = os.path.join(audio_dir, f"{cid}.mp3")

            if os.path.exists(outfile) and os.path.getsize(outfile) > 10000:
                print(f"[{i+1}] SKIP {cid} (already downloaded)")
                continue

            if cid in tasks:
                print(f"[{i+1}] SKIP {cid} (already submitted: {tasks[cid]})")
                continue

            body = {
                "customMode": True,
                "instrumental": False,
                "model": "V5",
                "prompt": c["lyrics"],
                "style": c["style_prompt"],
                "title": c["title"],
                "callBackUrl": "https://httpbin.org/post",
            }

            try:
                resp = curl_post(f"{BASE_URL}/api/v1/generate", body)
                if resp.get("code") == 200:
                    tid = resp["data"]["taskId"]
                    tasks[cid] = tid
                    print(f"[{i+1}] SUBMIT {cid}: taskId={tid}")
                else:
                    print(f"[{i+1}] FAIL {cid}: {resp.get('code')} {resp.get('msg')}")
            except Exception as e:
                print(f"[{i+1}] ERROR {cid}: {e}")

            with open(tasks_file, "w") as f:
                json.dump(tasks, f, indent=2)

            time.sleep(2)

        print(f"\nSubmitted {len(tasks)} tasks. Saved to {tasks_file}")

    elif action == "poll":
        if not os.path.exists(tasks_file):
            print("No tasks file. Run 'submit' first.")
            return

        tasks = json.load(open(tasks_file))
        done = 0
        pending = 0
        failed = 0

        for cid, tid in tasks.items():
            outfile = os.path.join(audio_dir, f"{cid}.mp3")
            if os.path.exists(outfile) and os.path.getsize(outfile) > 10000:
                done += 1
                continue

            try:
                result = subprocess.run(
                    ["curl", "-s", "--max-time", "15",
                     "-H", f"Authorization: Bearer {SUNO_API_KEY}",
                     f"{BASE_URL}/api/v1/generate/record-info?taskId={tid}"],
                    capture_output=True, text=True, timeout=20
                )
                resp = json.loads(result.stdout)
                if resp.get("code") != 200:
                    print(f"{cid}: API error")
                    pending += 1
                    continue

                status = resp["data"]["status"]
                if status in ("SUCCESS", "FIRST_SUCCESS", "CALLBACK_EXCEPTION"):
                    tracks = resp["data"].get("response", {}).get("sunoData", [])
                    if tracks:
                        url = tracks[0].get("audioUrl") or tracks[0].get("streamAudioUrl")
                        if url:
                            subprocess.run(["curl", "-sL", "--max-time", "60", "-o", outfile, url],
                                           check=True, timeout=65)
                            size = os.path.getsize(outfile)
                            print(f"{cid}: DOWNLOADED ({size // 1024}KB)")
                            done += 1
                            continue
                    pending += 1
                elif status in ("CREATE_TASK_FAILED", "GENERATE_AUDIO_FAILED", "SENSITIVE_WORD_ERROR"):
                    print(f"{cid}: FAILED ({status})")
                    failed += 1
                else:
                    print(f"{cid}: {status}")
                    pending += 1
            except Exception as e:
                print(f"{cid}: error: {e}")
                pending += 1

        print(f"\nDone: {done}, Pending: {pending}, Failed: {failed}")

    elif action == "status":
        if not os.path.exists(tasks_file):
            print("No tasks file.")
            return

        tasks = json.load(open(tasks_file))
        downloaded = sum(1 for cid in tasks
                        if os.path.exists(os.path.join(audio_dir, f"{cid}.mp3"))
                        and os.path.getsize(os.path.join(audio_dir, f"{cid}.mp3")) > 10000)
        print(f"Tasks: {len(tasks)}, Downloaded: {downloaded}, Remaining: {len(tasks) - downloaded}")

if __name__ == "__main__":
    main()
