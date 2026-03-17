#!/usr/bin/env python3
import json, os, random, shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "audio")
SUNO_DIR = os.path.join(AUDIO_DIR, "suno")
ACE_DIR = os.path.join(AUDIO_DIR, "ace-step")

CASES_FILE = os.path.join(DATA_DIR, "comparison_cases.json")
MAPPING_FILE = os.path.join(DATA_DIR, "track_mapping.json")

random.seed(42)


def main():
    with open(CASES_FILE) as f:
        cases = json.load(f)

    mapping = []
    ok = 0
    skip = 0

    for case in cases:
        cid = case["id"]
        suno_path = os.path.join(SUNO_DIR, f"{cid}.mp3")
        ace_path = os.path.join(ACE_DIR, f"{cid}.mp3")

        if not os.path.exists(suno_path):
            print(f"[SKIP] {cid}: no Suno track")
            skip += 1
            continue
        if not os.path.exists(ace_path):
            print(f"[SKIP] {cid}: no ACE-Step track")
            skip += 1
            continue

        ace_is_a = random.random() < 0.5

        if ace_is_a:
            src_a, src_b = ace_path, suno_path
            engine_a, engine_b = "ace-step", "suno"
        else:
            src_a, src_b = suno_path, ace_path
            engine_a, engine_b = "suno", "ace-step"

        dst_a = os.path.join(AUDIO_DIR, f"{cid}_a.mp3")
        dst_b = os.path.join(AUDIO_DIR, f"{cid}_b.mp3")

        shutil.copy2(src_a, dst_a)
        shutil.copy2(src_b, dst_b)

        mapping.append({
            "id": cid,
            "category": case["category"],
            "title": case["title"],
            "track_a_engine": engine_a,
            "track_b_engine": engine_b,
        })
        ok += 1
        print(f"[OK] {cid}: A={engine_a}, B={engine_b}")

    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    ace_count = sum(1 for m in mapping if m["track_a_engine"] == "ace-step")
    suno_count = sum(1 for m in mapping if m["track_a_engine"] == "suno")
    print(f"\n=== DONE === Paired: {ok}, Skipped: {skip}")
    print(f"A=ACE-Step: {ace_count}, A=Suno: {suno_count}")
    print(f"Mapping saved to {MAPPING_FILE}")


if __name__ == "__main__":
    main()
