import os, subprocess, csv, time

DATASET_DIR = os.environ.get("DATASET_DIR", "/home/runner/workspace/Music-Generation-API/dataset_v3")
CSV_PATH = "/home/runner/workspace/training/top100_russian_pop_2025.csv"
os.makedirs(DATASET_DIR, exist_ok=True)

TRACKS = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        TRACKS.append((row["Artist"], row["Title"]))

print(f"Loaded {len(TRACKS)} tracks from CSV")

def safe_filename(artist, track):
    s = f"{artist}_-_{track}"
    for ch in ['/', '\\', '"', "'", '?', '*', '<', '>', '|', ':', '&', '.', ',', '(', ')', '$']:
        s = s.replace(ch, '')
    s = s.replace(' ', '_')
    while '__' in s:
        s = s.replace('__', '_')
    return s

subprocess.run(["pip", "install", "-q", "-U", "yt-dlp"], check=True)

def try_soundcloud(query, outpath):
    cmd = [
        "yt-dlp",
        f"scsearch1:{query}",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "--no-playlist",
        "--output", outpath.replace(".mp3", ".%(ext)s"),
        "--no-warnings",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

def try_vk(query, outpath):
    cmd = [
        "yt-dlp",
        f"vksearch1:{query}",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "--no-playlist",
        "--output", outpath.replace(".mp3", ".%(ext)s"),
        "--no-warnings",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

def check_downloaded(outpath, fname, dataset_dir):
    if os.path.exists(outpath):
        return True
    for f in os.listdir(dataset_dir):
        if f.startswith(fname) and not f.endswith('.part') and not f.endswith('.ytdl'):
            src = os.path.join(dataset_dir, f)
            if os.path.getsize(src) > 10000:
                return True
    return False

downloaded = 0
failed = []
skipped = 0

for i, (artist, track) in enumerate(TRACKS, 1):
    fname = safe_filename(artist, track)
    outpath = os.path.join(DATASET_DIR, f"{fname}.mp3")

    if check_downloaded(outpath, fname, DATASET_DIR):
        print(f"[{i}/{len(TRACKS)}] SKIP (exists): {artist} - {track}")
        downloaded += 1
        skipped += 1
        continue

    query = f"{artist} {track}"
    print(f"[{i}/{len(TRACKS)}] Downloading: {artist} - {track}...")

    success = False

    for source_name, source_fn in [("SoundCloud", try_soundcloud), ("VK", try_vk)]:
        try:
            result = source_fn(query, outpath)
            if check_downloaded(outpath, fname, DATASET_DIR):
                for f in os.listdir(DATASET_DIR):
                    if f.startswith(fname) and not f.endswith('.part') and not f.endswith('.ytdl'):
                        fpath = os.path.join(DATASET_DIR, f)
                        size_mb = os.path.getsize(fpath) / 1024 / 1024
                        print(f"  OK: {size_mb:.1f}MB ({source_name})")
                        break
                downloaded += 1
                success = True
                break
            else:
                stderr = result.stderr[-200:] if result.stderr else ""
                print(f"  {source_name}: not found, trying next...")
        except subprocess.TimeoutExpired:
            print(f"  {source_name}: TIMEOUT, trying next...")
        except Exception as e:
            print(f"  {source_name}: ERROR {e}")

    if not success:
        print(f"  FAILED on all sources")
        failed.append(f"{artist} - {track}")

    time.sleep(1)

print(f"\n{'='*60}")
print(f"Downloaded: {downloaded}/{len(TRACKS)} (skipped: {skipped})")
print(f"Failed: {len(failed)}")
if failed:
    print("\nFailed tracks:")
    for f in failed:
        print(f"  - {f}")

files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.mp3', '.m4a', '.opus', '.ogg'))]
print(f"\nTotal audio files in {DATASET_DIR}: {len(files)}")
