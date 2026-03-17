#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
import tempfile
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_.')
    return name[:120]


def check_dependencies():
    for cmd in ['yt-dlp', 'ffmpeg']:
        result = subprocess.run(['which', cmd], capture_output=True)
        if result.returncode != 0:
            print(f"ERROR: {cmd} not found. Install it first.")
            sys.exit(1)


def search_youtube(query, max_results=5):
    cmd = [
        'yt-dlp',
        '--flat-playlist',
        '--print', '%(id)s\t%(title)s\t%(duration)s',
        f'ytsearch{max_results}:{query}',
        '--no-warnings',
        '--quiet',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return []

    entries = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) >= 2:
            video_id = parts[0]
            title = parts[1]
            try:
                duration = int(float(parts[2])) if len(parts) > 2 else 0
            except (ValueError, IndexError):
                duration = 0
            if 30 < duration < 600:
                entries.append({
                    'id': video_id,
                    'title': title,
                    'duration': duration,
                })
    return entries


def download_track(video_id, output_path, target_sr=48000):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, 'audio.%(ext)s')
        cmd = [
            'yt-dlp',
            '-x',
            '--audio-format', 'mp3',
            '--audio-quality', '0',
            '-o', tmp_file,
            '--no-playlist',
            '--no-warnings',
            '--quiet',
            f'https://www.youtube.com/watch?v={video_id}',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise Exception(f"yt-dlp failed: {result.stderr[:200]}")

        downloaded = None
        for f in os.listdir(tmpdir):
            if f.startswith('audio'):
                downloaded = os.path.join(tmpdir, f)
                break

        if not downloaded:
            raise Exception("No audio file downloaded")

        cmd_ffmpeg = [
            'ffmpeg', '-i', downloaded,
            '-ar', str(target_sr),
            '-ac', '2',
            '-b:a', '192k',
            '-y',
            output_path,
        ]
        result = subprocess.run(cmd_ffmpeg, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise Exception(f"ffmpeg failed: {result.stderr[:200]}")

    return os.path.exists(output_path)


def get_audio_duration(path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip())
    return 0


def collect_for_artist(artist, genre, output_dir, tracks_per_artist=7, existing_files=None):
    if existing_files is None:
        existing_files = set()

    query = f"{artist} {genre} official audio"
    results = search_youtube(query, max_results=tracks_per_artist + 5)

    if not results:
        query_simple = f"{artist} песня"
        results = search_youtube(query_simple, max_results=tracks_per_artist + 3)

    downloaded = []
    for entry in results:
        if len(downloaded) >= tracks_per_artist:
            break

        filename = sanitize_filename(f"{artist}_-_{entry['title']}")
        output_path = os.path.join(output_dir, f"{filename}.mp3")

        if filename in existing_files or os.path.exists(output_path):
            continue

        try:
            download_track(entry['id'], output_path)
            duration = get_audio_duration(output_path)
            if duration < 30:
                os.remove(output_path)
                continue

            downloaded.append({
                'artist': artist,
                'title': entry['title'],
                'file': f"{filename}.mp3",
                'duration': duration,
                'video_id': entry['id'],
            })
            existing_files.add(filename)
        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            continue

    return downloaded


def load_genre_config(config_path, genre):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if genre not in config:
        available = ', '.join(config.keys())
        print(f"ERROR: Genre '{genre}' not found. Available: {available}")
        sys.exit(1)

    return config[genre]


def main():
    parser = argparse.ArgumentParser(description='Collect music dataset for LoRA training')
    parser.add_argument('genre', help='Genre key from genre_artists.json (e.g., rap, rock, disco)')
    parser.add_argument('--config', default=None, help='Path to genre_artists.json')
    parser.add_argument('--output', default=None, help='Output directory (default: datasets/<genre>)')
    parser.add_argument('--tracks-per-artist', type=int, default=7, help='Max tracks per artist (default: 7)')
    parser.add_argument('--target-total', type=int, default=150, help='Target total tracks (default: 150)')
    parser.add_argument('--artists', nargs='+', help='Override artists list (space-separated)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous collection')
    args = parser.parse_args()

    check_dependencies()

    if args.config is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, '..', '..', 'ace-step-runpod-serverless', 'training', 'genre_artists.json'),
            os.path.join(script_dir, 'genre_artists.json'),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                args.config = p
                break
        if args.config is None:
            print("ERROR: genre_artists.json not found. Use --config to specify path.")
            sys.exit(1)

    genre_config = load_genre_config(args.config, args.genre)
    artists = args.artists or genre_config.get('artists', [])
    genre_label = genre_config.get('genre_label', args.genre)
    search_genre = genre_config.get('search_term', genre_label)

    if not artists:
        print(f"ERROR: No artists found for genre '{args.genre}'")
        sys.exit(1)

    if args.output is None:
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', args.genre)

    os.makedirs(args.output, exist_ok=True)

    progress_path = os.path.join(args.output, 'collection_progress.json')
    progress = {"downloaded": [], "failed_artists": []}
    existing_files = set()

    if args.resume and os.path.exists(progress_path):
        with open(progress_path, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        existing_files = {os.path.splitext(d['file'])[0] for d in progress['downloaded']}
        print(f"Resuming: {len(progress['downloaded'])} tracks already downloaded")

    for f in os.listdir(args.output):
        if f.endswith('.mp3'):
            existing_files.add(os.path.splitext(f)[0])

    total_downloaded = len(existing_files)
    print(f"\n=== Collecting '{genre_label}' dataset ===")
    print(f"Artists: {len(artists)}")
    print(f"Target: {args.target_total} tracks")
    print(f"Tracks per artist: {args.tracks_per_artist}")
    print(f"Output: {args.output}")
    print(f"Already have: {total_downloaded} tracks")
    print()

    for i, artist in enumerate(artists):
        if total_downloaded >= args.target_total:
            print(f"\nTarget reached ({args.target_total} tracks)")
            break

        remaining = args.target_total - total_downloaded
        tracks_for_artist = min(args.tracks_per_artist, remaining)

        print(f"[{i+1}/{len(artists)}] {artist} (need {tracks_for_artist} tracks) ... ", end='', flush=True)

        try:
            tracks = collect_for_artist(
                artist, search_genre, args.output,
                tracks_per_artist=tracks_for_artist,
                existing_files=existing_files,
            )
            if tracks:
                progress['downloaded'].extend(tracks)
                total_downloaded += len(tracks)
                print(f"OK ({len(tracks)} tracks, total: {total_downloaded})")
            else:
                print("NO TRACKS FOUND")
                if artist not in progress['failed_artists']:
                    progress['failed_artists'].append(artist)
        except Exception as e:
            print(f"ERROR: {str(e)[:100]}")
            if artist not in progress['failed_artists']:
                progress['failed_artists'].append(artist)

        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    audio_count = len([f for f in os.listdir(args.output) if f.endswith('.mp3')])
    print(f"\n=== Collection complete ===")
    print(f"Total audio files: {audio_count}")
    print(f"Output directory: {args.output}")
    if progress['failed_artists']:
        print(f"Failed artists: {progress['failed_artists']}")
    print(f"\nNext step: generate captions with:")
    print(f"  python scripts/generate_captions_gemini.py {args.output} --genre {args.genre}")


if __name__ == '__main__':
    main()
