#!/usr/bin/env python3
import os
import sys
import time
import json
import glob
import base64
import argparse
import subprocess
import tempfile
from pathlib import Path

import httpx

SUPPORTED_EXTENSIONS = {'.mp3', '.m4a', '.opus', '.ogg', '.wav', '.flac'}
MIME_TYPES = {
    '.mp3': 'audio/mpeg',
    '.m4a': 'audio/mp4',
    '.opus': 'audio/ogg',
    '.ogg': 'audio/ogg',
    '.wav': 'audio/wav',
    '.flac': 'audio/flac',
}

GENRE_CONFIGS = {
    "russianpop": {
        "genre_focus": "Genre: pop ballad, dance-pop, R&B pop, electro-pop, folk-pop, trap-pop, indie pop, synth-pop, etc.",
        "instruments_focus": "Instruments: list what you actually hear - piano, acoustic guitar, electric guitar, synths, pads, strings, brass, electronic drums, live drums, 808 bass, deep bass, etc.",
        "examples": [
            "russianpop, dance-pop, female vocal smooth and confident, synths and electronic drums and deep bass and claps, energetic and uplifting, uptempo 125 BPM, catchy chorus with club-ready beat and modern production, Russian lyrics",
            "russianpop, pop ballad, male vocal emotional tenor, piano and strings and soft pads, melancholic and romantic, slow 75 BPM, intimate verse building to powerful chorus, Russian lyrics",
            "russianpop, trap-pop, male vocal with autotune, 808 bass and hi-hats and atmospheric synths, dark and moody, mid-tempo 95 BPM, modern trap-influenced production with melodic hook, Russian lyrics",
        ],
    },
    "russianrap": {
        "genre_focus": "Genre: boom bap, trap, drill, cloud rap, mumble rap, conscious rap, gangsta rap, melodic rap, old school hip-hop, etc.",
        "instruments_focus": "Instruments: list what you actually hear - 808 bass, sub-bass, hi-hats (fast/slow), snares, kicks, sampled loops, piano chops, guitar loops, synth leads, pads, vocal chops, brass stabs, etc.",
        "examples": [
            "russianrap, trap, male vocal aggressive flow, 808 bass and fast hi-hats and heavy kicks and dark synth pads, aggressive and intense, uptempo 140 BPM, hard-hitting trap beat with rapid-fire delivery, Russian lyrics",
            "russianrap, cloud rap, male vocal with autotune melodic, atmospheric synths and reverbed 808 bass and soft hi-hats, dreamy and melancholic, mid-tempo 95 BPM, spacey production with emotional melodic rap, Russian lyrics",
            "russianrap, boom bap, male vocal sharp lyrical delivery, sampled piano loop and boom bap drums and vinyl crackle, nostalgic and raw, mid-tempo 90 BPM, old-school inspired beat with complex rhyme schemes, Russian lyrics",
        ],
    },
    "russiandisco": {
        "genre_focus": "Genre: italo disco, euro disco, synth-pop disco, hi-NRG, dance-pop, nu-disco, Soviet disco, retro disco, etc.",
        "instruments_focus": "Instruments: list what you actually hear - synths, drum machine, bass guitar, electric guitar, strings, brass section, handclaps, cowbell, sequenced bass, vocoder, analog synths, etc.",
        "examples": [
            "russiandisco, euro disco, female vocal bright and catchy, synths and drum machine and sequenced bass and handclaps, uplifting and danceable, uptempo 130 BPM, retro disco production with infectious chorus and 80s feel, Russian lyrics",
            "russiandisco, synth-pop disco, male vocal smooth and romantic, analog synths and drum machine and electric bass and strings, romantic and nostalgic, uptempo 120 BPM, 80s-inspired synth-driven dance track, Russian lyrics",
            "russiandisco, nu-disco, female vocal energetic, modern synths and four-on-the-floor kick and funky bass and filtered pads, fun and groovy, uptempo 125 BPM, modern take on classic disco with polished production, Russian lyrics",
        ],
    },
    "russianshanson": {
        "genre_focus": "Genre: chanson, blatnyak, criminal chanson, lyrical chanson, urban chanson, prison song, romantic chanson, author song chanson, etc.",
        "instruments_focus": "Instruments: list what you actually hear - acoustic guitar, seven-string guitar, accordion, piano, bass, violin, mandolin, bayan, light percussion, brushed drums, etc.",
        "examples": [
            "russianshanson, criminal chanson, male vocal raspy and emotional baritone, acoustic guitar and accordion and light bass, melancholic and raw, mid-tempo 100 BPM, storytelling delivery with intimate stripped-back arrangement, Russian lyrics",
            "russianshanson, lyrical chanson, male vocal deep and powerful, piano and strings and acoustic guitar and soft drums, nostalgic and sentimental, slow 80 BPM, emotional ballad with orchestral touches and expressive vocal, Russian lyrics",
            "russianshanson, urban chanson, male vocal gritty tenor, seven-string guitar and bayan and bass and brushed drums, bittersweet and reflective, mid-tempo 95 BPM, traditional chanson with modern production elements, Russian lyrics",
        ],
    },
    "russianrock": {
        "genre_focus": "Genre: post-punk, punk rock, hard rock, alternative rock, indie rock, grunge, progressive rock, folk rock, new wave, garage rock, etc.",
        "instruments_focus": "Instruments: list what you actually hear - electric guitar (distorted/clean/overdriven), bass guitar, live drums, acoustic guitar, keyboards, organ, synthesizer, tambourine, harmonica, etc.",
        "examples": [
            "russianrock, post-punk, male vocal baritone cold and detached, jangly electric guitar and driving bass and tight drums, dark and atmospheric, mid-tempo 110 BPM, cold wave influenced post-punk with reverb-drenched guitars, Russian lyrics",
            "russianrock, hard rock, male vocal powerful and raspy, heavy distorted electric guitar and bass guitar and powerful drums, aggressive and energetic, fast 145 BPM, heavy riff-driven rock with anthemic chorus, Russian lyrics",
            "russianrock, alternative rock, female vocal ethereal and emotional, clean electric guitar and distorted guitar layers and bass and drums, melancholic and dreamy, mid-tempo 105 BPM, 90s alternative influenced with dynamic quiet-loud shifts, Russian lyrics",
        ],
    },
    "russianclassical": {
        "genre_focus": "Genre: symphony, concerto, sonata, chamber music, ballet, opera, choral, orchestral suite, piano solo, string quartet, etc.",
        "instruments_focus": "Instruments: list what you actually hear - full orchestra, strings (violin/viola/cello/double bass), woodwinds (flute/oboe/clarinet/bassoon), brass (trumpet/horn/trombone/tuba), timpani, percussion, piano, harp, choir, etc.",
        "examples": [
            "russianclassical, symphony, orchestral, full orchestra with prominent strings and brass fanfares and timpani, dramatic and majestic, allegro 130 BPM, sweeping romantic-era orchestral work with powerful dynamic contrasts, instrumental",
            "russianclassical, piano concerto, solo piano with orchestra, grand piano and strings and woodwinds and soft brass, virtuosic and emotional, moderato 100 BPM, romantic piano concerto with brilliant passages and lyrical melodies, instrumental",
            "russianclassical, ballet suite, orchestral, strings and flute and harp and celesta and light percussion, graceful and enchanting, andante 80 BPM, delicate ballet music with elegant melodic lines and colorful orchestration, instrumental",
        ],
    },
    "russianelectro": {
        "genre_focus": "Genre: techno, house, synthwave, electro-pop, industrial, IDM, ambient electronic, drum and bass, breakbeat, trance, experimental electronic, etc.",
        "instruments_focus": "Instruments: list what you actually hear - synthesizers, drum machine, sequencer, bass synth, arpeggiator, sampler, vocoder, distorted synths, modular synths, sub-bass, electronic percussion, etc.",
        "examples": [
            "russianelectro, synth-pop, female vocal icy and hypnotic, analog synths and drum machine and pulsing bass and arpeggiator, dark and futuristic, uptempo 125 BPM, retro-futuristic synth-pop with cold wave influences, Russian lyrics",
            "russianelectro, industrial techno, male vocal distorted and aggressive, heavy distorted synths and pounding kick and industrial percussion, dark and intense, fast 140 BPM, hard industrial electronic with abrasive textures, Russian lyrics",
            "russianelectro, experimental electronic, no vocals, modular synths and glitchy percussion and ambient pads and sub-bass, eerie and atmospheric, mid-tempo 100 BPM, experimental electronic composition with textural sound design, instrumental",
        ],
    },
    "russianjazz": {
        "genre_focus": "Genre: bebop, cool jazz, fusion, smooth jazz, vocal jazz, swing, free jazz, latin jazz, jazz-funk, etc.",
        "instruments_focus": "Instruments: list what you actually hear - saxophone (alto/tenor/soprano), trumpet, piano, double bass, drums with brushes/sticks, vibraphone, guitar (clean archtop), trombone, clarinet, organ, etc.",
        "examples": [
            "russianjazz, bebop, male saxophone lead, tenor saxophone and piano comping and walking double bass and swinging drums with brushes, energetic and virtuosic, uptempo 180 BPM, classic bebop with complex improvisation and tight rhythm section, instrumental",
            "russianjazz, vocal jazz, female vocal smooth and sultry, piano trio and double bass and brushed drums and soft trumpet, romantic and intimate, slow 70 BPM, classic jazz standard interpretation with sophisticated harmony, Russian lyrics",
            "russianjazz, fusion, instrumental, electric guitar and Rhodes piano and bass guitar and drums and saxophone, groovy and sophisticated, mid-tempo 110 BPM, jazz-rock fusion with complex time signatures and extended solos, instrumental",
        ],
    },
    "russianfolk": {
        "genre_focus": "Genre: traditional folk, folk-pop, ethno-folk, choral folk, dance folk, ritual song, bylina, chastushka, folk-rock, etc.",
        "instruments_focus": "Instruments: list what you actually hear - balalaika, domra, bayan, accordion, gusli, zhaleika, wooden flute, tambourine, spoons, choir, fiddle, gudok, etc.",
        "examples": [
            "russianfolk, traditional folk, female choir polyphonic, balalaika and bayan and tambourine and wooden spoons, festive and joyful, uptempo 120 BPM, traditional Russian polyphonic singing with authentic folk instrumentation, Russian lyrics",
            "russianfolk, ethno-folk, female vocal powerful and clear, balalaika and domra and drums and modern bass, energetic and modern, uptempo 130 BPM, traditional melodies reimagined with contemporary arrangement, Russian lyrics",
            "russianfolk, folk ballad, female vocal pure and resonant, gusli and wooden flute and soft strings, wistful and serene, slow 70 BPM, ancient folk melody with meditative quality and natural acoustic space, Russian lyrics",
        ],
    },
    "russianacoustic": {
        "genre_focus": "Genre: bard song, singer-songwriter, acoustic ballad, campfire song, unplugged, author song, acoustic folk, fingerstyle, etc.",
        "instruments_focus": "Instruments: list what you actually hear - acoustic guitar (nylon/steel-string), voice, harmonica, mandolin, violin, cello, flute, light percussion, cajon, etc.",
        "examples": [
            "russianacoustic, bard song, male vocal warm baritone, nylon-string acoustic guitar fingerpicking, intimate and reflective, slow 75 BPM, classic Russian bard tradition with poetic lyrics and minimal arrangement, Russian lyrics",
            "russianacoustic, singer-songwriter, male vocal gentle tenor, steel-string acoustic guitar strumming and harmonica, nostalgic and wistful, mid-tempo 100 BPM, heartfelt storytelling with simple honest arrangement, Russian lyrics",
            "russianacoustic, acoustic ballad, female vocal clear soprano, acoustic guitar and cello and soft violin, emotional and tender, slow 65 BPM, delicate acoustic arrangement with expressive vocal performance, Russian lyrics",
        ],
    },
    "russianphonk": {
        "genre_focus": "Genre: drift phonk, phonk house, Memphis phonk, dark phonk, cowbell phonk, aggressive phonk, etc.",
        "instruments_focus": "Instruments: list what you actually hear - distorted 808 bass, cowbell, hi-hats, phonk kick, chopped vocal samples, dark synth stabs, vinyl crackle, Memphis-style samples, brass stabs, drift bass, etc.",
        "examples": [
            "russianphonk, drift phonk, no vocals, distorted 808 bass and cowbell and aggressive hi-hats and dark synth stabs, aggressive and dark, fast 150 BPM, hard-hitting drift phonk with heavy bass distortion and cowbell pattern, instrumental",
            "russianphonk, phonk house, chopped vocal samples, deep bass and four-on-the-floor kick and cowbell and filtered synths, dark and groovy, uptempo 130 BPM, club-ready phonk house with Memphis-inspired vocal chops, instrumental",
            "russianphonk, Memphis phonk, male vocal distorted, lo-fi 808 bass and rattling hi-hats and vinyl crackle and dark pads, menacing and raw, mid-tempo 70 BPM half-time, lo-fi Memphis inspired phonk with raw gritty aesthetic, Russian lyrics",
        ],
    },
    "russianrnb": {
        "genre_focus": "Genre: contemporary R&B, neo-soul, slow jam, R&B pop, alternative R&B, trap-soul, etc.",
        "instruments_focus": "Instruments: list what you actually hear - smooth synth pads, electric piano (Rhodes/Wurlitzer), bass guitar, programmed drums, snaps, 808 bass, layered vocals, guitar (clean/muted), strings, etc.",
        "examples": [
            "russianrnb, contemporary R&B, male vocal smooth and soulful, Rhodes piano and smooth bass and programmed drums and lush pads, romantic and sensual, mid-tempo 90 BPM, modern R&B with silky vocal runs and polished production, Russian lyrics",
            "russianrnb, trap-soul, male vocal emotional with light autotune, 808 bass and soft hi-hats and atmospheric pads and guitar, melancholic and moody, slow 75 BPM, trap-influenced R&B with vulnerable emotional delivery, Russian lyrics",
            "russianrnb, neo-soul, female vocal warm and rich, electric piano and live bass and drums with ghost notes and horn section, groovy and warm, mid-tempo 95 BPM, organic neo-soul with live instrumentation feel, Russian lyrics",
        ],
    },
    "russianballad": {
        "genre_focus": "Genre: power ballad, pop ballad, rock ballad, orchestral ballad, piano ballad, romantic ballad, dramatic ballad, etc.",
        "instruments_focus": "Instruments: list what you actually hear - piano, strings (orchestra/quartet), acoustic guitar, electric guitar (clean/light distortion), soft drums, bass, choir, harp, celesta, etc.",
        "examples": [
            "russianballad, power ballad, female vocal powerful soprano, piano and full orchestra strings and electric guitar building and drums, dramatic and emotional, slow building to 85 BPM, intimate piano opening building to soaring orchestral climax, Russian lyrics",
            "russianballad, piano ballad, male vocal emotional tenor, solo piano and soft strings and gentle cello, tender and heartfelt, slow 65 BPM, minimalist piano-driven ballad with raw vocal emotion, Russian lyrics",
            "russianballad, orchestral ballad, male vocal rich baritone, full orchestra and piano and choir, epic and cinematic, slow 70 BPM, grand orchestral arrangement with sweeping dynamic arc from quiet to triumphant, Russian lyrics",
        ],
    },
}

VALID_TRIGGER_WORDS = list(GENRE_CONFIGS.keys())


def build_caption_prompt(trigger_word):
    config = GENRE_CONFIGS[trigger_word]

    lyrics_note = "Russian lyrics" if trigger_word != "russianclassical" else "Russian lyrics (if vocals present, otherwise 'instrumental')"

    examples_text = "\n".join(config["examples"])

    prompt = f"""Listen to this audio track carefully and describe it for an AI music generation training dataset.

Respond with ONLY a single-line caption in this exact format:
{trigger_word}, <genre/subgenre>, <vocal type>, <instruments>, <mood/emotion>, <tempo feel>, <additional characteristics>, {lyrics_note}

Rules:
1. Start with the trigger word "{trigger_word}"
2. Be specific and accurate about what you actually hear
3. Keep it as ONE line, comma-separated descriptors
4. Use English for the description

Be specific about:
- {config["genre_focus"]}
- Vocal: male/female, duet, tenor/alto/soprano, raspy/smooth/powerful/breathy/autotuned, choir, or no vocals
- {config["instruments_focus"]}
- Mood: melancholic, uplifting, romantic, aggressive, nostalgic, dreamy, energetic, emotional, dark, playful, etc.
- Tempo: slow (~70-90 BPM), mid-tempo (~90-110 BPM), uptempo (~110-130 BPM), fast (~130+ BPM)
- Production: modern, retro, minimalist, layered, polished, lo-fi, stadium, intimate, etc.

Example outputs:
{examples_text}

Respond with ONLY the caption line, nothing else."""

    return prompt


MAX_FILE_SIZE_BYTES = 7 * 1024 * 1024
PROGRESS_FILE = "caption_progress.json"


def get_audio_files(dataset_dir):
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(dataset_dir, f"*{ext}")))
    files.sort()
    return files


def load_progress(dataset_dir):
    progress_path = os.path.join(dataset_dir, PROGRESS_FILE)
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(dataset_dir, progress):
    progress_path = os.path.join(dataset_dir, PROGRESS_FILE)
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def prepare_audio_for_api(audio_path):
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run([
            'ffmpeg', '-i', audio_path,
            '-ss', '15', '-t', '45',
            '-ac', '1', '-ar', '16000', '-b:a', '48k',
            '-y', tmp_path
        ], capture_output=True, check=True)
        with open(tmp_path, 'rb') as f:
            data = f.read()
        return data, 'audio/mpeg'
    finally:
        os.unlink(tmp_path)


def generate_caption_http(audio_path, trigger_word):
    base_url = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY", "")

    caption_prompt = build_caption_prompt(trigger_word)

    audio_data, mime_type = prepare_audio_for_api(audio_path)

    audio_b64 = base64.standard_b64encode(audio_data).decode('utf-8')

    url = f"{base_url}/models/gemini-2.5-flash:generateContent"

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": caption_prompt},
                {"inlineData": {"mimeType": mime_type, "data": audio_b64}}
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 8192
        }
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    for attempt in range(10):
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload, headers=headers)

            if response.status_code == 429:
                wait_time = min(60 * (2 ** attempt), 900)
                print(f"RATE LIMITED (attempt {attempt+1}/10), waiting {wait_time}s ... ", end='', flush=True)
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip()

        except httpx.TimeoutException:
            if attempt < 9:
                wait_time = 60
                print(f"TIMEOUT (attempt {attempt+1}/10), waiting {wait_time}s ... ", end='', flush=True)
                time.sleep(wait_time)
                continue
            raise

    raise Exception("Max retries exceeded (all 429)")


def main():
    parser = argparse.ArgumentParser(description='Generate captions for audio files using Gemini AI')
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    parser.add_argument('--genre', type=str, default='russianpop',
                        choices=VALID_TRIGGER_WORDS,
                        help=f'Genre trigger word (default: russianpop). Options: {", ".join(VALID_TRIGGER_WORDS)}')
    parser.add_argument('--force', action='store_true', help='Regenerate all captions (ignore progress)')
    parser.add_argument('--delay', type=float, default=15.0, help='Delay between requests in seconds (default: 15)')
    parser.add_argument('--start-from', type=int, default=0, help='Start from track number (0-indexed)')
    parser.add_argument('--batch', type=int, default=0, help='Process N tracks then exit (0=all)')
    args = parser.parse_args()

    trigger_word = args.genre
    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Directory not found: {dataset_dir}")
        sys.exit(1)

    print(f"Genre: {trigger_word}")

    audio_files = get_audio_files(dataset_dir)
    total = len(audio_files)
    print(f"Found {total} audio files in {dataset_dir}")

    progress = load_progress(dataset_dir) if not args.force else {"completed": [], "failed": []}

    completed = 0
    skipped = 0
    failed = 0

    for i, audio_path in enumerate(audio_files):
        if i < args.start_from:
            continue

        basename = os.path.basename(audio_path)
        txt_path = os.path.splitext(audio_path)[0] + '.txt'

        if basename in progress["completed"] and not args.force:
            skipped += 1
            continue

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"[{i+1}/{total}] {basename} ({file_size_mb:.1f} MB) ... ", end='', flush=True)

        try:
            caption = generate_caption_http(audio_path, trigger_word)
            if caption:
                if not caption.startswith(trigger_word):
                    caption = trigger_word + ', ' + caption
                caption = caption.replace('\n', ' ').strip()
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                if basename not in progress["completed"]:
                    progress["completed"].append(basename)
                if basename in progress["failed"]:
                    progress["failed"].remove(basename)
                save_progress(dataset_dir, progress)
                completed += 1
                print(f"OK ({len(caption)} chars)")
            else:
                failed += 1
                print("EMPTY RESPONSE")
                if basename not in progress["failed"]:
                    progress["failed"].append(basename)
                save_progress(dataset_dir, progress)

        except Exception as e:
            failed += 1
            error_msg = str(e)[:150]
            print(f"ERROR: {error_msg}")
            if basename not in progress["failed"]:
                progress["failed"].append(basename)
            save_progress(dataset_dir, progress)

        if args.batch > 0 and completed >= args.batch:
            print(f"\nBatch limit reached ({args.batch} tracks)")
            break

        if i < len(audio_files) - 1:
            time.sleep(args.delay)

    print(f"\nDone! Completed: {completed}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total with captions: {len(progress['completed'])}/{total}")
    if progress["failed"]:
        print(f"Failed tracks: {progress['failed']}")


if __name__ == '__main__':
    main()
