import os, subprocess, json, time

DATASET_DIR = "/workspace/dataset_v2"
os.makedirs(DATASET_DIR, exist_ok=True)

TRACKS = [
    ("Анна Асти", "По барам"),
    ("Анна Асти", "Царица"),
    ("Анна Асти", "Феникс"),
    ("Анна Асти", "Повело"),
    ("Анна Асти", "Целуешь другую"),
    ("Анна Асти", "Анечка"),
    ("Анна Асти", "Ночью на кухне"),
    ("Анна Асти", "Хищная"),
    ("Анна Асти", "Шоколад"),
    ("Анна Асти", "Высшие силы"),
    ("Анна Асти", "Преданный бывший"),
    ("Анна Асти", "Разбуди меня"),
    ("Анна Асти", "Без тебя"),
    ("Анна Асти", "Стекло"),
    ("Zivert", "Life"),
    ("Zivert", "Beverly Hills"),
    ("Zivert", "Ещё хочу"),
    ("Zivert", "Credo"),
    ("Zivert", "Многоточия"),
    ("Zivert", "Анестезия"),
    ("Клава Кока", "Мне пох"),
    ("Клава Кока", "Краш"),
    ("Клава Кока", "Покинула чат"),
    ("Клава Кока", "Влюблена"),
    ("Полина Гагарина", "Кукушка"),
    ("Полина Гагарина", "Миллион голосов"),
    ("Полина Гагарина", "Навек"),
    ("Полина Гагарина", "Выше головы"),
    ("Ёлка", "Прованс"),
    ("Ёлка", "Около тебя"),
    ("Ёлка", "Грею счастье"),
    ("Нюша", "Цунами"),
    ("Нюша", "Выбирать чудо"),
    ("Нюша", "Целуй"),
    ("Ханна", "Без тебя я не могу"),
    ("Ханна", "Потеряла голову"),
    ("Мот", "Капкан"),
    ("Мот", "Когда исчезнет слово"),
    ("Мот", "Соло на двоих"),
    ("Мот", "Ая"),
    ("Егор Крид", "Самая самая"),
    ("Егор Крид", "Голубые глаза"),
    ("Егор Крид", "Будильник"),
    ("Егор Крид", "Сердцеедка"),
    ("Макс Барских", "Туманы"),
    ("Макс Барских", "Неверная"),
    ("Макс Барских", "Моя любовь"),
    ("HammAli & Navai", "Птичка"),
    ("HammAli & Navai", "Пустите меня на танцпол"),
    ("HammAli & Navai", "Девочка-война"),
    ("Artik & Asti", "Никому не отдам"),
    ("Artik & Asti", "Грустный дэнс"),
    ("Artik & Asti", "Неделимы"),
    ("Джиган", "ДНК"),
    ("Джиган", "Дни и ночи"),
    ("Баста", "Сансара"),
    ("Баста", "Выпускной"),
    ("Баста", "Медлячок"),
    ("Тимати", "Мой лучший друг"),
    ("Тимати и Мот", "Молодость"),
    ("Дима Билан", "Не молчи"),
    ("Дима Билан", "Молния"),
    ("Дима Билан", "Невозможное возможно"),
    ("Сергей Лазарев", "Это всё она"),
    ("Сергей Лазарев", "Идеальный мир"),
    ("Loboda", "Твои глаза"),
    ("Loboda", "SuperSTAR"),
    ("Loboda", "Случайная"),
    ("Ольга Бузова", "Мало половин"),
    ("Ольга Бузова", "Водица"),
    ("Монатик", "Кружит"),
    ("Монатик", "Vitamin D"),
    ("Монатик", "Каждый раз"),
    ("Jony", "Комета"),
    ("Jony", "Звезда"),
    ("Jony", "Аллея"),
    ("Miyagi", "Minor"),
    ("Miyagi", "Колибри"),
    ("Rauf & Faik", "Детство"),
    ("Rauf & Faik", "Я люблю тебя"),
    ("Тима Белорусских", "Мокрые кроссы"),
    ("Тима Белорусских", "Незабудка"),
    ("Инстасамка", "Попа как у Ким"),
    ("Инстасамка", "За деньги да"),
    ("Niletto", "Любимка"),
    ("Niletto", "Краш"),
    ("Andro", "Удиви"),
    ("Andro", "Она"),
    ("Света", "А мне нравится"),
    ("Серебро", "Мало тебя"),
    ("Серебро", "Перепутала"),
    ("Лёша Свик", "Не забывай"),
    ("Лёша Свик", "Самолёты"),
    ("Мальбэк ft. Сюзанна", "Равнодушие"),
    ("Ваня Дмитриенко", "Венера-Юпитер"),
    ("Назима", "Бабл Гам"),
    ("Jakone", "Ты мой"),
    ("Мари Краймбрери", "Туси сам"),
    ("Мари Краймбрери", "Дыши"),
    ("Бьянка", "Музыка"),
]

def safe_filename(artist, track):
    s = f"{artist} - {track}"
    for ch in ['/', '\\', '"', "'", '?', '*', '<', '>', '|', ':', '&', '.']:
        s = s.replace(ch, '_')
    return s

print("Installing dependencies...")
subprocess.run(["pip", "install", "-q", "-U", "yt-dlp"], check=True)

has_ffmpeg = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0
if not has_ffmpeg:
    print("Installing ffmpeg...")
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], capture_output=True)

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
        if f.startswith(fname) and f != fname + ".mp3":
            src = os.path.join(dataset_dir, f)
            os.rename(src, outpath)
            return True
    return False

downloaded = 0
failed = []

for i, (artist, track) in enumerate(TRACKS, 1):
    fname = safe_filename(artist, track)
    outpath = os.path.join(DATASET_DIR, f"{fname}.mp3")

    if os.path.exists(outpath):
        print(f"[{i}/100] SKIP (exists): {artist} - {track}")
        downloaded += 1
        continue

    query = f"{artist} {track}"
    print(f"[{i}/100] Downloading: {artist} - {track}...")

    success = False

    for source_name, source_fn in [("SoundCloud", try_soundcloud), ("VK", try_vk)]:
        try:
            result = source_fn(query, outpath)
            if check_downloaded(outpath, fname, DATASET_DIR):
                size_mb = os.path.getsize(outpath) / 1024 / 1024
                print(f"  OK: {size_mb:.1f}MB ({source_name})")
                downloaded += 1
                success = True
                break
            else:
                stderr = result.stderr[-150:] if result.stderr else ""
                if "no results" in stderr.lower() or "unable" in stderr.lower():
                    print(f"  {source_name}: not found, trying next...")
                else:
                    print(f"  {source_name}: failed, trying next...")
        except subprocess.TimeoutExpired:
            print(f"  {source_name}: TIMEOUT, trying next...")
        except Exception as e:
            print(f"  {source_name}: ERROR {e}")

    if not success:
        print(f"  FAILED on all sources")
        failed.append(f"{artist} - {track}")

    time.sleep(1)

print(f"\n{'='*60}")
print(f"Downloaded: {downloaded}/100")
print(f"Failed: {len(failed)}")
if failed:
    print("\nFailed tracks:")
    for f in failed:
        print(f"  - {f}")

files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".mp3")]
print(f"\nTotal files in {DATASET_DIR}: {len(files)}")
