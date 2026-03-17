import os, glob, time, json, torch, numpy as np, soundfile as sf
from torch.nn import functional as F
from scipy.signal import resample as scipy_resample

DATASET = "/workspace/dataset"
OUTPUT = "/workspace/loras/russian-pop-v1"
STEPS = 700
RANK = 64
LR = 1e-5
MAX_DURATION = 30
SAMPLE_RATE = 48000
CHECKPOINT_EVERY = 200
GRAD_CLIP = 1.0

CAPTIONS = {
    "Anna_Asti_-_Anechka": "russianpop Russian pop ballad, female vocals, emotional, slow tempo, piano, gentle, intimate, nostalgic, tender vocal performance",
    "Anna_Asti_-_Bez_tebya": "russianpop Russian pop, female vocals, emotional breakup song, dramatic, melancholic, powerful chorus, synthesizer, mid-tempo",
    "Anna_Asti_-_Carica": "russianpop Russian dance-pop, female vocals, empowering, upbeat, club beat, confident, strong bass, catchy hook, modern production",
    "Anna_Asti_-_Celuyesh_druguyu": "russianpop Russian pop, female vocals, jealousy, heartbreak, dramatic, emotional dynamics, dance-pop beat, intense chorus",
    "Anna_Asti_-_Feniks": "russianpop Russian pop, female vocals, dramatic, powerful, cinematic build-ups, emotional, anthemic chorus, rebirth theme",
    "Anna_Asti_-_Hischnaya": "russianpop Russian dance-pop, female vocals, aggressive, confident, dark pop, heavy bass, club beat, fierce energy",
    "Anna_Asti_-_Nochyu_na_kuhne": "russianpop Russian pop ballad, female vocals, intimate, night mood, gentle, acoustic feel, warm, emotional, slow tempo",
    "Anna_Asti_-_Po_baram": "russianpop Russian dance-pop, female vocals, upbeat, party, club anthem, energetic, strong bass, fun, catchy chorus",
    "Anna_Asti_-_Povelo": "russianpop Russian pop, female vocals, upbeat, playful, dance-pop, modern production, catchy melody, feel-good, light",
    "Anna_Asti_-_Predannyj_byvshij": "russianpop Russian pop, female vocals, dramatic, about ex-lover, emotional, tense, powerful vocal runs, modern beat",
    "Anna_Asti_-_Razbudi_menya": "russianpop Russian pop, female vocals, emotional, dreamy, wake up theme, mid-tempo, atmospheric synths, powerful chorus",
    "Anna_Asti_-_Shokolad": "russianpop Russian pop, female vocals, playful, sweet, dance-pop, fun production, catchy, light-hearted, upbeat tempo",
    "Anna_Asti_-_Steklo": "russianpop Russian pop, female vocals, fragile, emotional, dramatic, powerful chorus, modern production, vulnerability",
    "Anna_Asti_-_Vysshie_sily": "russianpop Russian pop, female vocals, dramatic, spiritual, powerful, cinematic, epic chorus, emotional dynamics, atmospheric",
    "Andro_-_Ona": "russianpop Russian pop R&B, male vocals, smooth, romantic, mid-tempo, warm synths, love song, gentle groove",
    "Andro_-_Udivi": "russianpop Russian pop R&B, male vocals, smooth, groovy, romantic, mid-tempo, modern R&B production, warm bass",
    "Artik_Asti_-_Grustnyj_dens": "russianpop Russian pop, male-female duet, melancholic dance, bittersweet, dance-pop beat, emotional contrast, catchy melody",
    "Artik_Asti_-_Nedelimy": "russianpop Russian pop, male-female duet, romantic, uplifting, synth-pop, emotional, powerful chorus, love anthem",
    "Artik_Asti_-_Nikomu_ne_otdam": "russianpop Russian pop, male-female duet, possessive love, dramatic, dance-pop, powerful vocals, intense emotion",
    "Basta_-_Medlyachok": "russianpop Russian rap-pop, male vocals, slow dance song, romantic, gentle beat, emotional rap, warm melody, nostalgic",
    "Basta_-_Sansara": "russianpop Russian rap-pop, male vocals, philosophical, deep lyrics, mid-tempo, atmospheric, emotional, introspective, piano",
    "Basta_-_Vypusknoj": "russianpop Russian rap-pop, male vocals, graduation anthem, nostalgic, emotional, uplifting chorus, piano, sentimental",
    "Byanka_-_Muzyka": "russianpop Russian dance-pop, female vocals, upbeat, club anthem, energetic, party mood, catchy hook, strong beat",
    "Dima_Bilan_-_Molniya": "russianpop Russian pop, male vocals, powerful, dramatic, lightning energy, epic production, strong chorus, anthemic",
    "Dima_Bilan_-_Ne_molchi": "russianpop Russian pop ballad, male vocals, emotional plea, dramatic, orchestral elements, powerful tenor, romantic",
    "Dima_Bilan_-_Nevozmozhnoe_vozmozhno": "russianpop Russian pop, male vocals, inspirational, uplifting, epic chorus, orchestral pop, powerful, anthemic",
    "Djigan_-_Dni_i_nochi": "russianpop Russian rap-pop, male vocals, romantic, day and night theme, smooth beat, catchy hook, mid-tempo",
    "Djigan_-_DNK": "russianpop Russian rap-pop, male vocals, family theme, emotional, mid-tempo, warm production, heartfelt, sincere",
    "Egor_Kreed_-_Budilnik": "russianpop Russian pop, male vocals, morning theme, upbeat, playful, dance-pop, catchy melody, fun production",
    "Egor_Kreed_-_Golubye_glaza": "russianpop Russian pop, male vocals, romantic, blue eyes theme, mid-tempo, gentle, smooth vocals, love song",
    "Egor_Kreed_-_Samaya_samaya": "russianpop Russian pop, male vocals, romantic dedication, upbeat, catchy, dance-pop, feel-good, sweet lyrics",
    "Egor_Kreed_-_Serdceedka": "russianpop Russian pop, male vocals, romantic, playful, mid-tempo, catchy hook, modern production, love theme",
    "Elka_-_Greyu_schastye": "russianpop Russian pop, female vocals, warm, uplifting, happiness theme, acoustic elements, gentle, mid-tempo, cozy",
    "Elka_-_Okolo_tebya": "russianpop Russian pop, female vocals, romantic, gentle, near you theme, soft production, intimate, warm melody",
    "Elka_-_Provans": "russianpop Russian pop, female vocals, dreamy, romantic, French vibes, light, breezy, acoustic guitar, summer mood",
    "HammAli_Navai_-_Aylove": "russianpop Russian pop R&B, male duet, romantic, smooth, mid-tempo, warm synths, love declaration, modern production",
    "HammAli_Navai_-_Devochka_vojna": "russianpop Russian pop, male duet, dramatic, girl-war metaphor, emotional, intense, powerful chorus, dark undertones",
    "HammAli_Navai_-_Ptichka": "russianpop Russian pop, male duet, gentle, bird metaphor, romantic, mid-tempo, warm melody, tender vocals",
    "Hanna_-_Bez_tebya_ya_ne_mogu": "russianpop Russian pop, female vocals, emotional, can't live without you, dramatic, powerful chorus, dance-pop beat",
    "Hanna_-_Poteryala_golovu": "russianpop Russian dance-pop, female vocals, lost my mind, upbeat, catchy, club beat, energetic, fun",
    "Instasamka_-_Popa_kak_u_Kim": "russianpop Russian trap-pop, female vocals, provocative, bass-heavy, club banger, confident, aggressive, modern trap beat",
    "Instasamka_-_Za_dengi_da": "russianpop Russian trap-pop, female vocals, provocative, materialistic, heavy bass, trap beat, confident, bold",
    "Jakone_-_Ty_moj": "russianpop Russian pop R&B, male vocals, possessive love, smooth, mid-tempo, romantic, warm production, gentle",
    "Jony_-_Alleya": "russianpop Russian pop, male vocals, romantic walk, gentle, mid-tempo, dreamy synths, soft vocals, atmospheric",
    "Jony_-_Kometa": "russianpop Russian pop, male vocals, cosmic love metaphor, emotional, mid-tempo, atmospheric synths, dreamy, tender",
    "Jony_-_Zvezda": "russianpop Russian pop, male vocals, star metaphor, romantic, emotional, gentle production, mid-tempo, warm",
    "Klava_Koka_-_Krash": "russianpop Russian dance-pop, female vocals, crush theme, upbeat, catchy, modern pop, energetic, playful",
    "Klava_Koka_-_Mne_poh": "russianpop Russian dance-pop, female vocals, confident, don't care attitude, upbeat, bass-heavy, empowering, bold",
    "Klava_Koka_-_Pokinula_chat": "russianpop Russian pop, female vocals, left the chat, modern theme, catchy, mid-tempo, playful, social media",
    "Klava_Koka_-_Vlyublena": "russianpop Russian pop, female vocals, in love, romantic, upbeat, dance-pop, catchy melody, sweet, joyful",
    "Lesha_Svik_-_Ne_zabyvaj": "russianpop Russian pop, male vocals, don't forget me, emotional, mid-tempo, synth-pop, nostalgic, warm chorus",
    "Lesha_Svik_-_Samolyoty": "russianpop Russian pop, male vocals, airplanes metaphor, dreamy, emotional, atmospheric synths, mid-tempo, romantic",
    "Loboda_-_Sluchajnaya": "russianpop Russian pop, female vocals, accidental love, dramatic, powerful, dance-pop, intense chorus, emotional dynamics",
    "Loboda_-_SuperSTAR": "russianpop Russian dance-pop, female vocals, superstar energy, empowering, upbeat, club beat, confident, glamorous",
    "Loboda_-_Tvoi_glaza": "russianpop Russian pop, female vocals, your eyes theme, dramatic, emotional, powerful chorus, mid-tempo, intense",
    "Malbek_Syuzanna_-_Ravnodushie": "russianpop Russian pop, male-female duet, indifference theme, emotional, dramatic, mid-tempo, bittersweet, modern production",
    "Mari_Krajmbreri_-_Dyshi": "russianpop Russian pop, female vocals, breathe theme, emotional, gentle, mid-tempo, atmospheric, intimate, tender",
    "Mari_Krajmbreri_-_Tusi_sam": "russianpop Russian dance-pop, female vocals, party alone, upbeat, club beat, energetic, catchy, fun",
    "Max_Barskih_-_Hlop": "russianpop Russian dance-pop, male vocals, clap rhythm, upbeat, club banger, energetic, catchy hook, strong beat",
    "Max_Barskih_-_Nevernaya": "russianpop Russian pop, male vocals, unfaithful theme, dramatic, emotional, dance-pop beat, powerful chorus, intense",
    "Max_Barskih_-_Tumany": "russianpop Russian pop, male vocals, fog metaphor, atmospheric, emotional, mid-tempo, dreamy, melancholic, synth layers",
    "Miyagi_-_Kolibri": "russianpop Russian reggae-pop, male vocals, hummingbird metaphor, laid-back groove, warm bass, chill vibes, smooth flow",
    "Miyagi_-_Minor": "russianpop Russian hip-hop pop, male vocals, minor key, dark mood, deep bass, atmospheric, introspective, smooth rap",
    "Monatik_-_Kazhdyj_raz": "russianpop Russian pop funk, male vocals, every time theme, groovy, funky bass, upbeat, smooth vocals, dance",
    "Monatik_-_Kruzhit": "russianpop Russian pop funk, male vocals, spinning theme, groovy, funky, upbeat, dance energy, catchy rhythm",
    "Monatik_-_Vitamin_D": "russianpop Russian pop funk, male vocals, sunshine vibes, groovy, upbeat, funky bass, feel-good, summer energy",
    "Mot_-_Absent": "russianpop Russian rap-pop, male vocals, absence theme, emotional, mid-tempo, atmospheric beat, introspective, smooth flow",
    "Mot_-_Kapkan": "russianpop Russian rap-pop, male vocals, trap metaphor, emotional, dramatic, mid-tempo, atmospheric synths, deep lyrics",
    "Mot_-_Kogda_ischeznet_slovo": "russianpop Russian rap-pop, male vocals, words disappear, emotional, philosophical, mid-tempo, atmospheric, introspective",
    "Mot_-_Solo_na_dvoih": "russianpop Russian pop, male-female duet, solo for two, romantic, emotional, mid-tempo, warm melody, intimate",
    "Nazima_-_Bubble_Gum": "russianpop Russian dance-pop, female vocals, playful, sweet, bubblegum pop, upbeat, catchy, fun production, light",
    "Niletto_-_Krash": "russianpop Russian pop, male vocals, crush theme, upbeat, catchy, dance-pop, modern production, youthful energy",
    "Niletto_-_Lyubimka": "russianpop Russian pop, male vocals, darling theme, upbeat, catchy hook, dance-pop, feel-good, playful, summer hit",
    "Nyusha_-_Celuy": "russianpop Russian dance-pop, female vocals, kiss me, upbeat, sensual, club beat, catchy, energetic, flirty",
    "Nyusha_-_Cunami": "russianpop Russian dance-pop, female vocals, tsunami metaphor, powerful, dramatic, heavy beat, intense chorus, dance energy",
    "Nyusha_-_Vybirat_chudo": "russianpop Russian pop, female vocals, choosing miracles, uplifting, emotional, mid-tempo, warm, inspirational, hopeful",
    "Olga_Buzova_-_Malo_polovin": "russianpop Russian pop, female vocals, not enough halves, emotional, dance-pop, catchy, heartbreak, mid-tempo, dramatic",
    "Olga_Buzova_-_Vodica": "russianpop Russian dance-pop, female vocals, water theme, upbeat, summer pop, catchy hook, fun, light, playful",
    "Polina_Gagarina_-_Kukushka": "russianpop Russian pop ballad, female vocals, cuckoo theme, powerful, emotional, dramatic build, orchestral, war movie soundtrack",
    "Polina_Gagarina_-_Million_golosov": "russianpop Russian pop, female vocals, million voices, anthemic, powerful, Eurovision, uplifting, epic chorus, inspirational",
    "Polina_Gagarina_-_Navek": "russianpop Russian pop ballad, female vocals, forever theme, romantic, emotional, orchestral elements, powerful vocals, dramatic",
    "Polina_Gagarina_-_Vyshe_golovy": "russianpop Russian pop, female vocals, above the head, uplifting, empowering, mid-tempo, inspirational, powerful chorus",
    "Rauf_Faik_-_Detstvo": "russianpop Russian pop, male duet, childhood nostalgia, gentle, emotional, acoustic guitar, warm, tender, bittersweet",
    "Rauf_Faik_-_Ya_lyublyu_tebya": "russianpop Russian pop, male duet, I love you, romantic, gentle, mid-tempo, warm melody, sincere, tender",
    "Serebro_-_Malo_tebya": "russianpop Russian dance-pop, female group, not enough of you, upbeat, club beat, catchy, energetic, sexy",
    "Serebro_-_Pereputala": "russianpop Russian dance-pop, female group, mixed up, upbeat, catchy hook, club pop, energetic, fun, flirty",
    "Sergey_Lazarev_-_Eto_vsyo_ona": "russianpop Russian pop, male vocals, it's all her, dramatic, emotional, powerful tenor, dance-pop, intense chorus",
    "Sergey_Lazarev_-_Idealnyj_mir": "russianpop Russian pop, male vocals, perfect world, uplifting, emotional, anthemic chorus, synth-pop, inspirational",
    "Sveta_-_A_mne_nravitsya": "russianpop Russian dance-pop, female vocals, I like it, upbeat, catchy, fun, party pop, energetic, playful",
    "Tima_Belorusskih_-_Mokrye_krossy": "russianpop Russian pop, male vocals, wet sneakers, emotional, romantic, mid-tempo, gentle production, youthful, tender",
    "Tima_Belorusskih_-_Nezabudka": "russianpop Russian pop, male vocals, forget-me-not flower, gentle, romantic, mid-tempo, warm melody, nostalgic, sweet",
    "Timati_-_Moj_luchshij_drug": "russianpop Russian rap-pop, male vocals, best friend theme, emotional, mid-tempo, warm beat, heartfelt, sincere",
    "Timati_Mot_-_Molodost": "russianpop Russian rap-pop, male duet, youth theme, nostalgic, emotional, mid-tempo, warm production, reflective",
    "Vanya_Dmitrienko_-_Venera_Jupiter": "russianpop Russian pop, male vocals, Venus Jupiter metaphor, dreamy, romantic, mid-tempo, atmospheric, cosmic, gentle",
    "Zivert_-_Anesteziya": "russianpop Russian electro-pop, female vocals, anesthesia metaphor, dark pop, atmospheric, electronic, moody, hypnotic beat",
    "Zivert_-_Beverly_Hills": "russianpop Russian electro-pop, female vocals, glamorous, upbeat, modern pop, stylish production, confident, catchy hook",
    "Zivert_-_Credo": "russianpop Russian electro-pop, female vocals, credo theme, powerful, dramatic, electronic, intense, dark atmosphere, bold",
    "Zivert_-_Eshche_hochu": "russianpop Russian pop, female vocals, want more, sensual, mid-tempo, modern production, emotional, catchy melody",
    "Zivert_-_Life": "russianpop Russian electro-pop, female vocals, life theme, upbeat, positive energy, electronic, catchy, modern, feel-good",
    "Zivert_-_Mnogotochiya": "russianpop Russian pop, female vocals, ellipsis metaphor, emotional, atmospheric, mid-tempo, introspective, moody synths",
}

DEFAULT_CAPTION = "russianpop Russian pop, vocals, dance-pop, emotional, modern production, synthesizer, catchy melody, mid-tempo"

def get_caption(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    for key, caption in CAPTIONS.items():
        if key.lower() in base.lower():
            return caption
    return DEFAULT_CAPTION

files = sorted(glob.glob(os.path.join(DATASET, "*.mp3")) + glob.glob(os.path.join(DATASET, "*.wav")))
print(f"Found {len(files)} tracks")
if not files:
    print("ERROR: No audio files found")
    exit(1)

print("\nTrack -> Caption mapping:")
for f in files:
    cap = get_caption(f)
    print(f"  {os.path.basename(f)}")
    print(f"    -> {cap[:80]}...")
print()

print("Loading ACE-Step pipeline...")
from acestep.pipeline_ace_step import ACEStepPipeline

pipe = ACEStepPipeline(device_id=0, dtype="bfloat16")
pipe.load_checkpoint()
print("Pipeline loaded!")

dit = pipe.ace_step_transformer
dcae = pipe.music_dcae

dcae.eval()
for p in dcae.parameters():
    p.requires_grad = False

from peft import LoKrConfig, get_peft_model

target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
lokr_config = LoKrConfig(
    r=RANK,
    alpha=RANK,
    target_modules=target_modules,
    module_dropout=0.0,
)
dit = get_peft_model(dit, lokr_config)
trainable = sum(p.numel() for p in dit.parameters() if p.requires_grad)
total = sum(p.numel() for p in dit.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

print("Pre-computing text embeddings for all captions...")
caption_cache = {}
unique_captions = set(get_caption(f) for f in files)
for cap in unique_captions:
    hidden, mask = pipe.get_text_embeddings([cap])
    caption_cache[cap] = (hidden.detach(), mask.detach())
print(f"Cached {len(caption_cache)} unique caption embeddings")

lyric_dummy = torch.zeros(1, 300, dtype=torch.long, device=pipe.device)
lyric_mask_dummy = torch.zeros(1, 300, dtype=torch.long, device=pipe.device)
speaker_dummy = torch.zeros(1, 512, dtype=pipe.dtype, device=pipe.device)

opt = torch.optim.AdamW([p for p in dit.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-7)

def load_audio(path):
    audio, sr = sf.read(path)
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio])
    else:
        audio = audio.T
        if audio.shape[0] > 2:
            audio = audio[:2]
        elif audio.shape[0] == 1:
            audio = np.concatenate([audio, audio], axis=0)
    if sr != SAMPLE_RATE:
        new_len = int(audio.shape[1] * SAMPLE_RATE / sr)
        audio = np.stack([scipy_resample(audio[0], new_len), scipy_resample(audio[1], new_len)])
    max_samples = MAX_DURATION * SAMPLE_RATE
    if audio.shape[1] > max_samples:
        start = np.random.randint(0, audio.shape[1] - max_samples)
        audio = audio[:, start:start+max_samples]
    wav = torch.FloatTensor(audio).unsqueeze(0)
    return wav

def encode_audio(wav):
    with torch.no_grad():
        wav = wav.to(pipe.device).to(pipe.dtype)
        latents, lengths = dcae.encode(wav)
    return latents

print(f"\nStarting training: {STEPS} steps, rank={RANK}, lr={LR}")
print(f"Output: {OUTPUT}")
print("-" * 60)

dit.train()
t0 = time.time()
step = 0
epoch = 0
losses = []

while step < STEPS:
    epoch += 1
    np.random.shuffle(files)
    for f in files:
        if step >= STEPS:
            break
        try:
            wav = load_audio(f)
            latents = encode_audio(wav)
            caption = get_caption(f)
            text_hidden, text_mask = caption_cache[caption]
            noise = torch.randn_like(latents)
            t = torch.rand(1, device=pipe.device, dtype=pipe.dtype).clamp(0.01, 0.99)
            noisy_latents = (1 - t) * latents + t * noise
            velocity_target = noise - latents
            timestep_input = (t * 1000).long()
            frame_length = noisy_latents.shape[-1]
            attention_mask = torch.ones(1, frame_length, device=pipe.device, dtype=torch.long)
            pred = dit(
                hidden_states=noisy_latents,
                attention_mask=attention_mask,
                encoder_text_hidden_states=text_hidden,
                text_attention_mask=text_mask,
                speaker_embeds=speaker_dummy,
                lyric_token_idx=lyric_dummy,
                lyric_mask=lyric_mask_dummy,
                timestep=timestep_input,
            ).sample
            loss = F.mse_loss(pred, velocity_target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dit.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            if step % 10 == 0:
                avg_loss = np.mean(losses[-50:])
                el = time.time() - t0
                eta = el / step * (STEPS - step)
                lr_now = sched.get_last_lr()[0]
                print(f"Step {step}/{STEPS} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | LR: {lr_now:.2e} | {el:.0f}s | ETA: {eta:.0f}s | {os.path.basename(f)}")
            if step % CHECKPOINT_EVERY == 0:
                ckpt_dir = os.path.join(OUTPUT, f"checkpoint-{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                dit.save_pretrained(ckpt_dir)
                print(f"  Checkpoint saved: {ckpt_dir}")
        except Exception as e:
            print(f"Skip {os.path.basename(f)}: {e}")
            import traceback
            traceback.print_exc()
            continue

elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"Final avg loss: {np.mean(losses[-50:]):.4f}")

os.makedirs(OUTPUT, exist_ok=True)
dit.save_pretrained(OUTPUT)

metadata = {
    "name": "russian-pop-v1",
    "trigger": "russianpop",
    "rank": RANK,
    "steps": STEPS,
    "lr": LR,
    "tracks": len(files),
    "unique_captions": len(caption_cache),
    "final_loss": round(float(np.mean(losses[-50:])), 4),
    "time_seconds": round(elapsed),
    "method": "LoKr",
    "target_modules": target_modules,
    "improvements": "100-track Russian pop dataset, per-track captions, 700 steps",
}
json.dump(metadata, open(os.path.join(OUTPUT, "metadata.json"), "w"), ensure_ascii=False, indent=2)
print(f"Saved to {OUTPUT}")

for x in sorted(os.listdir(OUTPUT)):
    fp = os.path.join(OUTPUT, x)
    if os.path.isfile(fp):
        print(f"  {x} ({os.path.getsize(fp)/1024/1024:.1f}MB)")
    else:
        print(f"  {x}/")
