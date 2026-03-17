import numpy as np
np.float = np.float64
np.int = np.int64

import torch, sys, os, time, json, gc
import soundfile as sf
from safetensors.torch import load_file, save_file
from loguru import logger
import pyloudnorm as pyln
from pedalboard import Pedalboard, Compressor, Gain, LowShelfFilter, HighShelfFilter, PeakFilter, Limiter, HighpassFilter, LowpassFilter
import subprocess

BASE_DIR = '/workspace/ace-step-v15-checkpoint/acestep-v15-base'
sys.path.insert(0, BASE_DIR)

DEVICE = 'cuda'
DTYPE = torch.bfloat16
SR = 48000
DOWNSAMPLE = 1920
DURATION = 60.0
TLEN = int(DURATION * SR / DOWNSAMPLE)

OUTPUT_DIR = '/workspace/batch_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def master_audio(audio_np, sr, target_lufs=-14.0):
    audio_t = audio_np.astype(np.float32)
    eq_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30.0),
        LowShelfFilter(cutoff_frequency_hz=80.0, gain_db=1.5),
        PeakFilter(cutoff_frequency_hz=250.0, gain_db=-2.5, q=1.0),
        PeakFilter(cutoff_frequency_hz=400.0, gain_db=-1.5, q=0.8),
        PeakFilter(cutoff_frequency_hz=3000.0, gain_db=1.5, q=0.7),
        PeakFilter(cutoff_frequency_hz=5000.0, gain_db=1.0, q=0.8),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=2.5),
        LowpassFilter(cutoff_frequency_hz=20000.0),
    ])
    audio_t = eq_board(audio_t, sr)
    comp_board = Pedalboard([
        Compressor(threshold_db=-20.0, ratio=3.0, attack_ms=10.0, release_ms=100.0),
    ])
    audio_t = comp_board(audio_t, sr)
    mid = (audio_t[0] + audio_t[1]) / 2.0
    side = (audio_t[0] - audio_t[1]) / 2.0
    side *= 1.3
    audio_t = np.array([mid + side, mid - side])
    limit_board = Pedalboard([
        Gain(gain_db=2.0),
        Limiter(threshold_db=-1.0, release_ms=50.0),
    ])
    audio_t = limit_board(audio_t, sr)
    audio_out = audio_t.T
    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio_out)
    if not np.isinf(current_lufs):
        audio_out = pyln.normalize.loudness(audio_out, current_lufs, target_lufs)
    peak = np.max(np.abs(audio_out))
    if peak > 0.99:
        audio_out = audio_out * (0.99 / peak)
    return audio_out

def main():
    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel
    from diffusers.models import AutoencoderOobleck
    from configuration_acestep_v15 import AceStepConfig
    from modeling_acestep_v15_base import AceStepConditionGenerationModel

    with open('/root/batch_cases.json') as f:
        cases = json.load(f)

    logger.info(f"=== BATCH GENERATION: {len(cases)} tracks @ {DURATION}s ===")

    logger.info("Loading models...")
    vae = AutoencoderOobleck.from_pretrained('/workspace/ace-step-v15-checkpoint/vae').to(DEVICE, dtype=torch.float32)
    vae.eval()
    qwen_path = '/workspace/ace-step-v15-checkpoint/Qwen3-Embedding-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)
    text_enc = AutoModel.from_pretrained(qwen_path, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE)
    text_enc.eval()
    config = AceStepConfig.from_pretrained(BASE_DIR)
    base_sd = load_file(os.path.join(BASE_DIR, 'model.safetensors'))
    model = AceStepConditionGenerationModel(config)
    model.load_state_dict(base_sd, strict=False)
    model = model.to(DEVICE, dtype=DTYPE)
    model.eval()
    silence_latent = torch.load(os.path.join(BASE_DIR, 'silence_latent.pt'), weights_only=True).transpose(1, 2).to(DEVICE, dtype=DTYPE).detach()

    logger.info("Applying MACAN LoRA alpha=64...")
    model.decoder = PeftModel.from_pretrained(model.decoder, '/workspace/lora_fixed_base', is_trainable=False).to(DEVICE, dtype=DTYPE)
    model.decoder.eval()
    logger.info(f"Models loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
    GEN_PROMPT = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n"

    category_bpm = {
        'street': '140', 'melodic': '90', 'aggressive': '150',
        'lyrical': '85', 'hit': '120',
    }
    category_key = {
        'street': 'Cm', 'melodic': 'Am', 'aggressive': 'Dm',
        'lyrical': 'Em', 'hit': 'Gm',
    }

    total_start = time.time()
    results = []

    for i, case in enumerate(cases):
        cid = case['id']
        logger.info(f"[{i+1}/{len(cases)}] Generating {cid} ({case['category']})...")

        caption = f"voice_macan, {case['style_prompt']}"
        bpm = category_bpm.get(case['category'], '120')
        key = category_key.get(case['category'], 'Am')
        metas = f"bpm: {bpm}, key: {key}, time_signature: 4/4, genre: russian rap"
        text_prompt = GEN_PROMPT.format(INSTRUCTION, caption, metas)
        lyrics_text = f"# Languages\nru\n\n# Lyric\n{case['lyrics']}<|endoftext|>"

        text_inputs = tokenizer(text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt")
        text_ids = text_inputs['input_ids'].to(DEVICE)
        text_mask = text_inputs['attention_mask'].to(DEVICE).bool()
        lyric_inputs = tokenizer(lyrics_text, padding="longest", truncation=True, max_length=2048, return_tensors="pt")
        lyric_ids = lyric_inputs['input_ids'].to(DEVICE)
        lyric_mask = lyric_inputs['attention_mask'].to(DEVICE).bool()

        with torch.no_grad():
            text_hs = text_enc(text_ids, lyric_attention_mask=None).last_hidden_state
            lyric_hs = text_enc.embed_tokens(lyric_ids)

        bs = 1
        ref = silence_latent[:, :750, :].expand(bs, -1, -1)
        rom = torch.arange(bs, device=DEVICE, dtype=torch.long)
        src = silence_latent[:, :TLEN, :].expand(bs, -1, -1).clone()
        cm = torch.ones(bs, TLEN, dtype=torch.bool, device=DEVICE).unsqueeze(-1).repeat(1, 1, 64).to(DTYPE)
        ic = torch.zeros(bs, device=DEVICE, dtype=DTYPE)
        attn_mask = torch.ones(bs, TLEN, device=DEVICE, dtype=DTYPE)

        seed = 1000 + i
        t0 = time.time()
        with torch.no_grad():
            result = model.generate_audio(
                text_hidden_states=text_hs, text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hs, lyric_attention_mask=lyric_mask,
                refer_audio_acoustic_hidden_states_packed=ref,
                refer_audio_order_mask=rom, src_latents=src, chunk_masks=cm,
                is_covers=ic, silence_latent=silence_latent,
                attention_mask=attn_mask,
                seed=seed, infer_steps=60, diffusion_guidance_sale=7.0,
                infer_method="ode", use_cache=True, use_progress_bar=False,
            )
        gen_time = time.time() - t0

        lat = result["target_latents"]
        lat_vae = lat.transpose(1, 2).to(torch.float32)
        with torch.no_grad():
            audio = vae.decode(lat_vae).sample
        audio_np = audio[0].cpu().float().numpy()

        mastered = master_audio(audio_np, SR)

        wav_path = f'{OUTPUT_DIR}/{cid}_{case["ace_track"]}.wav'
        sf.write(wav_path, mastered, SR, subtype='PCM_24')

        mp3_path = f'{OUTPUT_DIR}/{cid}_{case["ace_track"]}.mp3'
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-codec:a', 'libmp3lame', '-b:a', '320k',
            '-ar', '48000', mp3_path
        ], capture_output=True)

        os.remove(wav_path)

        logger.info(f"  Done in {gen_time:.1f}s, saved {mp3_path}")
        results.append({'id': cid, 'track': case['ace_track'], 'gen_time': gen_time})

    total_time = time.time() - total_start
    logger.info(f"=== ALL DONE: {len(results)} tracks in {total_time:.0f}s ({total_time/60:.1f}min) ===")

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
