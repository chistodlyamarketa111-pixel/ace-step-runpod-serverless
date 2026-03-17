import numpy as np
np.float = np.float64
np.int = np.int64

import torch, sys, os, time, soundfile as sf, math, json, gc
from safetensors.torch import load_file, save_file
from loguru import logger
import pyloudnorm as pyln
from pedalboard import Pedalboard, Compressor, Gain, LowShelfFilter, HighShelfFilter, PeakFilter, Limiter, HighpassFilter, LowpassFilter

BASE_DIR = '/workspace/ace-step-v15-checkpoint/acestep-v15-base'
sys.path.insert(0, BASE_DIR)

DEVICE = 'cuda'
DTYPE = torch.bfloat16
SR = 48000
DOWNSAMPLE = 1920

CAPTION = "voice_macan, Russian hip-hop/rap track with emotional male vocals, atmospheric synthesizer pads, deep 808 bass, crisp hi-hats, mid-tempo trap beat 85 BPM minor key, melancholic yet triumphant"
LYRICS = """[chorus]
\u042f \u043f\u043e\u0434\u043d\u044f\u043b\u0441\u044f \u0441 \u043d\u0438\u0437\u043e\u0432 \u2014 \u0442\u0435\u043f\u0435\u0440\u044c \u0432\u0438\u0436\u0443 \u0432\u0435\u0440\u0448\u0438\u043d\u044b,
\u0413\u0434\u0435 \u043a\u043e\u0433\u0434\u0430-\u0442\u043e \u0431\u044b\u043b \u0434\u044b\u043c \u2014 \u0442\u0435\u043f\u0435\u0440\u044c \u0441\u0432\u0435\u0442 \u0432\u0438\u0442\u0440\u0438\u043d\u044b.
\u041c\u044b \u043c\u0435\u0447\u0442\u0430\u043b\u0438 \u043e \u0434\u043d\u0435, \u043a\u043e\u0433\u0434\u0430 \u0432\u044b\u0439\u0434\u0435\u043c \u043d\u0430 \u0441\u0432\u0435\u0442,
\u0418 \u0442\u0435\u043f\u0435\u0440\u044c \u0432\u0435\u0441\u044c \u043c\u043e\u0439 \u043f\u0443\u0442\u044c \u2014 \u044d\u0442\u043e \u043c\u043e\u0439 \u043c\u0430\u043d\u0438\u0444\u0435\u0441\u0442.
\u0421 \u043d\u0438\u0437\u043e\u0432 \u043a \u0432\u0435\u0440\u0445\u0430\u043c \u2014 \u044d\u0442\u043e \u043d\u0435 \u043f\u0440\u043e\u0441\u0442\u043e \u0441\u043b\u043e\u0432\u0430,
\u041a\u0430\u0436\u0434\u044b\u0439 \u0448\u0440\u0430\u043c \u043d\u0430 \u0434\u0443\u0448\u0435 \u2014 \u044d\u0442\u043e \u043c\u043e\u044f \u0433\u043b\u0430\u0432\u0430.
\u042f \u043f\u043e\u0434\u043d\u044f\u043b\u0441\u044f \u0441 \u043d\u0438\u0437\u043e\u0432, \u0438 \u0437\u0430\u043f\u043e\u043c\u043d\u0438 \u043d\u0430\u0432\u0435\u043a:
\u0415\u0441\u043b\u0438 \u0432\u0435\u0440\u0438\u0448\u044c \u0432 \u0441\u0435\u0431\u044f \u2014 \u0442\u044b \u0443\u0436\u0435 \u0447\u0435\u043b\u043e\u0432\u0435\u043a.

[verse]
\u042f \u0440\u043e\u0441 \u0442\u0430\u043c, \u0433\u0434\u0435 \u0434\u0432\u043e\u0440 \u0438 \u0431\u0435\u0442\u043e\u043d\u043d\u044b\u0435 \u0441\u0442\u0435\u043d\u044b,
\u0413\u0434\u0435 \u043a\u0430\u0436\u0434\u044b\u0439 \u0432\u0442\u043e\u0440\u043e\u0439 \u0433\u043e\u0432\u043e\u0440\u0438\u043b: \u0411\u0435\u0437 \u0441\u0438\u0441\u0442\u0435\u043c\u044b.
\u0413\u0434\u0435 \u043d\u043e\u0447\u044c\u044e \u0441\u0438\u0440\u0435\u043d\u044b, \u0433\u0434\u0435 \u0441\u0435\u0440\u044b\u0439 \u0440\u0430\u0439\u043e\u043d,
\u0413\u0434\u0435 \u0432\u0435\u0440\u0430 \u2014 \u0435\u0434\u0438\u043d\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u043c\u043e\u0439 \u043c\u0438\u043b\u043b\u0438\u043e\u043d.
\u041a\u0430\u0440\u043c\u0430\u043d \u0431\u044b\u043b \u043f\u0443\u0441\u0442\u043e\u0439, \u043d\u043e \u0433\u043b\u0430\u0437\u0430 \u2014 \u043a\u0430\u043a \u043e\u0433\u043e\u043d\u044c,
\u042f \u0448\u0451\u043b \u043f\u0440\u043e\u0442\u0438\u0432 \u0432\u0435\u0442\u0440\u0430, \u0434\u0435\u0440\u0436\u0430\u043b \u0441\u0432\u043e\u0439 \u0437\u0430\u043a\u043e\u043d.
\u041d\u0438\u043a\u0442\u043e \u043d\u0435 \u043f\u043e\u0432\u0435\u0440\u0438\u043b, \u043d\u0438\u043a\u0442\u043e \u043d\u0435 \u043f\u043e\u043c\u043e\u0433,
\u041d\u043e \u044f \u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0430\u043b \u2014 \u0448\u0430\u0433 \u0437\u0430 \u0448\u0430\u0433\u043e\u043c, \u0432 \u043f\u043e\u0442\u043e\u043a.
\u041d\u0438 \u0441\u0432\u044f\u0437\u0435\u0439, \u043d\u0438 \u0434\u0435\u043d\u0435\u0433, \u043d\u0438 \u0433\u0440\u043e\u043c\u043a\u0438\u0445 \u0438\u043c\u0451\u043d,
\u0422\u043e\u043b\u044c\u043a\u043e \u0434\u043e\u0440\u043e\u0433\u0430 \u0438 \u0432\u043d\u0443\u0442\u0440\u0435\u043d\u043d\u0438\u0439 \u0437\u0432\u043e\u043d.
\u042f \u043f\u0430\u0434\u0430\u043b \u0438 \u0441\u043d\u043e\u0432\u0430 \u0432\u0441\u0442\u0430\u0432\u0430\u043b \u043d\u0430 \u043d\u043e\u0433\u0430\u0445,
\u0418 \u043a\u0430\u0436\u0434\u044b\u0439 \u043f\u0440\u043e\u0432\u0430\u043b \u043f\u0440\u0435\u0432\u0440\u0430\u0449\u0430\u043b \u0432 \u043d\u043e\u0432\u044b\u0439 \u0448\u0430\u0433.

[chorus]
\u042f \u043f\u043e\u0434\u043d\u044f\u043b\u0441\u044f \u0441 \u043d\u0438\u0437\u043e\u0432 \u2014 \u0442\u0435\u043f\u0435\u0440\u044c \u0432\u0438\u0436\u0443 \u0432\u0435\u0440\u0448\u0438\u043d\u044b,
\u0413\u0434\u0435 \u043a\u043e\u0433\u0434\u0430-\u0442\u043e \u0431\u044b\u043b \u0434\u044b\u043c \u2014 \u0442\u0435\u043f\u0435\u0440\u044c \u0441\u0432\u0435\u0442 \u0432\u0438\u0442\u0440\u0438\u043d\u044b.
\u041c\u044b \u043c\u0435\u0447\u0442\u0430\u043b\u0438 \u043e \u0434\u043d\u0435, \u043a\u043e\u0433\u0434\u0430 \u0432\u044b\u0439\u0434\u0435\u043c \u043d\u0430 \u0441\u0432\u0435\u0442,
\u0418 \u0442\u0435\u043f\u0435\u0440\u044c \u0432\u0435\u0441\u044c \u043c\u043e\u0439 \u043f\u0443\u0442\u044c \u2014 \u044d\u0442\u043e \u043c\u043e\u0439 \u043c\u0430\u043d\u0438\u0444\u0435\u0441\u0442.
\u0421 \u043d\u0438\u0437\u043e\u0432 \u043a \u0432\u0435\u0440\u0445\u0430\u043c \u2014 \u044d\u0442\u043e \u043d\u0435 \u043f\u0440\u043e\u0441\u0442\u043e \u0441\u043b\u043e\u0432\u0430,
\u041a\u0430\u0436\u0434\u044b\u0439 \u0448\u0440\u0430\u043c \u043d\u0430 \u0434\u0443\u0448\u0435 \u2014 \u044d\u0442\u043e \u043c\u043e\u044f \u0433\u043b\u0430\u0432\u0430.
\u042f \u043f\u043e\u0434\u043d\u044f\u043b\u0441\u044f \u0441 \u043d\u0438\u0437\u043e\u0432, \u0438 \u0437\u0430\u043f\u043e\u043c\u043d\u0438 \u043d\u0430\u0432\u0435\u043a:
\u0415\u0441\u043b\u0438 \u0432\u0435\u0440\u0438\u0448\u044c \u0432 \u0441\u0435\u0431\u044f \u2014 \u0442\u044b \u0443\u0436\u0435 \u0447\u0435\u043b\u043e\u0432\u0435\u043a.

[verse]
\u041c\u044b \u0434\u0435\u043b\u0430\u043b\u0438 \u043f\u043b\u0430\u043d\u044b \u043f\u043e\u0434 \u043b\u0430\u043c\u043f\u043e\u0439 \u0432\u043e \u0434\u0432\u043e\u0440\u0435,
\u0413\u0434\u0435 \u0445\u043e\u043b\u043e\u0434\u043d\u044b\u0439 \u0430\u0441\u0444\u0430\u043b\u044c\u0442 \u0431\u044b\u043b \u043a\u0430\u043a \u0434\u043e\u043c \u0432 \u044f\u043d\u0432\u0430\u0440\u0435.
\u042f \u0432\u0438\u0434\u0435\u043b, \u043a\u0430\u043a \u043b\u044e\u0434\u0438 \u0441\u0434\u0430\u0432\u0430\u043b\u0438\u0441\u044c \u0432 \u043f\u0443\u0442\u0438,
\u041d\u043e \u044f \u0433\u043e\u0432\u043e\u0440\u0438\u043b \u0441\u0435\u0431\u0435: \u0422\u043e\u043b\u044c\u043a\u043e \u0438\u0434\u0438.
\u0414\u0440\u0443\u0437\u044c\u044f \u0443\u0445\u043e\u0434\u0438\u043b\u0438, \u043c\u0435\u043d\u044f\u043b\u0438\u0441\u044c \u0433\u043e\u0434\u0430,
\u041d\u043e \u0446\u0435\u043b\u044c \u043e\u0441\u0442\u0430\u0432\u0430\u043b\u0430\u0441\u044c \u0441\u043e \u043c\u043d\u043e\u0439 \u043d\u0430\u0432\u0441\u0435\u0433\u0434\u0430.
\u042f \u0437\u043d\u0430\u043b, \u0447\u0442\u043e \u0434\u043e\u0440\u043e\u0433\u0430 \u0432\u0435\u0434\u0451\u0442 \u043a \u0432\u044b\u0441\u043e\u0442\u0435,
\u0415\u0441\u043b\u0438 \u0441\u0435\u0440\u0434\u0446\u0435 \u0433\u043e\u0440\u0438\u0442, \u0430 \u043d\u0435 \u0442\u043e\u043d\u0435\u0442 \u0432\u043e \u043c\u0433\u043b\u0435.
\u0422\u0435\u043f\u0435\u0440\u044c \u043c\u043e\u0438 \u0442\u0440\u0435\u043a\u0438 \u043b\u0435\u0442\u044f\u0442 \u043f\u043e \u0441\u0442\u0440\u0430\u043d\u0435,
\u0418 \u043a\u0430\u0436\u0434\u044b\u0439 \u0440\u0430\u0439\u043e\u043d \u0443\u0437\u043d\u0430\u0451\u0442 \u043e\u0431\u043e \u043c\u043d\u0435.
\u041d\u043e \u044f \u043d\u0435 \u0437\u0430\u0431\u044b\u043b, \u0433\u0434\u0435 \u0431\u044b\u043b \u043f\u0435\u0440\u0432\u044b\u0439 \u043c\u043e\u0439 \u0441\u0442\u0430\u0440\u0442 \u2014
\u0413\u0434\u0435 \u043c\u0435\u0447\u0442\u0430 \u0437\u0430\u0440\u043e\u0434\u0438\u043b\u0430\u0441\u044c, \u043a\u0430\u043a \u043f\u0435\u0440\u0432\u044b\u0439 \u0443\u0434\u0430\u0440.

[chorus]
\u042f \u043f\u043e\u0434\u043d\u044f\u043b\u0441\u044f \u0441 \u043d\u0438\u0437\u043e\u0432 \u2014 \u0442\u0435\u043f\u0435\u0440\u044c \u0432\u0438\u0436\u0443 \u0432\u0435\u0440\u0448\u0438\u043d\u044b,
\u0413\u0434\u0435 \u043a\u043e\u0433\u0434\u0430-\u0442\u043e \u0431\u044b\u043b \u0434\u044b\u043c \u2014 \u0442\u0435\u043f\u0435\u0440\u044c \u0441\u0432\u0435\u0442 \u0432\u0438\u0442\u0440\u0438\u043d\u044b.
\u041c\u044b \u043c\u0435\u0447\u0442\u0430\u043b\u0438 \u043e \u0434\u043d\u0435, \u043a\u043e\u0433\u0434\u0430 \u0432\u044b\u0439\u0434\u0435\u043c \u043d\u0430 \u0441\u0432\u0435\u0442,
\u0418 \u0442\u0435\u043f\u0435\u0440\u044c \u0432\u0435\u0441\u044c \u043c\u043e\u0439 \u043f\u0443\u0442\u044c \u2014 \u044d\u0442\u043e \u043c\u043e\u0439 \u043c\u0430\u043d\u0438\u0444\u0435\u0441\u0442.
\u0421 \u043d\u0438\u0437\u043e\u0432 \u043a \u0432\u0435\u0440\u0445\u0430\u043c \u2014 \u044d\u0442\u043e \u043d\u0435 \u043f\u0440\u043e\u0441\u0442\u043e \u0441\u043b\u043e\u0432\u0430,
\u041a\u0430\u0436\u0434\u044b\u0439 \u0448\u0440\u0430\u043c \u043d\u0430 \u0434\u0443\u0448\u0435 \u2014 \u044d\u0442\u043e \u043c\u043e\u044f \u0433\u043b\u0430\u0432\u0430.
\u042f \u043f\u043e\u0434\u043d\u044f\u043b\u0441\u044f \u0441 \u043d\u0438\u0437\u043e\u0432, \u0438 \u0437\u0430\u043f\u043e\u043c\u043d\u0438 \u043d\u0430\u0432\u0435\u043a:
\u0415\u0441\u043b\u0438 \u0432\u0435\u0440\u0438\u0448\u044c \u0432 \u0441\u0435\u0431\u044f \u2014 \u0442\u044b \u0443\u0436\u0435 \u0447\u0435\u043b\u043e\u0432\u0435\u043a."""

def master_audio(audio_np, sr, target_lufs=-14.0):
    logger.info("=== MASTERING ===")
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
    logger.info(f"  Pre-norm LUFS: {current_lufs:.1f}")
    if not np.isinf(current_lufs):
        audio_out = pyln.normalize.loudness(audio_out, current_lufs, target_lufs)
    peak = np.max(np.abs(audio_out))
    if peak > 0.99:
        audio_out = audio_out * (0.99 / peak)
    final_lufs = meter.integrated_loudness(audio_out)
    logger.info(f"  Final LUFS: {final_lufs:.1f}, peak: {np.max(np.abs(audio_out)):.4f}")
    return audio_out

def main():
    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel
    from diffusers.models import AutoencoderOobleck
    from configuration_acestep_v15 import AceStepConfig
    from modeling_acestep_v15_base import AceStepConditionGenerationModel

    logger.info("=== LOADING BASE MODEL (48kHz) ===")
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
    logger.info(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    logger.info("=== APPLYING MACAN LoRA alpha=64 ===")
    fixed_dir = '/workspace/lora_fixed_base'
    os.makedirs(fixed_dir, exist_ok=True)
    lora_sd = load_file('/workspace/lora_output_v15/final/adapter_model.safetensors')
    fixed_sd = {k.replace('base_model.model.decoder.', 'base_model.model.'): v for k, v in lora_sd.items()}
    save_file(fixed_sd, f'{fixed_dir}/adapter_model.safetensors')
    with open('/workspace/lora_output_v15/final/adapter_config.json') as f:
        cfg = json.load(f)
    fixed_targets = list(set([t.replace('decoder.', '', 1) if t.startswith('decoder.') else t for t in cfg['target_modules']]))
    cfg['target_modules'] = fixed_targets
    cfg['lora_alpha'] = 64
    with open(f'{fixed_dir}/adapter_config.json', 'w') as f:
        json.dump(cfg, f)
    model.decoder = PeftModel.from_pretrained(model.decoder, fixed_dir, is_trainable=False).to(DEVICE, dtype=DTYPE)
    model.decoder.eval()

    INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
    GEN_PROMPT = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n"
    metas = "bpm: 85, key: Dm, time_signature: 4/4, genre: russian rap"
    text_prompt = GEN_PROMPT.format(INSTRUCTION, CAPTION, metas)
    lyrics_text = f"# Languages\nru\n\n# Lyric\n{LYRICS}<|endoftext|>"
    text_inputs = tokenizer(text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt")
    text_ids = text_inputs['input_ids'].to(DEVICE)
    text_mask = text_inputs['attention_mask'].to(DEVICE).bool()
    lyric_inputs = tokenizer(lyrics_text, padding="longest", truncation=True, max_length=2048, return_tensors="pt")
    lyric_ids = lyric_inputs['input_ids'].to(DEVICE)
    lyric_mask = lyric_inputs['attention_mask'].to(DEVICE).bool()
    with torch.no_grad():
        text_hs = text_enc(text_ids, lyric_attention_mask=None).last_hidden_state
        lyric_hs = text_enc.embed_tokens(lyric_ids)

    duration = 180.0
    tlen = int(duration * SR / DOWNSAMPLE)
    bs = 1
    ref = silence_latent[:, :750, :].expand(bs, -1, -1)
    rom = torch.arange(bs, device=DEVICE, dtype=torch.long)
    src = silence_latent[:, :tlen, :].expand(bs, -1, -1).clone()
    cm = torch.ones(bs, tlen, dtype=torch.bool, device=DEVICE).unsqueeze(-1).repeat(1, 1, 64).to(DTYPE)
    ic = torch.zeros(bs, device=DEVICE, dtype=DTYPE)
    attn_mask = torch.ones(bs, tlen, device=DEVICE, dtype=DTYPE)

    logger.info(f"=== GENERATING @ {SR}Hz, tlen={tlen} ===")
    t0 = time.time()
    with torch.no_grad():
        result = model.generate_audio(
            text_hidden_states=text_hs, text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hs, lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=ref,
            refer_audio_order_mask=rom, src_latents=src, chunk_masks=cm,
            is_covers=ic, silence_latent=silence_latent,
            attention_mask=attn_mask,
            seed=42, infer_steps=60, diffusion_guidance_sale=7.0,
            infer_method="ode", use_cache=True, use_progress_bar=True,
        )
    logger.info(f"Generation: {time.time()-t0:.1f}s")
    lat = result["target_latents"]
    logger.info(f"Latents: std={lat.std():.4f}, mean={lat.mean():.4f}")

    lat_vae = lat.transpose(1, 2).to(torch.float32)
    with torch.no_grad():
        audio = vae.decode(lat_vae).sample
    audio_np = audio[0].cpu().float().numpy()

    sf.write('/workspace/track_final_raw.wav', audio_np.T, SR)
    logger.info(f"Raw saved: duration={audio_np.shape[1]/SR:.1f}s")

    mastered = master_audio(audio_np, SR)
    sf.write('/workspace/track_final_mastered.wav', mastered, SR, subtype='PCM_24')
    logger.info(f"Mastered saved: duration={mastered.shape[0]/SR:.1f}s")
    logger.info("=== ALL DONE ===")

if __name__ == '__main__':
    main()
