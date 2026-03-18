import numpy as np
np.float = np.float64
np.int = np.int64

import torch, os, sys, time, json, io, base64, gc, shutil
import soundfile as sf
from safetensors.torch import load_file, save_file
from loguru import logger
import pyloudnorm as pyln
from pedalboard import Pedalboard, Compressor, Gain, LowShelfFilter, HighShelfFilter, PeakFilter, Limiter, HighpassFilter, LowpassFilter

CHECKPOINT_DIR = '/workspace/ace-step-v15-checkpoint'
BASE_DIR = os.path.join(CHECKPOINT_DIR, 'acestep-v15-base')
LORA_DIR = '/workspace/lora_fixed_base'

DEVICE = 'cuda'
DTYPE = torch.bfloat16
SR = 48000
DOWNSAMPLE = 1920

MODEL = None
VAE = None
TOKENIZER = None
TEXT_ENC = None
SILENCE_LATENT = None


def download_models():
    from huggingface_hub import snapshot_download

    if not os.path.exists(os.path.join(BASE_DIR, 'model.safetensors')):
        logger.info("Downloading ACE-Step v1.5 BASE...")
        snapshot_download('ACE-Step/acestep-v15-base', local_dir=BASE_DIR)

        sed_target = os.path.join(BASE_DIR, 'modeling_acestep_v15_base.py')
        if os.path.exists(sed_target):
            with open(sed_target, 'r') as f:
                code = f.read()
            code = code.replace('mask_cat.argsort(', 'mask_cat.int().argsort(')
            with open(sed_target, 'w') as f:
                f.write(code)
            logger.info("Patched bool sort CUDA bug")

    vae_dir = os.path.join(CHECKPOINT_DIR, 'vae')
    if not os.path.exists(vae_dir):
        logger.info("Downloading VAE...")
        snapshot_download('ACE-Step/ACE-Step-v1-3.5B', local_dir=CHECKPOINT_DIR, allow_patterns='vae/*')

    qwen_dir = os.path.join(CHECKPOINT_DIR, 'Qwen3-Embedding-0.6B')
    if not os.path.exists(qwen_dir):
        logger.info("Downloading Qwen3-Embedding-0.6B...")
        snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir=qwen_dir)

    if not os.path.exists(os.path.join(LORA_DIR, 'adapter_model.safetensors')):
        logger.info("Downloading MACAN LoRA v5...")
        tmp = '/tmp/macan_lora'
        snapshot_download('ruslanmusinrusmus/macan-lora-v5-acestep-v15', local_dir=tmp, allow_patterns='lora_fixed_base/*')
        src = os.path.join(tmp, 'lora_fixed_base')
        if os.path.exists(LORA_DIR):
            shutil.rmtree(LORA_DIR)
        shutil.copytree(src, LORA_DIR)
        shutil.rmtree(tmp, ignore_errors=True)

    logger.info("All models downloaded")


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


def load_models():
    global MODEL, VAE, TOKENIZER, TEXT_ENC, SILENCE_LATENT

    download_models()

    sys.path.insert(0, BASE_DIR)
    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel
    from diffusers.models import AutoencoderOobleck
    from configuration_acestep_v15 import AceStepConfig
    from modeling_acestep_v15_base import AceStepConditionGenerationModel

    logger.info("Loading models into GPU...")
    VAE = AutoencoderOobleck.from_pretrained(os.path.join(CHECKPOINT_DIR, 'vae')).to(DEVICE, dtype=torch.float32)
    VAE.eval()
    qwen_path = os.path.join(CHECKPOINT_DIR, 'Qwen3-Embedding-0.6B')
    TOKENIZER = AutoTokenizer.from_pretrained(qwen_path)
    TEXT_ENC = AutoModel.from_pretrained(qwen_path, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE)
    TEXT_ENC.eval()
    config = AceStepConfig.from_pretrained(BASE_DIR)
    base_sd = load_file(os.path.join(BASE_DIR, 'model.safetensors'))
    MODEL = AceStepConditionGenerationModel(config)
    MODEL.load_state_dict(base_sd, strict=False)
    MODEL = MODEL.to(DEVICE, dtype=DTYPE)
    MODEL.eval()
    SILENCE_LATENT = torch.load(os.path.join(BASE_DIR, 'silence_latent.pt'), weights_only=True).transpose(1, 2).to(DEVICE, dtype=DTYPE).detach()

    MODEL.decoder = PeftModel.from_pretrained(MODEL.decoder, LORA_DIR, is_trainable=False).to(DEVICE, dtype=DTYPE)
    MODEL.decoder.eval()
    logger.info(f"Models loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")


def generate(job):
    try:
        if MODEL is None:
            load_models()
    except Exception as e:
        import traceback
        error_msg = f"Model loading failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"error": error_msg}

    inp = job.get("input", {})
    caption = inp.get("caption", "voice_macan, Russian hip-hop/rap track with emotional male vocals, atmospheric synthesizer pads, deep 808 bass, crisp hi-hats, mid-tempo trap beat 85 BPM minor key")
    lyrics = inp.get("lyrics", "")
    duration = inp.get("duration", 180.0)
    seed = inp.get("seed", 42)
    infer_steps = inp.get("infer_steps", 60)
    guidance = inp.get("guidance", 7.0)
    lora_alpha = inp.get("lora_alpha", 64)
    bpm = inp.get("bpm", "85")
    key = inp.get("key", "Dm")
    do_master = inp.get("mastering", True)

    INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
    GEN_PROMPT = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n"
    metas = f"bpm: {bpm}, key: {key}, time_signature: 4/4, genre: russian rap"
    text_prompt = GEN_PROMPT.format(INSTRUCTION, caption, metas)
    lyrics_text = f"# Languages\nru\n\n# Lyric\n{lyrics}<|endoftext|>"

    text_inputs = TOKENIZER(text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt")
    text_ids = text_inputs['input_ids'].to(DEVICE)
    text_mask = text_inputs['attention_mask'].to(DEVICE).bool()
    lyric_inputs = TOKENIZER(lyrics_text, padding="longest", truncation=True, max_length=2048, return_tensors="pt")
    lyric_ids = lyric_inputs['input_ids'].to(DEVICE)
    lyric_mask = lyric_inputs['attention_mask'].to(DEVICE).bool()

    with torch.no_grad():
        text_hs = TEXT_ENC(text_ids, lyric_attention_mask=None).last_hidden_state
        lyric_hs = TEXT_ENC.embed_tokens(lyric_ids)

    tlen = int(duration * SR / DOWNSAMPLE)
    bs = 1
    ref = SILENCE_LATENT[:, :750, :].expand(bs, -1, -1)
    rom = torch.arange(bs, device=DEVICE, dtype=torch.long)
    src = SILENCE_LATENT[:, :tlen, :].expand(bs, -1, -1).clone()
    cm = torch.ones(bs, tlen, dtype=torch.bool, device=DEVICE).unsqueeze(-1).repeat(1, 1, 64).to(DTYPE)
    ic = torch.zeros(bs, device=DEVICE, dtype=DTYPE)
    attn_mask = torch.ones(bs, tlen, device=DEVICE, dtype=DTYPE)

    t0 = time.time()
    with torch.no_grad():
        result = MODEL.generate_audio(
            text_hidden_states=text_hs, text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hs, lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=ref,
            refer_audio_order_mask=rom, src_latents=src, chunk_masks=cm,
            is_covers=ic, silence_latent=SILENCE_LATENT,
            attention_mask=attn_mask,
            seed=seed, infer_steps=infer_steps, diffusion_guidance_sale=guidance,
            infer_method="ode", use_cache=True, use_progress_bar=False,
        )
    gen_time = time.time() - t0

    lat = result["target_latents"]
    lat_vae = lat.transpose(1, 2).to(torch.float32)
    with torch.no_grad():
        audio = VAE.decode(lat_vae).sample
    audio_np = audio[0].cpu().float().numpy()

    if do_master:
        audio_out = master_audio(audio_np, SR)
    else:
        audio_out = audio_np.T

    buf = io.BytesIO()
    sf.write(buf, audio_out, SR, subtype='PCM_24', format='WAV')
    audio_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "audio_base64": audio_b64,
        "sample_rate": SR,
        "duration": audio_out.shape[0] / SR,
        "generation_time": gen_time,
        "format": "wav_24bit",
    }

try:
    import runpod
    runpod.serverless.start({"handler": generate})
except ImportError:
    if __name__ == "__main__":
        result = generate({"input": {"lyrics": "test", "duration": 10}})
        print(f"Generated {result['duration']:.1f}s in {result['generation_time']:.1f}s")
