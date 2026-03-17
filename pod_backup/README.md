# ACE-Step v1.5 + MACAN LoRA v5 — Pod Backup

## Quick Start (new pod)
```bash
# Upload this entire pod_backup/ folder to the pod, then:
cd /path/to/pod_backup
bash setup_pod.sh

# Generate a 3-minute track with mastering:
python3 /root/gen_final.py
```

## What's Inside

### LoRA Adapter (MACAN voice)
- `lora_fixed_base/` — LoRA adapter with fixed keys for BASE model (ready to use)
- `lora_original/` — Original LoRA adapter as trained (needs key remapping)
- Alpha=64 (scale=1.0), 168MB

### Scripts
- `scripts/gen_final.py` — Full pipeline: load model + LoRA → generate 180s → master → save WAV
- `scripts/gen_base_lora_48k.py` — Generation only (no mastering)
- `scripts/mastering.py` — Standalone mastering (EQ + compression + stereo widening + limiter + LUFS -14)

### Patches (reference)
- `patches/configuration_acestep_v15.py` — Patched config (layer_type_validation)
- `patches/modeling_acestep_v15_base.py` — Patched model (bool sort CUDA fix)
- `patches/config.json` — Model config JSON

## Critical Parameters
| Parameter | Value |
|-----------|-------|
| Sample rate | 48000 Hz |
| Downsample ratio | 1920 |
| tlen formula | `int(duration * 48000 / 1920)` |
| LoRA key fix | `base_model.model.decoder.` → `base_model.model.` |
| LoRA alpha | 64 |
| Bool sort fix | `mask_cat.int().argsort()` |
| Mastering LUFS | -14.0 |
| Output format | 24-bit PCM WAV |

## Required Packages
```
torch==2.4.1 (CUDA 12.4)
transformers==5.3.0
peft==0.18.1
diffusers==0.32.2
safetensors, accelerate, pedalboard, pyloudnorm, soundfile, loguru, soxr
```

## HuggingFace Models (auto-downloaded by setup_pod.sh)
- `ACE-Step/acestep-v15-base` — Base model (4.5GB)
- `ACE-Step/ACE-Step-v1-3.5B` — VAE only (324MB)
- `Qwen/Qwen3-Embedding-0.6B` — Text encoder (1.2GB)

## For RunPod Serverless
Use `setup_pod.sh` as the start script in your serverless template.
The script is idempotent — it skips downloads if files already exist on the volume.
