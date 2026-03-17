#!/bin/bash
set -e

echo "============================================"
echo " ACE-Step v1.5 + MACAN LoRA — Pod Bootstrap"
echo "============================================"

echo "[1/6] Installing Python packages..."
pip install -q \
    torch==2.4.1 torchaudio==2.4.1 \
    transformers==5.3.0 \
    peft==0.18.1 \
    diffusers==0.32.2 \
    safetensors==0.7.0 \
    accelerate==1.13.0 \
    pedalboard==0.9.22 \
    pyloudnorm==0.2.0 \
    soundfile==0.13.1 \
    loguru==0.7.3 \
    soxr==1.0.0

echo "[2/6] Downloading ACE-Step v1.5 BASE model..."
mkdir -p /workspace/ace-step-v15-checkpoint
if [ ! -f /workspace/ace-step-v15-checkpoint/acestep-v15-base/model.safetensors ]; then
    pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ACE-Step/acestep-v15-base', local_dir='/workspace/ace-step-v15-checkpoint/acestep-v15-base')
"
    echo "  Base model downloaded."
else
    echo "  Base model already exists, skipping."
fi

echo "[3/6] Downloading VAE..."
if [ ! -d /workspace/ace-step-v15-checkpoint/vae ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ACE-Step/ACE-Step-v1-3.5B', local_dir='/tmp/ace_full', allow_patterns='vae/*')
import shutil
shutil.move('/tmp/ace_full/vae', '/workspace/ace-step-v15-checkpoint/vae')
shutil.rmtree('/tmp/ace_full', ignore_errors=True)
"
    echo "  VAE downloaded."
else
    echo "  VAE already exists, skipping."
fi

echo "[4/6] Downloading Qwen3-Embedding-0.6B..."
if [ ! -d /workspace/ace-step-v15-checkpoint/Qwen3-Embedding-0.6B ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='/workspace/ace-step-v15-checkpoint/Qwen3-Embedding-0.6B')
"
    echo "  Qwen downloaded."
else
    echo "  Qwen already exists, skipping."
fi

echo "[5/6] Applying patches..."
BASEDIR="/workspace/ace-step-v15-checkpoint/acestep-v15-base"
sed -i 's/mask_cat\.argsort(/mask_cat.int().argsort(/' "$BASEDIR/modeling_acestep_v15_base.py"
echo "  Bool sort CUDA bug patched."

echo "[6/6] Setting up LoRA and scripts..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p /workspace/lora_fixed_base
cp "$SCRIPT_DIR/lora_fixed_base/adapter_model.safetensors" /workspace/lora_fixed_base/
cp "$SCRIPT_DIR/lora_fixed_base/adapter_config.json" /workspace/lora_fixed_base/

mkdir -p /workspace/lora_output_v15/final
cp "$SCRIPT_DIR/lora_original/adapter_model.safetensors" /workspace/lora_output_v15/final/
cp "$SCRIPT_DIR/lora_original/adapter_config.json" /workspace/lora_output_v15/final/

cp "$SCRIPT_DIR/scripts/gen_final.py" /root/gen_final.py
cp "$SCRIPT_DIR/scripts/gen_base_lora_48k.py" /root/gen_base_lora_48k.py
cp "$SCRIPT_DIR/scripts/mastering.py" /root/mastering.py

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " To generate a track:"
echo "   python3 /root/gen_final.py"
echo ""
echo " Key parameters:"
echo "   SR=48000, downsample_ratio=1920"
echo "   LoRA alpha=64 (scale=1.0)"
echo "   Infer steps=60, guidance=7.0"
echo "   Mastering: LUFS -14, 24-bit PCM"
echo "============================================"
