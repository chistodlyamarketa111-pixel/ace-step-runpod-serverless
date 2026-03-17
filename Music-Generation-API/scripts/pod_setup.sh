#!/bin/bash
set -e

echo "=== ACE-Step Pod Setup ==="
echo "Installing dependencies..."
pip install -q git+https://github.com/ACE-Step/ACE-Step.git peft soundfile scipy huggingface_hub

echo ""
echo "=== Checking LoRAs ==="
mkdir -p /workspace/loras

LORAS=("anna-asti-v2:ruslanmusinrusmus/anna-asti-v2" "russian-pop-lora:kemendev/russian-pop-lora")

for entry in "${LORAS[@]}"; do
    NAME="${entry%%:*}"
    REPO="${entry##*:}"
    DIR="/workspace/loras/$NAME"
    if [ -d "$DIR" ] && [ -f "$DIR/adapter_config.json" ]; then
        echo "  $NAME — already on volume"
    else
        echo "  Downloading $NAME from HuggingFace..."
        mkdir -p "$DIR"
        python3 -c "
from huggingface_hub import hf_hub_download
import os
for f in ['adapter_config.json', 'adapter_model.safetensors']:
    try:
        src = hf_hub_download('$REPO', f, local_dir='$DIR')
        print(f'  OK: {f}')
    except Exception as e:
        print(f'  Skip {f}: {e}')
"
    fi
done

echo ""
echo "=== Available LoRAs ==="
ls /workspace/loras/

echo ""
echo "=== Setup complete! ==="
echo "Run training: python3 /workspace/train.py"
