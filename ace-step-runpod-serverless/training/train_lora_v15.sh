#!/bin/bash

LORA_NAME="${1:-russianpop}"
DATASET_DIR="${2:-/workspace/dataset}"
RANK="${3:-32}"
LR="${4:-5e-5}"
EPOCHS="${5:-1}"
MODEL_VARIANT="${6:-turbo}"

ACE_STEP_DIR="/workspace/ace-step-1.5"
CHECKPOINT_DIR="/workspace/checkpoints"
TENSOR_DIR="/workspace/preprocessed_tensors/${LORA_NAME}"
LORA_OUTPUT_DIR="/workspace/lora_output/${LORA_NAME}"
FINAL_OUTPUT_DIR="/workspace/${LORA_NAME}"
TRAINING_LOG="/workspace/training_${LORA_NAME}.log"

echo "=== ACE-Step v1.5 LoRA Training (Side-Step) ==="
echo "LoRA name:       ${LORA_NAME}"
echo "Dataset:         ${DATASET_DIR}"
echo "Rank:            ${RANK}"
echo "Learning rate:   ${LR}"
echo "Model variant:   ${MODEL_VARIANT}"
echo "Epochs:          ${EPOCHS}"
echo ""

if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_DIR}"
    echo "Expected structure:"
    echo "  audio.mp3 + audio_prompt.txt + audio_lyrics.txt"
    echo "  OR"
    echo "  audio.mp3 + audio.txt (caption)"
    exit 1
fi

AUDIO_COUNT=$(find "${DATASET_DIR}" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.m4a" \) | wc -l)
echo "Found ${AUDIO_COUNT} audio files"

echo ""
echo "=== Step 1: Clone ACE-Step 1.5 ==="
if [ ! -d "${ACE_STEP_DIR}" ]; then
    git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git "${ACE_STEP_DIR}"
fi

echo ""
echo "=== Step 2: Install dependencies ==="
cd "${ACE_STEP_DIR}"
pip install --no-cache-dir -e ".[train]" 2>&1 | tail -5
pip install --no-cache-dir safetensors peft 2>&1 | tail -2

echo ""
echo "=== Step 3: Download checkpoints ==="
if [ ! -d "${CHECKPOINT_DIR}/acestep-v15-turbo" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ACE-Step/Ace-Step1.5', local_dir='${CHECKPOINT_DIR}')
snapshot_download('ACE-Step/acestep-v15-turbo', local_dir='${CHECKPOINT_DIR}/acestep-v15-turbo')
" 2>&1 | tail -5
fi
echo "Checkpoints ready"

echo ""
echo "=== Step 4: Preprocess audio (VAE + text encoding) ==="
cd "${ACE_STEP_DIR}"

python3 train.py fixed \
    --preprocess \
    --audio-dir "${DATASET_DIR}" \
    --tensor-output "${TENSOR_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --model-variant "${MODEL_VARIANT}" \
    --max-duration 240 \
    2>&1 | tee -a "${TRAINING_LOG}"

TENSOR_COUNT=$(find "${TENSOR_DIR}" -name "*.pt" 2>/dev/null | wc -l)
echo "Preprocessed ${TENSOR_COUNT} tensors"

if [ "${TENSOR_COUNT}" -eq 0 ]; then
    echo "ERROR: No tensors generated. Check audio files and logs."
    exit 1
fi

echo ""
echo "=== Step 5: Train LoRA ==="
echo "Starting at $(date)"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

python3 -u train.py fixed \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --model-variant "${MODEL_VARIANT}" \
    --dataset-dir "${TENSOR_DIR}" \
    --output-dir "${LORA_OUTPUT_DIR}" \
    --rank "${RANK}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    2>&1 | tee -a "${TRAINING_LOG}"

echo ""
echo "=== Step 6: Collect and verify LoRA ==="

ADAPTER_FILE=$(find "${LORA_OUTPUT_DIR}" -name "adapter_model.safetensors" -o -name "pytorch_lora_weights.safetensors" 2>/dev/null | head -1)

if [ -z "${ADAPTER_FILE}" ]; then
    echo "ERROR: No adapter safetensors found!"
    echo "Contents of ${LORA_OUTPUT_DIR}:"
    find "${LORA_OUTPUT_DIR}" -type f 2>/dev/null | head -20
    exit 1
fi

mkdir -p "${FINAL_OUTPUT_DIR}"
cp "${ADAPTER_FILE}" "${FINAL_OUTPUT_DIR}/adapter_model.safetensors"

CONFIG_FILE=$(find "${LORA_OUTPUT_DIR}" -name "adapter_config.json" 2>/dev/null | head -1)
if [ -n "${CONFIG_FILE}" ]; then
    cp "${CONFIG_FILE}" "${FINAL_OUTPUT_DIR}/adapter_config.json"
else
    cat > "${FINAL_OUTPUT_DIR}/adapter_config.json" << CONFIG_EOF
{
    "peft_type": "LORA",
    "r": ${RANK},
    "lora_alpha": $((RANK * 2)),
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "use_rslora": true,
    "base_model_name_or_path": "ACE-Step/Ace-Step1.5",
    "task_type": null
}
CONFIG_EOF
fi

python3 << 'VERIFY_EOF'
import json, os, sys
from safetensors import safe_open

output_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FINAL_OUTPUT_DIR", ".")
st_path = os.path.join(output_dir, "adapter_model.safetensors")
cfg_path = os.path.join(output_dir, "adapter_config.json")

with open(cfg_path) as f:
    cfg = json.load(f)
print(f"Config: peft_type={cfg.get('peft_type')}, r={cfg.get('r')}, target_modules={cfg.get('target_modules')}")

with safe_open(st_path, framework="pt") as f:
    keys = sorted(f.keys())
    print(f"Keys: {len(keys)} total")
    for k in keys[:4]:
        print(f"  {k}: {list(f.get_tensor(k).shape)}")

old_modules = {"to_q", "to_k", "to_v", "to_out.0"}
if old_modules & set(cfg.get("target_modules", [])):
    print("WARNING: target_modules still use old ACE-Step v1 names!")
    sys.exit(1)

has_2560 = False
with safe_open(st_path, framework="pt") as f:
    for k in f.keys():
        shape = list(f.get_tensor(k).shape)
        if 2560 in shape:
            has_2560 = True
            print(f"WARNING: tensor {k} has dim 2560 (v1 architecture)")
            break

if has_2560:
    print("ERROR: LoRA appears to be for ACE-Step v1, not v1.5!")
    sys.exit(1)

print("Verification PASSED: LoRA compatible with ACE-Step v1.5")
VERIFY_EOF

echo ""
echo "=== Step 7: Upload to HuggingFace ==="
python3 << UPLOAD_EOF
import os
from huggingface_hub import HfApi
token = os.environ.get("HF_TOKEN", "")
if not token:
    print("No HF_TOKEN set, skipping upload")
else:
    api = HfApi(token=token)
    repo_id = f"ruslanmusinrusmus/${os.environ.get('LORA_NAME', 'custom')}"
    try:
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(
            folder_path="${FINAL_OUTPUT_DIR}",
            repo_id=repo_id,
            commit_message="LoRA trained on ACE-Step v1.5 (Side-Step)",
        )
        print(f"Uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
        print("You can upload manually later")
UPLOAD_EOF

echo ""
echo "=== Training complete! ==="
echo "Finished at $(date)"
echo "LoRA saved to: ${FINAL_OUTPUT_DIR}"
ls -lh "${FINAL_OUTPUT_DIR}/"
echo ""
echo "To use with API: lora_name='${LORA_NAME}'"
