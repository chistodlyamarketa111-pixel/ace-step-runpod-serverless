FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV ACESTEP_DOWNLOAD_SOURCE=huggingface
ENV HF_HOME=/workspace/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    ffmpeg libsndfile1 git curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python3 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip install uv

RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /app/ace-step

WORKDIR /app/ace-step

RUN uv pip install --system --no-cache-dir \
    "transformers>=4.51.0,<4.58.0" \
    diffusers \
    "scipy>=1.10.1" \
    "soundfile>=0.13.1" \
    "loguru>=0.7.3" \
    "einops>=0.8.1" \
    "accelerate>=1.12.0" \
    "numba" \
    "vector-quantize-pytorch>=1.27.15" \
    "peft>=0.18.0" \
    "lycoris-lora" \
    toml \
    huggingface_hub

RUN uv pip install --system --no-cache-dir ./acestep/third_parts/nano-vllm || true

RUN uv pip install --system --no-cache-dir --no-deps -e /app/ace-step

RUN pip install --no-cache-dir "runpod>=1.8.0"

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/Ace-Step1.5', local_dir='/app/checkpoints')"

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('ACE-Step/acestep-v15-sft', local_dir='/app/checkpoints/acestep-v15-sft'); \
    snapshot_download('ACE-Step/acestep-v15-base', local_dir='/app/checkpoints/acestep-v15-base'); \
    snapshot_download('ACE-Step/acestep-v15-turbo-shift3', local_dir='/app/checkpoints/acestep-v15-turbo-shift3')"

ENV ACESTEP_CHECKPOINT_DIR=/app/checkpoints

COPY download_loras.py /tmp/download_loras.py
RUN python /tmp/download_loras.py
RUN rm /tmp/download_loras.py

ENV ACESTEP_LORA_DIR=/app/loras
ENV NETWORK_VOLUME_LORA_DIR=/runpod-volume/loras

WORKDIR /app

COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

CMD ["python", "-u", "/app/handler.py"]
