FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV ACESTEP_DOWNLOAD_SOURCE=huggingface
ENV HF_HOME=/tmp/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

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
    toml \
    huggingface_hub \
    fastapi \
    uvicorn

RUN uv pip install --system --no-cache-dir ./acestep/third_parts/nano-vllm || true
RUN uv pip install --system --no-cache-dir --no-deps -e /app/ace-step

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/Ace-Step1.5', local_dir='/app/checkpoints')"

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('ACE-Step/acestep-v15-sft', local_dir='/app/checkpoints/acestep-v15-sft'); \
    snapshot_download('ACE-Step/acestep-v15-base', local_dir='/app/checkpoints/acestep-v15-base'); \
    snapshot_download('ACE-Step/acestep-v15-turbo-shift3', local_dir='/app/checkpoints/acestep-v15-turbo-shift3')"

ENV ACESTEP_CHECKPOINT_DIR=/app/checkpoints

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 80

CMD ["python", "-u", "/app/app.py"]
