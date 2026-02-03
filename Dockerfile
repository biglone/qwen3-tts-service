ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        sox \
        libsox-dev \
        git \
        cmake \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# Install base API dependencies (exclude qwen-tts to avoid pulling torchaudio/torch from PyPI).
RUN grep -v '^qwen-tts' /app/requirements.txt > /tmp/requirements.base.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements.base.txt \
    && rm /tmp/requirements.base.txt
# Install qwen-tts dependencies except torchaudio/torch (built separately).
RUN python -m pip install --no-cache-dir \
        transformers==4.57.3 \
        accelerate==1.12.0 \
        gradio==5.3.0 \
        onnxruntime==1.23.2 \
        sox==1.5.0
# Build torchaudio from source against NVIDIA PyTorch to keep GPU support.
RUN git clone --depth 1 --branch v2.9.0 https://github.com/pytorch/audio.git /tmp/torchaudio \
    && cd /tmp/torchaudio \
    && python -m pip install --no-deps --no-build-isolation . \
    && cd / \
    && rm -rf /tmp/torchaudio
# Install qwen-tts itself without pulling dependencies.
RUN python -m pip install --no-cache-dir --no-deps qwen-tts==0.0.5
# Avoid torchvision import errors in transformers when torchvision ops are unavailable.
RUN python -m pip uninstall -y torchvision || true

COPY app /app/app

EXPOSE 8000

ENV QWEN_TTS_MODEL_CACHE_MAX=1 \
    QWEN_TTS_PRELOAD=custom \
    QWEN_TTS_DEVICE_MAP=auto \
    QWEN_TTS_DTYPE=float16

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
