ARG BASE_IMAGE=qwen3-tts:cu130
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        sox \
        libsox-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt /srv/requirements.txt
# Install API dependencies (exclude qwen-tts to avoid pulling torch/torchaudio from PyPI).
RUN grep -v '^qwen-tts' /srv/requirements.txt > /tmp/requirements.base.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements.base.txt \
    && rm /tmp/requirements.base.txt
COPY app /srv/app

EXPOSE 8000

ENV QWEN_TTS_MODEL_CACHE_MAX=1 \
    QWEN_TTS_PRELOAD=custom \
    QWEN_TTS_DEVICE_MAP=cuda:0 \
    QWEN_TTS_DTYPE=bfloat16 \
    QWEN_TTS_ATTN_IMPLEMENTATION=flash_attention_2 \
    QWEN_TTS_FLASH_ATTN=true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
