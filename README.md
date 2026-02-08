# Qwen3-TTS HTTP Service (Docker, Jetson)
[![CI](https://github.com/biglone/qwen3-tts-service/actions/workflows/ci.yml/badge.svg)](https://github.com/biglone/qwen3-tts-service/actions/workflows/ci.yml)

HTTP service built on `qwen-tts` with:
- CustomVoice (built-in voices)
- VoiceDesign (prompt-driven voice)
- VoiceClone (reference audio)
- Sync and streaming endpoints
- Output: `wav` / `mp3`

## Structure
```
qwen3-tts-service/
  app/
    main.py
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## Quick start

### 1) Build
```
# Requires the Qwen3-TTS demo image (qwen3-tts:cu130) built in ../Qwen3-TTS
docker build -t qwen3-tts-service:latest --build-arg BASE_IMAGE=qwen3-tts:cu130 .
```

### 2) Run (Jetson GPU)
```
docker run --rm -p 8000:8000 \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HOME=/hf-cache \
  -e TRANSFORMERS_CACHE=/hf-cache \
  -e QWEN_TTS_CUSTOM_MODEL_ID=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  -v /home/Biglone/workspace/Qwen3-TTS/models:/models:ro \
  -v $(pwd)/models:/hf-cache \
  qwen3-tts-service:latest
```

Or with compose:
```
docker compose up --build
```
Then visit: `http://localhost:18180`

## API (v1)

### Base URL
`http://localhost:8000` (or `http://localhost:18180` if started via compose)

### Auth
None (add a reverse proxy if needed)

### Health
`GET /healthz`

### Voices
`GET /v1/voices`

### Models
`GET /v1/models`

### CustomVoice
`POST /v1/tts/custom`

Body:
```json
{
  "text": "Hello from Qwen3-TTS",
  "voice": "Vivian",
  "instruct": "Say it in a very warm tone",
  "language": "auto",
  "speed": 1.0,
  "output_format": "wav"
}
```

Streaming: `POST /v1/tts/custom/stream`

### Segment Streaming (SSE)
`POST /v1/tts/custom/stream_segments`

Server-Sent Events stream that yields base64 audio chunks as they are generated.

Example:
```
curl -N -H "Content-Type: application/json" \
  -d '{"text":"Hello there.","voice":"vivian","output_format":"wav"}' \
  http://localhost:8000/v1/tts/custom/stream_segments
```

### VoiceDesign
`POST /v1/tts/design`

Body:
```json
{
  "text": "Please read this summary",
  "prompt_text": "Warm, calm female voice, medium pace",
  "speed": 1.0,
  "output_format": "mp3"
}
```

Streaming: `POST /v1/tts/design/stream`

Segment streaming: `POST /v1/tts/design/stream_segments`

### VoiceClone
`POST /v1/tts/clone` (multipart/form-data)

Fields:
- `text`
- `ref_audio` (file)
- `ref_text` (optional)
- `x_vector_only_mode` (optional, default false)
- `speed` (0.25 - 4.0)
- `output_format` (`wav` / `mp3`)

Example:
```
curl -X POST http://localhost:8000/v1/tts/clone \
  -F "text=Hello, cloned voice" \
  -F "ref_audio=@/path/to/ref.wav" \
  -F "ref_text=reference transcript" \
  -F "output_format=wav" \
  --output clone.wav
```

Streaming: `POST /v1/tts/clone/stream`

Segment streaming: `POST /v1/tts/clone/stream_segments`

## Notes

- `/stream` endpoints return chunked audio after synthesis completes.
- If future `qwen-tts` releases expose native streaming, you can wire it in `app/main.py`.
- Default models:
  - CustomVoice: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - VoiceDesign: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
  - VoiceClone: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

## Environment variables

- `QWEN_TTS_CUSTOM_MODEL_ID`
- `QWEN_TTS_DESIGN_MODEL_ID`
- `QWEN_TTS_CLONE_MODEL_ID`
- `QWEN_TTS_MODEL_CACHE_MAX` (default 1)
- `QWEN_TTS_PRELOAD` (e.g. `custom,design`)
- `QWEN_TTS_DTYPE` (`float16`/`bfloat16`/`float32`)
- `QWEN_TTS_DEVICE_MAP` (default `auto`)
- `QWEN_TTS_MP3_BITRATE` (default `192k`)
- `QWEN_TTS_STREAM_CHUNK_SIZE` (default 65536)
- `QWEN_TTS_SERIALIZE` (default `true`)
- `QWEN_TTS_STREAM_SEGMENT_MAX_CHARS` (default `120`)
- `QWEN_TTS_STREAM_SEGMENT_MIN_CHARS` (default `20`)
- `QWEN_TTS_STREAM_RETURN_FULL` (default `true`)
- `QWEN_TTS_STREAM_KEEPALIVE_SECONDS` (default `8`)
- `QWEN_TTS_STREAM_BATCH_SIZE` (default `1`)
- `QWEN_TTS_STREAM_RETURN_LINK` (default `false`)
- `QWEN_TTS_STREAM_DOWNLOAD_DIR` (default `/tmp/qwen3_tts_stream`)
- `QWEN_TTS_STREAM_DOWNLOAD_TTL` (default `3600`)
- `QWEN_TTS_MAX_NEW_TOKENS_PER_CHAR` (default unset)
- `QWEN_TTS_MAX_NEW_TOKENS_MIN` (default `128`)
- `QWEN_TTS_AUTO_LANGUAGE` (default `true`)
- `QWEN_TTS_DEFAULT_DO_SAMPLE` (default unset; uses model generate_config)
- `QWEN_TTS_DEFAULT_TOP_P` (default unset; uses model generate_config)
- `QWEN_TTS_DEFAULT_TEMPERATURE` (default unset; uses model generate_config)
- `QWEN_TTS_ATTN_IMPLEMENTATION` (e.g. `flash_attention_2`)
- `QWEN_TTS_FLASH_ATTN` (default `true`)
- `QWEN_TTS_TRIM_SILENCE` (default `false`)
- `QWEN_TTS_TRIM_THRESHOLD_DB` (default `-45`)
- `QWEN_TTS_TRIM_PAD_MS` (default `80`)
- `QWEN_TTS_WARMUP` (default `true`)
- `QWEN_TTS_WARMUP_TEXT` (default `hello`)
- `QWEN_TTS_WARMUP_VOICE` (default `vivian`)
- `QWEN_TTS_WARMUP_LANGUAGE` (optional)
- `QWEN_TTS_WARMUP_SPEED` (default `1.0`)
