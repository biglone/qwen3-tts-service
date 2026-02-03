# Qwen3-TTS HTTP Service (Docker, Jetson)

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
docker build -t qwen3-tts:latest .
```

### 2) Run (Jetson GPU)
```
docker run --rm -p 8000:8000 \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HOME=/models \
  -e TRANSFORMERS_CACHE=/models \
  -v $(pwd)/models:/models \
  qwen3-tts:latest
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
  "language": "auto",
  "speed": 1.0,
  "output_format": "wav"
}
```

Streaming: `POST /v1/tts/custom/stream`

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

### VoiceClone
`POST /v1/tts/clone` (multipart/form-data)

Fields:
- `text`
- `ref_audio` (file)
- `ref_text` (optional)
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
