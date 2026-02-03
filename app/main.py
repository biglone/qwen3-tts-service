from __future__ import annotations

import gc
import inspect
import io
import os
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel

CUSTOM_VOICES = [
    "vivian",
    "serena",
    "uncle_fu",
    "dylan",
    "eric",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
]

CUSTOM_MODEL_ID = os.getenv(
    "QWEN_TTS_CUSTOM_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)
DESIGN_MODEL_ID = os.getenv(
    "QWEN_TTS_DESIGN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)
CLONE_MODEL_ID = os.getenv(
    "QWEN_TTS_CLONE_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)

MODEL_CACHE_MAX = int(os.getenv("QWEN_TTS_MODEL_CACHE_MAX", "1"))
PRELOAD_MODES = [
    item.strip()
    for item in os.getenv("QWEN_TTS_PRELOAD", "").split(",")
    if item.strip()
]
DEVICE_MAP = os.getenv("QWEN_TTS_DEVICE_MAP", "auto")
DTYPE_NAME = os.getenv("QWEN_TTS_DTYPE", "float16")
MP3_BITRATE = os.getenv("QWEN_TTS_MP3_BITRATE", "192k")
STREAM_CHUNK_SIZE = int(os.getenv("QWEN_TTS_STREAM_CHUNK_SIZE", "65536"))
SERIALIZE_INFERENCE = os.getenv("QWEN_TTS_SERIALIZE", "true").lower() == "true"
HF_CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
WARMUP_ENABLED = os.getenv("QWEN_TTS_WARMUP", "true").lower() == "true"
WARMUP_TEXT = os.getenv("QWEN_TTS_WARMUP_TEXT", "hello")
WARMUP_VOICE = os.getenv("QWEN_TTS_WARMUP_VOICE", "vivian")
WARMUP_LANGUAGE = os.getenv("QWEN_TTS_WARMUP_LANGUAGE")


MODEL_IDS = {
    "custom": CUSTOM_MODEL_ID,
    "design": DESIGN_MODEL_ID,
    "clone": CLONE_MODEL_ID,
}


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


WARMUP_SPEED = _get_float_env("QWEN_TTS_WARMUP_SPEED", 1.0)


def _resolve_local_model_path(model_id: str) -> str:
    try:
        return snapshot_download(
            model_id,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Local cache missing for {model_id}. Please download models first."
        ) from exc


class CustomVoiceRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = Field("vivian")
    language: Optional[str] = None
    speed: float = Field(1.0, ge=0.25, le=4.0)
    output_format: str = Field("wav", pattern="^(wav|mp3)$")
    extra_params: Optional[Dict[str, Any]] = None


class VoiceDesignRequest(BaseModel):
    text: str = Field(..., min_length=1)
    prompt_text: str = Field(..., min_length=1)
    speed: float = Field(1.0, ge=0.25, le=4.0)
    output_format: str = Field("wav", pattern="^(wav|mp3)$")
    extra_params: Optional[Dict[str, Any]] = None


class ModelManager:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, Qwen3TTSModel] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, mode: str) -> Qwen3TTSModel:
        if mode not in MODEL_IDS:
            raise KeyError(f"Unknown mode: {mode}")
        model_id = MODEL_IDS[mode]
        with self._lock:
            if model_id in self._cache:
                self._cache.move_to_end(model_id)
                return self._cache[model_id]
            self._evict_if_needed()
            model = self._load_model(model_id)
            self._cache[model_id] = model
            return model

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= self._max_size and self._cache:
            _, old_model = self._cache.popitem(last=False)
            try:
                del old_model
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_model(self, model_id: str) -> Qwen3TTSModel:
        dtype = _get_torch_dtype(DTYPE_NAME)
        model_path = _resolve_local_model_path(model_id)
        return Qwen3TTSModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=DEVICE_MAP,
        )


MODEL_MANAGER = ModelManager(MODEL_CACHE_MAX)
INFER_LOCK = threading.Lock()

app = FastAPI(title="Qwen3 TTS Service", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    for mode in PRELOAD_MODES:
        if mode not in MODEL_IDS:
            continue
        MODEL_MANAGER.get(mode)
    if WARMUP_ENABLED:
        threading.Thread(target=_warmup_task, daemon=True).start()


def _warmup_task() -> None:
    text = WARMUP_TEXT.strip()
    if not text:
        return
    try:
        model = MODEL_MANAGER.get("custom")
        voice = WARMUP_VOICE.strip().lower()
        if voice not in CUSTOM_VOICES:
            voice = CUSTOM_VOICES[0]
        kwargs: Dict[str, Any] = {
            "text": text,
            "voice": voice,
            "speaker": voice,
            "speed": WARMUP_SPEED,
            "non_streaming_mode": True,
        }
        if WARMUP_LANGUAGE:
            kwargs["language"] = WARMUP_LANGUAGE
        kwargs = _filter_kwargs(model.generate_custom_voice, kwargs)
        _run_inference(model.generate_custom_voice, **kwargs)
    except Exception:
        return


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/voices")
def list_voices() -> Dict[str, Any]:
    return {"data": CUSTOM_VOICES}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {"data": MODEL_IDS}


@app.post("/v1/tts/custom")
def tts_custom(req: CustomVoiceRequest) -> Response:
    wav, sr = _run_custom_voice(req)
    audio_bytes = _encode_audio(wav, sr, req.output_format)
    return _audio_response(audio_bytes, req.output_format)


@app.post("/v1/tts/custom/stream")
def tts_custom_stream(req: CustomVoiceRequest) -> StreamingResponse:
    wav, sr = _run_custom_voice(req)
    audio_bytes = _encode_audio(wav, sr, req.output_format)
    return _audio_stream_response(audio_bytes, req.output_format)


@app.post("/v1/tts/design")
def tts_design(req: VoiceDesignRequest) -> Response:
    wav, sr = _run_voice_design(req)
    audio_bytes = _encode_audio(wav, sr, req.output_format)
    return _audio_response(audio_bytes, req.output_format)


@app.post("/v1/tts/design/stream")
def tts_design_stream(req: VoiceDesignRequest) -> StreamingResponse:
    wav, sr = _run_voice_design(req)
    audio_bytes = _encode_audio(wav, sr, req.output_format)
    return _audio_stream_response(audio_bytes, req.output_format)


@app.post("/v1/tts/clone")
def tts_clone(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    speed: float = Form(1.0),
    output_format: str = Form("wav"),
) -> Response:
    wav, sr = _run_voice_clone(text, ref_audio, ref_text, speed)
    audio_bytes = _encode_audio(wav, sr, output_format)
    return _audio_response(audio_bytes, output_format)


@app.post("/v1/tts/clone/stream")
def tts_clone_stream(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    speed: float = Form(1.0),
    output_format: str = Form("wav"),
) -> StreamingResponse:
    wav, sr = _run_voice_clone(text, ref_audio, ref_text, speed)
    audio_bytes = _encode_audio(wav, sr, output_format)
    return _audio_stream_response(audio_bytes, output_format)


def _run_custom_voice(req: CustomVoiceRequest) -> Tuple[np.ndarray, int]:
    model = MODEL_MANAGER.get("custom")
    voice = req.voice.strip().lower()
    if voice not in CUSTOM_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice: {req.voice}. Supported: {CUSTOM_VOICES}",
        )
    kwargs: Dict[str, Any] = {
        "text": req.text,
        "voice": voice,
        "speaker": voice,
        "speed": req.speed,
        "non_streaming_mode": True,
    }
    if req.language:
        kwargs["language"] = req.language
    if req.extra_params:
        kwargs.update(req.extra_params)
    kwargs = _filter_kwargs(model.generate_custom_voice, kwargs)
    return _run_inference(model.generate_custom_voice, **kwargs)


def _run_voice_design(req: VoiceDesignRequest) -> Tuple[np.ndarray, int]:
    model = MODEL_MANAGER.get("design")
    kwargs: Dict[str, Any] = {
        "text": req.text,
        "prompt_text": req.prompt_text,
        "speed": req.speed,
        "non_streaming_mode": True,
    }
    if req.extra_params:
        kwargs.update(req.extra_params)
    kwargs = _filter_kwargs(model.generate_voice_design, kwargs)
    return _run_inference(model.generate_voice_design, **kwargs)


def _run_voice_clone(
    text: str,
    ref_audio: UploadFile,
    ref_text: Optional[str],
    speed: float,
) -> Tuple[np.ndarray, int]:
    if speed < 0.25 or speed > 4.0:
        raise HTTPException(status_code=400, detail="speed must be in [0.25, 4.0]")
    model = MODEL_MANAGER.get("clone")
    if ref_audio is None:
        raise HTTPException(status_code=400, detail="ref_audio is required")

    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = Path(ref_audio.filename or "ref.wav").suffix
        if not suffix:
            suffix = ".wav"
        ref_path = Path(tmpdir) / f"ref{suffix}"
        with ref_path.open("wb") as f:
            f.write(ref_audio.file.read())

        kwargs: Dict[str, Any] = {
            "text": text,
            "ref_audio": str(ref_path),
            "speed": speed,
            "non_streaming_mode": True,
        }
        if ref_text:
            kwargs["ref_text"] = ref_text
        kwargs = _filter_kwargs(model.generate_voice_clone, kwargs)
        return _run_inference(model.generate_voice_clone, **kwargs)


def _run_inference(fn, **kwargs) -> Tuple[np.ndarray, int]:
    if SERIALIZE_INFERENCE:
        with INFER_LOCK:
            return fn(**kwargs)
    return fn(**kwargs)


def _encode_audio(wav: Any, sr: int, fmt: str) -> bytes:
    fmt = fmt.lower().strip()
    if isinstance(wav, (list, tuple)) and wav:
        wav = wav[0]
    wav_np = np.asarray(wav, dtype=np.float32)
    if fmt == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, wav_np, sr, format="WAV")
        return buffer.getvalue()
    if fmt == "mp3":
        return _encode_mp3(wav_np, sr)
    raise HTTPException(status_code=400, detail="output_format must be wav or mp3")


def _encode_mp3(wav: np.ndarray, sr: int) -> bytes:
    if not _ffmpeg_available():
        raise HTTPException(status_code=500, detail="ffmpeg is not available")
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "input.wav"
        mp3_path = Path(tmpdir) / "output.mp3"
        sf.write(wav_path, wav, sr, format="WAV")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            MP3_BITRATE,
            str(mp3_path),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=500, detail="ffmpeg failed") from exc
        return mp3_path.read_bytes()


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _audio_response(audio_bytes: bytes, fmt: str) -> Response:
    media_type = _media_type(fmt)
    headers = {"X-Audio-Format": fmt}
    return Response(content=audio_bytes, media_type=media_type, headers=headers)


def _audio_stream_response(audio_bytes: bytes, fmt: str) -> StreamingResponse:
    media_type = _media_type(fmt)
    headers = {"X-Audio-Format": fmt}
    return StreamingResponse(
        _iter_bytes(audio_bytes, STREAM_CHUNK_SIZE),
        media_type=media_type,
        headers=headers,
    )


def _iter_bytes(data: bytes, chunk_size: int) -> Iterable[bytes]:
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def _media_type(fmt: str) -> str:
    fmt = fmt.lower()
    if fmt == "wav":
        return "audio/wav"
    if fmt == "mp3":
        return "audio/mpeg"
    return "application/octet-stream"


def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _get_torch_dtype(name: str):
    key = name.lower()
    if key in {"float16", "fp16"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")
