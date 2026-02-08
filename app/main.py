from __future__ import annotations

import base64
import concurrent.futures
import gc
import inspect
import io
import json
import logging
import os
import re
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
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
STREAM_SEGMENT_MAX_CHARS = int(os.getenv("QWEN_TTS_STREAM_SEGMENT_MAX_CHARS", "120"))
STREAM_SEGMENT_MIN_CHARS = int(os.getenv("QWEN_TTS_STREAM_SEGMENT_MIN_CHARS", "20"))
STREAM_RETURN_FULL = os.getenv("QWEN_TTS_STREAM_RETURN_FULL", "true").lower() == "true"
STREAM_KEEPALIVE_SECONDS = float(os.getenv("QWEN_TTS_STREAM_KEEPALIVE_SECONDS", "8"))
MAX_NEW_TOKENS = os.getenv("QWEN_TTS_MAX_NEW_TOKENS")
STREAM_BATCH_SIZE = max(int(os.getenv("QWEN_TTS_STREAM_BATCH_SIZE", "1")), 1)
STREAM_RETURN_LINK = os.getenv("QWEN_TTS_STREAM_RETURN_LINK", "false").lower() == "true"
STREAM_DOWNLOAD_DIR = Path(
    os.getenv("QWEN_TTS_STREAM_DOWNLOAD_DIR", "/tmp/qwen3_tts_stream")
).resolve()
STREAM_DOWNLOAD_TTL = int(os.getenv("QWEN_TTS_STREAM_DOWNLOAD_TTL", "3600"))
TRIM_SILENCE = os.getenv("QWEN_TTS_TRIM_SILENCE", "false").lower() == "true"
TRIM_THRESHOLD_DB = float(os.getenv("QWEN_TTS_TRIM_THRESHOLD_DB", "-45"))
TRIM_PAD_MS = float(os.getenv("QWEN_TTS_TRIM_PAD_MS", "80"))
AUTO_LANGUAGE = os.getenv("QWEN_TTS_AUTO_LANGUAGE", "false").lower() == "true"

def _get_optional_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "none", "null", "unset"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return default


def _get_optional_bool(name: str, default: Optional[bool]) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "none", "null", "unset"}:
        return None
    return raw in {"1", "true", "yes", "y", "on"}


DEFAULT_DO_SAMPLE = _get_optional_bool("QWEN_TTS_DEFAULT_DO_SAMPLE", None)
DEFAULT_TOP_P = _get_optional_float("QWEN_TTS_DEFAULT_TOP_P", None)
DEFAULT_TEMPERATURE = _get_optional_float("QWEN_TTS_DEFAULT_TEMPERATURE", None)
MAX_NEW_TOKENS_PER_CHAR = _get_optional_float("QWEN_TTS_MAX_NEW_TOKENS_PER_CHAR", None)
MAX_NEW_TOKENS_MIN = int(os.getenv("QWEN_TTS_MAX_NEW_TOKENS_MIN", "128"))
ATTN_IMPLEMENTATION = os.getenv("QWEN_TTS_ATTN_IMPLEMENTATION")
USE_FLASH_ATTN = os.getenv("QWEN_TTS_FLASH_ATTN", "true").lower() == "true"


MODEL_IDS = {
    "custom": CUSTOM_MODEL_ID,
    "design": DESIGN_MODEL_ID,
    "clone": CLONE_MODEL_ID,
}

_SENTENCE_BREAKS = set("。！？!?；;，,、\n")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


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
    path = Path(model_id).expanduser()
    if path.exists():
        return str(path.resolve())
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
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    language: Optional[str] = None
    speed: float = Field(1.0, ge=0.25, le=4.0)
    output_format: str = Field("wav", pattern="^(wav|mp3)$")
    extra_params: Optional[Dict[str, Any]] = None


class VoiceDesignRequest(BaseModel):
    text: str = Field(..., min_length=1)
    prompt_text: str = Field(..., min_length=1)
    language: Optional[str] = None
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
        attn_impl = ATTN_IMPLEMENTATION
        if not attn_impl and USE_FLASH_ATTN:
            attn_impl = "flash_attention_2"
        if attn_impl and not torch.cuda.is_available():
            attn_impl = None
        kwargs: Dict[str, Any] = {
            "device_map": DEVICE_MAP,
            "dtype": dtype,
            "torch_dtype": dtype,
        }
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            **kwargs,
        )
        try:
            param = next(model.parameters())
            logger.info("Loaded %s on %s", model_id, param.device)
        except Exception:
            logger.info("Loaded %s", model_id)
        return model


MODEL_MANAGER = ModelManager(MODEL_CACHE_MAX)
INFER_LOCK = threading.Lock()
STREAM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
DOWNLOAD_LOCK = threading.Lock()
DOWNLOAD_INDEX: Dict[str, Tuple[Path, float, str, str]] = {}

STREAM_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Qwen3 TTS Service", version="1.0")
WEB_DIR = Path(__file__).resolve().parent / "web"
ASSETS_DIR = WEB_DIR / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


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
        voice = (WARMUP_VOICE or "").strip()
        try:
            speaker = _resolve_speaker(model, voice)
        except HTTPException:
            speaker = _resolve_speaker(model, (CUSTOM_VOICES[0] if CUSTOM_VOICES else ""))
        kwargs: Dict[str, Any] = {
            "text": text,
            "speaker": speaker,
            "speed": WARMUP_SPEED,
            "non_streaming_mode": True,
        }
        if WARMUP_LANGUAGE:
            kwargs["language"] = WARMUP_LANGUAGE
        kwargs = _filter_kwargs(model.generate_custom_voice, kwargs)
        _run_inference(model.generate_custom_voice, **kwargs)
    except Exception:
        return


def _segment_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    segments: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _SENTENCE_BREAKS:
            segment = "".join(buf).strip()
            if segment:
                segments.append(segment)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)

    if not segments:
        return [text]

    expanded: list[str] = []
    for seg in segments:
        if len(seg) <= STREAM_SEGMENT_MAX_CHARS:
            expanded.append(seg)
            continue
        for i in range(0, len(seg), STREAM_SEGMENT_MAX_CHARS):
            expanded.append(seg[i : i + STREAM_SEGMENT_MAX_CHARS])

    merged: list[str] = []
    buffer = ""
    for seg in expanded:
        if not buffer:
            buffer = seg
            continue
        if len(buffer) < STREAM_SEGMENT_MIN_CHARS:
            buffer += seg
            continue
        if len(buffer) + len(seg) <= STREAM_SEGMENT_MAX_CHARS:
            buffer += seg
            continue
        merged.append(buffer)
        buffer = seg
    if buffer:
        merged.append(buffer)
    return merged


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def _to_wav_array(wav: Any) -> np.ndarray:
    if isinstance(wav, (list, tuple)) and wav:
        wav = wav[0]
    return np.asarray(wav, dtype=np.float32)


def _trim_silence(wav: np.ndarray, sr: int) -> np.ndarray:
    if not TRIM_SILENCE:
        return wav
    if wav.size == 0:
        return wav
    if wav.ndim > 1:
        amp = np.max(np.abs(wav), axis=1)
    else:
        amp = np.abs(wav)
    threshold = 10 ** (TRIM_THRESHOLD_DB / 20.0)
    idx = np.where(amp > threshold)[0]
    if idx.size == 0:
        return wav
    pad = int(sr * TRIM_PAD_MS / 1000.0)
    start = max(int(idx[0]) - pad, 0)
    end = min(int(idx[-1]) + pad + 1, len(wav))
    return wav[start:end]


def _exc_detail(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc)


def _auto_language(text: str, provided: Optional[str]) -> Optional[str]:
    if provided:
        return _normalize_language(provided)
    if not AUTO_LANGUAGE:
        return None
    if _CJK_RE.search(text):
        return "chinese"
    return None


def _normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    value = str(language).strip()
    if not value:
        return None
    return value.lower()

def _get_supported_speakers(model: Qwen3TTSModel) -> Optional[list[str]]:
    getter = getattr(getattr(model, "model", None), "get_supported_speakers", None)
    if callable(getter):
        try:
            speakers = list(getter())
        except Exception:
            speakers = None
        if speakers:
            return [str(s) for s in speakers]
    return None


def _resolve_speaker(model: Qwen3TTSModel, speaker: Optional[str]) -> str:
    raw = (speaker or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="speaker is required")
    speakers = _get_supported_speakers(model) or CUSTOM_VOICES
    mapping = {str(s).lower(): str(s) for s in speakers}
    key = raw.lower()
    if key in mapping:
        return mapping[key]
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported speaker: {raw}. Supported: {speakers}",
    )


def _apply_default_sampling(kwargs: Dict[str, Any]) -> None:
    if DEFAULT_DO_SAMPLE is not None and "do_sample" not in kwargs:
        kwargs["do_sample"] = DEFAULT_DO_SAMPLE
    if kwargs.get("do_sample"):
        if DEFAULT_TOP_P is not None and "top_p" not in kwargs:
            kwargs["top_p"] = DEFAULT_TOP_P
        if DEFAULT_TEMPERATURE is not None and "temperature" not in kwargs:
            kwargs["temperature"] = DEFAULT_TEMPERATURE


def _compute_max_new_tokens(text: str, cap_override: Optional[int] = None) -> Optional[int]:
    cap = cap_override
    if cap is None and MAX_NEW_TOKENS:
        try:
            cap = int(MAX_NEW_TOKENS)
        except ValueError:
            cap = None
    per_char = MAX_NEW_TOKENS_PER_CHAR
    if not per_char or per_char <= 0:
        return cap
    est = max(MAX_NEW_TOKENS_MIN, int(len(text) * per_char))
    if cap is not None:
        return min(cap, est)
    return est


def _cleanup_downloads() -> None:
    now = time.time()
    with DOWNLOAD_LOCK:
        expired = [k for k, (_, exp, _, _) in DOWNLOAD_INDEX.items() if exp <= now]
        for key in expired:
            path, _, _, _ = DOWNLOAD_INDEX.pop(key, (None, None, None, None))
            if path and path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass


def _register_download(audio_bytes: bytes, fmt: str) -> str:
    _cleanup_downloads()
    file_id = uuid.uuid4().hex
    filename = f"qwen3_tts_{file_id}.{fmt}"
    path = STREAM_DOWNLOAD_DIR / filename
    path.write_bytes(audio_bytes)
    expires_at = time.time() + STREAM_DOWNLOAD_TTL
    media_type = _media_type(fmt)
    with DOWNLOAD_LOCK:
        DOWNLOAD_INDEX[file_id] = (path, expires_at, media_type, filename)
    return file_id


def _register_download_path(path: Path, fmt: str) -> str:
    _cleanup_downloads()
    file_id = uuid.uuid4().hex
    filename = f"qwen3_tts_{file_id}.{fmt}"
    target = STREAM_DOWNLOAD_DIR / filename
    if path != target:
        try:
            path.replace(target)
        except Exception:
            shutil.move(str(path), str(target))
    expires_at = time.time() + STREAM_DOWNLOAD_TTL
    media_type = _media_type(fmt)
    with DOWNLOAD_LOCK:
        DOWNLOAD_INDEX[file_id] = (target, expires_at, media_type, filename)
    return file_id


def _new_temp_wav_path() -> Path:
    fd, tmp_path = tempfile.mkstemp(
        prefix="qwen3_tts_stream_", suffix=".wav", dir=STREAM_DOWNLOAD_DIR
    )
    os.close(fd)
    return Path(tmp_path)


def _open_stream_writer(sr: int, wav_np: np.ndarray) -> Tuple[sf.SoundFile, Path]:
    path = _new_temp_wav_path()
    channels = 1 if wav_np.ndim == 1 else wav_np.shape[1]
    writer = sf.SoundFile(path, mode="w", samplerate=sr, channels=channels, format="WAV")
    return writer, path


def _finalize_stream_file(path: Path, fmt: str) -> Path:
    fmt = fmt.lower()
    if fmt == "wav":
        return path
    if fmt == "mp3":
        mp3_path = path.with_suffix(".mp3")
        _convert_wav_to_mp3(path, mp3_path)
        try:
            path.unlink()
        except Exception:
            pass
        return mp3_path
    return path


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> Response:
    if WEB_DIR.exists():
        return FileResponse(WEB_DIR / "index.html")
    return Response(content="Qwen3 TTS Service", media_type="text/plain")


@app.get("/v1/voices")
def list_voices() -> Dict[str, Any]:
    try:
        model = MODEL_MANAGER.get("custom")
    except Exception:
        model = None
    if model:
        speakers = _get_supported_speakers(model)
        if speakers:
            return {"data": speakers}
    return {"data": CUSTOM_VOICES}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {"data": MODEL_IDS}


@app.get("/v1/tts/download/{file_id}")
def download_tts(file_id: str) -> FileResponse:
    _cleanup_downloads()
    with DOWNLOAD_LOCK:
        entry = DOWNLOAD_INDEX.get(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Download not found or expired.")
    path, expires_at, media_type, filename = entry
    if time.time() > expires_at or not path.exists():
        raise HTTPException(status_code=404, detail="Download not found or expired.")
    headers = {"Cache-Control": "no-store"}
    return FileResponse(path, media_type=media_type, filename=filename, headers=headers)


@app.post("/v1/tts/custom")
def tts_custom(req: CustomVoiceRequest) -> Response:
    wav, sr = _run_custom_voice(req)
    wav_np = _trim_silence(_to_wav_array(wav), sr)
    audio_bytes = _encode_audio(wav_np, sr, req.output_format)
    return _audio_response(audio_bytes, req.output_format)


@app.post("/v1/tts/custom/stream")
def tts_custom_stream(req: CustomVoiceRequest) -> StreamingResponse:
    wav, sr = _run_custom_voice(req, streaming=True)
    wav_np = _trim_silence(_to_wav_array(wav), sr)
    audio_bytes = _encode_audio(wav_np, sr, req.output_format)
    return _audio_stream_response(audio_bytes, req.output_format)


@app.post("/v1/tts/custom/stream_segments")
def tts_custom_stream_segments(req: CustomVoiceRequest) -> StreamingResponse:
    return _stream_custom_segments(req)


@app.post("/v1/tts/design")
def tts_design(req: VoiceDesignRequest) -> Response:
    wav, sr = _run_voice_design(req)
    wav_np = _trim_silence(_to_wav_array(wav), sr)
    audio_bytes = _encode_audio(wav_np, sr, req.output_format)
    return _audio_response(audio_bytes, req.output_format)


@app.post("/v1/tts/design/stream")
def tts_design_stream(req: VoiceDesignRequest) -> StreamingResponse:
    wav, sr = _run_voice_design(req, streaming=True)
    wav_np = _trim_silence(_to_wav_array(wav), sr)
    audio_bytes = _encode_audio(wav_np, sr, req.output_format)
    return _audio_stream_response(audio_bytes, req.output_format)


@app.post("/v1/tts/design/stream_segments")
def tts_design_stream_segments(req: VoiceDesignRequest) -> StreamingResponse:
    return _stream_design_segments(req)


@app.post("/v1/tts/clone")
def tts_clone(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    speed: float = Form(1.0),
    x_vector_only_mode: bool = Form(False),
    output_format: str = Form("wav"),
) -> Response:
    wav, sr = _run_voice_clone(text, ref_audio, ref_text, language, speed, x_vector_only_mode)
    wav_np = _trim_silence(_to_wav_array(wav), sr)
    audio_bytes = _encode_audio(wav_np, sr, output_format)
    return _audio_response(audio_bytes, output_format)


@app.post("/v1/tts/clone/stream")
def tts_clone_stream(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    speed: float = Form(1.0),
    x_vector_only_mode: bool = Form(False),
    output_format: str = Form("wav"),
) -> StreamingResponse:
    wav, sr = _run_voice_clone(
        text,
        ref_audio,
        ref_text,
        language,
        speed,
        x_vector_only_mode,
        streaming=True,
    )
    wav_np = _trim_silence(_to_wav_array(wav), sr)
    audio_bytes = _encode_audio(wav_np, sr, output_format)
    return _audio_stream_response(audio_bytes, output_format)


@app.post("/v1/tts/clone/stream_segments")
def tts_clone_stream_segments(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    speed: float = Form(1.0),
    x_vector_only_mode: bool = Form(False),
    output_format: str = Form("wav"),
) -> StreamingResponse:
    return _stream_clone_segments(
        text,
        ref_audio,
        ref_text,
        language,
        speed,
        x_vector_only_mode,
        output_format,
    )


def _run_custom_voice(
    req: CustomVoiceRequest,
    streaming: bool = False,
) -> Tuple[np.ndarray, int]:
    model = MODEL_MANAGER.get("custom")
    speaker = _resolve_speaker(model, req.speaker or req.voice)
    kwargs: Dict[str, Any] = {
        "text": req.text,
        "speaker": speaker,
        "speed": req.speed,
        "non_streaming_mode": True,
    }
    instruct = (req.instruct or "").strip()
    if instruct:
        kwargs["instruct"] = instruct
    language = _auto_language(req.text, req.language)
    if language:
        kwargs["language"] = language
    _apply_default_sampling(kwargs)
    if req.extra_params:
        kwargs.update(req.extra_params)
    max_tokens = _compute_max_new_tokens(req.text)
    if max_tokens:
        kwargs["max_new_tokens"] = max_tokens
    kwargs = _filter_kwargs(model.generate_custom_voice, kwargs)
    return _run_inference(model.generate_custom_voice, **kwargs)


def _run_custom_voice_batch(
    req: CustomVoiceRequest,
    texts: list[str],
    streaming: bool = False,
) -> Tuple[list[np.ndarray], int]:
    model = MODEL_MANAGER.get("custom")
    speaker = _resolve_speaker(model, req.speaker or req.voice)
    batch = len(texts)
    kwargs: Dict[str, Any] = {
        "text": texts,
        "speaker": [speaker] * batch,
        "speed": req.speed,
        "non_streaming_mode": True,
    }
    instruct = (req.instruct or "").strip()
    if instruct:
        kwargs["instruct"] = [instruct] * batch
    language = _auto_language("".join(texts), req.language)
    if language:
        kwargs["language"] = [language] * batch
    _apply_default_sampling(kwargs)
    if req.extra_params:
        kwargs.update(req.extra_params)
    max_tokens = _compute_max_new_tokens(max(texts, key=len) if texts else "")
    if max_tokens:
        kwargs["max_new_tokens"] = max_tokens
    kwargs = _filter_kwargs(model.generate_custom_voice, kwargs)
    return _run_inference(model.generate_custom_voice, **kwargs)


def _run_voice_design(
    req: VoiceDesignRequest,
    streaming: bool = False,
) -> Tuple[np.ndarray, int]:
    model = MODEL_MANAGER.get("design")
    kwargs: Dict[str, Any] = {
        "text": req.text,
        "instruct": req.prompt_text,
        "speed": req.speed,
        "non_streaming_mode": True,
    }
    language = _auto_language(req.text, req.language)
    if language:
        kwargs["language"] = language
    _apply_default_sampling(kwargs)
    if req.extra_params:
        kwargs.update(req.extra_params)
    max_tokens = _compute_max_new_tokens(req.text)
    if max_tokens:
        kwargs["max_new_tokens"] = max_tokens
    kwargs = _filter_kwargs(model.generate_voice_design, kwargs)
    return _run_inference(model.generate_voice_design, **kwargs)


def _run_voice_design_batch(
    req: VoiceDesignRequest,
    texts: list[str],
    streaming: bool = False,
) -> Tuple[list[np.ndarray], int]:
    model = MODEL_MANAGER.get("design")
    batch = len(texts)
    kwargs: Dict[str, Any] = {
        "text": texts,
        "instruct": [req.prompt_text] * batch,
        "speed": req.speed,
        "non_streaming_mode": True,
    }
    language = _auto_language("".join(texts), req.language)
    if language:
        kwargs["language"] = [language] * batch
    _apply_default_sampling(kwargs)
    if req.extra_params:
        kwargs.update(req.extra_params)
    max_tokens = _compute_max_new_tokens(max(texts, key=len) if texts else "")
    if max_tokens:
        kwargs["max_new_tokens"] = max_tokens
    kwargs = _filter_kwargs(model.generate_voice_design, kwargs)
    return _run_inference(model.generate_voice_design, **kwargs)


def _run_voice_clone(
    text: str,
    ref_audio: UploadFile,
    ref_text: Optional[str],
    language: Optional[str],
    speed: float,
    x_vector_only_mode: bool,
    streaming: bool = False,
) -> Tuple[np.ndarray, int]:
    if speed < 0.25 or speed > 4.0:
        raise HTTPException(status_code=400, detail="speed must be in [0.25, 4.0]")
    if ref_audio is None:
        raise HTTPException(status_code=400, detail="ref_audio is required")

    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = Path(ref_audio.filename or "ref.wav").suffix
        if not suffix:
            suffix = ".wav"
        ref_path = Path(tmpdir) / f"ref{suffix}"
        with ref_path.open("wb") as f:
            f.write(ref_audio.file.read())
        return _run_voice_clone_from_path(
            text,
            ref_path,
            ref_text=ref_text,
            language=language,
            speed=speed,
            x_vector_only_mode=x_vector_only_mode,
            streaming=streaming,
        )


def _run_voice_clone_from_path(
    text: str,
    ref_path: Path,
    ref_text: Optional[str],
    language: Optional[str],
    speed: float,
    x_vector_only_mode: bool,
    streaming: bool = False,
) -> Tuple[np.ndarray, int]:
    if speed < 0.25 or speed > 4.0:
        raise HTTPException(status_code=400, detail="speed must be in [0.25, 4.0]")
    if not x_vector_only_mode and not (ref_text and ref_text.strip()):
        raise HTTPException(
            status_code=400,
            detail="ref_text is required when x_vector_only_mode is false.",
        )
    model = MODEL_MANAGER.get("clone")
    kwargs: Dict[str, Any] = {
        "text": text,
        "ref_audio": str(ref_path),
        "x_vector_only_mode": x_vector_only_mode,
    }
    language = _auto_language(text, language)
    if language:
        kwargs["language"] = language
    if ref_text:
        kwargs["ref_text"] = ref_text
    _apply_default_sampling(kwargs)
    max_tokens = _compute_max_new_tokens(text)
    if max_tokens:
        kwargs["max_new_tokens"] = max_tokens
    kwargs = _filter_kwargs(model.generate_voice_clone, kwargs)
    return _run_inference(model.generate_voice_clone, **kwargs)


def _run_inference(fn, **kwargs) -> Tuple[np.ndarray, int]:
    if SERIALIZE_INFERENCE:
        with INFER_LOCK:
            return fn(**kwargs)
    return fn(**kwargs)


def _stream_custom_segments(req: CustomVoiceRequest) -> StreamingResponse:
    segments = _segment_text(req.text)
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    def event_stream():
        total = len(segments)
        if total == 0:
            yield _sse_event("error", {"detail": "Empty text"})
            return
        try:
            yield _sse_event("meta", {"segments": total, "format": req.output_format})
            wav_parts: list[np.ndarray] = []
            sample_rate: Optional[int] = None
            total_frames = 0
            writer = None
            writer_path: Optional[Path] = None
            collect_full = STREAM_RETURN_FULL
            for start in range(0, total, STREAM_BATCH_SIZE):
                batch_segments = segments[start : start + STREAM_BATCH_SIZE]
                seg_req = CustomVoiceRequest(
                    text=batch_segments[0],
                    voice=req.voice,
                    speaker=req.speaker,
                    instruct=req.instruct,
                    language=req.language,
                    speed=req.speed,
                    output_format=req.output_format,
                    extra_params=req.extra_params,
                )
                future = STREAM_EXECUTOR.submit(
                    _run_custom_voice_batch,
                    seg_req,
                    batch_segments,
                    True,
                )
                while True:
                    try:
                        wavs, sr = future.result(timeout=STREAM_KEEPALIVE_SECONDS)
                        break
                    except concurrent.futures.TimeoutError:
                        yield _sse_event(
                            "keepalive",
                            {"index": start + 1, "total": total},
                        )
                sample_rate = sr
                for offset, segment in enumerate(batch_segments):
                    seg_index = start + offset + 1
                    wav_np = _to_wav_array(wavs[offset])
                    if TRIM_SILENCE and (seg_index == 1 or seg_index == total):
                        wav_np = _trim_silence(wav_np, sr)
                    if collect_full:
                        wav_parts.append(wav_np)
                    if STREAM_RETURN_LINK:
                        if writer is None:
                            writer, writer_path = _open_stream_writer(sr, wav_np)
                        writer.write(wav_np)
                    audio_bytes = _encode_audio(wav_np, sr, req.output_format)
                    total_frames += len(wav_np)
                    payload = {
                        "index": seg_index,
                        "total": total,
                        "text": segment,
                        "format": req.output_format,
                        "sample_rate": sr,
                        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                        "duration_ms": int(len(wav_np) / max(sr, 1) * 1000),
                    }
                    yield _sse_event("chunk", payload)
            if (collect_full or STREAM_RETURN_LINK) and sample_rate:
                audio_bytes = None
                if collect_full and wav_parts:
                    full_wav = np.concatenate(wav_parts)
                    audio_bytes = _encode_audio(full_wav, sample_rate, req.output_format)
                payload = {
                    "format": req.output_format,
                    "sample_rate": sample_rate,
                    "duration_ms": int(total_frames / max(sample_rate, 1) * 1000),
                }
                if STREAM_RETURN_FULL:
                    payload["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
                if STREAM_RETURN_LINK:
                    if writer and writer_path:
                        writer.close()
                        final_path = _finalize_stream_file(writer_path, req.output_format)
                        file_id = _register_download_path(final_path, req.output_format)
                        payload["download_url"] = f"/v1/tts/download/{file_id}"
                    elif audio_bytes:
                        file_id = _register_download(audio_bytes, req.output_format)
                        payload["download_url"] = f"/v1/tts/download/{file_id}"
                yield _sse_event("final", payload)
            yield _sse_event("done", {"segments": total})
        except Exception as exc:
            if writer:
                try:
                    writer.close()
                except Exception:
                    pass
            if writer_path and writer_path.exists():
                try:
                    writer_path.unlink()
                except Exception:
                    pass
            logger.exception("custom stream_segments failed")
            yield _sse_event("error", {"detail": _exc_detail(exc)})
            return

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


def _stream_design_segments(req: VoiceDesignRequest) -> StreamingResponse:
    segments = _segment_text(req.text)
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    def event_stream():
        total = len(segments)
        if total == 0:
            yield _sse_event("error", {"detail": "Empty text"})
            return
        try:
            yield _sse_event("meta", {"segments": total, "format": req.output_format})
            wav_parts: list[np.ndarray] = []
            sample_rate: Optional[int] = None
            total_frames = 0
            writer = None
            writer_path: Optional[Path] = None
            collect_full = STREAM_RETURN_FULL
            for start in range(0, total, STREAM_BATCH_SIZE):
                batch_segments = segments[start : start + STREAM_BATCH_SIZE]
                seg_req = VoiceDesignRequest(
                    text=batch_segments[0],
                    prompt_text=req.prompt_text,
                    speed=req.speed,
                    output_format=req.output_format,
                    extra_params=req.extra_params,
                )
                future = STREAM_EXECUTOR.submit(
                    _run_voice_design_batch,
                    seg_req,
                    batch_segments,
                    True,
                )
                while True:
                    try:
                        wavs, sr = future.result(timeout=STREAM_KEEPALIVE_SECONDS)
                        break
                    except concurrent.futures.TimeoutError:
                        yield _sse_event(
                            "keepalive",
                            {"index": start + 1, "total": total},
                        )
                sample_rate = sr
                for offset, segment in enumerate(batch_segments):
                    seg_index = start + offset + 1
                    wav_np = _to_wav_array(wavs[offset])
                    if TRIM_SILENCE and (seg_index == 1 or seg_index == total):
                        wav_np = _trim_silence(wav_np, sr)
                    if collect_full:
                        wav_parts.append(wav_np)
                    if STREAM_RETURN_LINK:
                        if writer is None:
                            writer, writer_path = _open_stream_writer(sr, wav_np)
                        writer.write(wav_np)
                    audio_bytes = _encode_audio(wav_np, sr, req.output_format)
                    total_frames += len(wav_np)
                    payload = {
                        "index": seg_index,
                        "total": total,
                        "text": segment,
                        "format": req.output_format,
                        "sample_rate": sr,
                        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                        "duration_ms": int(len(wav_np) / max(sr, 1) * 1000),
                    }
                    yield _sse_event("chunk", payload)
            if (collect_full or STREAM_RETURN_LINK) and sample_rate:
                audio_bytes = None
                if collect_full and wav_parts:
                    full_wav = np.concatenate(wav_parts)
                    audio_bytes = _encode_audio(full_wav, sample_rate, req.output_format)
                payload = {
                    "format": req.output_format,
                    "sample_rate": sample_rate,
                    "duration_ms": int(total_frames / max(sample_rate, 1) * 1000),
                }
                if STREAM_RETURN_FULL:
                    payload["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
                if STREAM_RETURN_LINK:
                    if writer and writer_path:
                        writer.close()
                        final_path = _finalize_stream_file(writer_path, req.output_format)
                        file_id = _register_download_path(final_path, req.output_format)
                        payload["download_url"] = f"/v1/tts/download/{file_id}"
                    elif audio_bytes:
                        file_id = _register_download(audio_bytes, req.output_format)
                        payload["download_url"] = f"/v1/tts/download/{file_id}"
                yield _sse_event("final", payload)
            yield _sse_event("done", {"segments": total})
        except Exception as exc:
            if writer:
                try:
                    writer.close()
                except Exception:
                    pass
            if writer_path and writer_path.exists():
                try:
                    writer_path.unlink()
                except Exception:
                    pass
            logger.exception("design stream_segments failed")
            yield _sse_event("error", {"detail": _exc_detail(exc)})
            return

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


def _stream_clone_segments(
    text: str,
    ref_audio: UploadFile,
    ref_text: Optional[str],
    language: Optional[str],
    speed: float,
    x_vector_only_mode: bool,
    output_format: str,
) -> StreamingResponse:
    segments = _segment_text(text)
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    def event_stream():
        total = len(segments)
        if total == 0:
            yield _sse_event("error", {"detail": "Empty text"})
            return
        try:
            yield _sse_event("meta", {"segments": total, "format": output_format})
            wav_parts: list[np.ndarray] = []
            sample_rate: Optional[int] = None
            total_frames = 0
            writer = None
            writer_path: Optional[Path] = None
            collect_full = STREAM_RETURN_FULL
            with tempfile.TemporaryDirectory() as tmpdir:
                suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
                ref_path = Path(tmpdir) / f"ref{suffix}"
                with ref_path.open("wb") as f:
                    f.write(ref_audio.file.read())
                for idx, segment in enumerate(segments, start=1):
                    future = STREAM_EXECUTOR.submit(
                        _run_voice_clone_from_path,
                        segment,
                        ref_path,
                        ref_text,
                        language,
                        speed,
                        x_vector_only_mode,
                        True,
                    )
                    while True:
                        try:
                            wav, sr = future.result(timeout=STREAM_KEEPALIVE_SECONDS)
                            break
                        except concurrent.futures.TimeoutError:
                            yield _sse_event("keepalive", {"index": idx, "total": total})
                    wav_np = _to_wav_array(wav)
                    if TRIM_SILENCE and (idx == 1 or idx == total):
                        wav_np = _trim_silence(wav_np, sr)
                    sample_rate = sr
                    if collect_full:
                        wav_parts.append(wav_np)
                    if STREAM_RETURN_LINK:
                        if writer is None:
                            writer, writer_path = _open_stream_writer(sr, wav_np)
                        writer.write(wav_np)
                    audio_bytes = _encode_audio(wav_np, sr, output_format)
                    total_frames += len(wav_np)
                    payload = {
                        "index": idx,
                        "total": total,
                        "text": segment,
                        "format": output_format,
                        "sample_rate": sr,
                        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                        "duration_ms": int(len(wav_np) / max(sr, 1) * 1000),
                    }
                    yield _sse_event("chunk", payload)
            if (collect_full or STREAM_RETURN_LINK) and sample_rate:
                audio_bytes = None
                if collect_full and wav_parts:
                    full_wav = np.concatenate(wav_parts)
                    audio_bytes = _encode_audio(full_wav, sample_rate, output_format)
                payload = {
                    "format": output_format,
                    "sample_rate": sample_rate,
                    "duration_ms": int(total_frames / max(sample_rate, 1) * 1000),
                }
                if STREAM_RETURN_FULL:
                    payload["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
                if STREAM_RETURN_LINK:
                    if writer and writer_path:
                        writer.close()
                        final_path = _finalize_stream_file(writer_path, output_format)
                        file_id = _register_download_path(final_path, output_format)
                        payload["download_url"] = f"/v1/tts/download/{file_id}"
                    elif audio_bytes:
                        file_id = _register_download(audio_bytes, output_format)
                        payload["download_url"] = f"/v1/tts/download/{file_id}"
                yield _sse_event("final", payload)
            yield _sse_event("done", {"segments": total})
        except Exception as exc:
            if writer:
                try:
                    writer.close()
                except Exception:
                    pass
            if writer_path and writer_path.exists():
                try:
                    writer_path.unlink()
                except Exception:
                    pass
            logger.exception("clone stream_segments failed")
            yield _sse_event("error", {"detail": _exc_detail(exc)})
            return

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


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


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    if not _ffmpeg_available():
        raise HTTPException(status_code=500, detail="ffmpeg is not available")
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
import time
import uuid
