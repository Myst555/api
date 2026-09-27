import asyncio
import hashlib
import importlib
import logging
import os
import re
import tempfile

from firebase_service import send_push_notification
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Any
from pronunciation import analyze_pronunciation as analyze_pronunciation_text
from transcribe import transcribe_audio_file

_EDGE_TTS_MODULE: Any = None

app = FastAPI()
_LOG = logging.getLogger(__name__)

_MAX_CONCURRENT_INFERENCES = max(1, int(os.getenv("MAX_CONCURRENT_INFERENCES", "2")))
_MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))
_INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "90"))
_TRANSCRIBE_TIMEOUT_SECONDS = float(os.getenv("TRANSCRIBE_TIMEOUT_SECONDS", "60"))
_UPLOAD_CHUNK_BYTES = int(os.getenv("UPLOAD_CHUNK_BYTES", str(1024 * 1024)))
_MAX_TTS_TEXT_LENGTH = int(os.getenv("MAX_TTS_TEXT_LENGTH", "1200"))
_EDGE_TTS_VOICE_DEFAULT = os.getenv("EDGE_TTS_VOICE", "en-PH-RosaNeural")
_EDGE_TTS_VOICE_EN_PH = os.getenv("EDGE_TTS_VOICE_EN_PH", _EDGE_TTS_VOICE_DEFAULT)
_EDGE_TTS_VOICE_EN = os.getenv("EDGE_TTS_VOICE_EN", _EDGE_TTS_VOICE_DEFAULT)
_EDGE_TTS_VOICE_FIL = os.getenv("EDGE_TTS_VOICE_FIL", "fil-PH-AngeloNeural")
_EDGE_TTS_RATE = os.getenv("EDGE_TTS_RATE", "+0%")
_EDGE_TTS_PITCH = os.getenv("EDGE_TTS_PITCH", "+0Hz")
_TTS_CACHE_DIR = os.path.join(tempfile.gettempdir(), "readsmart_tts_cache")

_INFERENCE_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_INFERENCES)
_LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2})?$")


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "en"
    voice: str | None = None


def _run_inference(temp_path: str, reference_text: str, language_hint: str) -> tuple[str, dict]:
    result = transcribe_audio_file(
        temp_path,
        beam_size=1,
        vad_filter=True,
        language_hint=language_hint or None,
        timeout_seconds=_TRANSCRIBE_TIMEOUT_SECONDS,
    )
    recognized_text = result.get("text", "")
    analysis = analyze_pronunciation_text(reference_text, recognized_text, temp_path)
    return recognized_text, analysis


def _normalize_language_hint(language: str) -> str:
    candidate = (language or "").strip().lower()
    if not candidate:
        return ""
    if _LANGUAGE_PATTERN.fullmatch(candidate):
        return candidate
    raise HTTPException(status_code=422, detail="Invalid language format")


async def _write_upload_to_temp_file(audio: UploadFile) -> tuple[str, int]:
    suffix = os.path.splitext(audio.filename or "upload.wav")[1] or ".wav"
    size_bytes = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        while True:
            chunk = await audio.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > _MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds {_MAX_UPLOAD_SIZE_BYTES} bytes",
                )
            temp_file.write(chunk)

    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return temp_path, size_bytes


async def _cleanup_temp_file(path: str) -> None:
    if not path:
        return
    for _ in range(3):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            await asyncio.sleep(0.1)
        except Exception as exc:
            _LOG.warning("temp_file_cleanup_failed", extra={"path": path, "error": str(exc)})
            return


def _cleanup_temp_file_sync(path: str) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as exc:  # pragma: no cover - best effort cleanup
        _LOG.warning("temp_file_cleanup_failed", extra={"path": path, "error": str(exc)})


def _resolve_edge_voice(language: str, voice: str | None = None) -> str:
    if voice and voice.strip():
        return voice.strip()
    normalized = (language or "").strip().lower()
    if normalized in {"en-ph", "en_ph"}:
        return _EDGE_TTS_VOICE_EN_PH
    if normalized.startswith("en"):
        return _EDGE_TTS_VOICE_EN
    if normalized.startswith("fil") or normalized.startswith("tl"):
        return _EDGE_TTS_VOICE_FIL
    return _EDGE_TTS_VOICE_DEFAULT


def _ensure_tts_cache_dir() -> None:
    os.makedirs(_TTS_CACHE_DIR, exist_ok=True)


def _tts_cache_key(text: str, language: str, voice: str | None = None) -> str:
    resolved_voice = _resolve_edge_voice(language, voice)
    payload = (
        f"{resolved_voice}|{_EDGE_TTS_RATE}|{_EDGE_TTS_PITCH}|{text}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tts_cache_path(cache_key: str) -> str:
    return os.path.join(_TTS_CACHE_DIR, f"{cache_key}.mp3")


def _normalize_tts_text(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"[\t\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _ensure_edge_tts_installed() -> None:
    _get_edge_tts_module()


def _get_edge_tts_module() -> Any:
    global _EDGE_TTS_MODULE
    if _EDGE_TTS_MODULE is not None:
        return _EDGE_TTS_MODULE
    try:
        _EDGE_TTS_MODULE = importlib.import_module("edge_tts")
        return _EDGE_TTS_MODULE
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Edge TTS is unavailable. Install edge-tts.",
        ) from exc


async def _synthesize_to_file(text: str, language: str, voice: str | None = None) -> str:
    _ensure_edge_tts_installed()
    _ensure_tts_cache_dir()

    cache_key = _tts_cache_key(text, language, voice)
    cached_path = _tts_cache_path(cache_key)
    if os.path.exists(cached_path):
        return cached_path

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        output_path = temp_file.name

    try:
        edge_tts_module = _get_edge_tts_module()
        communicate = edge_tts_module.Communicate(
            text=text,
            voice=_resolve_edge_voice(language, voice),
            rate=_EDGE_TTS_RATE,
            pitch=_EDGE_TTS_PITCH,
        )
        await communicate.save(output_path)
        os.replace(output_path, cached_path)
        return cached_path
    finally:
        _cleanup_temp_file_sync(output_path)


@app.get("/")
async def root():
    return {"message": "Api Working!"}


@app.post("/analyze-pronunciation")
async def analyze_pronunciation(
    audio: UploadFile = File(...),
    reference_text: str = Form(""),
    language: str = Form("en"),
):
    language_hint = _normalize_language_hint(language)

    temp_path = ""
    size_bytes = 0
    recognized_text = ""
    analysis = {}
    try:
        temp_path, size_bytes = await _write_upload_to_temp_file(audio)

        async with _INFERENCE_SEMAPHORE:
            recognized_text, analysis = await asyncio.wait_for(
                run_in_threadpool(
                    _run_inference,
                    temp_path,
                    reference_text,
                    language_hint,
                ),
                timeout=_INFERENCE_TIMEOUT_SECONDS,
            )
    except HTTPException:
        raise
    except asyncio.TimeoutError as exc:
        _LOG.warning("inference_timeout", extra={"filename": audio.filename})
        raise HTTPException(status_code=504, detail="Inference timed out") from exc
    except Exception as exc:
        _LOG.exception("inference_failed", extra={"filename": audio.filename})
        raise HTTPException(status_code=500, detail="Transcription failed") from exc
    finally:
        await _cleanup_temp_file(temp_path)

    return {
        "filename": audio.filename,
        "content_type": audio.content_type,
        "size_bytes": size_bytes,
        "status": "transcribed",
        "language": language,
        "recognized_text": recognized_text,
        "analysis": analysis,
    }


@app.get("/tts-voices")
async def get_tts_voices():
    return [
        {"id": "en-PH-RosaNeural", "name": "Rosa", "accent": "Philippines", "gender": "Female", "locale": "en-PH"},
        {"id": "en-PH-JamesNeural", "name": "James", "accent": "Philippines", "gender": "Male", "locale": "en-PH"},
        {"id": "en-US-AvaNeural", "name": "Ava", "accent": "United States", "gender": "Female", "locale": "en-US"},
        {"id": "en-US-AndrewNeural", "name": "Andrew", "accent": "United States", "gender": "Male", "locale": "en-US"},
        {"id": "en-US-JennyNeural", "name": "Jenny", "accent": "United States", "gender": "Female", "locale": "en-US"},
        {"id": "en-US-GuyNeural", "name": "Guy", "accent": "United States", "gender": "Male", "locale": "en-US"},
        {"id": "en-GB-SoniaNeural", "name": "Sonia", "accent": "United Kingdom", "gender": "Female", "locale": "en-GB"},
        {"id": "en-GB-RyanNeural", "name": "Ryan", "accent": "United Kingdom", "gender": "Male", "locale": "en-GB"},
        {"id": "en-AU-NatashaNeural", "name": "Natasha", "accent": "Australia", "gender": "Female", "locale": "en-AU"},
        {"id": "en-AU-WilliamNeural", "name": "William", "accent": "Australia", "gender": "Male", "locale": "en-AU"},
        {"id": "fil-PH-AngeloNeural", "name": "Angelo", "accent": "Philippines (Tagalog)", "gender": "Male", "locale": "fil-PH"},
        {"id": "fil-PH-BlessicaNeural", "name": "Blessica", "accent": "Philippines (Tagalog)", "gender": "Female", "locale": "fil-PH"},
    ]


@app.post("/synthesize-speech")
async def synthesize_speech(payload: SynthesizeRequest):
    text = _normalize_tts_text(payload.text)
    language_hint = _normalize_language_hint(payload.language)
    if not text:
        raise HTTPException(status_code=422, detail="Text is required")

    if len(text) > _MAX_TTS_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds max length of {_MAX_TTS_TEXT_LENGTH} characters",
        )

    try:
        async with _INFERENCE_SEMAPHORE:
            audio_path = await asyncio.wait_for(
                _synthesize_to_file(text, language_hint, payload.voice),
                timeout=_INFERENCE_TIMEOUT_SECONDS,
            )
    except HTTPException:
        raise
    except asyncio.TimeoutError as exc:
        _LOG.warning("tts_timeout")
        raise HTTPException(status_code=504, detail="Speech synthesis timed out") from exc
    except Exception as exc:
        _LOG.exception("tts_failed")
        raise HTTPException(status_code=500, detail="Speech synthesis failed") from exc

    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename="reference.mp3",
    )

class NotificationRequest(BaseModel):
    token: str
    title: str
    body: str
    data: dict[str, str] = Field(default_factory=dict)


@app.post("/test-notification")
async def test_notification(request: NotificationRequest):
    try:
        message_id = send_push_notification(
            token=request.token,
            title=request.title,
            body=request.body,
            data=request.data,
        )

        return {
            "success": True,
            "message_id": message_id,
        }

    except Exception as exc:
        _LOG.exception("notification_failed")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
