import concurrent.futures
import importlib
import os
import threading
from typing import Optional

from faster_whisper import WhisperModel

_MODEL: Optional[WhisperModel] = None
_MODEL_LOCK = threading.Lock()

os.environ["HF_HOME"] = r"D:\whisper\huggingFaceCache"
_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
_MODEL_ROOT = os.getenv("WHISPER_MODEL_DIR", ".whisperModel")
_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "10"))
_PREFERRED_DEVICE = os.getenv("WHISPER_DEVICE", "cuda").strip().lower()
_TRANSCRIBE_TIMEOUT_SECONDS = float(os.getenv("TRANSCRIBE_TIMEOUT_SECONDS", "60"))


def _cuda_available() -> bool:
    try:
        torch = importlib.import_module("torch")
        return bool(torch.cuda.is_available())  # type: ignore[union-attr]
    except Exception:
        return False


def _device_candidates() -> list[str]:
    if _PREFERRED_DEVICE == "cuda":
        return ["cuda", "cpu"]
    if _PREFERRED_DEVICE == "cpu":
        return ["cpu"]
    return ["cuda", "cpu"] if _cuda_available() else ["cpu"]


def _compute_type_for_device(device: str) -> str:
    return "int8_float32" if device == "cuda" else "int8"


def _build_model() -> WhisperModel:
    last_error: Optional[Exception] = None
    for device in _device_candidates():
        try:
            return WhisperModel(
                _MODEL_SIZE,
                device=device,
                compute_type=_compute_type_for_device(device),
                cpu_threads=_CPU_THREADS,
                download_root=_MODEL_ROOT,
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to initialize WhisperModel: {last_error}")


def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = _build_model()
    return _MODEL


def _run_transcription(
    audio_path: str,
    beam_size: int,
    vad_filter: bool,
    language_hint: Optional[str],
) -> dict:
    model = _get_model()

    transcribe_kwargs = {
        "beam_size": beam_size,
        "vad_filter": vad_filter,
    }
    if language_hint:
        transcribe_kwargs["language"] = language_hint

    segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    segment_items = []
    text_parts = []
    for s in segments:
        text = s.text.strip()
        segment_items.append(
            {
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": text,
            }
        )
        text_parts.append(text)

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "text": " ".join(part for part in text_parts if part).strip(),
        "segments": segment_items,
    }


def transcribe_audio_file(
    audio_path: str,
    beam_size: int = 3,
    vad_filter: bool = True,
    language_hint: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> dict:
    timeout = _TRANSCRIBE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _run_transcription,
            audio_path,
            beam_size,
            vad_filter,
            language_hint,
        )
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(f"transcription_timeout_after_{timeout}s") from exc