import mimetypes
import os
from typing import Optional

import requests


# Cloudflare configuration
_CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
_CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "@cf/openai/whisper")
_TRANSCRIBE_TIMEOUT_SECONDS = float(
    os.getenv("TRANSCRIBE_TIMEOUT_SECONDS", "60")
)


def transcribe_audio_file(
    audio_path: str,
    beam_size: int = 3,
    vad_filter: bool = True,
    language_hint: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> dict:
    """
    Transcribe audio using Cloudflare Workers AI Whisper.

    The function keeps the same interface as the previous
    faster-whisper implementation so api.py does not need
    to change.
    """

    timeout = (
        _TRANSCRIBE_TIMEOUT_SECONDS
        if timeout_seconds is None
        else timeout_seconds
    )

    if not _CLOUDFLARE_ACCOUNT_ID:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID is not configured."
        )

    if not _CLOUDFLARE_API_TOKEN:
        raise RuntimeError(
            "CLOUDFLARE_API_TOKEN is not configured."
        )

    # Cloudflare expects the model name in the URL.
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{_CLOUDFLARE_ACCOUNT_ID}/ai/run/{_WHISPER_MODEL}"
    )

    content_type = mimetypes.guess_type(audio_path)[0]

    if not content_type:
        content_type = "application/octet-stream"

    headers = {
        "Authorization": f"Bearer {_CLOUDFLARE_API_TOKEN}",
        "Content-Type": content_type,
    }

    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()

    response = requests.post(
        url,
        headers=headers,
        data=audio_data,
        timeout=timeout,
    )

    # Give us Cloudflare's actual error instead of only
    # "400 Client Error".
    if not response.ok:
        try:
            error_data = response.json()
        except ValueError:
            error_data = response.text

        raise RuntimeError(
            f"Cloudflare transcription failed "
            f"(HTTP {response.status_code}): {error_data}"
        )

    result = response.json()

    if not result.get("success", True):
        raise RuntimeError(
            f"Cloudflare transcription failed: "
            f"{result.get('errors', 'Unknown error')}"
        )

    whisper_result = result.get("result", {})

    text = whisper_result.get("text", "").strip()

    words = whisper_result.get("words", [])

    return {
        "language": "en" if language_hint == "en" else "unknown",
        "language_probability": 0.0,
        "text": text,
        "segments": [],
        "words": words,
        "word_count": whisper_result.get("word_count", len(words)),
        "vtt": whisper_result.get("vtt", ""),
    }
