import json
import os
from typing import Optional

import requests

# Cloudflare configuration
_CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
_CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "@cf/meta/whisper")
_TRANSCRIBE_TIMEOUT_SECONDS = float(os.getenv("TRANSCRIBE_TIMEOUT_SECONDS", "60"))


def transcribe_audio_file(
    audio_path: str,
    beam_size: int = 3,
    vad_filter: bool = True,
    language_hint: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> dict:
    """
    Transcribes audio using Cloudflare Workers AI (Whisper).
    Acts as a drop-in replacement for the local faster-whisper implementation.
    """
    timeout = _TRANSCRIBE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds

    if not _CLOUDFLARE_ACCOUNT_ID or not _CLOUDFLARE_API_TOKEN:
        raise RuntimeError(
            "Cloudflare credentials not configured. "
            "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment variables."
        )

    url = f"https://api.cloudflare.com/client/v4/accounts/{_CLOUDFLARE_ACCOUNT_ID}/ai/run"

    headers = {
        "Authorization": f"Bearer {_CLOUDFLARE_API_TOKEN}",
        "cf-aig-gateway-id": "default",
        "Content-Type": "application/json",
    }

    # Read the audio file and convert to base64 for JSON transmission
    import base64
    with open(audio_path, "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

    payload = {
        "model": _WHISPER_MODEL,
        "input": {
            "audio": audio_data,
        },
    }
    
    if language_hint:
        payload["input"]["language"] = language_hint

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()

    # Cloudflare returns: {"result": {"text": "...", "language": "..."}, "success": true}
    if not result.get("success"):
        raise RuntimeError(
            f"Cloudflare transcription failed: {result.get('errors', 'Unknown error')}"
        )

    whisper_result = result.get("result", {})
    return {
        "language": whisper_result.get("language", "unknown"),
        "language_probability": whisper_result.get("language_probability", 0.0),
        "text": whisper_result.get("text", "").strip(),
        "segments": [],  # Cloudflare Whisper doesn't return segments by default
    }

