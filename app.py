"""
Hugging Face Spaces entry point.

Mounts the existing FastAPI application (api.py) onto Gradio's
internal FastAPI server so all REST endpoints remain available
at their original paths (e.g. /analyze-pronunciation, /synthesize-speech).

Gradio serves a lightweight landing page at the root (/).
"""

import gradio as gr
from api import app as fastapi_app

# -- Gradio UI (landing page) ------------------------------------------------
with gr.Blocks(title="ReadSmart API") as demo:
    gr.Markdown("# 📖 ReadSmart API")
    gr.Markdown(
        "This space hosts the ReadSmart pronunciation analysis & TTS API.\n\n"
        "### Available Endpoints\n"
        "| Method | Path | Description |\n"
        "|--------|------|-------------|\n"
        "| `GET` | `/api/` | Health check |\n"
        "| `POST` | `/api/analyze-pronunciation` | Analyze pronunciation from audio |\n"
        "| `POST` | `/api/synthesize-speech` | Text-to-speech synthesis |\n"
        "| `GET` | `/api/tts-voices` | List available TTS voices |\n\n"
        "Append `/api/docs` to the Space URL for the interactive Swagger UI."
    )

# Mount the FastAPI app under /api so it doesn't conflict with Gradio's root
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
