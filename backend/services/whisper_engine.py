"""
Whisper Speech-to-Text engine — wraps OpenAI Whisper for file and streaming transcription.
"""

from __future__ import annotations
import time
import tempfile
import os
from pathlib import Path
from typing import Optional, AsyncGenerator

from loguru import logger

_whisper_model = None
_model_name: str = ""


def _load_model(model_name: str = "base", device: str = "cpu"):
    """Lazy-load the Whisper model (called once on first use)."""
    global _whisper_model, _model_name
    if _whisper_model is None or _model_name != model_name:
        import whisper
        logger.info(f"Loading Whisper model '{model_name}' on {device}...")
        t0 = time.time()
        _whisper_model = whisper.load_model(model_name, device=device)
        _model_name = model_name
        logger.info(f"Whisper model loaded in {time.time()-t0:.2f}s")
    return _whisper_model


async def transcribe_file(
    audio_bytes: bytes,
    *,
    language: str = "en",
    model_name: str = "base",
    device: str = "cpu",
) -> dict:
    """
    Transcribe an uploaded audio file.

    Returns:
        {
            "text": str,
            "language": str,
            "segments": [...],
            "duration": float,
        }
    """
    import asyncio

    def _run():
        model = _load_model(model_name, device)
        # Write to a temp file — Whisper needs a filepath
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            result = model.transcribe(
                tmp_path,
                language=language if language != "auto" else None,
                fp16=False,
            )
            return {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "duration": result["segments"][-1]["end"] if result.get("segments") else 0.0,
            }
        finally:
            os.unlink(tmp_path)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


def get_whisper_status(model_name: str = "base") -> dict:
    """Return model load status without blocking."""
    try:
        import whisper
        available_models = whisper.available_models()
        return {
            "status": "ready" if _whisper_model is not None else "not_loaded",
            "model": model_name,
            "available_models": list(available_models),
        }
    except ImportError:
        return {"status": "unavailable", "model": model_name, "error": "whisper not installed"}
