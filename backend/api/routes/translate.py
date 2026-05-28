"""
Translation REST API routes.

POST /translate/text   — translate typed text to ISL animation
POST /translate/voice  — upload audio file, transcribe, then translate
GET  /translation/history — fetch history
"""

from __future__ import annotations
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.responses import JSONResponse
from loguru import logger

from models.schemas import (
    TranslateTextRequest,
    TranslationResult,
    TranslationHistoryItem,
    NLPBreakdown,
    TokenAnalysis,
)
from services import nlp_engine, animation_engine, whisper_engine, cache_service
from config import settings

router = APIRouter(prefix="/translate", tags=["Translation"])


# ── POST /translate/text ───────────────────────────────────────────────────────

@router.post("/text", response_model=TranslationResult)
async def translate_text(req: TranslateTextRequest):
    t_start = time.perf_counter()
    cache_key = f"text:{req.text.strip().lower()}:{req.isl_grammar}"

    # Check cache
    cached = cache_service.cache_get(cache_key)
    if cached:
        logger.info(f"Cache hit for: {req.text[:50]}")
        return JSONResponse(content=cached)

    try:
        # NLP processing
        nlp_result = nlp_engine.process_text(req.text, apply_isl_grammar=req.isl_grammar)

        # Animation generation
        anim = animation_engine.generate_animation(
            gloss_sequence=nlp_result["gloss_sequence"],
            fps=settings.animation_fps,
            hold_frames=20,
            interp_frames=settings.interpolation_frames,
            idle_animation=settings.idle_animation_enabled,
            source_text=req.text,
        )

        processing_ms = (time.perf_counter() - t_start) * 1000

        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "original_text": req.text,
            "simplified_text": nlp_result["simplified"],
            "gloss_sequence": nlp_result["gloss_sequence"],
            "nlp_breakdown": {
                "original": nlp_result["original"],
                "simplified": nlp_result["simplified"],
                "gloss_sequence": nlp_result["gloss_sequence"],
                "tokens": nlp_result["tokens"],
                "sentence_structure": nlp_result["sentence_structure"],
            },
            "animation": anim,
            "confidence": _calc_confidence(nlp_result),
            "processing_time_ms": round(processing_ms, 2),
        }

        cache_service.cache_set(cache_key, result)
        cache_service.add_to_history(result)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /translate/voice ──────────────────────────────────────────────────────

@router.post("/voice", response_model=TranslationResult)
async def translate_voice(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/ogg/webm)"),
    language: str = Form(default="en"),
    isl_grammar: bool = Form(default=True),
):
    t_start = time.perf_counter()

    try:
        audio_bytes = await audio.read()
        logger.info(f"Received audio: {audio.filename}, {len(audio_bytes)/1024:.1f}KB")

        # Transcribe
        whisper_result = await whisper_engine.transcribe_file(
            audio_bytes,
            language=language,
            model_name=settings.whisper_model,
            device=settings.whisper_device,
        )
        transcribed_text = whisper_result["text"]
        logger.info(f"Transcribed: '{transcribed_text}'")

        if not transcribed_text.strip():
            raise HTTPException(status_code=422, detail="Could not transcribe audio — no speech detected")

        # NLP
        nlp_result = nlp_engine.process_text(transcribed_text, apply_isl_grammar=isl_grammar)

        # Animation
        anim = animation_engine.generate_animation(
            gloss_sequence=nlp_result["gloss_sequence"],
            fps=settings.animation_fps,
            hold_frames=20,
            interp_frames=settings.interpolation_frames,
            source_text=transcribed_text,
        )

        processing_ms = (time.perf_counter() - t_start) * 1000

        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "original_text": transcribed_text,
            "transcribed_text": transcribed_text,
            "simplified_text": nlp_result["simplified"],
            "gloss_sequence": nlp_result["gloss_sequence"],
            "nlp_breakdown": {
                "original": nlp_result["original"],
                "simplified": nlp_result["simplified"],
                "gloss_sequence": nlp_result["gloss_sequence"],
                "tokens": nlp_result["tokens"],
                "sentence_structure": nlp_result["sentence_structure"],
            },
            "animation": anim,
            "confidence": _calc_confidence(nlp_result),
            "processing_time_ms": round(processing_ms, 2),
        }

        cache_service.add_to_history(result)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /translation/history ───────────────────────────────────────────────────

@router.get("/history", response_model=list[TranslationHistoryItem])
async def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return cache_service.get_history(limit=limit, offset=offset)


# ── DELETE /translation/history ────────────────────────────────────────────────

@router.delete("/history")
async def clear_history():
    cache_service.clear_history()
    return {"message": "History cleared"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _calc_confidence(nlp_result: dict) -> float:
    """Simple heuristic confidence: ratio of kept tokens."""
    tokens = nlp_result.get("tokens", [])
    if not tokens:
        return 0.5
    kept = sum(1 for t in tokens if t["kept_in_isl"])
    total = sum(1 for t in tokens if t["token"].isalpha())
    if total == 0:
        return 0.5
    ratio = kept / total
    return round(min(0.95, max(0.3, ratio)), 3)
