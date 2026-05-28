"""
System status route.
GET /system/status — health check, model info, uptime
"""

import time
from fastapi import APIRouter
from models.schemas import SystemStatus, ModelStatus
from services import whisper_engine, cache_service
from services.sign_generator import get_available_signs
from config import settings

router = APIRouter(prefix="/system", tags=["System"])

_start_time = time.time()


@router.get("/status", response_model=SystemStatus)
async def get_status():
    uptime = time.time() - _start_time

    # Whisper status
    w_status = whisper_engine.get_whisper_status(settings.whisper_model)
    whisper_model_status = ModelStatus(
        name=f"Whisper ({settings.whisper_model})",
        status="ready" if w_status["status"] == "ready" else "not_loaded",
        version=settings.whisper_model,
        message=w_status.get("error"),
    )

    # NLP status
    try:
        import nltk
        nltk_ok = True
    except ImportError:
        nltk_ok = False

    nlp_status = ModelStatus(
        name="NLP Engine (NLTK + spaCy)",
        status="ready" if nltk_ok else "unavailable",
        version="3.8",
    )

    # SLT status
    try:
        import sign_language_translator
        slt_version = getattr(sign_language_translator, "__version__", "unknown")
        slt_status = ModelStatus(name="sign-language-translator", status="ready", version=slt_version)
    except ImportError:
        slt_status = ModelStatus(name="sign-language-translator", status="unavailable",
                                  message="Procedural fallback active")

    # Sign library
    sign_count = len(get_available_signs())
    signs_status = ModelStatus(
        name="ISL Pose Library",
        status="ready",
        version=f"{sign_count} signs",
    )

    cache_stats = cache_service.get_cache_stats()

    overall = "healthy"
    if not nltk_ok:
        overall = "degraded"

    return SystemStatus(
        status=overall,
        version=settings.app_version,
        uptime_seconds=round(uptime, 1),
        models=[whisper_model_status, nlp_status, slt_status, signs_status],
    )


@router.get("/signs")
async def list_signs():
    """List all known ISL gloss tokens."""
    return {"signs": get_available_signs(), "count": len(get_available_signs())}


@router.get("/cache")
async def cache_stats():
    return cache_service.get_cache_stats()
