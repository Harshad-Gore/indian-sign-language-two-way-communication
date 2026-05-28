"""
Animation generation route.
POST /animation/generate — generate animation from a raw gloss sequence
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from models.schemas import AnimationGenerateRequest, AnimationData
from services import animation_engine
from config import settings

router = APIRouter(prefix="/animation", tags=["Animation"])


@router.post("/generate", response_model=AnimationData)
async def generate_animation(req: AnimationGenerateRequest):
    if not req.gloss_sequence:
        raise HTTPException(status_code=422, detail="gloss_sequence must not be empty")
    try:
        result = animation_engine.generate_animation(
            gloss_sequence=req.gloss_sequence,
            fps=req.fps,
            hold_frames=20,
            interp_frames=req.interpolation_frames,
            idle_animation=settings.idle_animation_enabled,
            source_text=req.source_text,
        )
        return result
    except Exception as e:
        logger.error(f"Animation generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
