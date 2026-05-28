"""
Pydantic schemas for all API request/response models.
"""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ── Shared ────────────────────────────────────────────────────────────────────

class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float
    visibility: float = 1.0


class PoseFrame(BaseModel):
    """One frame of the skeleton animation — 75 landmark points."""
    frame_index: int
    timestamp_ms: float
    body: list[LandmarkPoint] = Field(default_factory=list)     # 33 body
    left_hand: list[LandmarkPoint] = Field(default_factory=list)  # 21 left
    right_hand: list[LandmarkPoint] = Field(default_factory=list) # 21 right


class AnimationData(BaseModel):
    """Complete animation: sequence of pose frames at a given FPS."""
    fps: int = 30
    total_frames: int
    duration_ms: float
    frames: list[PoseFrame]
    gloss_timeline: list[GlossTimestamp] = Field(default_factory=list)


class GlossTimestamp(BaseModel):
    gloss: str
    start_frame: int
    end_frame: int


# ── NLP Output ────────────────────────────────────────────────────────────────

class TokenAnalysis(BaseModel):
    token: str
    pos: str
    dep: str
    lemma: str
    is_stopword: bool
    kept_in_isl: bool


class NLPBreakdown(BaseModel):
    original: str
    simplified: str
    gloss_sequence: list[str]
    tokens: list[TokenAnalysis]
    sentence_structure: str


# ── Translation ───────────────────────────────────────────────────────────────

class TranslateTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = "en"
    isl_grammar: bool = True


class TranslateVoiceRequest(BaseModel):
    language: str = "en"
    isl_grammar: bool = True


class TranslationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    original_text: str
    transcribed_text: Optional[str] = None
    simplified_text: str
    gloss_sequence: list[str]
    nlp_breakdown: NLPBreakdown
    animation: AnimationData
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float


class TranslationHistoryItem(BaseModel):
    id: str
    timestamp: datetime
    original_text: str
    gloss_sequence: list[str]
    confidence: float


# ── Animation ─────────────────────────────────────────────────────────────────

class AnimationGenerateRequest(BaseModel):
    gloss_sequence: list[str]
    fps: int = Field(default=30, ge=10, le=60)
    interpolation_frames: int = Field(default=10, ge=3, le=30)
    source_text: Optional[str] = None


# ── System ────────────────────────────────────────────────────────────────────

class ModelStatus(BaseModel):
    name: str
    status: str   # "ready" | "loading" | "error" | "unavailable"
    version: Optional[str] = None
    message: Optional[str] = None


class SystemStatus(BaseModel):
    status: str   # "healthy" | "degraded" | "error"
    version: str
    uptime_seconds: float
    models: list[ModelStatus]


# ── WebSocket ─────────────────────────────────────────────────────────────────

class WSMessage(BaseModel):
    type: str
    payload: Any


class WSTranslationEvent(BaseModel):
    event: str
    data: Any
