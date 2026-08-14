"""
Sign Language Recognition API — receives hand/pose landmarks, returns recognized signs.
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from loguru import logger
import numpy as np
from config import settings
try:
    from groq import Groq, AsyncGroq
except ImportError:
    Groq = None
    AsyncGroq = None

router = APIRouter(prefix="/api/recognize", tags=["Recognition"])

# ── Add ISL module to path ────────────────────────────────────────────────────
ISL_ROOT = Path(__file__).parent.parent.parent.parent / "isl"
if str(ISL_ROOT) not in sys.path:
    # Keep backend path precedence so uvicorn "main:app" still resolves backend/main.py.
    sys.path.append(str(ISL_ROOT))

# ── Lazy-loaded recognizer ────────────────────────────────────────────────────
_recognizer = None
_class_names: List[str] = []


def _get_recognizer():
    """Lazy-load the ML recognizer (loads model on first call)."""
    global _recognizer, _class_names
    if _recognizer is not None:
        return _recognizer

    model_path = ISL_ROOT / "models" / "best_model.pt"
    if not model_path.exists():
        raise RuntimeError(f"ISL model not found at {model_path}")

    try:
        from src.ml.recognizer import MLRecognizer
        _recognizer = MLRecognizer(
            model_path=str(model_path),
            window_size=30,
            predict_interval=1,   # predict every frame when called from API
            device="cpu",
        )
        _class_names = _recognizer.class_names
        logger.info(f"ISL recognizer loaded: {_recognizer.num_classes} classes, device={_recognizer.device}")
        return _recognizer
    except Exception as e:
        logger.error(f"Failed to load ISL recognizer: {e}")
        raise


# ── Request / Response models ─────────────────────────────────────────────────

class LandmarkFrame(BaseModel):
    """Single frame of landmarks from MediaPipe."""
    right_hand: Optional[List[List[float]]] = None   # (21, 3) or null
    left_hand: Optional[List[List[float]]] = None     # (21, 3) or null
    pose: Optional[List[List[float]]] = None           # (33, 4) or null


class RecognitionResult(BaseModel):
    sign: Optional[str] = None
    confidence: float = 0.0
    top_k: List[dict] = []
    frame_count: int = 0
    buffer_size: int = 0


class RecognizerStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    loaded: bool
    num_classes: int = 0
    class_names: List[str] = []
    model_path: str = ""
    window_size: int = 30
    confidence_threshold: float = 0.6


class SentenceCompletionRequest(BaseModel):
    words: List[str]


class SentenceCompletionResponse(BaseModel):
    sentence: str


class VisionRecognitionRequest(BaseModel):
    image_base64: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/frame", response_model=RecognitionResult)
async def recognize_frame(frame: LandmarkFrame):
    """
    Add a frame of landmarks and get the current prediction.
    Send landmarks every ~2 frames from the frontend (~15fps).
    """
    try:
        recognizer = _get_recognizer()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Recognizer not available: {e}")

    # Convert to numpy arrays
    right_hand = np.array(frame.right_hand, dtype=np.float32) if frame.right_hand else None
    left_hand = np.array(frame.left_hand, dtype=np.float32) if frame.left_hand else None
    pose = np.array(frame.pose, dtype=np.float32) if frame.pose else None

    # Validate shapes
    if right_hand is not None and right_hand.shape != (21, 3):
        right_hand = None
    if left_hand is not None and left_hand.shape != (21, 3):
        left_hand = None
    if pose is not None and pose.shape[0] != 33:
        pose = None

    # Add frame and predict
    recognizer.add_frame(right_hand, left_hand, pose)
    sign, confidence = recognizer.predict(force=True)

    top_k = [
        {"sign": name, "confidence": round(conf, 4)}
        for name, conf in recognizer.get_top_k(5)
    ]

    return RecognitionResult(
        sign=sign,
        confidence=round(confidence, 4),
        top_k=top_k,
        frame_count=recognizer.frame_count,
        buffer_size=len(recognizer.frame_buffer),
    )


@router.post("/reset")
async def reset_recognizer():
    """Clear the frame buffer and reset state."""
    try:
        recognizer = _get_recognizer()
        recognizer.reset()
        return {"status": "ok", "message": "Recognizer reset"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/status", response_model=RecognizerStatus)
async def recognizer_status():
    """Get recognizer model information."""
    try:
        recognizer = _get_recognizer()
        return RecognizerStatus(
            loaded=True,
            num_classes=recognizer.num_classes,
            class_names=recognizer.class_names,
            model_path=str(ISL_ROOT / "models" / "best_model.pt"),
            window_size=recognizer.window_size,
            confidence_threshold=recognizer.confidence_threshold,
        )
    except Exception:
        return RecognizerStatus(
            loaded=False,
            model_path=str(ISL_ROOT / "models" / "best_model.pt"),
        )


@router.get("/classes")
async def get_classes():
    """Get all recognized sign class names."""
    try:
        recognizer = _get_recognizer()
        return {
            "classes": recognizer.class_names,
            "count": recognizer.num_classes,
        }
    except Exception:
        # Fall back to class_names.json
        class_file = ISL_ROOT / "models" / "class_names.json"
        if class_file.exists():
            with open(class_file) as f:
                names = json.load(f)
            return {"classes": names, "count": len(names)}
        return {"classes": [], "count": 0}


@router.post("/complete-sentence", response_model=SentenceCompletionResponse)
async def complete_sentence(req: SentenceCompletionRequest):
    """Uses Groq LLM to convert a sequence of ISL glosses into a natural sentence."""
    if AsyncGroq is None:
        raise HTTPException(status_code=500, detail="Groq library not installed")
        
    api_key = settings.groq_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not set")
    
    prompt = f"Convert the following sequence of raw sign language words into a natural, grammatically correct conversational English sentence. Fix pronouns, verb tenses, and add missing prepositions. Do not add extra fabricated meaning. Output ONLY the resulting sentence string without quotes or explanations.\n\nWords: {' '.join(req.words)}"
    
    try:
        client = AsyncGroq(api_key=api_key)
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional Sign Language to English translator. Always output only the translated sentence."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=settings.groq_text_model,
            temperature=0.2,
            max_tokens=128
        )
        sentence = chat_completion.choices[0].message.content.strip()
        # Remove quotes if the LLM added them
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]
            
        return SentenceCompletionResponse(sentence=sentence)
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision", response_model=RecognitionResult)
async def recognize_vision(req: VisionRecognitionRequest):
    """Uses Groq Vision model to identify static ISL signs (A-Z, 0-9) from a base64 image."""
    if AsyncGroq is None:
        raise HTTPException(status_code=500, detail="Groq library not installed")
        
    api_key = settings.groq_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not set")
        
    try:
        client = AsyncGroq(api_key=api_key)
        
        # Ensure we have the correct base64 prefix stripped if present
        b64_data = req.image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
            
        prompt = (
            "You are an expert in Indian Sign Language (ISL). "
            "Analyze this image and identify the static sign being performed. "
            "It is highly likely to be a letter (A-Z) or a number (0-9). "
            "Respond with EXACTLY ONE word or character representing the sign. "
            "If there is no clear sign, respond with the exact word 'None'. "
            "Do not provide any other explanation or text."
        )

        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_data}",
                            },
                        },
                    ],
                }
            ],
            model=settings.groq_vision_model,
            temperature=0.1,
            max_tokens=10,
        )
        
        sign = chat_completion.choices[0].message.content.strip().replace('"', '')
        
        # Post-process
        if sign.lower() == "none" or len(sign) > 15:
            sign = None
        else:
            sign = sign.upper() # Standardize ISL glosses to uppercase
            
        return RecognitionResult(
            sign=sign,
            confidence=0.95, # High confidence for explicit vision trigger
            top_k=[],
            frame_count=0,
            buffer_size=0,
        )
        
    except Exception as e:
        logger.error(f"Groq Vision API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
