"""SiGML translation routes — converts English text to ISL sign sequences."""

from __future__ import annotations

import json
import re
import string
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from config import settings
from services import whisper_engine

try:
    from groq import AsyncGroq
except ImportError:  # pragma: no cover - optional dependency
    AsyncGroq = None


router = APIRouter(prefix="/api/sigml", tags=["sigml"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/

WORDS_PATH = BASE_DIR / "words.txt"
CATALOG_PATH = BASE_DIR / "sigmlFiles.json"

# ISL stop-words — auxiliary/modal verbs that don't appear in sign language
STOP_WORDS = frozenset({
    "am", "are", "is", "was", "were", "be", "being", "been",
    "have", "has", "had", "does", "did", "do",
    "could", "should", "would", "can", "shall", "will",
    "may", "might", "must", "let",
    "a", "an", "the", "to",
})


# ── Data loading ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_valid_words() -> set[str]:
    """Load the set of words that have corresponding .sigml sign files."""
    if not WORDS_PATH.exists():
        logger.warning(f"words.txt not found at {WORDS_PATH}")
        return set()
    with WORDS_PATH.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


@lru_cache(maxsize=1)
def _load_catalog() -> list[dict]:
    """Load the sign file catalog from sigmlFiles.json."""
    if not CATALOG_PATH.exists():
        logger.warning(f"sigmlFiles.json not found at {CATALOG_PATH}")
        return []
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Translation logic ────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize input text: collapse whitespace, strip special chars."""
    normalized = " ".join((text or "").strip().split())
    return re.sub(r"[^\w\s.,'?!\-]", "", normalized)


def _tokenize_and_filter(text: str) -> list[str]:
    """Tokenize, lowercase, and remove stop words + punctuation."""
    tokens = []
    for word in text.split():
        # Strip surrounding punctuation
        cleaned = word.strip(string.punctuation).lower()
        if not cleaned:
            continue
        if cleaned in STOP_WORDS:
            continue
        tokens.append(cleaned)
    return tokens


def _format_sign_token(token: str) -> str:
    """Format for asset path: single char → UPPERCASE, words → lowercase."""
    return token.upper() if len(token) == 1 else token.lower()


def _expand_to_sign_sequence(tokens: list[str]) -> list[dict]:
    """Convert normalized tokens into sign sequence with asset paths."""
    valid_words = _load_valid_words()
    sequence = []
    idx = 1

    for token in tokens:
        token_lower = token.lower()
        if token_lower in valid_words:
            formatted = _format_sign_token(token_lower)
            # Determine the correct filename
            # Single chars use uppercase file (A.sigml), words use lowercase (hello.sigml)
            if len(token_lower) == 1:
                asset_name = token_lower.upper()
            else:
                asset_name = token_lower
            sequence.append({
                "id": idx,
                "value": formatted,
                "kind": "sign",
                "asset": f"/SignFiles/{asset_name}.sigml",
            })
            idx += 1
        else:
            # Fingerspell: break into individual characters
            for char in token:
                if char.isalnum():
                    display = char.upper()
                    sequence.append({
                        "id": idx,
                        "value": display,
                        "kind": "fingerspell",
                        "asset": f"/SignFiles/{display}.sigml",
                    })
                    idx += 1

    return sequence


def translate_text(text: str) -> dict:
    """Full translation pipeline: text → ISL sign sequence."""
    cleaned = _clean_text(text)
    if not cleaned:
        return {
            "input": text,
            "gloss": "",
            "tokenCount": 0,
            "sequence": [],
        }

    tokens = _tokenize_and_filter(cleaned)
    sequence = _expand_to_sign_sequence(tokens)
    gloss = " ".join(item["value"] for item in sequence)

    return {
        "input": cleaned,
        "gloss": gloss,
        "tokenCount": len(sequence),
        "sequence": sequence,
    }


# ── Pydantic models ──────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str


class LanguageTranslateRequest(BaseModel):
    text: str
    target_language: str
    source_language: str = "auto"


class LanguageTranslateResponse(BaseModel):
    input: str
    translatedText: str
    sourceLanguage: str
    targetLanguage: str


class VoiceSigmlResponse(BaseModel):
    input: str
    transcribedText: str
    detectedLanguage: str
    duration: float
    gloss: str
    tokenCount: int
    sequence: list[dict]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/translate")
async def api_translate(req: TranslateRequest):
    """Translate English text to ISL sign sequence."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Please provide text to translate.")

    result = translate_text(req.text)
    logger.info(f"SiGML translate: '{req.text[:50]}' → {result['tokenCount']} tokens")
    return result


@router.post("/voice", response_model=VoiceSigmlResponse)
async def api_voice_to_sigml(
    audio: UploadFile = File(..., description="Audio file such as wav, mp3, ogg, webm, or flac."),
    language: str = Form(default="en"),
):
    """Transcribe uploaded speech and translate the transcript to ISL sign sequence."""
    filename = audio.filename or "audio"
    suffix = Path(filename).suffix.lower()
    allowed_suffixes = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}

    if suffix and suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio type. Upload wav, mp3, m4a, ogg, webm, or flac.",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Please upload a non-empty audio file.")
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file is too large. Keep it below 25 MB.")

    try:
        whisper_result = await whisper_engine.transcribe_file(
            audio_bytes=audio_bytes,
            language=(language or "en").split("-")[0],
            model_name=settings.whisper_model,
            device=settings.whisper_device,
        )
    except Exception as exc:
        logger.error(f"Voice-to-SiGML transcription failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    transcribed_text = (whisper_result.get("text") or "").strip()
    if not transcribed_text:
        raise HTTPException(status_code=422, detail="Whisper could not detect clear speech in this audio.")

    result = translate_text(transcribed_text)
    duration = float(whisper_result.get("duration") or 0.0)
    logger.info(
        f"Voice SiGML translate: '{filename}' -> {result['tokenCount']} tokens "
        f"from {duration:.2f}s audio"
    )

    return VoiceSigmlResponse(
        input=result["input"],
        transcribedText=transcribed_text,
        detectedLanguage=str(whisper_result.get("language") or language or "en"),
        duration=duration,
        gloss=result["gloss"],
        tokenCount=result["tokenCount"],
        sequence=result["sequence"],
    )


@router.get("/catalog")
async def api_catalog():
    """Return the full catalog of available sign files."""
    catalog = _load_catalog()
    return {"count": len(catalog), "items": catalog}


@router.post("/translate-language", response_model=LanguageTranslateResponse)
async def api_translate_language(req: LanguageTranslateRequest):
    """Translate plain text into another language for accessibility output."""
    text = (req.text or "").strip()
    target_language = (req.target_language or "").strip()
    source_language = (req.source_language or "auto").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Please provide text to translate.")
    if not target_language:
        raise HTTPException(status_code=400, detail="Please provide a target language.")
    if AsyncGroq is None:
        raise HTTPException(status_code=503, detail="Groq SDK is not installed.")
    if not settings.groq_api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is required for language translation.")

    prompt = (
        "Translate the text into the requested target language for accessibility. "
        "Preserve the meaning, keep names unchanged, and output only the translated text.\n\n"
        f"Source language: {source_language}\n"
        f"Target language: {target_language}\n"
        f"Text: {text}"
    )

    try:
        client = AsyncGroq(api_key=settings.groq_api_key)
        completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise multilingual accessibility translator. Output only the translation.",
                },
                {"role": "user", "content": prompt},
            ],
            model=settings.groq_text_model,
            temperature=0.1,
            max_tokens=256,
        )
        translated = completion.choices[0].message.content.strip().strip('"')
        return LanguageTranslateResponse(
            input=text,
            translatedText=translated,
            sourceLanguage=source_language,
            targetLanguage=target_language,
        )
    except Exception as exc:
        logger.error(f"Language translation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/words")
async def api_words():
    """Return the list of known ISL sign words."""
    words = sorted(_load_valid_words())
    return {"count": len(words), "words": words}
