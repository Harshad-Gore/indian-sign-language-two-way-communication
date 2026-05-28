"""
Application configuration — reads from .env file or environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ─────────────────────────────────────────────────────────────────
    app_name: str = "ISL Translation System"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Whisper ──────────────────────────────────────────────────────────────
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = "base"
    whisper_device: Literal["cpu", "cuda"] = "cpu"
    whisper_language: str = "en"

    # ── NLP ──────────────────────────────────────────────────────────────────
    spacy_model: str = "en_core_web_sm"

    # ── Animation ────────────────────────────────────────────────────────────
    animation_fps: int = 30
    interpolation_frames: int = 10        # frames between signs
    idle_animation_enabled: bool = True

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./isl_history.db"

    # ── Cache ────────────────────────────────────────────────────────────────
    cache_max_size: int = 256

    # ── Groq (optional) ─────────────────────────────────────────────────────
    groq_api_key: Optional[str] = None
    groq_text_model: str = "llama-3.3-70b-versatile"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ── Sign Language Translator ─────────────────────────────────────────────
    slt_dataset_dir: str = "./slt_data"

    # ── ISL Landmark Dataset (local) ─────────────────────────────────────────
    isl_landmarks_dir: str = "../isl/extracted_data"
    isl_dataset_fps: int = 15

    # ── ISL Motion Clips (from videos) ───────────────────────────────────────
    isl_motion_clips_dir: str = "../isl/motion_clips"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
