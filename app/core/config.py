"""
Central application configuration.
All secrets/settings come from environment variables (loaded via python-dotenv).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "JournalistAI Transcription"
    DEBUG: bool = False

    STORAGE_ROOT: Path = Path("./storage")
    MAX_UPLOAD_BYTES: int = 2 * 1024 * 1024 * 1024
    MAX_DURATION_SECONDS: float = 3 * 3600

    DATABASE_URL: str = "sqlite:///./storage/db.sqlite3"

    TRANSCRIPTION_PROVIDER: str = "whisper"
    WHISPER_MODEL_SIZE: str = "large-v3"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 1
    WHISPER_CPU_THREADS: int = 0
    WHISPER_WORD_TIMESTAMPS: bool = False

    ANALYSIS_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-pro"

    HF_TOKEN: str = ""
    ENABLE_DIARIZATION: bool = False

    FFMPEG_BIN: str = "ffmpeg"
    FFPROBE_BIN: str = "ffprobe"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
