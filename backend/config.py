"""Central configuration loaded from environment / .env (no secrets in code)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _path(name: str, default: str) -> Path:
    raw = os.getenv(name) or default
    p = Path(raw)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    lm_base_url: str = (os.getenv("LM_STUDIO_BASE_URL") or "http://localhost:1234/v1").rstrip("/")
    lm_model: str = os.getenv("LM_STUDIO_MODEL") or "qwen2.5-7b-instruct-1m"
    lm_api_key: str = os.getenv("LM_STUDIO_API_KEY") or "lm-studio"
    lm_timeout: float = _float("LM_STUDIO_TIMEOUT", 180.0)

    embedding_model: str = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = os.getenv("RERANKER_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = _bool("RERANKER_ENABLED", True)

    chroma_path: Path = _path("CHROMA_PATH", "./chroma_db")
    upload_path: Path = _path("UPLOAD_PATH", "./uploads")
    eval_path: Path = _path("EVAL_PATH", "./evaluation_results")
    collection_name: str = os.getenv("CHROMA_COLLECTION") or "documents"

    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)
    top_k: int = _int("TOP_K", 8)
    candidate_k: int = _int("CANDIDATE_K", 20)
    rerank_top_k: int = _int("RERANK_TOP_K", 5)
    min_similarity: float = _float("MIN_SIMILARITY", 0.20)
    history_window: int = _int("HISTORY_WINDOW", 6)

    max_file_size_mb: int = _int("MAX_FILE_SIZE_MB", 50)
    debug: bool = _bool("DEBUG", True)
    database_url: str = os.getenv("DATABASE_URL") or ""

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()
