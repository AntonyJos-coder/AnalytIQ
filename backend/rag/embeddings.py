"""Local sentence-transformers embeddings (no external API)."""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any, Sequence

from backend.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """Lazy-loading wrapper around a SentenceTransformer model. Vectors are normalized."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: Any = None
        self._lock = threading.Lock()
        self.error: str = ""

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def is_ready(self) -> bool:
        try:
            self._get_model()
            self.error = ""
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("Embedding model unavailable: %s", self.error)
            return False

    def embed_query(self, text: str) -> list[float]:
        vec = self._get_model().encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._get_model().encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def embed_query(text: str) -> list[float]:
    return get_embedder().embed_query(text)


def embed_chunks(texts: Sequence[str]) -> list[list[float]]:
    return get_embedder().embed_documents(texts)
