"""Local CrossEncoder reranker with a safe fallback when the model is unavailable."""
from __future__ import annotations

import logging
import threading
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None, enabled: bool | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self.enabled = settings.reranker_enabled if enabled is None else enabled
        self._model: Any = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.enabled and not self._failed

    def _load(self) -> Any:
        if self._model is None and not self._failed:
            with self._lock:
                if self._model is None and not self._failed:
                    try:
                        from sentence_transformers import CrossEncoder

                        self._model = CrossEncoder(self.model_name)
                    except Exception as exc:
                        logger.warning("Reranker unavailable (%s). Falling back to fusion order.", exc)
                        self._failed = True
        return self._model

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or settings.rerank_top_k
        if not chunks:
            return []
        model = self._load() if self.enabled else None
        if model is None:
            return chunks[:top_k]
        scores = model.predict([(query, c["text"]) for c in chunks], show_progress_bar=False)
        ranked = []
        for chunk, score in zip(chunks, scores):
            ranked.append({**chunk, "rerank_score": round(float(score), 4)})
        ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
