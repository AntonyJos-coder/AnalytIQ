"""BM25 keyword retrieval over the chunks stored in Chroma (good for names, SKUs, codes)."""
from __future__ import annotations

import re
import threading
from typing import Any

from rank_bm25 import BM25Okapi

from backend.rag.retriever import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "are", "was",
    "were", "be", "been", "it", "its", "this", "that", "these", "those", "what", "which", "who",
    "why", "how", "when", "where", "did", "do", "does", "has", "have", "had", "at", "as", "from",
    "about", "say", "said", "says", "me", "my", "i", "you", "we", "they", "their", "than", "then",
}


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._chunks: list[dict[str, Any]] = []
        self._tokens: list[list[str]] = []
        self._index: BM25Okapi | None = None
        self._built_for = -1
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        self._built_for = -1

    def _ensure_index(self) -> None:
        current = self.store.count()
        if current == self._built_for and self._index is not None:
            return
        with self._lock:
            self._chunks = self.store.all_chunks()
            self._tokens = [tokenize(c["text"]) for c in self._chunks]
            self._index = BM25Okapi(self._tokens) if self._tokens else None
            self._built_for = current

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        self._ensure_index()
        terms = content_terms(query)
        if self._index is None or not terms:
            return []
        scores = self._index.get_scores(terms)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in order:
            chunk_terms = set(self._tokens[i])
            coverage = sum(1 for t in set(terms) if t in chunk_terms) / len(set(terms))
            if coverage == 0:
                continue
            c = self._chunks[i]
            results.append(
                {
                    "text": c["text"],
                    "file": c.get("file"),
                    "page": c.get("page"),
                    "chunk_id": c.get("chunk_id"),
                    "document_id": c.get("document_id", ""),
                    "bm25_score": round(float(scores[i]), 4),
                    "coverage": round(coverage, 3),
                }
            )
        return results
