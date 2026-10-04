"""Chunking that never drops citation metadata."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from backend.config import settings

Splitter = Callable[[str], list[str]]


class TextChunker:
    """Character-window chunker that prefers paragraph/sentence boundaries.

    A custom ``splitter`` (e.g. token-aware or semantic) can be injected later
    without touching the rest of the pipeline.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        splitter: Splitter | None = None,
    ) -> None:
        self.chunk_size = settings.chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self._splitter = splitter or self._window_split

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _window_split(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                window = text[start:end]
                cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
                if cut > self.chunk_size * 0.5:
                    end = start + cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def split_text(self, text: str) -> list[str]:
        return self._splitter(self._normalize(text))

    def chunk_pages(self, pages: list[dict[str, Any]], document_id: str = "") -> list[dict[str, Any]]:
        """Turn loader pages into chunks: text, file, page, chunk_id (+ document_id)."""
        chunks: list[dict[str, Any]] = []
        for page in pages:
            for index, piece in enumerate(self.split_text(page["text"])):
                raw = f"{document_id}|{page['file']}|{page['page']}|{index}|{piece}"
                chunks.append(
                    {
                        "text": piece,
                        "file": page["file"],
                        "page": page["page"],
                        "chunk_id": hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20],
                        "chunk_index": index,
                        "document_id": document_id,
                    }
                )
        return chunks
