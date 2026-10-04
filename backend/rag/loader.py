"""Document loading. Every returned page keeps its file name and page number."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import pymupdf  # PyMuPDF (do not use the deprecated `import fitz`)

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Raised when a document cannot be read (missing, unsupported, corrupt, empty)."""


class DocumentLoader:
    """Loads PDF, TXT, Markdown, CSV and JSON files into page dictionaries.

    Each page is ``{"text": str, "page": int, "file": str}``. Empty pages are skipped.
    OCR is intentionally not part of V1.
    """

    SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".csv", ".json"}
    CSV_PREVIEW_ROWS = 200

    def load(self, path: str | Path, display_name: str | None = None) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise DocumentLoadError(f"File not found: {p.name}")
        ext = p.suffix.lower()
        if ext not in self.SUPPORTED:
            raise DocumentLoadError(f"Unsupported file type: {ext or '(none)'}")
        if p.stat().st_size == 0:
            raise DocumentLoadError(f"File is empty: {p.name}")

        name = display_name or p.name
        if ext == ".pdf":
            pages = self._load_pdf(p, name)
        elif ext in {".txt", ".md", ".markdown"}:
            pages = self._single_page(self._read_text(p), name)
        elif ext == ".csv":
            pages = self._single_page(self._csv_to_text(p), name)
        else:
            pages = self._single_page(self._json_to_text(p), name)

        if not pages:
            raise DocumentLoadError(f"No readable text found in {name}")
        return pages

    # ------------------------------------------------------------------ helpers
    def _load_pdf(self, path: Path, name: str) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        try:
            with pymupdf.open(str(path)) as doc:
                if doc.needs_pass:
                    raise DocumentLoadError(f"PDF is password protected: {name}")
                for index, page in enumerate(doc, start=1):
                    text = (page.get_text("text") or "").strip()
                    if text:
                        pages.append({"text": text, "page": index, "file": name})
        except DocumentLoadError:
            raise
        except Exception as exc:  # corrupted / not a PDF
            logger.warning("Could not open PDF %s: %s", name, exc)
            raise DocumentLoadError(f"Could not read PDF (corrupted or invalid): {name}") from exc
        return pages

    @staticmethod
    def _read_text(path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeError, OSError):
                continue
        raise DocumentLoadError(f"Could not decode text file: {path.name}")

    def _csv_to_text(self, path: Path) -> str:
        text = self._read_text(path)
        rows = list(csv.reader(text.splitlines()))
        if not rows:
            return ""
        header = rows[0]
        lines = []
        for row in rows[1 : self.CSV_PREVIEW_ROWS + 1]:
            lines.append("; ".join(f"{h}: {v}" for h, v in zip(header, row)))
        return "\n".join(lines)

    def _json_to_text(self, path: Path) -> str:
        try:
            return json.dumps(json.loads(self._read_text(path)), indent=2, ensure_ascii=False)
        except json.JSONDecodeError as exc:
            raise DocumentLoadError(f"Invalid JSON file: {path.name}") from exc

    @staticmethod
    def _single_page(text: str, name: str) -> list[dict[str, Any]]:
        text = text.strip()
        return [{"text": text, "page": 1, "file": name}] if text else []
