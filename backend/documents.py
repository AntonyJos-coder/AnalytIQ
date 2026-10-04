"""Uploaded-document registry and filename safety helpers."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

TABLE_EXT = {".csv", ".xlsx"}
TEXT_EXT = {".pdf", ".txt", ".md"}
ALLOWED_EXT = TABLE_EXT | TEXT_EXT


def safe_filename(name: str) -> str:
    """Strip any path components and unsafe characters from a user-supplied name."""
    base = Path((name or "").replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._\- ()]+", "_", base).strip(" .")
    return base[:120] or "file"


def kind_for(ext: str) -> str:
    return "table" if ext in TABLE_EXT else "pdf" if ext == ".pdf" else "text"


class DocumentRegistry:
    """JSON-backed registry. Files are stored under generated ids, never user-chosen paths."""

    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.path = upload_dir / "registry.json"
        self._lock = threading.Lock()
        self._docs: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._docs = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._docs = {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._docs, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.get("uploaded_at", ""))

    def get(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._docs.get(doc_id)

    def find_by_name(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            return next((d for d in self._docs.values() if d["name"].lower() == name.lower()), None)

    def put(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._docs[entry["id"]] = entry
            self._save()

    def remove(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._docs.pop(doc_id, None)
            self._save()
            return entry

    def file_path(self, entry: dict[str, Any]) -> Path:
        return self.upload_dir / entry["stored_name"]

    def has_kind(self, *kinds: str) -> bool:
        with self._lock:
            return any(d["kind"] in kinds for d in self._docs.values())
