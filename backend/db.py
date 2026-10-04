"""Optional PostgreSQL query history. Disabled (no-op) when DATABASE_URL is empty."""
from __future__ import annotations

import json
import logging
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS query_history (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    session_id TEXT,
    question TEXT,
    route TEXT,
    answer TEXT,
    sources JSONB,
    latency_ms INTEGER
)
"""


class HistoryStore:
    def __init__(self, url: str | None = None) -> None:
        self.url = settings.database_url if url is None else url
        self.enabled = bool(self.url)
        self.error = ""
        if self.enabled:
            try:
                with self._connect() as conn:
                    conn.execute(DDL)
            except Exception as exc:
                self.enabled = False
                self.error = f"{type(exc).__name__}: {exc}"
                logger.warning("PostgreSQL history disabled: %s", self.error)

    def _connect(self):  # noqa: ANN202
        import psycopg

        return psycopg.connect(self.url, autocommit=True, connect_timeout=3)

    def log(self, session_id: str, question: str, route: str, answer: str, sources: list[Any], latency_ms: int) -> None:
        if not self.enabled:
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO query_history (session_id, question, route, answer, sources, latency_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (session_id, question, route, answer, json.dumps(sources), latency_ms),
                )
        except Exception as exc:
            logger.warning("Could not write query history: %s", type(exc).__name__)

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "error": self.error}
