"""Holds every uploaded table and picks the right one for a question."""
from __future__ import annotations

import logging
import threading
from typing import Any

from backend.analytics.csv_analyzer import CSVAnalyzer
from backend.analytics.sql_engine import SQLEngine, UnsafeSQLError
from backend.rag.generator import LMStudioClient, LMStudioError

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    def __init__(self, llm: LMStudioClient | None = None) -> None:
        self.llm = llm
        self._tables: dict[str, list[CSVAnalyzer]] = {}
        self._lock = threading.Lock()

    def add(self, doc_id: str, analyzers: list[CSVAnalyzer]) -> None:
        with self._lock:
            self._tables[doc_id] = analyzers

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._tables.pop(doc_id, None)

    def has_tables(self) -> bool:
        return any(self._tables.values())

    def summaries(self) -> list[dict[str, Any]]:
        return [a.summary() for group in self._tables.values() for a in group]

    def analyze(self, question: str, use_llm_fallback: bool = True) -> dict[str, Any]:
        analyzers = [a for group in list(self._tables.values()) for a in group]
        if not analyzers:
            return {"ok": False, "reason": "No CSV or Excel data has been uploaded yet."}
        best = max(analyzers, key=lambda a: a.table_score(question))
        result = best.analyze(question)
        if result:
            return result
        reason = best.last_issue or "The question could not be mapped to a calculation."
        if use_llm_fallback and self.llm is not None:
            try:
                fallback = SQLEngine(best).answer(question, self.llm)
                if fallback:
                    return fallback
            except (LMStudioError, UnsafeSQLError) as exc:
                logger.info("SQL fallback unavailable: %s", exc)
            except Exception as exc:  # malformed SQL from the model, etc.
                logger.info("SQL fallback failed: %s", type(exc).__name__)
        return {"ok": False, "reason": reason, "table": best.label}
