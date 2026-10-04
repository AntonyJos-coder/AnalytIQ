"""Lazy construction of all long-lived services (single instance per process)."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.analytics.csv_analyzer import AnalyticsError, load_csv
from backend.analytics.engine import AnalyticsEngine
from backend.analytics.excel_analyzer import load_excel
from backend.config import settings
from backend.db import HistoryStore
from backend.documents import DocumentRegistry
from backend.pipeline import AnalystPipeline
from backend.rag.bm25_retriever import BM25Retriever
from backend.rag.chunker import TextChunker
from backend.rag.conversation import ConversationStore
from backend.rag.embeddings import get_embedder
from backend.rag.generator import LMStudioClient
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.loader import DocumentLoader
from backend.rag.query_rewriter import QueryRewriter
from backend.rag.reranker import Reranker
from backend.rag.retriever import VectorStore
from backend.router import QuestionRouter

logger = logging.getLogger(__name__)


class Services:
    def __init__(self) -> None:
        settings.upload_path.mkdir(parents=True, exist_ok=True)
        self.embedder = get_embedder()
        self.vector_store = VectorStore(embedder=self.embedder)
        self.bm25 = BM25Retriever(self.vector_store)
        self.reranker = Reranker()
        self.hybrid = HybridRetriever(self.vector_store, self.bm25, self.reranker)
        self.llm = LMStudioClient()
        self.engine = AnalyticsEngine(self.llm)
        self.conversations = ConversationStore()
        self.rewriter = QueryRewriter(self.llm, self.conversations)
        self.router = QuestionRouter(self.llm)
        self.registry = DocumentRegistry(settings.upload_path)
        self.history = HistoryStore()
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.pipeline = AnalystPipeline(self.hybrid, self.llm, self.engine, self.conversations, self.rewriter,
                                        self.router, self.registry, self.history)

    # -------------------------------------------------------------- ingestion
    def ingest(self, entry: dict[str, Any]) -> dict[str, Any]:
        """(Re)index one registered document. Returns stats; raises on unreadable files."""
        path: Path = self.registry.file_path(entry)
        if entry["kind"] == "table":
            if entry["name"].lower().endswith(".csv"):
                analyzers = [load_csv(path, entry["name"])]
            else:
                analyzers = load_excel(path, entry["name"])
            self.engine.add(entry["id"], analyzers)
            return {"rows": sum(len(a.df) for a in analyzers), "tables": len(analyzers)}
        pages = self.loader.load(path, display_name=entry["name"])
        chunks = self.chunker.chunk_pages(pages, document_id=entry["id"])
        self.vector_store.delete_document(entry["id"])
        added = self.vector_store.add_chunks(chunks)
        self.bm25.invalidate()
        return {"pages": len(pages), "chunks": len(chunks), "added": added}

    def remove(self, entry: dict[str, Any]) -> None:
        if entry["kind"] == "table":
            self.engine.remove(entry["id"])
        else:
            self.vector_store.delete_document(entry["id"])
            self.bm25.invalidate()
        try:
            self.registry.file_path(entry).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete stored file: %s", exc)
        self.registry.remove(entry["id"])

    def restore(self) -> None:
        """On startup: reload tables into memory and re-index documents missing from Chroma."""
        for entry in self.registry.all():
            try:
                if entry["kind"] == "table":
                    self.ingest(entry)
                elif self.vector_store.document_chunk_count(entry["id"]) == 0:
                    self.ingest(entry)
            except Exception as exc:
                logger.warning("Could not restore %s: %s", entry.get("name"), type(exc).__name__)


@lru_cache(maxsize=1)
def get_services() -> Services:
    return Services()
