"""Persistent ChromaDB vector store + plain vector retrieval."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.rag.embeddings import Embedder, get_embedder

logger = logging.getLogger(__name__)
_BATCH = 256


class VectorStore:
    """Chroma collection (cosine space). ``embedder`` only needs embed_query/embed_documents."""

    def __init__(self, path: Path | None = None, embedder: Any = None, collection_name: str | None = None) -> None:
        self.path = Path(path or settings.chroma_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder: Embedder = embedder or get_embedder()
        self.client = chromadb.PersistentClient(
            path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------ write
    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Embed and upsert chunks. Existing chunk ids are skipped (no duplicates)."""
        added = 0
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            ids = [c["chunk_id"] for c in batch]
            existing = set(self.collection.get(ids=ids, include=[]).get("ids", []))
            fresh = [c for c in batch if c["chunk_id"] not in existing]
            if not fresh:
                continue
            vectors = self.embedder.embed_documents([c["text"] for c in fresh])
            self.collection.add(
                ids=[c["chunk_id"] for c in fresh],
                embeddings=vectors,
                documents=[c["text"] for c in fresh],
                metadatas=[
                    {
                        "file": c["file"],
                        "page": int(c["page"]),
                        "chunk_id": c["chunk_id"],
                        "chunk_index": int(c.get("chunk_index", 0)),
                        "document_id": c.get("document_id", ""),
                    }
                    for c in fresh
                ],
            )
            added += len(fresh)
        return added

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return self.collection.count()

    def document_chunk_count(self, document_id: str) -> int:
        return len(self.collection.get(where={"document_id": document_id}, include=[]).get("ids", []))

    def all_chunks(self) -> list[dict[str, Any]]:
        data = self.collection.get(include=["documents", "metadatas"])
        out = []
        for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            out.append({"text": doc, **meta})
        return out

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        total = self.count()
        if total == 0 or not query.strip():
            return []
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append(
                {
                    "text": doc,
                    "file": meta.get("file"),
                    "page": meta.get("page"),
                    "chunk_id": meta.get("chunk_id"),
                    "document_id": meta.get("document_id", ""),
                    "score": round(1.0 - float(dist), 4),
                    "vector_score": round(1.0 - float(dist), 4),
                }
            )
        return results

    def ping(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception as exc:
            logger.error("Chroma heartbeat failed: %s", exc)
            return False


class Retriever:
    """Plain vector retrieval returning structured, citation-ready results."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.store.search(question, top_k or settings.top_k)
