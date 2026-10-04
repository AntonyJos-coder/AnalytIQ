from __future__ import annotations

import time
from typing import Any

from backend.config import settings
from backend.rag.bm25_retriever import BM25Retriever
from backend.rag.reranker import Reranker
from backend.rag.retriever import VectorStore


RRF_K = 60
MIN_BM25_COVERAGE = 0.5


def _ms(start: float) -> float:
    """
    Convert elapsed perf_counter time to milliseconds.
    """
    return round(
        (time.perf_counter() - start) * 1000,
        2,
    )


class HybridRetriever:

    def __init__(
        self,
        store: VectorStore,
        bm25: BM25Retriever,
        reranker: Reranker,
    ) -> None:

        self.store = store
        self.bm25 = bm25
        self.reranker = reranker

    # =====================================================
    # Reciprocal Rank Fusion
    # =====================================================

    @staticmethod
    def fuse(
        vector_hits: list[dict[str, Any]],
        bm25_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Reciprocal Rank Fusion.

        Duplicate chunks are identified using chunk_id and merged.
        """

        merged: dict[str, dict[str, Any]] = {}

        # -------------------------------------------------
        # Vector results
        # -------------------------------------------------

        for rank, hit in enumerate(
            vector_hits,
            start=1,
        ):

            chunk_id = hit["chunk_id"]

            entry = merged.setdefault(
                chunk_id,
                dict(hit),
            )

            entry["rrf_score"] = (
                entry.get("rrf_score", 0.0)
                + 1.0 / (RRF_K + rank)
            )

            entry["vector_rank"] = rank

        # -------------------------------------------------
        # BM25 results
        # -------------------------------------------------

        for rank, hit in enumerate(
            bm25_hits,
            start=1,
        ):

            chunk_id = hit["chunk_id"]

            entry = merged.setdefault(
                chunk_id,
                dict(hit),
            )

            for key in (
                "bm25_score",
                "coverage",
            ):
                if key in hit:
                    entry[key] = hit[key]

            entry["rrf_score"] = (
                entry.get("rrf_score", 0.0)
                + 1.0 / (RRF_K + rank)
            )

            entry["bm25_rank"] = rank

        # -------------------------------------------------
        # Sort by fused score
        # -------------------------------------------------

        fused = sorted(
            merged.values(),
            key=lambda chunk: chunk["rrf_score"],
            reverse=True,
        )

        for chunk in fused:

            chunk["rrf_score"] = round(
                chunk["rrf_score"],
                5,
            )

        return fused

    # =====================================================
    # Relevance gate
    # =====================================================

    @staticmethod
    def is_relevant(
        chunk: dict[str, Any],
    ) -> bool:
        """
        A chunk survives if either:

        - vector similarity is high enough
        OR
        - BM25 token coverage is high enough
        """

        vector_score = chunk.get(
            "vector_score"
        )

        if (
            vector_score is not None
            and vector_score
            >= settings.min_similarity
        ):
            return True

        return (
            chunk.get("coverage", 0.0)
            >= MIN_BM25_COVERAGE
        )

    # =====================================================
    # Retrieval
    # =====================================================

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        rerank_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the complete hybrid retrieval pipeline and record
        timing for every important stage.
        """

        total_start = time.perf_counter()

        top_k = top_k or settings.top_k
        candidate_k = (
            candidate_k
            or settings.candidate_k
        )
        rerank_k = (
            rerank_k
            or settings.rerank_top_k
        )

        # =================================================
        # Vector search
        # =================================================

        start = time.perf_counter()

        vector_hits = self.store.search(
            query,
            top_k,
        )

        vector_ms = _ms(start)

        # =================================================
        # BM25
        # =================================================

        start = time.perf_counter()

        bm25_hits = self.bm25.search(
            query,
            top_k,
        )

        bm25_ms = _ms(start)

        # =================================================
        # RRF fusion
        # =================================================

        start = time.perf_counter()

        fused_all = self.fuse(
            vector_hits,
            bm25_hits,
        )

        fused = fused_all[:candidate_k]

        fusion_ms = _ms(start)

        # =================================================
        # Relevance threshold
        # =================================================

        start = time.perf_counter()

        relevant = [
            chunk
            for chunk in fused
            if self.is_relevant(chunk)
        ]

        threshold_ms = _ms(start)

        # =================================================
        # Reranking
        # =================================================

        start = time.perf_counter()

        if relevant:

            final = self.reranker.rerank(
                query,
                relevant,
                rerank_k,
            )

        else:

            final = []

        rerank_ms = _ms(start)

        # =================================================
        # Final scores
        # =================================================

        for chunk in final:

            chunk["score"] = chunk.get(
                "rerank_score",
                chunk.get(
                    "vector_score",
                    chunk.get(
                        "rrf_score",
                        0.0,
                    ),
                ),
            )

        total_ms = _ms(total_start)

        # =================================================
        # Result
        # =================================================

        return {

            "query": query,

            "chunks": final,

            "debug": {

                # -----------------------------------------
                # Result counts
                # -----------------------------------------

                "vector_hits": len(
                    vector_hits
                ),

                "bm25_hits": len(
                    bm25_hits
                ),

                "fused": len(
                    fused
                ),

                "after_threshold": len(
                    relevant
                ),

                "returned": len(
                    final
                ),

                "reranker_used": (
                    self.reranker.available
                    and bool(relevant)
                ),

                # -----------------------------------------
                # Performance
                # -----------------------------------------

                "timing_ms": {

                    "vector_search":
                        vector_ms,

                    "bm25_search":
                        bm25_ms,

                    "rrf_fusion":
                        fusion_ms,

                    "relevance_filter":
                        threshold_ms,

                    "reranking":
                        rerank_ms,

                    "retrieval_total":
                        total_ms,
                },
            },
        }