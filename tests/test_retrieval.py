import pytest

from backend.rag.bm25_retriever import BM25Retriever
from backend.rag.chunker import TextChunker
from backend.rag.generator import build_sources, validate_citations
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.reranker import Reranker
from backend.rag.retriever import Retriever, VectorStore

DOC = [
    {"text": "Product B revenue declined in August because inventory shortages limited supply.", "page": 12,
     "file": "Management_Report.pdf"},
    {"text": "The marketing team launched a new brand campaign across social channels.", "page": 3,
     "file": "Management_Report.pdf"},
    {"text": "SKU AB-1234 requires special cold storage handling in the warehouse.", "page": 5,
     "file": "Operations.pdf"},
]


@pytest.fixture()
def store(tmp_path, fake_embedder):
    s = VectorStore(path=tmp_path / "chroma", embedder=fake_embedder, collection_name="test_docs")
    s.add_chunks(TextChunker(800, 100).chunk_pages(DOC, document_id="doc1"))
    return s


def test_vector_search_returns_citation_metadata(store):
    hits = Retriever(store).retrieve("why did Product B revenue decline", top_k=3)
    assert hits[0]["file"] == "Management_Report.pdf" and hits[0]["page"] == 12
    assert 0 < hits[0]["score"] <= 1


def test_no_duplicate_chunks_on_reindex(store):
    before = store.count()
    added = store.add_chunks(TextChunker(800, 100).chunk_pages(DOC, document_id="doc1"))
    assert added == 0 and store.count() == before


def test_delete_document(store):
    assert store.document_chunk_count("doc1") == 3
    store.delete_document("doc1")
    assert store.count() == 0 and store.search("anything") == []


def test_bm25_finds_exact_sku(store):
    hits = BM25Retriever(store).search("AB-1234", top_k=3)
    assert hits and hits[0]["file"] == "Operations.pdf" and hits[0]["page"] == 5


def test_rrf_merges_and_deduplicates():
    v = [{"chunk_id": "a", "text": "a", "vector_score": 0.9}, {"chunk_id": "b", "text": "b", "vector_score": 0.5}]
    k = [{"chunk_id": "b", "text": "b", "bm25_score": 3.0, "coverage": 1.0},
         {"chunk_id": "c", "text": "c", "bm25_score": 1.0, "coverage": 0.5}]
    fused = HybridRetriever.fuse(v, k)
    ids = [c["chunk_id"] for c in fused]
    assert sorted(ids) == ["a", "b", "c"] and len(ids) == 3
    assert ids[0] == "b"  # found by both retrievers -> highest fused score


def test_hybrid_retrieval_and_insufficient_evidence(store):
    hybrid = HybridRetriever(store, BM25Retriever(store), Reranker(enabled=False))
    good = hybrid.retrieve("Product B revenue decline inventory", rerank_k=2)
    assert good["chunks"][0]["page"] == 12 and len(good["chunks"]) <= 2
    unrelated = hybrid.retrieve("quantum chromodynamics lattice simulation")
    assert unrelated["chunks"] == []  # below relevance threshold -> caller must say "insufficient evidence"


def test_citations_come_from_metadata_and_invalid_ids_are_removed():
    chunks = [{"file": "A.pdf", "page": 2, "text": "x"}, {"file": "B.pdf", "page": 9, "text": "y"}]
    clean, sources, verified = validate_citations("Fact one [S2]. Invented [S7].", chunks)
    assert "[S7]" not in clean and verified
    assert sources == [{"file": "B.pdf", "page": 9, "type": "document"}]
    _, fallback_sources, verified2 = validate_citations("No citations here.", chunks)
    assert not verified2 and fallback_sources == build_sources(chunks)
