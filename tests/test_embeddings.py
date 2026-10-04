import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from backend.rag.embeddings import Embedder


@pytest.fixture(scope="module")
def embedder():
    e = Embedder()
    if not e.is_ready():
        pytest.skip(f"Embedding model could not be loaded (offline?): {e.error}")
    return e


def test_vectors_are_normalized_and_have_expected_dimension(embedder):
    vec = np.array(embedder.embed_query("Why did August sales decrease?"))
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-3


def test_batch_embedding_shape(embedder):
    vecs = embedder.embed_documents(["one", "two", "three"])
    assert len(vecs) == 3 and all(len(v) == 384 for v in vecs)
    assert embedder.embed_documents([]) == []


def test_semantic_similarity_ordering(embedder):
    q = np.array(embedder.embed_query("Why did revenue fall last month?"))
    near = np.array(embedder.embed_query("Sales dropped because of inventory shortages."))
    far = np.array(embedder.embed_query("The cat sat on the warm windowsill."))
    assert q @ near > q @ far
