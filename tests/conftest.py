import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEmbedder:
    """Deterministic bag-of-words embedder so retrieval tests run without downloading a model."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.DIM)
        for tok in text.lower().split():
            tok = "".join(ch for ch in tok if ch.isalnum())
            if tok:
                v[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts, batch_size: int = 32):
        return [self._vec(t) for t in texts]


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
