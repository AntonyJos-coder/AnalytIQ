import pytest

from backend.rag.chunker import TextChunker


def pages(text, page=7, file="August_Report.pdf"):
    return [{"text": text, "page": page, "file": file}]


def test_metadata_preserved_on_every_chunk():
    text = " ".join(f"Sentence number {i} talks about sales." for i in range(200))
    chunks = TextChunker(300, 50).chunk_pages(pages(text), document_id="doc1")
    assert len(chunks) > 3
    assert all(c["file"] == "August_Report.pdf" and c["page"] == 7 and c["document_id"] == "doc1" for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_chunks_respect_size_and_overlap():
    text = "word " * 500
    chunks = TextChunker(200, 40).chunk_pages(pages(text))
    assert all(len(c["text"]) <= 200 for c in chunks)
    first_tail, second_head = chunks[0]["text"][-30:], chunks[1]["text"][:60]
    assert first_tail.strip().split()[-1] in second_head  # overlapping context is carried over


def test_short_text_is_single_chunk_and_ids_are_stable():
    a = TextChunker(800, 150).chunk_pages(pages("Short text."), document_id="d")
    b = TextChunker(800, 150).chunk_pages(pages("Short text."), document_id="d")
    assert len(a) == 1 and a[0]["chunk_id"] == b[0]["chunk_id"]


def test_ids_differ_across_pages():
    pgs = pages("Same text") + pages("Same text", page=8)
    chunks = TextChunker().chunk_pages(pgs)
    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]


def test_invalid_configuration():
    with pytest.raises(ValueError):
        TextChunker(100, 100)
    with pytest.raises(ValueError):
        TextChunker(0, 0)


def test_custom_splitter_can_be_injected():
    chunker = TextChunker(splitter=lambda t: t.split("|"))
    out = chunker.chunk_pages(pages("a|b|c"))
    assert [c["text"] for c in out] == ["a", "b", "c"]
