import pymupdf
import pytest

from backend.rag.loader import DocumentLoader, DocumentLoadError


def make_pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_pdf_pages_keep_metadata_and_skip_empty(tmp_path):
    pdf = tmp_path / "Report.pdf"
    make_pdf(pdf, ["First page about revenue", "", "Third page about inventory"])
    pages = DocumentLoader().load(pdf)
    assert [p["page"] for p in pages] == [1, 3]  # empty page 2 skipped, numbering preserved
    assert all(p["file"] == "Report.pdf" for p in pages)
    assert "inventory" in pages[1]["text"]


def test_display_name_overrides_file_name(tmp_path):
    pdf = tmp_path / "abc123.pdf"
    make_pdf(pdf, ["hello world"])
    assert DocumentLoader().load(pdf, display_name="August_Report.pdf")[0]["file"] == "August_Report.pdf"


def test_missing_file(tmp_path):
    with pytest.raises(DocumentLoadError, match="not found"):
        DocumentLoader().load(tmp_path / "nope.pdf")


def test_unsupported_extension(tmp_path):
    f = tmp_path / "x.exe"
    f.write_bytes(b"data")
    with pytest.raises(DocumentLoadError, match="Unsupported"):
        DocumentLoader().load(f)


def test_corrupted_pdf(tmp_path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"this is definitely not a pdf")
    with pytest.raises(DocumentLoadError):
        DocumentLoader().load(f)


def test_empty_file_and_blank_pdf(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with pytest.raises(DocumentLoadError, match="empty"):
        DocumentLoader().load(empty)
    blank = tmp_path / "blank.pdf"
    make_pdf(blank, ["", ""])
    with pytest.raises(DocumentLoadError, match="No readable text"):
        DocumentLoader().load(blank)


def test_text_markdown_csv_json(tmp_path):
    (tmp_path / "a.md").write_text("# Title\nbody", encoding="utf-8")
    (tmp_path / "b.csv").write_text("name,qty\nwidget,3\n", encoding="utf-8")
    (tmp_path / "c.json").write_text('{"k": "v"}', encoding="utf-8")
    loader = DocumentLoader()
    assert "Title" in loader.load(tmp_path / "a.md")[0]["text"]
    assert "name: widget" in loader.load(tmp_path / "b.csv")[0]["text"]
    assert '"k"' in loader.load(tmp_path / "c.json")[0]["text"]
    (tmp_path / "d.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        loader.load(tmp_path / "d.json")
