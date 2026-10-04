"""End-to-end API tests with a fake embedder and LM Studio intentionally unreachable."""
import io

import pandas as pd
import pymupdf
import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeEmbedder
from tests.test_analytics import ROWS


class ReadyFakeEmbedder(FakeEmbedder):
    error = ""

    def is_ready(self) -> bool:
        return True


def pdf_bytes(pages):
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from backend import services as services_module
    from backend.config import settings

    overrides = {"upload_path": tmp_path / "uploads", "chroma_path": tmp_path / "chroma",
                 "lm_base_url": "http://127.0.0.1:9/v1"}
    originals = {k: getattr(settings, k) for k in overrides}
    for name, value in overrides.items():
        object.__setattr__(settings, name, value)  # Settings is a frozen dataclass
    monkeypatch.setattr(services_module, "get_embedder", lambda: ReadyFakeEmbedder())
    services_module.get_services.cache_clear()
    from backend.main import app

    yield TestClient(app)
    services_module.get_services.cache_clear()
    for name, value in originals.items():
        object.__setattr__(settings, name, value)


def csv_bytes():
    df = pd.DataFrame(ROWS, columns=["Date", "Product", "Region", "Revenue"])
    df["Product"] = "Product " + df["Product"]
    return df.to_csv(index=False).encode()


def upload(client, name, data):
    return client.post("/api/upload", files={"file": (name, io.BytesIO(data))})


def test_health_survives_missing_lm_studio(client):
    body = client.get("/api/health").json()
    assert body["backend"] == "online" and body["lm_studio"] is False
    assert "LM Studio" in body["lm_studio_message"] and body["vector_database"] is True


def test_upload_validation(client):
    assert upload(client, "evil.exe", b"x").status_code == 415
    assert upload(client, "empty.csv", b"").status_code == 400
    assert upload(client, "broken.pdf", b"not a pdf").status_code == 422
    r = upload(client, "..\\..\\secret\\report.csv", csv_bytes())
    assert r.status_code == 200 and r.json()["document"]["name"] == "report.csv"
    assert "stored_name" not in r.json()["document"]


def test_duplicate_and_delete(client):
    first = upload(client, "sales.csv", csv_bytes()).json()
    again = upload(client, "copy.csv", csv_bytes()).json()
    assert again["duplicate"] is True and again["document"]["id"] == first["document"]["id"]
    doc_id = first["document"]["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.get("/api/documents").json()["documents"] == []
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_rag_with_citations_when_llm_is_down(client):
    upload(client, "Management_Report.pdf", pdf_bytes(
        ["Intro page about the company.", "Product B revenue declined in August due to inventory shortages."]))
    body = client.post("/api/ask", json={"question": "What does the report say about inventory shortages?"}).json()
    assert body["route"]["route"] == "RAG"
    assert {"file": "Management_Report.pdf", "page": 2, "type": "document"} in body["sources"]
    assert any("LM Studio" in w for w in body["warnings"])


def test_rag_says_insufficient_when_nothing_relevant(client):
    upload(client, "Management_Report.pdf", pdf_bytes(["Product B revenue declined due to inventory shortages."]))
    body = client.post("/api/ask", json={"question": "What does the report say about quantum chromodynamics?"}).json()
    assert "couldn't find enough evidence" in body["answer"] and body["sources"] == []


def test_analytics_route_is_exact(client):
    upload(client, "sales.csv", csv_bytes())
    body = client.post("/api/ask", json={"question": "What was total revenue in August?"}).json()
    assert body["route"]["route"] == "ANALYTICS" and "2,280.00" in body["answer"]
    assert body["sources"] == [{"file": "sales.csv", "type": "data"}]


def test_hybrid_combines_calculation_and_documents_and_followups(client):
    upload(client, "sales.csv", csv_bytes())
    upload(client, "Management_Report.pdf", pdf_bytes(
        ["Overview.", "Product B revenue decline in August was driven by inventory shortages."]))
    body = client.post("/api/ask", json={
        "question": "Which product had the largest revenue decline and what explanation did management give in the report?",
        "session_id": "s1"}).json()
    assert body["route"]["route"] == "HYBRID"
    assert "CALCULATED:" in body["answer"] and "Product B" in body["answer"] and "-27.0%" in body["answer"]
    assert "DOCUMENT EVIDENCE:" in body["answer"]
    assert body["chart"] is not None
    assert any(s.get("page") == 2 for s in body["sources"])
    assert "Product B" in body["debug"]["retrieval_query"]
    follow = client.post("/api/ask", json={"question": "Why?", "session_id": "s1"}).json()
    assert follow["route"]["route"] == "HYBRID" and "Product B" in follow["rewritten_question"]
