"""DataAnalyst AI - FastAPI application."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.analytics.csv_analyzer import AnalyticsError
from backend.config import ROOT_DIR, settings
from backend.documents import ALLOWED_EXT, kind_for, safe_filename
from backend.evaluation.evaluator import RagEvaluator
from backend.rag.loader import DocumentLoadError
from backend.services import Services, get_services

logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dataanalyst")
FRONTEND_DIR = ROOT_DIR / "frontend"
CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    def warm() -> None:
        try:
            svc = get_services()
            svc.restore()
            svc.embedder.is_ready()
        except Exception as exc:  # never crash startup
            logger.error("Startup warm-up failed: %s", exc)

    threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(title="DataAnalyst AI", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=64)


class EvalRequest(BaseModel):
    dataset: list[dict[str, Any]] | None = None
    generate: bool = True


def services() -> Services:
    try:
        return get_services()
    except Exception as exc:
        logger.exception("Service initialisation failed")
        raise HTTPException(500, f"Backend failed to initialise: {type(exc).__name__}: {exc}") from exc


def public_doc(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k not in {"stored_name", "sha256"}}


# ---------------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


# --------------------------------------------------------------------- health
@app.get("/api/health")
def health() -> dict[str, Any]:
    svc = services()
    lm = svc.llm.status()
    embedding_ok = svc.embedder.is_ready()
    return {
        "backend": "online",
        "lm_studio": lm["reachable"],
        "lm_studio_message": lm["message"],
        "embedding_model": embedding_ok,
        "embedding_message": "" if embedding_ok else svc.embedder.error,
        "vector_database": svc.vector_store.ping(),
        "reranker": svc.reranker.available,
        "postgres": svc.history.status(),
        "debug": settings.debug,
    }


@app.get("/api/model-status")
def model_status() -> dict[str, Any]:
    return services().llm.status()


# ------------------------------------------------------------------ documents
@app.get("/api/documents")
def list_documents() -> dict[str, Any]:
    return {"documents": [public_doc(d) for d in services().registry.all()]}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    svc = services()
    name = safe_filename(file.filename or "")
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}")

    tmp = settings.upload_path / f"upload_{uuid.uuid4().hex}.tmp"
    size, sha = 0, hashlib.sha256()
    try:
        with tmp.open("wb") as out:
            while True:
                block = file.file.read(CHUNK)
                if not block:
                    break
                size += len(block)
                if size > settings.max_file_bytes:
                    raise HTTPException(413, f"File is larger than {settings.max_file_size_mb} MB.")
                sha.update(block)
                out.write(block)
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")

        digest = sha.hexdigest()
        doc_id = digest[:16]
        existing = svc.registry.get(doc_id)
        if existing:
            tmp.unlink(missing_ok=True)
            return {"document": public_doc(existing), "duplicate": True, "replaced": False,
                    "message": f"{existing['name']} is already uploaded."}

        replaced = False
        same_name = svc.registry.find_by_name(name)
        if same_name:
            svc.remove(same_name)
            replaced = True

        stored = f"{doc_id}{ext}"
        final = settings.upload_path / stored
        tmp.replace(final)
        entry: dict[str, Any] = {
            "id": doc_id, "name": name, "kind": kind_for(ext), "size": size, "sha256": digest,
            "stored_name": stored, "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            entry.update(svc.ingest(entry))
        except (DocumentLoadError, AnalyticsError) as exc:
            final.unlink(missing_ok=True)
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            final.unlink(missing_ok=True)
            logger.exception("Ingestion failed")
            raise HTTPException(500, f"Could not process the file: {type(exc).__name__}: {exc}") from exc
        svc.registry.put(entry)
        return {"document": public_doc(entry), "duplicate": False, "replaced": replaced,
                "message": f"{name} processed successfully."}
    finally:
        tmp.unlink(missing_ok=True)


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, Any]:
    svc = services()
    entry = svc.registry.get(document_id)
    if not entry:
        raise HTTPException(404, "Document not found.")
    svc.remove(entry)
    return {"deleted": document_id}


@app.post("/api/documents/{document_id}/reindex")
def reindex_document(document_id: str) -> dict[str, Any]:
    svc = services()
    entry = svc.registry.get(document_id)
    if not entry:
        raise HTTPException(404, "Document not found.")
    try:
        entry.update(svc.ingest(entry))
    except (DocumentLoadError, AnalyticsError) as exc:
        raise HTTPException(422, str(exc)) from exc
    svc.registry.put(entry)
    return {"document": public_doc(entry)}


# ------------------------------------------------------------------------ ask
@app.post("/api/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    svc = services()
    if not svc.embedder.is_ready() and svc.registry.has_kind("pdf", "text"):
        raise HTTPException(503, f"Embedding model is unavailable: {svc.embedder.error}")
    try:
        return svc.pipeline.ask(req.question.strip(), req.session_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(500, f"Could not answer: {type(exc).__name__}: {exc}") from exc


@app.post("/api/analyze")
def analyze(req: AskRequest) -> dict[str, Any]:
    """Structured analytics only (no document retrieval)."""
    svc = services()
    if not svc.engine.has_tables():
        raise HTTPException(400, "Upload a CSV or Excel file first.")
    return svc.pipeline.ask(req.question.strip(), req.session_id, force_route="ANALYTICS")


@app.post("/api/conversation/clear")
def clear_conversation(req: AskRequest | None = None) -> dict[str, str]:
    services().conversations.clear(req.session_id if req else "default")
    return {"status": "cleared"}


# ----------------------------------------------------------------- evaluation
@app.post("/api/evaluate")
def evaluate(req: EvalRequest) -> dict[str, Any]:
    svc = services()
    dataset = req.dataset
    if dataset is None:
        path = ROOT_DIR / "evaluation_data" / "eval_dataset.json"
        if not path.exists():
            raise HTTPException(400, "Send a 'dataset' in the request or create evaluation_data/eval_dataset.json.")
        dataset = json.loads(path.read_text(encoding="utf-8"))
    try:
        return RagEvaluator(svc.hybrid, svc.llm).run(dataset, req.generate)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/evaluations")
def evaluations() -> dict[str, Any]:
    return {"reports": RagEvaluator.list_reports()}
