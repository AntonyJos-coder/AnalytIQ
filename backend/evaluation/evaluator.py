"""Local RAG evaluation: retrieval hit rate, citation accuracy, relevance, grounding, latency."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.rag.generator import INSUFFICIENT, LMStudioClient, LMStudioError
from backend.rag.hybrid_retriever import HybridRetriever

_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_STOP = {"the", "and", "was", "were", "for", "that", "with", "this", "from", "have", "has", "had", "are", "its", "their"}


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def _matches(item: dict[str, Any], file: str | None, page: Any) -> bool:
    ok_file = not item.get("expected_file") or (file or "").lower() == str(item["expected_file"]).lower()
    ok_page = item.get("expected_page") is None or page == item["expected_page"]
    return ok_file and ok_page


class RagEvaluator:
    def __init__(self, hybrid: HybridRetriever, llm: LMStudioClient) -> None:
        self.hybrid, self.llm = hybrid, llm

    def evaluate_item(self, item: dict[str, Any], generate: bool) -> dict[str, Any]:
        t0 = time.perf_counter()
        retrieved = self.hybrid.retrieve(item["question"])
        chunks = retrieved["chunks"]
        hit = any(_matches(item, c["file"], c["page"]) for c in chunks)
        row: dict[str, Any] = {
            "question": item["question"], "retrieval_hit": hit, "retrieved": [
                {"file": c["file"], "page": c["page"], "score": c.get("score")} for c in chunks],
            "answer": None, "citation_correct": None, "answer_relevance": None, "faithfulness": None,
        }
        if generate and chunks:
            try:
                gen = self.llm.generate_answer(item["question"], chunks)
                answer = gen["answer"]
                row["answer"] = answer
                row["citation_correct"] = any(_matches(item, s["file"], s["page"]) for s in gen["sources"])
                expected = _terms(item.get("expected_answer", ""))
                row["answer_relevance"] = round(len(expected & _terms(answer)) / len(expected), 3) if expected else None
                answer_terms = _terms(answer)
                context_terms = _terms(" ".join(c["text"] for c in chunks))
                row["faithfulness"] = None if INSUFFICIENT in answer or not answer_terms else round(
                    len(answer_terms & context_terms) / len(answer_terms), 3)
            except LMStudioError as exc:
                row["generation_error"] = str(exc)
        row["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return row

    def run(self, dataset: list[dict[str, Any]], generate: bool = True) -> dict[str, Any]:
        if not dataset:
            raise ValueError("The evaluation dataset is empty.")
        for i, item in enumerate(dataset):
            if "question" not in item:
                raise ValueError(f"Item {i} is missing 'question'.")
        generate = generate and self.llm.is_reachable()
        rows = [self.evaluate_item(item, generate) for item in dataset]

        def mean(key: str) -> float | None:
            vals = [r[key] for r in rows if r.get(key) is not None]
            if not vals:
                return None
            return round(sum(float(v) for v in vals) / len(vals), 3)

        summary = {
            "questions": len(rows),
            "generation_enabled": generate,
            "retrieval_hit_rate": mean("retrieval_hit"),
            "citation_accuracy": mean("citation_correct"),
            "answer_relevance": mean("answer_relevance"),
            "faithfulness": mean("faithfulness"),
            "avg_latency_ms": mean("latency_ms"),
        }
        report = {"created_at": datetime.now().isoformat(timespec="seconds"), "summary": summary, "results": rows}
        self._save(report)
        return report

    @staticmethod
    def _save(report: dict[str, Any]) -> None:
        settings.eval_path.mkdir(parents=True, exist_ok=True)
        name = "eval_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        Path(settings.eval_path / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["saved_as"] = name

    @staticmethod
    def list_reports() -> list[dict[str, Any]]:
        if not settings.eval_path.exists():
            return []
        out = []
        for p in sorted(settings.eval_path.glob("eval_*.json"), reverse=True)[:20]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append({"file": p.name, "created_at": data.get("created_at"), "summary": data.get("summary")})
            except (OSError, json.JSONDecodeError):
                continue
        return out
