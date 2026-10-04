"""LM Studio (OpenAI-compatible) client. All local-LLM calls live in this module."""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

import httpx

from backend.config import settings


logger = logging.getLogger(__name__)


# =========================================================
# Messages
# =========================================================

INSUFFICIENT = (
    "I couldn't find enough evidence in the uploaded "
    "documents to answer this reliably."
)

LM_DOWN = (
    "LM Studio is not reachable. "
    "Start the LM Studio local server."
)


# =========================================================
# Context limits
# =========================================================

# Retrieval still returns the normal number of chunks.
# These limits only control how much evidence is sent to Qwen.

RAG_CONTEXT_MAX_CHARS = 5000
RAG_MAX_PASSAGES = 4

HYBRID_CONTEXT_MAX_CHARS = 4000
HYBRID_MAX_PASSAGES = 4

MAX_PASSAGE_CHARS = 2000
MIN_PASSAGES = 2


# =========================================================
# System prompts
# =========================================================

SYSTEM_RAG = (
    "Answer the question using ONLY the numbered evidence passages. "
    "Do not invent facts, numbers, file names or page numbers. "
    "Cite supporting evidence inline using [S1], [S2], etc. "
    "Only cite source IDs that were provided. "
    "Give a concise, direct answer. "
    "If the evidence does not answer the question, reply exactly: "
    + INSUFFICIENT
)


SYSTEM_HYBRID = (
    "A calculated result was already computed exactly by code. "
    "Do not change or invent calculated numbers. "
    "Using ONLY the numbered document passages, briefly explain "
    "what the documents say that relates to the calculated result. "
    "Use 2-4 sentences and cite evidence using [S1], [S2], etc. "
    "Only cite source IDs that were provided. "
    "If the passages do not explain the result, reply exactly: "
    + INSUFFICIENT
)


# =========================================================
# Exception
# =========================================================

class LMStudioError(Exception):
    """
    LM Studio is unreachable or returned an invalid response.
    """


# =========================================================
# Citation helpers
# =========================================================

def build_sources(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build citations from retrieval metadata.

    File/page information is never trusted from model output.
    """

    seen: set[tuple[Any, Any]] = set()
    sources: list[dict[str, Any]] = []

    for chunk in chunks:
        key = (
            chunk.get("file"),
            chunk.get("page"),
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "file": chunk.get("file"),
                "page": chunk.get("page"),
                "type": "document",
            }
        )

    return sources


# =========================================================
# Context budgeting
# =========================================================

def select_context_chunks(
    chunks: list[dict[str, Any]],
    max_chars: int,
    max_passages: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Select the highest-ranked retrieved passages for the LLM.

    Important:
    - Retrieval itself is NOT changed.
    - The UI can still show all retrieved passages.
    - Only the evidence sent to Qwen is reduced.
    - Retrieval order is preserved.
    """

    if not chunks:
        return [], {
            "retrieved_chunks": 0,
            "chunks_sent_to_llm": 0,
            "original_context_chars": 0,
            "context_chars": 0,
            "context_budget_chars": max_chars,
            "max_passages": max_passages,
            "truncated_passages": 0,
        }

    selected: list[dict[str, Any]] = []

    total_chars = 0

    original_chars = sum(
        len(str(chunk.get("text", "")))
        for chunk in chunks
    )

    truncated_passages = 0

    for chunk in chunks:
        if len(selected) >= max_passages:
            break

        original_text = str(
            chunk.get("text", "")
        ).strip()

        if not original_text:
            continue

        remaining = max_chars - total_chars

        if remaining <= 0:
            break

        # Never send an unnecessarily huge individual passage.
        allowed = min(
            MAX_PASSAGE_CHARS,
            remaining,
        )

        text = original_text[:allowed]

        if len(text) < len(original_text):
            truncated_passages += 1

        selected_chunk = dict(chunk)
        selected_chunk["text"] = text

        selected.append(selected_chunk)

        total_chars += len(text)

        # Keep at least MIN_PASSAGES when possible.
        # After that, stop when the context budget is full.
        if (
            len(selected) >= MIN_PASSAGES
            and total_chars >= max_chars
        ):
            break

    # Safety fallback: if unusual empty chunks prevented selection,
    # use the first non-empty chunk.
    if not selected:
        for chunk in chunks:
            text = str(
                chunk.get("text", "")
            ).strip()

            if not text:
                continue

            selected_chunk = dict(chunk)

            selected_chunk["text"] = text[
                :min(MAX_PASSAGE_CHARS, max_chars)
            ]

            selected.append(selected_chunk)

            if len(selected_chunk["text"]) < len(text):
                truncated_passages += 1

            total_chars = len(
                selected_chunk["text"]
            )

            break

    debug = {
        "retrieved_chunks": len(chunks),
        "chunks_sent_to_llm": len(selected),
        "original_context_chars": original_chars,
        "context_chars": total_chars,
        "context_budget_chars": max_chars,
        "max_passages": max_passages,
        "truncated_passages": truncated_passages,
    }

    return selected, debug


def format_context(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Convert selected chunks into numbered evidence passages.

    The numbering here is the exact numbering the LLM sees,
    which keeps citation validation aligned.
    """

    passages: list[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        file_name = chunk.get(
            "file",
            "unknown",
        )

        page = chunk.get(
            "page",
            "unknown",
        )

        text = str(
            chunk.get(
                "text",
                "",
            )
        ).strip()

        passages.append(
            f"[S{index}] "
            f"({file_name}, page {page})\n"
            f"{text}"
        )

    return "\n\n".join(passages)


def validate_citations(
    answer: str,
    chunks: list[dict[str, Any]],
) -> tuple[
    str,
    list[dict[str, Any]],
    bool,
]:
    """
    Validate [S#] references.

    Invented source IDs are removed.

    Returns:
        clean answer
        sources
        citations_verified
    """

    valid = set(
        range(
            1,
            len(chunks) + 1,
        )
    )

    # Correct citation regex:
    # matches [S1], [S2], [S3], etc.
    cited = {
        int(number)
        for number in re.findall(
            r"\[S(\d+)\]",
            answer,
        )
    }

    clean = answer

    # -----------------------------------------------------
    # Remove invented citation IDs
    # -----------------------------------------------------

    for number in cited - valid:
        clean = clean.replace(
            f"[S{number}]",
            "",
        )

    # -----------------------------------------------------
    # Resolve valid citations
    # -----------------------------------------------------

    used_chunks = [
        chunks[number - 1]
        for number in sorted(
            cited & valid
        )
    ]

    if used_chunks:
        return (
            clean.strip(),
            build_sources(
                used_chunks
            ),
            True,
        )

    # Model answered without citations.
    # Expose selected retrieved sources but mark as unverified.

    return (
        clean.strip(),
        build_sources(chunks),
        False,
    )


# =========================================================
# LM Studio client
# =========================================================

class LMStudioClient:

    def __init__(self) -> None:
        self.base_url = settings.lm_base_url

        self.headers = {
            "Authorization":
                f"Bearer {settings.lm_api_key}"
        }

        self._resolved: str | None = None
        self._lock = threading.Lock()

    # =====================================================
    # Model discovery
    # =====================================================

    def list_models(
        self,
    ) -> list[str]:

        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=5.0,
            )

            response.raise_for_status()

            return [
                model["id"]
                for model
                in response.json().get(
                    "data",
                    [],
                )
            ]

        except (
            httpx.HTTPError,
            ValueError,
            KeyError,
        ) as exc:

            raise LMStudioError(
                LM_DOWN
            ) from exc

    def resolve_model(
        self,
        force: bool = False,
    ) -> str:
        """
        Resolve the actual model ID exposed by LM Studio.
        """

        if (
            self._resolved
            and not force
        ):
            return self._resolved

        models = self.list_models()

        if not models:
            raise LMStudioError(
                "LM Studio is running but no model is loaded. "
                "Load Qwen2.5-7B-Instruct-1M."
            )

        wanted = re.sub(
            r"[^a-z0-9]",
            "",
            settings.lm_model.lower(),
        )

        chosen: str | None = None

        # -------------------------------------------------
        # Exact normalized match
        # -------------------------------------------------

        for model in models:
            normalized = re.sub(
                r"[^a-z0-9]",
                "",
                model.lower(),
            )

            if normalized == wanted:
                chosen = model
                break

        # -------------------------------------------------
        # Fuzzy Qwen/instruct match
        # -------------------------------------------------

        if chosen is None:
            for model in models:
                normalized = re.sub(
                    r"[^a-z0-9]",
                    "",
                    model.lower(),
                )

                if (
                    wanted in normalized
                    or (
                        "qwen" in normalized
                        and "instruct"
                        in normalized
                    )
                ):
                    chosen = model
                    break

        # -------------------------------------------------
        # Final fallback
        # -------------------------------------------------

        if chosen is None:
            chosen = next(
                (
                    model
                    for model in models
                    if "embed"
                    not in model.lower()
                ),
                models[0],
            )

        with self._lock:
            self._resolved = chosen

        return chosen

    # =====================================================
    # Status
    # =====================================================

    def status(
        self,
    ) -> dict[str, Any]:

        try:
            models = self.list_models()

            model = self.resolve_model(
                force=True
            )

            return {
                "reachable": True,
                "model": model,
                "configured_model":
                    settings.lm_model,
                "available_models":
                    models,
                "base_url":
                    self.base_url,
                "message":
                    "LM Studio is connected.",
            }

        except LMStudioError as exc:
            return {
                "reachable": False,
                "model": None,
                "configured_model":
                    settings.lm_model,
                "available_models": [],
                "base_url":
                    self.base_url,
                "message": str(exc),
            }

    def is_reachable(
        self,
    ) -> bool:

        try:
            self.list_models()
            return True

        except LMStudioError:
            return False

    # =====================================================
    # Chat + diagnostics
    # =====================================================

    def chat_with_debug(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 300,
    ) -> dict[str, Any]:
        """
        Send one chat-completion request to LM Studio and
        preserve timing/token diagnostics.

        This method does not change the generated content.
        """

        model = self.resolve_model()

        message_chars = [
            {
                "role": message.get("role", ""),
                "chars": len(
                    message.get("content", "")
                ),
            }
            for message in messages
        ]

        prompt_chars = sum(
            item["chars"]
            for item in message_chars
        )

        request_started = time.perf_counter()

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=settings.lm_timeout,
            )

            api_time_ms = round(
                (
                    time.perf_counter()
                    - request_started
                )
                * 1000,
                2,
            )

            if response.status_code >= 400:
                raise LMStudioError(
                    "LM Studio returned HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:200]}"
                )

            data = response.json()

            choices = data.get(
                "choices",
                [],
            )

            if not choices:
                raise LMStudioError(
                    "Unexpected response format "
                    "from LM Studio."
                )

            first_choice = choices[0]

            content = (
                first_choice
                .get("message", {})
                .get("content", "")
                or ""
            ).strip()

            finish_reason = first_choice.get(
                "finish_reason"
            )

            usage = data.get(
                "usage",
                {},
            ) or {}

            prompt_tokens = usage.get(
                "prompt_tokens"
            )

            completion_tokens = usage.get(
                "completion_tokens"
            )

            total_tokens = usage.get(
                "total_tokens"
            )

            completion_tokens_per_api_second = None

            if (
                isinstance(completion_tokens, (int, float))
                and completion_tokens > 0
                and api_time_ms > 0
            ):
                completion_tokens_per_api_second = round(
                    completion_tokens
                    / (api_time_ms / 1000),
                    2,
                )

            debug = {
                "model": model,
                "api_time_ms": api_time_ms,
                "prompt_chars": prompt_chars,
                "message_chars": message_chars,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "finish_reason": finish_reason,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "completion_tokens_per_api_second":
                    completion_tokens_per_api_second,
            }

            return {
                "content": content,
                "debug": debug,
            }

        except httpx.ConnectError as exc:
            self._resolved = None

            raise LMStudioError(
                LM_DOWN
            ) from exc

        except httpx.TimeoutException as exc:
            raise LMStudioError(
                "LM Studio took too long to respond."
            ) from exc

        except (
            KeyError,
            IndexError,
            ValueError,
        ) as exc:
            raise LMStudioError(
                "Unexpected response format "
                "from LM Studio."
            ) from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 300,
    ) -> str:
        """
        Compatibility wrapper.

        Existing router/query-rewriter/analytics code can continue
        calling chat() and receiving a plain string.
        """

        result = self.chat_with_debug(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return result["content"]

    # =====================================================
    # RAG generation
    # =====================================================

    def generate_answer(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        history_text: str = "",
    ) -> dict[str, Any]:

        if not chunks:
            return {
                "answer": INSUFFICIENT,
                "sources": [],
                "citations_verified": True,
                "debug": {
                    "generation_ms": 0.0,
                    "history_chars": len(history_text),
                    "question_chars": len(question),
                    "context": {
                        "retrieved_chunks": 0,
                        "chunks_sent_to_llm": 0,
                        "original_context_chars": 0,
                        "context_chars": 0,
                        "context_budget_chars":
                            RAG_CONTEXT_MAX_CHARS,
                        "max_passages":
                            RAG_MAX_PASSAGES,
                        "truncated_passages": 0,
                    },
                    "lm_studio": None,
                },
            }

        # -------------------------------------------------
        # Select evidence for Qwen
        # -------------------------------------------------

        llm_chunks, context_debug = (
            select_context_chunks(
                chunks,
                max_chars=RAG_CONTEXT_MAX_CHARS,
                max_passages=RAG_MAX_PASSAGES,
            )
        )

        history = (
            f"Recent conversation:\n"
            f"{history_text}\n\n"
            if history_text
            else ""
        )

        context = format_context(
            llm_chunks
        )

        user = (
            f"{history}"
            f"Evidence passages:\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer directly and concisely. "
            "Use only the evidence above and cite "
            "the supporting [S#] passage."
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_RAG,
            },
            {
                "role": "user",
                "content": user,
            },
        ]

        generation_start = (
            time.perf_counter()
        )

        chat_result = self.chat_with_debug(
            messages,
            temperature=0.1,
            max_tokens=300,
        )

        raw = chat_result["content"]

        generation_ms = round(
            (
                time.perf_counter()
                - generation_start
            )
            * 1000,
            2,
        )

        # IMPORTANT:
        # Validate against llm_chunks, not all retrieved chunks.

        answer, sources, verified = (
            validate_citations(
                raw,
                llm_chunks,
            )
        )

        debug = {
            "generation_ms":
                generation_ms,
            "history_chars":
                len(history_text),
            "question_chars":
                len(question),
            "context":
                context_debug,
            "lm_studio":
                chat_result["debug"],
        }

        if INSUFFICIENT in answer:
            return {
                "answer": INSUFFICIENT,
                "sources": [],
                "citations_verified": True,
                "debug": debug,
            }

        return {
            "answer": answer,
            "sources": sources,
            "citations_verified":
                verified,
            "debug": debug,
        }

    # =====================================================
    # HYBRID evidence generation
    # =====================================================

    def generate_evidence(
        self,
        question: str,
        calculated: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:

        if not chunks:
            return {
                "answer": INSUFFICIENT,
                "sources": [],
                "citations_verified": True,
                "debug": {
                    "generation_ms": 0.0,
                    "question_chars": len(question),
                    "calculated_chars": len(calculated),
                    "context": {
                        "retrieved_chunks": 0,
                        "chunks_sent_to_llm": 0,
                        "original_context_chars": 0,
                        "context_chars": 0,
                        "context_budget_chars":
                            HYBRID_CONTEXT_MAX_CHARS,
                        "max_passages":
                            HYBRID_MAX_PASSAGES,
                        "truncated_passages": 0,
                    },
                    "lm_studio": None,
                },
            }

        # -------------------------------------------------
        # Select evidence for Qwen
        # -------------------------------------------------

        llm_chunks, context_debug = (
            select_context_chunks(
                chunks,
                max_chars=HYBRID_CONTEXT_MAX_CHARS,
                max_passages=HYBRID_MAX_PASSAGES,
            )
        )

        context = format_context(
            llm_chunks
        )

        user = (
            "Calculated result "
            "(exact, do not alter):\n"
            f"{calculated}\n\n"
            "Document passages:\n"
            f"{context}\n\n"
            f"User question: {question}\n"
            "Briefly explain only what the passages "
            "say that relates to the calculated result."
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_HYBRID,
            },
            {
                "role": "user",
                "content": user,
            },
        ]

        generation_start = (
            time.perf_counter()
        )

        chat_result = self.chat_with_debug(
            messages,
            temperature=0.1,
            max_tokens=250,
        )

        raw = chat_result["content"]

        generation_ms = round(
            (
                time.perf_counter()
                - generation_start
            )
            * 1000,
            2,
        )

        # Validate against only the passages Qwen saw.

        answer, sources, verified = (
            validate_citations(
                raw,
                llm_chunks,
            )
        )

        debug = {
            "generation_ms":
                generation_ms,
            "question_chars":
                len(question),
            "calculated_chars":
                len(calculated),
            "context":
                context_debug,
            "lm_studio":
                chat_result["debug"],
        }

        if INSUFFICIENT in answer:
            return {
                "answer": INSUFFICIENT,
                "sources": [],
                "citations_verified": True,
                "debug": debug,
            }

        return {
            "answer": answer,
            "sources": sources,
            "citations_verified":
                verified,
            "debug": debug,
        }