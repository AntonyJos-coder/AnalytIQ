from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.rag.generator import LMStudioClient, LMStudioError


# =========================================================
# Routing signals
# =========================================================

DATA_OPS = re.compile(
    r"\b("
    r"total|sum|average|avg|mean|count|how many|number of|"
    r"highest|lowest|best[- ]selling|worst|top|bottom|"
    r"trend\w*|monthly|growth|grew|"
    r"declin\w*|decreas\w*|increas\w*|"
    r"drop\w*|fell|fall\w*|rose|rise|"
    r"percent\w*|compare|comparison|versus|vs|"
    r"maximum|minimum|largest|biggest|smallest|"
    r"weakest|strongest|"
    r"by (?:region|product|month|category|customer)|"
    r"per (?:region|product|month)"
    r")\b|%"
)


DATA_NOUNS = re.compile(
    r"\b("
    r"revenue|sales|units|profit|margin|"
    r"cost|costs|quantity|orders|income|expenses?"
    r")\b"
)


DOC_STRONG = re.compile(
    r"\b("
    r"report|reports|document|documents|pdf|"
    r"management|according|"
    r"says?|said|mention\w*|summar\w+|"
    r"discuss\w*|states?|stated|memo|minutes"
    r")\b"
)


DOC_WHY = re.compile(
    r"\b("
    r"why|reason\w*|cause[ds]?|"
    r"explain\w*|explanation|because|"
    r"attribut\w*|behind"
    r")\b"
)


# =========================================================
# Route result
# =========================================================

@dataclass
class RouteDecision:
    route: str
    method: str
    reason: str
    signals: dict[str, bool] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================================================
# Router
# =========================================================

class QuestionRouter:

    def __init__(
        self,
        llm: LMStudioClient | None = None
    ) -> None:

        self.llm = llm

    # -----------------------------------------------------
    # Public routing method
    # -----------------------------------------------------

    def route(
        self,
        question: str,
        has_docs: bool = True,
        has_tables: bool = True,
        previous_route: str | None = None,
    ) -> RouteDecision:

        question = question.strip()

        ql = question.lower()

        # -------------------------------------------------
        # Detect routing signals
        # -------------------------------------------------

        sig = {

            "data_ops":
                bool(DATA_OPS.search(ql)),

            "data_nouns":
                bool(DATA_NOUNS.search(ql)),

            "doc_strong":
                bool(DOC_STRONG.search(ql)),

            "doc_why":
                bool(DOC_WHY.search(ql)),

            "has_docs":
                has_docs,

            "has_tables":
                has_tables,
        }

        # =================================================
        # STEP 1
        # Availability-first routing
        # =================================================
        #
        # If only one type of data exists, there is no need
        # to wake up the LLM just to classify the question.
        # =================================================

        if has_docs and not has_tables:

            return RouteDecision(
                route="RAG",
                method="availability",
                reason="documents available; no tables uploaded",
                signals=sig,
            )

        if has_tables and not has_docs:

            return RouteDecision(
                route="ANALYTICS",
                method="availability",
                reason="tables available; no documents uploaded",
                signals=sig,
            )

        # No usable data source.
        #
        # Keep RAG as the neutral route. The higher-level
        # request handler can return the appropriate
        # "upload data" message.

        if not has_docs and not has_tables:

            return RouteDecision(
                route="RAG",
                method="availability",
                reason="no documents or tables uploaded",
                signals=sig,
            )

        # =================================================
        # STEP 2
        # Both documents AND tables are available.
        # Use deterministic routing rules.
        # =================================================

        # Example:
        #
        # "Which product declined most and what does the
        # management report say caused it?"
        #
        if (
            sig["data_ops"]
            and (
                sig["doc_strong"]
                or sig["doc_why"]
            )
        ):

            return RouteDecision(
                route="HYBRID",
                method="rules",
                reason=(
                    "calculation words + "
                    "explanation/document words"
                ),
                signals=sig,
            )

        # -------------------------------------------------
        # Pure calculation
        # -------------------------------------------------

        if (
            sig["data_ops"]
            and not sig["doc_strong"]
            and not sig["doc_why"]
        ):

            return RouteDecision(
                route="ANALYTICS",
                method="rules",
                reason="calculation words only",
                signals=sig,
            )

        # -------------------------------------------------
        # Explicit document request
        # -------------------------------------------------

        if (
            sig["doc_strong"]
            and not sig["data_ops"]
        ):

            return RouteDecision(
                route="RAG",
                method="rules",
                reason="explicit document reference",
                signals=sig,
            )

        # -------------------------------------------------
        # Metric + explanation
        #
        # Example:
        # "Why did revenue decline?"
        # -------------------------------------------------

        if (
            sig["data_nouns"]
            and sig["doc_why"]
        ):

            return RouteDecision(
                route="HYBRID",
                method="rules",
                reason="metric noun + why/explain",
                signals=sig,
            )

        # -------------------------------------------------
        # Metric only
        #
        # Example:
        # "Show revenue"
        # -------------------------------------------------

        if (
            sig["data_nouns"]
            and not sig["doc_why"]
            and not sig["doc_strong"]
        ):

            return RouteDecision(
                route="ANALYTICS",
                method="rules",
                reason="metric noun only",
                signals=sig,
            )

        # -------------------------------------------------
        # Conversational explanation following analytics
        #
        # Previous:
        #   Product B declined the most.
        #
        # Current:
        #   Why?
        # -------------------------------------------------

        if (
            sig["doc_why"]
            and previous_route
            in {"ANALYTICS", "HYBRID"}
        ):

            return RouteDecision(
                route="HYBRID",
                method="rules",
                reason=(
                    "'why' follow-up to an "
                    "analytics answer"
                ),
                signals=sig,
            )

        # -------------------------------------------------
        # General explanation question
        # -------------------------------------------------

        if sig["doc_why"]:

            return RouteDecision(
                route="RAG",
                method="rules",
                reason="explanation question",
                signals=sig,
            )

        # =================================================
        # STEP 3
        # Genuine ambiguity.
        #
        # BOTH documents and tables exist at this point.
        # Only now may we use the local LLM.
        # =================================================

        return self._ambiguous(
            question=question,
            has_docs=has_docs,
            has_tables=has_tables,
            sig=sig,
        )

    # =====================================================
    # Ambiguous routing
    # =====================================================

    def _ambiguous(
        self,
        question: str,
        has_docs: bool,
        has_tables: bool,
        sig: dict[str, bool],
    ) -> RouteDecision:

        # -------------------------------------------------
        # Extra safety:
        # Never call LLM when availability already decides.
        # -------------------------------------------------

        if has_docs and not has_tables:

            return RouteDecision(
                route="RAG",
                method="availability",
                reason=(
                    "ambiguous question; "
                    "only documents available"
                ),
                signals=sig,
            )

        if has_tables and not has_docs:

            return RouteDecision(
                route="ANALYTICS",
                method="availability",
                reason=(
                    "ambiguous question; "
                    "only tables available"
                ),
                signals=sig,
            )

        # -------------------------------------------------
        # LLM classification
        #
        # Only used when:
        #
        # documents = yes
        # tables    = yes
        #
        # AND deterministic rules could not decide.
        # -------------------------------------------------

        if (
            self.llm is not None
            and has_docs
            and has_tables
        ):

            try:

                prompt = (
                    "Classify the user's question into exactly "
                    "one category.\n\n"

                    "RAG = answer primarily from uploaded "
                    "documents/PDF text.\n"

                    "ANALYTICS = requires calculation, "
                    "aggregation, filtering, comparison, or "
                    "analysis of CSV/Excel data.\n"

                    "HYBRID = requires BOTH numerical/table "
                    "analysis AND documentary explanation.\n\n"

                    "Return ONLY one word:\n"
                    "RAG\n"
                    "ANALYTICS\n"
                    "HYBRID\n\n"

                    f"Question: {question}"
                )

                output = self.llm.chat(
                    [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0.0,
                    max_tokens=5,
                )

                output = (
                    output
                    .strip()
                    .upper()
                )

                # Exact result preferred.
                if output == "HYBRID":

                    return RouteDecision(
                        route="HYBRID",
                        method="llm",
                        reason=(
                            "ambiguous question classified "
                            "by local LLM"
                        ),
                        signals=sig,
                    )

                if output == "ANALYTICS":

                    return RouteDecision(
                        route="ANALYTICS",
                        method="llm",
                        reason=(
                            "ambiguous question classified "
                            "by local LLM"
                        ),
                        signals=sig,
                    )

                if output == "RAG":

                    return RouteDecision(
                        route="RAG",
                        method="llm",
                        reason=(
                            "ambiguous question classified "
                            "by local LLM"
                        ),
                        signals=sig,
                    )

            except LMStudioError:

                pass

        # =================================================
        # Final deterministic fallback
        # =================================================

        # Both data types exist, but the question contains
        # no strong calculation signal.
        #
        # RAG is safer because it does not invent a numerical
        # operation that the user never requested.

        return RouteDecision(
            route="RAG",
            method="default",
            reason=(
                "ambiguous question; no calculation "
                "signal detected, defaulting to documents"
            ),
            signals=sig,
        )