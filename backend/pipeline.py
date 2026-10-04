from __future__ import annotations



import calendar

import logging

import time

from typing import Any



from backend.analytics.engine import AnalyticsEngine

from backend.db import HistoryStore

from backend.documents import DocumentRegistry

from backend.rag.conversation import ConversationStore, Turn

from backend.rag.generator import (

    INSUFFICIENT,

    LM_DOWN,

    LMStudioClient,

    LMStudioError,

    build_sources,

)

from backend.rag.hybrid_retriever import HybridRetriever

from backend.rag.query_rewriter import QueryRewriter

from backend.router import QuestionRouter



logger = logging.getLogger(__name__)



DIRECTION_WORDS = {

    "decline": "decline decrease drop reason explanation",

    "growth": "growth increase reason explanation",

    "high": "strong performance reason",

    "low": "weak performance reason",

}





def elapsed_ms(start: float) -> float:

    """Return elapsed milliseconds rounded for readable debug output."""

    return round((time.perf_counter() - start) * 1000, 2)





def build_retrieval_query(

    question: str,

    focus: dict[str, Any] | None,

) -> str:

    """Turn an analytics result (e.g. 'Product B declined') into a document search query."""

    if not focus or not focus.get("entities"):

        return question



    parts = list(focus["entities"]) + [focus.get("metric") or ""]

    parts.append(

        DIRECTION_WORDS.get(

            focus.get("direction") or "",

            "explanation",

        )

    )



    periods = focus.get("periods") or []



    if periods:

        try:

            parts.append(

                calendar.month_name[

                    int(str(periods[-1])[-2:])

                ]

            )

        except (ValueError, IndexError):

            pass



    return " ".join(p for p in parts if p).strip()





def evidence_view(

    chunks: list[dict[str, Any]],

) -> list[dict[str, Any]]:

    return [

        {

            "file": c["file"],

            "page": c["page"],

            "score": c.get("score"),

            "text": c["text"][:500],

        }

        for c in chunks

    ]





class AnalystPipeline:

    def __init__(

        self,

        hybrid: HybridRetriever,

        llm: LMStudioClient,

        engine: AnalyticsEngine,

        conversations: ConversationStore,

        rewriter: QueryRewriter,

        router: QuestionRouter,

        registry: DocumentRegistry,

        history: HistoryStore,

    ) -> None:

        self.hybrid = hybrid

        self.llm = llm

        self.engine = engine



        self.conversations = conversations

        self.rewriter = rewriter

        self.router = router



        self.registry = registry

        self.history = history



    # ------------------------------------------------------------------ public



    def ask(

        self,

        question: str,

        session_id: str = "default",

        force_route: str | None = None,

    ) -> dict[str, Any]:



        pipeline_started = time.perf_counter()



        timing: dict[str, float] = {}



        # ---------------------------------------------------------- rewrite

        stage = time.perf_counter()



        rewrite = self.rewriter.rewrite(

            question,

            session_id,

        )



        timing["rewrite"] = elapsed_ms(stage)



        effective = rewrite["rewritten"]



        # ------------------------------------------------ conversation lookup

        stage = time.perf_counter()



        previous = self.conversations.turns(session_id)



        timing["conversation_lookup"] = elapsed_ms(stage)



        # ------------------------------------------------ source availability

        stage = time.perf_counter()



        has_docs = self.registry.has_kind(

            "pdf",

            "text",

        )



        has_tables = self.engine.has_tables()



        timing["source_check"] = elapsed_ms(stage)



        # ----------------------------------------------------------- routing

        stage = time.perf_counter()



        if force_route:

            from backend.router import RouteDecision



            decision = RouteDecision(

                force_route,

                "forced",

                "route forced by endpoint",

                {},

            )

        else:

            decision = self.router.route(

                effective,

                has_docs,

                has_tables,

                previous[-1].route if previous else None,

            )



        timing["routing"] = elapsed_ms(stage)



        # ------------------------------------------------------------ execute

        warnings: list[str] = []



        route_started = time.perf_counter()



        if not has_docs and not has_tables:

            result = self._empty(

                "Upload a PDF, CSV or Excel file first, then ask a question."

            )



        elif decision.route == "ANALYTICS":

            result = self._analytics(

                effective,

                warnings,

            )



        elif decision.route == "HYBRID":

            result = self._hybrid(

                effective,

                session_id,

                warnings,

            )



        else:

            result = self._rag(

                effective,

                session_id,

                warnings,

            )



        timing["route_execution"] = elapsed_ms(route_started)



        # ------------------------------------------------------- route timing

        route_timing = result.pop("_timing", {})



        if route_timing:

            timing.update(route_timing)



        # ------------------------------------------------------------- focus

        focus = result.pop("focus", None)



        entities = (

            (focus or {}).get("entities", [])

            if focus

            else []

        )



        # ------------------------------------------------ conversation store

        stage = time.perf_counter()



        self.conversations.add(

            session_id,

            Turn(

                question,

                result["answer"],

                decision.route,

                entities,

            ),

        )



        timing["conversation_store"] = elapsed_ms(stage)



        # -------------------------------------------------------- total so far

        latency = int(

            (time.perf_counter() - pipeline_started) * 1000

        )



        # ----------------------------------------------------------- history

        stage = time.perf_counter()



        self.history.log(

            session_id,

            question,

            decision.route,

            result["answer"],

            result["sources"],

            latency,

        )



        timing["history_log"] = elapsed_ms(stage)



        # True pipeline time including history logging

        timing["pipeline_total"] = elapsed_ms(

            pipeline_started

        )



        return {

            "question": question,

            "rewritten_question": effective,

            "route": decision.to_dict(),

            **result,

            "warnings": warnings + result.get("warnings", []),

            "latency_ms": round(timing["pipeline_total"]),

            "debug": {

                "rewrite": rewrite,

                "focus": focus,

                **result.get("debug", {}),

                "timing_ms": timing,

            },

        }



    # ------------------------------------------------------------------ routes



    @staticmethod

    def _empty(

        message: str,

    ) -> dict[str, Any]:



        return {

            "answer": message,

            "calculation": None,

            "data": None,

            "chart": None,

            "sources": [],

            "evidence": [],

            "debug": {},

            "_timing": {},

        }



    # --------------------------------------------------------------- analytics



    def _analytics(

        self,

        question: str,

        warnings: list[str],

    ) -> dict[str, Any]:



        started = time.perf_counter()



        stage = time.perf_counter()



        res = self.engine.analyze(question)



        analytics_ms = elapsed_ms(stage)



        total_ms = elapsed_ms(started)



        if not res.get("ok"):

            return {

                **self._empty(

                    "I couldn't calculate that from the uploaded data. "

                    + res.get("reason", "")

                ),

                "debug": {

                    "analytics_error": res.get("reason"),

                },

                "_timing": {

                    "analytics": analytics_ms,

                    "analytics_total": total_ms,

                },

            }



        return {

            "answer": res["answer"],

            "calculation": res["calculation"],

            "data": res["data"],

            "chart": res["chart"],

            "sources": [

                {

                    "file": res["source"],

                    "type": "data",

                }

            ],

            "evidence": [],

            "focus": res.get("focus"),

            "debug": {

                "analytics_intent": res["intent"],

                "steps": res["steps"],

                "table": res.get("table"),

            },

            "_timing": {

                "analytics": analytics_ms,

                "analytics_total": total_ms,

            },

        }



    # --------------------------------------------------------------------- RAG



    def _rag(

        self,

        question: str,

        session_id: str,

        warnings: list[str],

    ) -> dict[str, Any]:



        route_started = time.perf_counter()



        # --------------------------------------------------------- retrieval

        stage = time.perf_counter()



        retrieved = self.hybrid.retrieve(question)



        retrieval_ms = elapsed_ms(stage)



        chunks = retrieved["chunks"]



        base = {

            "calculation": None,

            "data": None,

            "chart": None,

            "evidence": evidence_view(chunks),

            "debug": {

                "retrieval": retrieved["debug"],

                "query": question,

            },

        }



        if not chunks:

            return {

                **base,

                "answer": INSUFFICIENT,

                "sources": [],

                "_timing": {

                    "retrieval": retrieval_ms,

                    "generation": 0.0,

                    "rag_total": elapsed_ms(route_started),

                },

            }



        # --------------------------------------------------------- generation

        generation_started = time.perf_counter()



        try:

            gen = self.llm.generate_answer(

                question,

                chunks,

                self.conversations.history_text(session_id),

            )



        except LMStudioError as exc:



            generation_ms = elapsed_ms(

                generation_started

            )



            warnings.append(str(exc))



            return {

                **base,

                "answer": (

                    f"{exc} Showing the retrieved passages instead."

                ),

                "sources": build_sources(chunks),

                "_timing": {

                    "retrieval": retrieval_ms,

                    "generation": generation_ms,

                    "rag_total": elapsed_ms(route_started),

                },

            }



        generation_ms = elapsed_ms(

            generation_started

        )



        if not gen["citations_verified"]:

            warnings.append(

                "The model did not cite specific passages; "

                "all retrieved passages are listed as sources."

            )



        # generator.py may already have its own timer.

        generator_debug = gen.get("debug", {})



        return {

            **base,

            "answer": gen["answer"],

            "sources": gen["sources"],

            "debug": {

                **base.get("debug", {}),

                "generator": generator_debug,

            },

            "_timing": {

                "retrieval": retrieval_ms,

                "generation": generation_ms,

                "generator_internal": generator_debug.get(

                    "generation_ms",

                    generation_ms,

                ),

                "rag_total": elapsed_ms(route_started),

            },

        }



    # ------------------------------------------------------------------ hybrid



    def _hybrid(

        self,

        question: str,

        session_id: str,

        warnings: list[str],

    ) -> dict[str, Any]:



        route_started = time.perf_counter()



        # --------------------------------------------------------- analytics

        stage = time.perf_counter()



        res = self.engine.analyze(question)



        analytics_ms = elapsed_ms(stage)



        if not res.get("ok"):



            warnings.append(

                "Calculation failed: "

                + res.get("reason", "unknown reason")

                + " Falling back to documents."

            )



            rag_result = self._rag(

                question,

                session_id,

                warnings,

            )



            fallback_timing = rag_result.get(

                "_timing",

                {},

            )



            rag_result["_timing"] = {

                "analytics": analytics_ms,

                **fallback_timing,

                "hybrid_total": elapsed_ms(route_started),

            }



            return rag_result



        # ------------------------------------------------ retrieval query

        stage = time.perf_counter()



        query = build_retrieval_query(

            question,

            res.get("focus"),

        )



        query_build_ms = elapsed_ms(stage)



        # --------------------------------------------------------- retrieval

        stage = time.perf_counter()



        retrieved = self.hybrid.retrieve(query)



        retrieval_ms = elapsed_ms(stage)



        chunks = retrieved["chunks"]



        calculated = res["answer"]



        generation_ms = 0.0

        generator_internal_ms = 0.0

        generator_debug: dict[str, Any] = {}



        # --------------------------------------------------------- generation

        if not chunks:



            evidence_text = INSUFFICIENT

            doc_sources = []



        else:



            generation_started = time.perf_counter()



            try:

                gen = self.llm.generate_evidence(

                    question,

                    calculated,

                    chunks,

                )



                generation_ms = elapsed_ms(

                    generation_started

                )



                generator_debug = gen.get("debug", {})

                generator_internal_ms = generator_debug.get(

                    "generation_ms",

                    generation_ms,

                )



                evidence_text = gen["answer"]

                doc_sources = gen["sources"]



                if (

                    not gen["citations_verified"]

                    and gen["sources"]

                ):

                    warnings.append(

                        "The model did not cite specific passages; "

                        "all retrieved passages are listed."

                    )



            except LMStudioError as exc:



                generation_ms = elapsed_ms(

                    generation_started

                )



                generator_internal_ms = generation_ms



                warnings.append(str(exc))



                evidence_text = (

                    f"{LM_DOWN} "

                    "The retrieved passages are listed below."

                )



                doc_sources = build_sources(chunks)



        # ---------------------------------------------------------- response

        answer = (

            f"CALCULATED:\n"

            f"{calculated}\n\n"

            f"DOCUMENT EVIDENCE:\n"

            f"{evidence_text}"

        )



        sources = [

            {

                "file": res["source"],

                "type": "data",

            }

        ] + doc_sources



        return {

            "answer": answer,

            "calculation": res["calculation"],

            "data": res["data"],

            "chart": res["chart"],

            "sources": sources,

            "evidence": evidence_view(chunks),

            "focus": res.get("focus"),

            "debug": {

                "analytics_intent": res["intent"],

                "steps": res["steps"],

                "retrieval_query": query,

                "retrieval": retrieved["debug"],

                "table": res.get("table"),

                "generator": generator_debug,

            },

            "_timing": {

                "analytics": analytics_ms,

                "retrieval_query_build": query_build_ms,

                "retrieval": retrieval_ms,

                "generation": generation_ms,

                "generator_internal": generator_internal_ms,

                "hybrid_total": elapsed_ms(route_started),

            },

        }