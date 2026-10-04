from __future__ import annotations
import re
from typing import Any

from backend.rag.conversation import ConversationStore
from backend.rag.generator import LMStudioClient, LMStudioError


# ---------------------------------------------------------
# Words that often refer to previous conversation context
# ---------------------------------------------------------

REFERENCE_WORDS = {
    "it",
    "its",
    "that",
    "this",
    "they",
    "them",
    "their",
    "those",
    "these",
    "he",
    "she",
    "his",
    "her",
}


# ---------------------------------------------------------
# Very short phrases that are clearly conversational
# follow-ups.
# ---------------------------------------------------------

FOLLOWUP_PHRASES = {
    "why",
    "why?",
    "how",
    "how?",
    "explain",
    "explain more",
    "tell me more",
    "more",
    "continue",
    "what about it",
    "what about that",
    "why is that",
    "why so",
    "how so",
    "how does it work",
    "what happened",
    "what happened next",
    "and why",
    "and how",
    "what about this",
    "what about them",
}


class QueryRewriter:
    """
    Converts genuinely context-dependent follow-up questions into
    standalone questions.

    Example:

        Previous:
            Which product performed worst?

        Current:
            Why?

        Rewritten:
            Why did Product B perform worst?

    But:

        Current:
            What is distance vector?

        remains:

            What is distance vector?
    """

    def __init__(
        self,
        llm: LMStudioClient,
        conversations: ConversationStore
    ) -> None:

        self.llm = llm
        self.conversations = conversations

    # ---------------------------------------------------------
    # Normalize question
    # ---------------------------------------------------------

    @staticmethod
    def _normalize(question: str) -> str:

        return re.sub(
            r"\s+",
            " ",
            question.strip().lower()
        )

    # ---------------------------------------------------------
    # Extract words
    # ---------------------------------------------------------

    @staticmethod
    def _words(question: str) -> list[str]:

        return re.findall(
            r"[a-zA-Z0-9']+",
            question.lower()
        )

    # ---------------------------------------------------------
    # Detect clearly self-contained questions
    # ---------------------------------------------------------

    @classmethod
    def is_self_contained(
        cls,
        question: str
    ) -> bool:
        """
        Determine whether the question appears meaningful without
        conversation history.

        This deliberately prefers NOT rewriting when uncertain.
        """

        normalized = cls._normalize(question)

        words = cls._words(question)

        if not words:
            return True

        # -----------------------------------------------------
        # Exact conversational follow-ups
        # -----------------------------------------------------

        if normalized in FOLLOWUP_PHRASES:
            return False

        # -----------------------------------------------------
        # Explicit reference pronouns usually require context
        # -----------------------------------------------------

        word_set = set(words)

        if REFERENCE_WORDS & word_set:

            # Example:
            #
            # "What is TCP and how does it work?"
            #
            # Although "it" exists, TCP is explicitly named.
            # Longer questions containing meaningful nouns should
            # generally remain untouched.

            if len(words) <= 6:
                return False

        # -----------------------------------------------------
        # Common standalone question patterns
        # -----------------------------------------------------

        standalone_patterns = [

            r"^what\s+(?:is|are|was|were)\s+.+",

            r"^who\s+(?:is|are|was|were)\s+.+",

            r"^where\s+(?:is|are|was|were)\s+.+",

            r"^when\s+(?:is|are|was|were|did|does)\s+.+",

            r"^why\s+(?:is|are|was|were|does|do|did)\s+.+",

            r"^how\s+(?:is|are|was|were|does|do|did|can|could)\s+.+",

            r"^define\s+.+",

            r"^explain\s+.+",

            r"^describe\s+.+",

            r"^compare\s+.+",

            r"^difference\s+between\s+.+",

            r"^what\s+does\s+.+",

            r"^what\s+causes\s+.+",

            r"^what\s+caused\s+.+",

            r"^tell\s+me\s+about\s+.+",
        ]

        for pattern in standalone_patterns:

            if re.match(
                pattern,
                normalized
            ):

                # Don't classify:
                # "What is it?"
                # as standalone.

                meaningful_words = [
                    word
                    for word in words
                    if word not in {
                        "what",
                        "is",
                        "are",
                        "was",
                        "were",
                        "why",
                        "how",
                        "does",
                        "do",
                        "did",
                        "the",
                        "a",
                        "an",
                        "of",
                        "about",
                    }
                ]

                if meaningful_words:

                    if not (
                        len(meaningful_words) == 1
                        and meaningful_words[0]
                        in REFERENCE_WORDS
                    ):
                        return True

        # -----------------------------------------------------
        # Questions with several meaningful words are normally
        # independent.
        # -----------------------------------------------------

        stop_words = {
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "can",
            "could",
            "would",
            "should",
            "the",
            "a",
            "an",
            "of",
            "to",
            "for",
            "in",
            "on",
            "and",
            "or",
        }

        meaningful = [
            word
            for word in words
            if word not in stop_words
            and word not in REFERENCE_WORDS
        ]

        if len(meaningful) >= 2:
            return True

        return False

    # ---------------------------------------------------------
    # Decide whether rewriting is needed
    # ---------------------------------------------------------

    @classmethod
    def needs_rewrite(
        cls,
        question: str
    ) -> bool:
        """
        Rewrite only questions that clearly depend on previous
        conversational context.
        """

        normalized = cls._normalize(question)

        words = cls._words(question)

        if not words:
            return False

        # Exact follow-up phrases.
        if normalized in FOLLOWUP_PHRASES:
            return True

        # Self-contained questions must never be rewritten.
        if cls.is_self_contained(question):
            return False

        # Very short questions containing a reference pronoun.
        #
        # Examples:
        #   "Why did it fail?"
        #   "What about that?"
        #   "How does it work?"
        #
        if (
            len(words) <= 8
            and bool(
                REFERENCE_WORDS
                & set(words)
            )
        ):
            return True

        return False

    # ---------------------------------------------------------
    # Rule fallback
    # ---------------------------------------------------------

    def _fallback_rewrite(
        self,
        question: str,
        session_id: str
    ) -> str:
        """
        Create a simple deterministic rewrite when the LLM
        cannot be reached.
        """

        entities = self.conversations.last_entities(
            session_id
        )

        if not entities:
            return question

        normalized_question = question.strip()

        # Don't append an entity that already appears.
        for entity in entities:

            if (
                entity.lower()
                in normalized_question.lower()
            ):
                return question

        entity_text = ", ".join(
            entities[:3]
        )

        clean_question = (
            normalized_question
            .rstrip("?")
            .strip()
        )

        return (
            f"{clean_question} "
            f"(regarding {entity_text})?"
        )

    # ---------------------------------------------------------
    # Rewrite
    # ---------------------------------------------------------

    def rewrite(
        self,
        question: str,
        session_id: str
    ) -> dict[str, Any]:
        """
        Return original + rewritten versions.

        The original question is never modified.
        """

        original = question.strip()

        # -----------------------------------------------------
        # No session = no conversational rewrite
        # -----------------------------------------------------

        if not session_id:

            return {
                "original": original,
                "rewritten": original,
                "changed": False,
                "method": "none",
            }

        # -----------------------------------------------------
        # No conversation history = nothing to resolve
        # -----------------------------------------------------

        if not self.conversations.has_history(
            session_id
        ):

            return {
                "original": original,
                "rewritten": original,
                "changed": False,
                "method": "none",
            }

        # -----------------------------------------------------
        # IMPORTANT:
        # Independent questions bypass the LLM completely.
        # -----------------------------------------------------

        if not self.needs_rewrite(original):

            return {
                "original": original,
                "rewritten": original,
                "changed": False,
                "method": "none",
            }

        history = self.conversations.history_text(
            session_id
        )

        if not history:

            return {
                "original": original,
                "rewritten": original,
                "changed": False,
                "method": "none",
            }

        fallback = self._fallback_rewrite(
            original,
            session_id
        )

        # -----------------------------------------------------
        # LLM rewrite
        # -----------------------------------------------------

        prompt = f"""
You rewrite conversational follow-up questions into standalone questions.

IMPORTANT RULES:

1. Use conversation history ONLY to resolve missing references.
2. Do NOT answer the question.
3. Do NOT change the topic.
4. Do NOT replace a new question with an older question.
5. Preserve the user's current intent.
6. If the question is already understandable by itself,
   return it unchanged.
7. Return ONLY one question.
8. Do not include explanations.

Conversation:
{history}

Current user question:
{original}

Standalone question:
""".strip()

        try:

            output = self.llm.chat(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
                max_tokens=80,
            )

            output = output.strip()

            if output:

                rewritten = (
                    output
                    .splitlines()[0]
                    .strip()
                    .strip('"')
                    .strip("'")
                )

                # Basic safety validation.
                if (
                    3 <= len(rewritten) <= 300
                ):

                    changed = (
                        rewritten.lower()
                        != original.lower()
                    )

                    return {
                        "original": original,
                        "rewritten": rewritten,
                        "changed": changed,
                        "method": "llm",
                    }

        except LMStudioError:

            pass

        # -----------------------------------------------------
        # LLM unavailable → deterministic fallback
        # -----------------------------------------------------

        changed = (
            fallback.lower()
            != original.lower()
        )

        return {
            "original": original,
            "rewritten": fallback,
            "changed": changed,
            "method": (
                "rule"
                if changed
                else "none"
            ),
        }