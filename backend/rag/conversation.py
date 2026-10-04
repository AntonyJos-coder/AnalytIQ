
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from backend.config import settings


MAX_SESSIONS = 100


@dataclass
class Turn:
    """
    Represents one completed conversation turn.
    """

    question: str
    answer: str
    route: str = ""
    entities: list[str] = field(default_factory=list)


class ConversationStore:
    """
    Thread-safe in-memory conversation store.

    Conversation history is separated by session_id and limited
    to prevent unlimited memory growth.
    """

    def __init__(self, window: int | None = None) -> None:
        self.window = window or settings.history_window

        self._sessions: dict[str, list[Turn]] = {}

        self._lock = threading.Lock()

    # ---------------------------------------------------------
    # Get conversation turns
    # ---------------------------------------------------------

    def turns(self, session_id: str) -> list[Turn]:
        """
        Return recent conversation turns for a session.
        """

        if not session_id:
            return []

        with self._lock:
            history = self._sessions.get(session_id, [])

            return list(history[-self.window :])

    # ---------------------------------------------------------
    # Add conversation turn
    # ---------------------------------------------------------

    def add(self, session_id: str, turn: Turn) -> None:
        """
        Add a completed turn to conversation history.
        """

        if not session_id:
            return

        with self._lock:

            history = self._sessions.setdefault(
                session_id,
                []
            )

            history.append(turn)

            # Keep some additional history internally,
            # while turns() exposes only the configured window.
            max_internal_turns = max(self.window * 2, 2)

            if len(history) > max_internal_turns:
                del history[:-max_internal_turns]

            # Prevent unlimited number of sessions.
            while len(self._sessions) > MAX_SESSIONS:

                oldest_session = next(
                    iter(self._sessions)
                )

                self._sessions.pop(
                    oldest_session,
                    None
                )

    # ---------------------------------------------------------
    # Latest turn
    # ---------------------------------------------------------

    def last_turn(
        self,
        session_id: str
    ) -> Turn | None:
        """
        Return the latest conversation turn.
        """

        if not session_id:
            return None

        with self._lock:

            history = self._sessions.get(
                session_id,
                []
            )

            if not history:
                return None

            return history[-1]

    # ---------------------------------------------------------
    # Conversation history for LLM
    # ---------------------------------------------------------

    def history_text(
        self,
        session_id: str,
        max_chars: int = 1500
    ) -> str:
        """
        Convert recent conversation history into text that can
        optionally be supplied to the query rewriter or LLM.
        """

        if not session_id:
            return ""

        lines: list[str] = []

        for turn in self.turns(session_id):

            question = turn.question.strip()

            answer = turn.answer.strip()

            # Limit old answers so large generated responses
            # do not consume the entire context.
            answer = answer[:300]

            lines.append(
                f"User: {question}\n"
                f"Assistant: {answer}"
            )

        history = "\n".join(lines)

        return history[-max_chars:]

    # ---------------------------------------------------------
    # Last known entities
    # ---------------------------------------------------------

    def last_entities(
        self,
        session_id: str
    ) -> list[str]:
        """
        Return entities from the most recent turn containing them.
        """

        if not session_id:
            return []

        for turn in reversed(
            self.turns(session_id)
        ):

            if turn.entities:
                return list(turn.entities)

        return []

    # ---------------------------------------------------------
    # Check whether session has history
    # ---------------------------------------------------------

    def has_history(
        self,
        session_id: str
    ) -> bool:
        """
        Return True if conversation history exists.
        """

        if not session_id:
            return False

        with self._lock:

            return bool(
                self._sessions.get(session_id)
            )

    # ---------------------------------------------------------
    # Clear one conversation
    # ---------------------------------------------------------

    def clear(
        self,
        session_id: str
    ) -> None:
        """
        Completely remove conversation history for one session.

        This should be called when the frontend user clicks
        'New conversation'.
        """

        if not session_id:
            return

        with self._lock:

            self._sessions.pop(
                session_id,
                None
            )

    # ---------------------------------------------------------
    # Debug helper
    # ---------------------------------------------------------

    def session_count(self) -> int:
        """
        Return number of active in-memory sessions.
        """

        with self._lock:
            return len(self._sessions)