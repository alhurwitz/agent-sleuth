"""Session — multi-turn ring buffer for conversation coherence.

Phase 4 will add ``session.save(path)`` / ``Session.load(path)`` and
``await session.flush()``.  The ``Session`` class and ``Turn`` dataclass
are stable; Phase 4 only extends them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from sleuth.llm.base import Message
from sleuth.types import Result, Source


@dataclass
class Turn:
    """One Q&A exchange stored in the session ring buffer."""

    query: str
    result: Result  # type: ignore[type-arg]
    citations: list[Source]


class Session:
    """In-memory ring buffer of recent conversation turns.

    Attributes:
        max_turns: Maximum number of turns retained.  Oldest turn is dropped
            when the buffer is full.  Defaults to 20 (spec §8).

    Note:
        Phase 4 adds ``save(path)``, ``Session.load(path)``, and
        ``flush()`` for persistence.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._max_turns = max_turns
        self._buffer: deque[Turn] = deque(maxlen=max_turns)

    @property
    def turns(self) -> list[Turn]:
        """Snapshot of current turns, oldest first."""
        return list(self._buffer)

    def add_turn(self, query: str, result: Result, citations: list[Source]) -> None:  # type: ignore[type-arg]
        """Append a turn; oldest turn is evicted when buffer is full."""
        self._buffer.append(Turn(query=query, result=result, citations=citations))

    def as_messages(self) -> list[Message]:
        """Return turns as alternating user/assistant Message objects.

        Suitable for passing to ``LLMClient.stream()`` as conversation history.
        """
        messages: list[Message] = []
        for turn in self._buffer:
            messages.append(Message(role="user", content=turn.query))
            messages.append(Message(role="assistant", content=turn.result.text))
        return messages
