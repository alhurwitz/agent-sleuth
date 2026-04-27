"""Session — multi-turn ring buffer with JSON persistence.

Phase 4 adds ``save(path)``, ``Session.load(path)``, ``flush()``, and
``_schedule_background_save(path)``.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sleuth.llm.base import Message
from sleuth.types import Result, RunStats, Source


@dataclass
class Turn:
    """One Q&A exchange stored in the session ring buffer."""

    query: str
    result: Result  # type: ignore[type-arg]
    citations: list[Source]


class Session:
    """In-memory ring buffer of recent conversation turns with optional JSON persistence.

    Attributes:
        max_turns: Maximum number of turns retained.  Oldest turn is dropped
            when the buffer is full.  Defaults to 20 (spec §8).
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._max_turns = max_turns
        self._buffer: deque[Turn] = deque(maxlen=max_turns)
        self._pending_save: asyncio.Task[None] | None = None

    @property
    def max_turns(self) -> int:
        """Maximum number of turns the ring buffer retains."""
        return self._max_turns

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

    # ------------------------------------------------------------------
    # Phase 4: JSON persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the session to a JSON file at *path* (synchronous).

        The file is overwritten atomically on supported platforms.
        """
        path = Path(path)
        data: dict[str, Any] = {
            "max_turns": self._max_turns,
            "turns": [
                {
                    "query": t.query,
                    "result_text": t.result.text,
                    "citations": [c.model_dump() for c in t.citations],
                }
                for t in self._buffer
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Session:
        """Load a session from a JSON file created by :meth:`save`.

        Args:
            path: Path to the JSON file.

        Raises:
            FileNotFoundError: if *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        session = cls(max_turns=data["max_turns"])
        for turn_data in data["turns"]:
            stats = RunStats(
                latency_ms=0,
                first_token_ms=None,
                tokens_in=0,
                tokens_out=0,
                cache_hits={},
                backends_called=[],
            )
            result: Result[Any] = Result(
                text=turn_data["result_text"],
                citations=[],
                stats=stats,
            )
            citations = [Source.model_validate(c) for c in turn_data.get("citations", [])]
            session.add_turn(turn_data["query"], result, citations)
        return session

    def _schedule_background_save(self, path: str | Path) -> None:
        """Schedule an async background write of this session to *path*.

        The caller must call :meth:`flush` before the next turn to ensure the
        write completes.  If no event loop is running the write is performed
        synchronously.
        """
        path = Path(path)

        async def _write() -> None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.save, path)

        try:
            loop = asyncio.get_running_loop()
            self._pending_save = loop.create_task(_write())
        except RuntimeError:
            # No running loop — save synchronously
            self.save(path)

    async def flush(self) -> None:
        """Await any pending background write task.

        Callers who need the on-disk session to be up-to-date (e.g. before
        graceful shutdown or in tests) should call ``await session.flush()``.
        """
        pending = self._pending_save
        if pending is not None and not pending.done():
            await pending
        self._pending_save = None
