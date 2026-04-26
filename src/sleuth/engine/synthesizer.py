"""Streaming synthesizer — converts chunks into a token + citation event stream.

Calls the LLM with the user query + retrieved chunks, streams ``ThinkingEvent``
(when the LLM supports reasoning), ``TokenEvent``, and ``CitationEvent``s.
Yields a ``DoneEvent`` as the final event.

Returns ``Result`` via the ``last_result`` property after the generator completes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from pydantic import BaseModel

from sleuth.events import CitationEvent, DoneEvent, ThinkingEvent, TokenEvent
from sleuth.llm.base import LLMClient, Message, ReasoningDelta, Stop, TextDelta
from sleuth.types import Chunk, Result, RunStats

logger = logging.getLogger("sleuth.engine.synthesizer")

_SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Answer the user's question based on the provided search results. "
    "Be concise and cite your sources."
)


def _build_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no search results available)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        loc = chunk.source.location
        title = chunk.source.title or loc
        parts.append(f"[{i}] {title}\n{chunk.text}")
    return "\n\n".join(parts)


SynthEvent = ThinkingEvent | TokenEvent | CitationEvent | DoneEvent


class Synthesizer:
    """Streaming LLM synthesizer.

    Args:
        llm: Any object satisfying the ``LLMClient`` Protocol.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._last_result: Result | None = None  # type: ignore[type-arg]

    @property
    def last_result(self) -> Result | None:  # type: ignore[type-arg]
        """The final ``Result`` built from the completed stream, or ``None`` before done."""
        return self._last_result

    async def synthesize(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        history: list[Message],
        stats_start_ms: float,
        schema: type[BaseModel] | None = None,
        backends_called: list[str] | None = None,
        cache_hits: dict[str, int] | None = None,
    ) -> AsyncIterator[SynthEvent]:
        """Stream synthesis events and yield a ``DoneEvent`` last.

        This is an async generator — iterate with ``async for``.

        Args:
            query: The user's search question.
            chunks: Merged, deduped chunks from the executor.
            history: Prior conversation turns as ``Message`` objects.
            stats_start_ms: ``time.monotonic() * 1000`` at run start.
            schema: Optional Pydantic schema for structured output.
            backends_called: Backend names that contributed chunks.
            cache_hits: Cache namespace → hit count dict for ``RunStats``.
        """
        _backends_called = backends_called or []
        _cache_hits = cache_hits or {}

        context = _build_context(chunks)
        messages: list[Message] = [
            Message(role="system", content=_SYSTEM_PROMPT),
            *history,
            Message(
                role="user",
                content=f"Question: {query}\n\nSearch results:\n{context}",
            ),
        ]

        text_parts: list[str] = []
        tokens_out = 0
        first_token_ms: int | None = None

        async for chunk in self._llm.stream(messages, schema=schema):
            now_ms = int(time.monotonic() * 1000)
            if isinstance(chunk, ReasoningDelta):
                if self._llm.supports_reasoning:
                    yield ThinkingEvent(type="thinking", text=chunk.text)
            elif isinstance(chunk, TextDelta):
                if first_token_ms is None:
                    first_token_ms = now_ms - int(stats_start_ms)
                text_parts.append(chunk.text)
                tokens_out += 1
                yield TokenEvent(type="token", text=chunk.text)
            elif isinstance(chunk, Stop):
                break

        # Emit citations for each chunk that contributed
        for idx, c in enumerate(chunks):
            yield CitationEvent(type="citation", index=idx, source=c.source)

        full_text = "".join(text_parts)
        elapsed_ms = int(time.monotonic() * 1000 - stats_start_ms)

        stats = RunStats(
            latency_ms=elapsed_ms,
            first_token_ms=first_token_ms,
            tokens_in=len(messages),  # approximation (message count, not real tokens)
            tokens_out=tokens_out,
            cache_hits=_cache_hits,
            backends_called=_backends_called,
        )

        self._last_result = Result(
            text=full_text,
            citations=[c.source for c in chunks],
            stats=stats,
        )

        yield DoneEvent(type="done", stats=stats)
