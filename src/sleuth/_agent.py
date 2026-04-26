"""Sleuth — the top-level agent class.

Wires Router → Executor → Synthesizer into a single ``aask`` async generator
and provides an ``ask`` sync wrapper.

Design notes (per spec §4):
- ``Sleuth(...)`` is constructed once; ``ask`` / ``aask`` are stateless unless
  ``session=`` is passed.
- ``fast_llm`` defaults to the main ``llm`` when not supplied (spec §15 #3 resolved
  as documentation-only — no literal default model is imported here).
- ``cache="default"`` maps to ``MemoryCache()`` in Phase 1; Phase 4 replaces this
  with a ``SqliteCache``.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
from collections.abc import AsyncIterator
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from sleuth.backends.base import Backend
from sleuth.engine.executor import Executor
from sleuth.engine.router import Router
from sleuth.engine.synthesizer import Synthesizer
from sleuth.events import DoneEvent, Event, TokenEvent
from sleuth.llm.base import LLMClient
from sleuth.memory.cache import Cache, MemoryCache
from sleuth.memory.session import Session
from sleuth.types import Depth, Length, Result

T = TypeVar("T", bound=BaseModel)

_DEFAULT_BACKEND_TIMEOUT_S = 8.0


class Sleuth:
    """Plug-and-play agentic search with reasoning, planning, citations, and observability.

    Args:
        llm: Any object satisfying the ``LLMClient`` Protocol.  Used for synthesis.
        backends: One or more ``Backend`` instances to search against.
        fast_llm: Optional faster LLM for routing/planning.  Defaults to ``llm``
            when not supplied — no built-in fast model is imported (spec §15 #3).
        cache: ``"default"`` (MemoryCache in Phase 1), a ``Cache`` instance, or ``None``
            to disable caching.
        semantic_cache: Reserved for Phase 4.  Pass ``False`` (default) or ``None``.
        session: Optional persistent ``Session`` for multi-turn conversations.
    """

    def __init__(
        self,
        llm: LLMClient,
        backends: list[Backend],
        *,
        fast_llm: LLMClient | None = None,
        cache: Cache | Literal["default"] | None = "default",
        semantic_cache: Any = False,
        session: Session | None = None,
    ) -> None:
        self._llm = llm
        self._fast_llm = fast_llm or llm
        self._backends = backends
        self._session = session

        if cache == "default":
            self._cache: Cache | None = MemoryCache()
        else:
            self._cache = cache

        self._router = Router()
        self._executor = Executor(backends=backends, timeout_s=_DEFAULT_BACKEND_TIMEOUT_S)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def aask(
        self,
        query: str,
        *,
        depth: Depth = "auto",
        max_iterations: int = 4,
        schema: type[BaseModel] | None = None,
        session: Session | None = None,
    ) -> AsyncIterator[Event]:
        """Run an async search and yield typed events.

        This is an async generator — iterate with ``async for``.

        Args:
            query: The user's question.
            depth: ``"auto"`` (default), ``"fast"``, or ``"deep"``.
                   ``"deep"`` is handled as ``"fast"`` in Phase 1 — Phase 3 adds the planner.
            max_iterations: Maximum planning iterations (deep mode, Phase 3).
            schema: Optional Pydantic model for structured output.
            session: Per-call session override; overrides the instance-level session.
        """
        resolved_session = session or self._session
        start_ms = time.monotonic() * 1000

        # 1. Route
        route_event = self._router.route(query, depth=depth)
        yield route_event

        # Phase 1: treat "deep" as "fast" — Phase 3 adds the planner

        # 2. Execute (single-query fan-out)
        search_events, chunks = await self._executor.run(query)
        for se in search_events:
            yield se

        # 3. Synthesize
        backends_called = [se.backend for se in search_events if se.error is None]
        synth = Synthesizer(llm=self._llm)
        history = resolved_session.as_messages() if resolved_session else []

        async for event in synth.synthesize(
            query=query,
            chunks=chunks,
            history=history,
            stats_start_ms=start_ms,
            schema=schema,
            backends_called=backends_called,
            cache_hits={},
        ):
            yield event

        # 4. Update session
        if resolved_session is not None and synth.last_result is not None:
            resolved_session.add_turn(query, synth.last_result, [c.source for c in chunks])

    def ask(
        self,
        query: str,
        *,
        depth: Depth = "auto",
        max_iterations: int = 4,
        schema: type[BaseModel] | None = None,
        session: Session | None = None,
    ) -> Result:  # type: ignore[type-arg]
        """Synchronous wrapper around ``aask``.

        Blocks until the run completes and returns a ``Result``.
        """
        return asyncio.run(self._collect(query=query, depth=depth, schema=schema, session=session))

    async def asummarize(
        self,
        target: str,
        *,
        length: Length = "standard",
        schema: type[BaseModel] | None = None,
    ) -> AsyncIterator[Event]:
        """Summarize a URL, file path, or topic.

        Phase 2 extension: when ``target`` is an existing file path AND a
        ``LocalFiles`` backend is configured, delegate to
        ``LocalFiles._get_summary(target, length)`` and emit the result as
        ``TokenEvent`` + ``DoneEvent`` rather than running the full engine.
        """
        # Route file-path targets through LocalFiles._get_summary when available.
        _target_path = pathlib.Path(target)
        target_exists = await asyncio.to_thread(_target_path.exists)
        if target_exists:
            for backend in self._backends:
                if hasattr(backend, "_get_summary"):
                    import time as _time

                    start_ms = _time.monotonic() * 1000
                    summary_text: str = await backend._get_summary(target, length=length)
                    for token in summary_text.split(" "):
                        yield TokenEvent(type="token", text=token + " ")
                    elapsed = int((_time.monotonic() * 1000) - start_ms)
                    from sleuth.types import RunStats

                    yield DoneEvent(
                        type="done",
                        stats=RunStats(
                            latency_ms=elapsed,
                            first_token_ms=elapsed,
                            tokens_in=0,
                            tokens_out=len(summary_text.split()),
                            cache_hits={},
                            backends_called=[backend.name],
                        ),
                    )
                    return
        # Fallback: run the full engine with a summarize query.
        async for event in self.aask(
            f"summarize: {target} (length={length})", depth="fast", schema=schema
        ):
            yield event

    def summarize(
        self,
        target: str,
        *,
        length: Length = "standard",
        schema: type[BaseModel] | None = None,
    ) -> Result:  # type: ignore[type-arg]
        """Synchronous wrapper around ``asummarize``."""
        return asyncio.run(self._collect_summarize(target=target, length=length, schema=schema))

    async def warm_index(self) -> None:
        """Eagerly index all LocalFiles backends.

        Phase 2: calls ``warm_index()`` on every backend that supports it.
        """
        for backend in self._backends:
            if hasattr(backend, "warm_index"):
                await backend.warm_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _collect(
        self,
        *,
        query: str,
        depth: Depth,
        schema: type[BaseModel] | None,
        session: Session | None,
    ) -> Result:  # type: ignore[type-arg]
        """Consume aask stream and return the final Result."""
        from sleuth.engine.synthesizer import Synthesizer as _Synth

        start_ms = time.monotonic() * 1000
        resolved_session = session or self._session

        # Route (result not used here; routing done for consistency)
        self._router.route(query, depth=depth)
        # Execute
        _, chunks = await self._executor.run(query)
        # Synthesize
        history = resolved_session.as_messages() if resolved_session else []
        synth = _Synth(llm=self._llm)
        async for _ in synth.synthesize(
            query=query,
            chunks=chunks,
            history=history,
            stats_start_ms=start_ms,
            schema=schema,
            backends_called=[],
            cache_hits={},
        ):
            pass
        result = synth.last_result
        assert result is not None, "Synthesizer did not produce a result"
        return result

    async def _collect_summarize(
        self,
        *,
        target: str,
        length: Length,
        schema: type[BaseModel] | None,
    ) -> Result:  # type: ignore[type-arg]
        query = f"summarize: {target} (length={length})"
        return await self._collect(query=query, depth="fast", schema=schema, session=None)
