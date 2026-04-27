"""Single-backend fan-out executor.

Fans search queries out to all registered backends in parallel, applies
per-backend timeouts, handles failures per spec §7.1, and de-duplicates
results by source location.

Phase 3 extensions:
  - Multi-query fan-out via ``execute_subqueries``
  - Speculative prefetch via ``execute_with_prefetch``
  - Reflect loop via ``reflect_loop``
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from sleuth.backends.base import Backend
from sleuth.errors import BackendError
from sleuth.events import SearchEvent
from sleuth.types import Chunk

if TYPE_CHECKING:
    from sleuth.engine.planner import Planner
    from sleuth.events import PlanStep

logger = logging.getLogger("sleuth.engine.executor")


class Executor:
    """Async fan-out over all registered backends for a single query.

    Args:
        backends: List of ``Backend`` instances to query in parallel.
        timeout_s: Per-backend timeout in seconds (default 8s per spec §7.1).
    """

    def __init__(self, backends: list[Backend], *, timeout_s: float = 8.0) -> None:
        self._backends = backends
        self._timeout_s = timeout_s

    async def run(self, query: str, *, k: int = 10) -> tuple[list[SearchEvent], list[Chunk]]:
        """Fan out ``query`` to all backends and return events + merged chunks.

        Returns:
            A tuple of (``SearchEvent`` list, deduplicated ``Chunk`` list).
            Never raises — per-backend errors are captured in ``SearchEvent.error``.
        """
        tasks = {
            asyncio.create_task(
                self._search_one(backend, query, k),
                name=f"executor:{backend.name}",
            ): backend
            for backend in self._backends
        }

        events: list[SearchEvent] = []
        all_chunks: list[Chunk] = []

        results = await asyncio.gather(*tasks.keys(), return_exceptions=True)

        for (_task, backend), result in zip(tasks.items(), results, strict=False):
            if isinstance(result, SearchEvent):
                # Error SearchEvent returned from _search_one
                events.append(result)
            elif isinstance(result, list):
                events.append(SearchEvent(type="search", backend=backend.name, query=query))
                all_chunks.extend(result[:k])  # honour k per backend
            else:
                # Unexpected exception (should not happen, but be safe)
                logger.error("Unexpected result from backend %s: %r", backend.name, result)
                events.append(
                    SearchEvent(
                        type="search",
                        backend=backend.name,
                        query=query,
                        error=repr(result),
                    )
                )

        deduped = self._deduplicate(all_chunks)
        return events, deduped

    async def _search_one(self, backend: Backend, query: str, k: int) -> list[Chunk] | SearchEvent:
        """Run a single backend search, wrapping errors into SearchEvent."""
        try:
            chunks = await asyncio.wait_for(
                backend.search(query, k),
                timeout=self._timeout_s,
            )
            return chunks
        except TimeoutError:
            logger.warning("Backend %s timed out after %.1fs", backend.name, self._timeout_s)
            return SearchEvent(
                type="search",
                backend=backend.name,
                query=query,
                error=f"timeout after {self._timeout_s}s",
            )
        except BackendError as exc:
            logger.warning("Backend %s error: %s", backend.name, exc)
            return SearchEvent(
                type="search",
                backend=backend.name,
                query=query,
                error=str(exc),
            )
        except Exception as exc:
            logger.error("Backend %s unexpected error: %s", backend.name, exc, exc_info=True)
            return SearchEvent(
                type="search",
                backend=backend.name,
                query=query,
                error=f"unexpected error: {exc}",
            )

    @staticmethod
    def _deduplicate(chunks: list[Chunk]) -> list[Chunk]:
        """Remove chunks with duplicate source locations, keeping first occurrence."""
        seen: set[str] = set()
        result: list[Chunk] = []
        for chunk in chunks:
            loc = chunk.source.location
            if loc not in seen:
                seen.add(loc)
                result.append(chunk)
        return result


# ---------------------------------------------------------------------------
# Phase 3: Multi-sub-query fan-out
# ---------------------------------------------------------------------------


async def execute_subqueries(
    subqueries: list[str],
    backends: list[Backend],
    *,
    k: int = 10,
    on_search_event: Callable[[SearchEvent], Any] | None = None,
) -> list[Chunk]:
    """Fan out all sub-queries across all backends in parallel; deduplicate by source.

    Args:
        subqueries: Sub-queries emitted by the Planner.
        backends: Backends to search. All backends are tried for every sub-query.
        k: Max results per backend per sub-query.
        on_search_event: Optional callback fired (synchronously) per SearchEvent.

    Returns:
        Deduplicated, merged list of Chunks from all searches.
    """

    async def _search_one(query: str, backend: Backend) -> list[Chunk]:
        event = SearchEvent(type="search", backend=backend.name, query=query)
        if on_search_event is not None:
            on_search_event(event)
        try:
            return await backend.search(query, k)
        except Exception as exc:
            logger.warning("backend %r failed for query %r: %s", backend.name, query, exc)
            return []

    tasks = [_search_one(query, backend) for query in subqueries for backend in backends]
    results: list[list[Chunk]] = await asyncio.gather(*tasks)

    # Flatten + deduplicate by source.location (first occurrence wins)
    seen: set[str] = set()
    merged: list[Chunk] = []
    for chunk_list in results:
        for chunk in chunk_list:
            loc = chunk.source.location
            if loc not in seen:
                seen.add(loc)
                merged.append(chunk)

    return merged


# ---------------------------------------------------------------------------
# Phase 3: Speculative prefetch
# ---------------------------------------------------------------------------


async def execute_with_prefetch(
    plan_steps: AsyncIterator[PlanStep],
    backends: list[Backend],
    *,
    k: int = 10,
    on_search_event: Callable[[SearchEvent], Any] | None = None,
) -> list[Chunk]:
    """Consume plan steps and start search tasks speculatively.

    As soon as the planner yields its first PlanStep, a search task is launched
    immediately — before waiting for the planner to finish streaming. This hides
    planner latency behind search latency (spec §11 point 3).

    Args:
        plan_steps: Async generator of PlanStep objects from Planner.plan().
        backends: Backends to search.
        k: Max results per backend per sub-query.
        on_search_event: Optional callback per SearchEvent.

    Returns:
        Deduplicated merged chunks from all sub-queries.
    """
    pending_tasks: list[asyncio.Task[list[Chunk]]] = []

    async def _launch(query: str) -> None:
        """Create and register a search task for `query`."""
        task: asyncio.Task[list[Chunk]] = asyncio.create_task(
            execute_subqueries(
                subqueries=[query],
                backends=backends,
                k=k,
                on_search_event=on_search_event,
            )
        )
        pending_tasks.append(task)

    async for step in plan_steps:
        if step.done:
            break
        if step.query:
            await _launch(step.query)
            # Yield control immediately so the launched task can start running
            # while the planner continues streaming — this is the speculative
            # prefetch: search and planner overlap in time.
            await asyncio.sleep(0)

    if not pending_tasks:
        return []

    results: list[list[Chunk]] = await asyncio.gather(*pending_tasks)

    # Flatten + deduplicate by source.location (first occurrence wins)
    seen: set[str] = set()
    merged: list[Chunk] = []
    for chunk_list in results:
        for chunk in chunk_list:
            loc = chunk.source.location
            if loc not in seen:
                seen.add(loc)
                merged.append(chunk)

    return merged


# ---------------------------------------------------------------------------
# Phase 3: Reflect loop with max_iterations
# ---------------------------------------------------------------------------


async def reflect_loop(
    query: str,
    planner: Planner,
    backends: list[Backend],
    *,
    max_iterations: int = 4,
    k: int = 10,
    on_search_event: Callable[[SearchEvent], Any] | None = None,
    on_plan_event: Callable[..., Any] | None = None,
) -> tuple[list[Chunk], int]:
    """Run the plan → search → reflect loop for deep mode.

    Each iteration:
      1. Call planner.plan(query, state) to get sub-queries.
      2. Fan out searches with speculative prefetch via execute_with_prefetch().
      3. Append result summaries to state.context_snippets for the next iteration.
      4. Stop if planner emits done=True in this iteration or max_iterations reached.

    Args:
        query: The original user query.
        planner: A Planner instance.
        backends: Backends to search.
        max_iterations: Maximum number of plan→search→reflect cycles.
        k: Max results per backend per sub-query.
        on_search_event: Optional callback per SearchEvent.
        on_plan_event: Optional callback per PlanEvent (forwarded to planner.plan()).

    Returns:
        (merged_chunks, iteration_count) — all chunks from all iterations,
        deduplicated by source.location; number of iterations executed.
    """
    from sleuth.engine.planner import _PlannerState
    from sleuth.events import PlanStep as _PlanStep

    state = _PlannerState()
    all_chunks: list[Chunk] = []
    seen_locations: set[str] = set()
    iterations = 0
    done = False

    while not done and iterations < max_iterations:
        iterations += 1

        # Collect plan steps for this iteration
        steps: list[_PlanStep] = []
        async for step in planner.plan(query, state, on_plan_event=on_plan_event):
            steps.append(step)

        # Determine if the planner explicitly requested termination.
        # Only stop if the LLM itself included done:true (not auto-appended).
        # planner._last_explicitly_done is set by Planner.plan() after _parse_steps.
        explicitly_done = getattr(planner, "_last_explicitly_done", False)

        real_queries = [s.query for s in steps if not s.done and s.query]
        if not real_queries:
            # No queries to run — planner is unconditionally done.
            done = True
            break

        if explicitly_done:
            # LLM explicitly said done — run this batch and stop.
            done = True

        # Fan out searches with speculative prefetch
        async def _step_gen(queries: list[str] = real_queries) -> AsyncIterator[_PlanStep]:
            for q in queries:
                yield _PlanStep(query=q)
            yield _PlanStep(query="", done=True)

        iter_chunks = await execute_with_prefetch(
            plan_steps=_step_gen(),
            backends=backends,
            k=k,
            on_search_event=on_search_event,
        )

        # Merge into global deduplicated set
        for chunk in iter_chunks:
            loc = chunk.source.location
            if loc not in seen_locations:
                seen_locations.add(loc)
                all_chunks.append(chunk)

        # Update planner state with result summaries for next iteration
        snippets = [c.text[:300] for c in iter_chunks]  # truncate for context window
        state.context_snippets.extend(snippets)

    return all_chunks, iterations
