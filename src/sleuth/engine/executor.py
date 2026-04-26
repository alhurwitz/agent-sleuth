"""Single-backend fan-out executor.

Fans search queries out to all registered backends in parallel, applies
per-backend timeouts, handles failures per spec §7.1, and de-duplicates
results by source location.

Phase 3 will extend this for:
  - Multi-query fan-out (planner emits multiple sub-queries)
  - Speculative prefetch (start backend search while planner is still streaming)
Keep this module focused on single-query fan-out.
"""

from __future__ import annotations

import asyncio
import logging

from sleuth.backends.base import Backend
from sleuth.errors import BackendError
from sleuth.events import SearchEvent
from sleuth.types import Chunk

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
