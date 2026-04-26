import asyncio

from sleuth.backends.base import Capability
from sleuth.engine.executor import Executor
from sleuth.errors import BackendError
from sleuth.events import SearchEvent
from sleuth.types import Chunk, Source


def _make_chunk(url: str, text: str = "content", score: float = 0.9) -> Chunk:
    return Chunk(
        text=text,
        source=Source(kind="url", location=url),
        score=score,
    )


class OkBackend:
    name = "ok"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [_make_chunk("https://ok.com/1"), _make_chunk("https://ok.com/2")]


class ErrorBackend:
    name = "error"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        raise BackendError("search failed")


class SlowBackend:
    name = "slow"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        await asyncio.sleep(10.0)  # will be cancelled by timeout
        return []


class DuplicateBackend:
    """Returns a chunk whose source URL overlaps with OkBackend."""

    name = "dup"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [_make_chunk("https://ok.com/1"), _make_chunk("https://dup.com/unique")]


async def test_single_backend_returns_chunks():
    executor = Executor(backends=[OkBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)


async def test_emits_search_event():
    executor = Executor(backends=[OkBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    search_events = [e for e in events if isinstance(e, SearchEvent)]
    assert len(search_events) == 1
    assert search_events[0].backend == "ok"
    assert search_events[0].error is None


async def test_error_backend_emits_error_search_event():
    executor = Executor(backends=[ErrorBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    search_events = [e for e in events if isinstance(e, SearchEvent)]
    assert len(search_events) == 1
    assert search_events[0].error is not None
    assert chunks == []


async def test_timeout_backend_emits_error_search_event():
    executor = Executor(backends=[SlowBackend()], timeout_s=0.05)
    events, chunks = await executor.run("query", k=10)
    search_events = [e for e in events if isinstance(e, SearchEvent)]
    assert search_events[0].error is not None
    assert "timeout" in search_events[0].error.lower()
    assert chunks == []


async def test_partial_failure_keeps_successful_results():
    executor = Executor(backends=[OkBackend(), ErrorBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    assert len(chunks) == 2  # OkBackend succeeded
    error_events = [e for e in events if isinstance(e, SearchEvent) and e.error]
    assert len(error_events) == 1


async def test_deduplication_by_source_location():
    executor = Executor(backends=[OkBackend(), DuplicateBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    locations = [c.source.location for c in chunks]
    assert len(locations) == len(set(locations)), "Duplicate source locations found"


async def test_k_limits_per_backend():
    executor = Executor(backends=[OkBackend()], timeout_s=5.0)
    _, chunks = await executor.run("query", k=1)
    assert len(chunks) <= 1
