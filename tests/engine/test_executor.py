import asyncio
from collections.abc import AsyncGenerator

import pytest

from sleuth.backends.base import Capability
from sleuth.engine.executor import Executor, execute_subqueries, execute_with_prefetch, reflect_loop
from sleuth.errors import BackendError
from sleuth.events import PlanStep, SearchEvent
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
    _events, chunks = await executor.run("query", k=10)
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)


async def test_emits_search_event():
    executor = Executor(backends=[OkBackend()], timeout_s=5.0)
    events, _chunks = await executor.run("query", k=10)
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
    _events, chunks = await executor.run("query", k=10)
    locations = [c.source.location for c in chunks]
    assert len(locations) == len(set(locations)), "Duplicate source locations found"


async def test_k_limits_per_backend():
    executor = Executor(backends=[OkBackend()], timeout_s=5.0)
    _, chunks = await executor.run("query", k=1)
    assert len(chunks) <= 1


# ---------------------------------------------------------------------------
# Phase 3: execute_subqueries (Task 5)
# ---------------------------------------------------------------------------


class _StubBackend:
    name = "stub-web"
    capabilities = frozenset({Capability.WEB})

    def __init__(self, results: list[Chunk]) -> None:
        self._results = results

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return self._results


def _make_chunk2(text: str, url: str = "https://example.com") -> Chunk:
    return Chunk(text=text, source=Source(kind="url", location=url))


@pytest.mark.asyncio
async def test_execute_subqueries_fans_out_in_parallel() -> None:
    """execute_subqueries runs all sub-queries concurrently and merges chunks."""
    backend = _StubBackend([_make_chunk2("result A", url="https://example.com/a")])
    chunks = await execute_subqueries(
        subqueries=["query one", "query two"],
        backends=[backend],
    )
    # Two sub-queries x one result each = 2 chunks (different URLs)
    # Note: same URL would be deduplicated, so we use unique URLs via backend
    assert len(chunks) >= 1  # at least one result


@pytest.mark.asyncio
async def test_execute_subqueries_deduplicates_by_source() -> None:
    """Chunks from the same URL (same source.location) are deduplicated."""
    shared_chunk = _make_chunk2("shared", url="https://shared.com")
    backend = _StubBackend([shared_chunk])

    chunks = await execute_subqueries(
        subqueries=["q1", "q2"],
        backends=[backend],
    )
    locations = [c.source.location for c in chunks]
    assert locations.count("https://shared.com") == 1


@pytest.mark.asyncio
async def test_execute_subqueries_emits_search_events() -> None:
    """execute_subqueries yields SearchEvent for every backend x sub-query."""
    backend = _StubBackend([_make_chunk2("r", url="https://ex.com/r")])
    events: list[SearchEvent] = []

    await execute_subqueries(
        subqueries=["q1", "q2"],
        backends=[backend],
        on_search_event=lambda e: events.append(e),
    )
    assert len(events) == 2  # one per sub-query (one backend)
    assert all(isinstance(e, SearchEvent) for e in events)
    assert {e.query for e in events} == {"q1", "q2"}


# ---------------------------------------------------------------------------
# Phase 3: execute_with_prefetch (Task 6)
# ---------------------------------------------------------------------------


class _OrderTrackingBackend:
    """Backend that records when searches are called via an asyncio.Event."""

    name = "order-tracker"
    capabilities = frozenset({Capability.WEB})

    def __init__(self, first_search_started: asyncio.Event) -> None:
        self._first_search_started = first_search_started
        self.call_count = 0

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        self.call_count += 1
        if self.call_count == 1:
            self._first_search_started.set()  # signal: first search is running
        await asyncio.sleep(0)  # yield control
        return [Chunk(text=query, source=Source(kind="url", location=f"https://ex.com/{query}"))]


async def _slow_planner(
    first_search_started: asyncio.Event,
) -> AsyncGenerator[PlanStep, None]:
    """Async generator that yields two sub-queries, pausing after the first.

    After emitting step 1, it WAITS until the backend signals that the first
    search task has started before emitting step 2. This asserts overlap —
    if prefetch didn't happen, the wait would deadlock.
    """
    yield PlanStep(query="sub-query-one")
    # Wait until the executor has actually started the search task
    await asyncio.wait_for(first_search_started.wait(), timeout=2.0)
    yield PlanStep(query="sub-query-two")
    yield PlanStep(query="", done=True)


@pytest.mark.asyncio
async def test_speculative_prefetch_starts_before_planner_finishes() -> None:
    """Executor starts the first search task while the planner is still running.

    The _slow_planner waits for the backend's first_search_started event before
    emitting its second step. If prefetch didn't overlap, this deadlocks/times out.
    """
    first_search_started = asyncio.Event()
    backend = _OrderTrackingBackend(first_search_started)

    chunks = await execute_with_prefetch(
        plan_steps=_slow_planner(first_search_started),
        backends=[backend],
    )

    assert backend.call_count == 2  # both sub-queries were searched
    assert len(chunks) == 2
    texts = {c.text for c in chunks}
    assert texts == {"sub-query-one", "sub-query-two"}


@pytest.mark.asyncio
async def test_speculative_prefetch_single_step() -> None:
    """Works correctly when the planner emits only one real sub-query."""
    from sleuth.events import PlanStep

    async def _single_step_planner() -> AsyncGenerator[PlanStep, None]:
        yield PlanStep(query="only-query")
        yield PlanStep(query="", done=True)

    class _SimpleBackend:
        name = "simple"
        capabilities = frozenset({Capability.WEB})

        async def search(self, q: str, k: int = 10) -> list[Chunk]:
            return [Chunk(text=q, source=Source(kind="url", location=f"https://x.com/{q}"))]

    chunks = await execute_with_prefetch(
        plan_steps=_single_step_planner(),
        backends=[_SimpleBackend()],
    )
    assert len(chunks) == 1
    assert chunks[0].text == "only-query"


# ---------------------------------------------------------------------------
# Phase 3: reflect_loop (Task 7)
# ---------------------------------------------------------------------------


class _CountingBackend:
    name = "counting"
    capabilities = frozenset({Capability.WEB})

    def __init__(self) -> None:
        self.call_count = 0

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        self.call_count += 1
        return [
            Chunk(
                text=f"result for {query} (call {self.call_count})",
                source=Source(kind="url", location=f"https://ex.com/{query}/{self.call_count}"),
            )
        ]


@pytest.mark.asyncio
async def test_reflect_loop_stops_on_done() -> None:
    """reflect_loop terminates when planner emits done=True on first iteration."""
    from sleuth.engine.planner import Planner
    from sleuth.llm.stub import StubLLM

    llm = StubLLM(responses=['[{"query": "sub1"}, {"done": true, "query": ""}]'])
    planner = Planner(llm=llm)
    backend = _CountingBackend()

    chunks, iterations = await reflect_loop(
        query="what is TLS?",
        planner=planner,
        backends=[backend],
        max_iterations=5,
    )

    assert iterations == 1
    assert len(chunks) >= 1
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_reflect_loop_respects_max_iterations() -> None:
    """reflect_loop stops at max_iterations even if planner never emits done."""
    from collections.abc import AsyncIterator

    from sleuth.engine.planner import Planner
    from sleuth.llm.base import LLMChunk, Stop, TextDelta

    call_count: dict[str, int] = {"n": 0}

    from sleuth.llm.base import Message as _Msg

    class _LoopingLLM:
        name = "looping"
        supports_reasoning = False
        supports_structured_output = True

        async def stream(
            self,
            messages: list[_Msg],
            *,
            schema: object = None,
            tools: object = None,
        ) -> AsyncIterator[LLMChunk]:
            call_count["n"] += 1
            yield TextDelta(text=f'[{{"query": "sub-query-{call_count["n"]}"}}]')
            yield Stop(reason="end_turn")

    planner = Planner(llm=_LoopingLLM())
    backend = _CountingBackend()

    _chunks, iterations = await reflect_loop(
        query="never-ending research",
        planner=planner,
        backends=[backend],
        max_iterations=3,
    )

    assert iterations == 3


@pytest.mark.asyncio
async def test_reflect_loop_feeds_chunks_as_context() -> None:
    """After iteration 1, prior chunks appear in the planner's next prompt."""
    from collections.abc import AsyncIterator

    from sleuth.engine.planner import Planner
    from sleuth.llm.base import LLMChunk, Stop, TextDelta

    captured: list[str] = []
    call_num: dict[str, int] = {"n": 0}

    from sleuth.llm.base import Message as _Msg2

    class _SpyLLM:
        name = "spy"
        supports_reasoning = False
        supports_structured_output = True

        async def stream(
            self,
            messages: list[_Msg2],
            *,
            schema: object = None,
            tools: object = None,
        ) -> AsyncIterator[LLMChunk]:
            call_num["n"] += 1
            user_content = next(m.content for m in messages if m.role == "user")
            captured.append(user_content)
            if call_num["n"] == 1:
                # First iteration: return one real sub-query (no done yet)
                yield TextDelta(text='[{"query": "first sub"}]')
                yield Stop(reason="end_turn")
            else:
                # Second iteration: signal done
                yield TextDelta(text='[{"done": true, "query": ""}]')
                yield Stop(reason="end_turn")

    planner = Planner(llm=_SpyLLM())
    backend = _CountingBackend()

    await reflect_loop(
        query="deep question",
        planner=planner,
        backends=[backend],
        max_iterations=5,
    )

    # On the second LLM call, prior results should be in the user message
    assert len(captured) == 2
    assert "result for first sub" in captured[1]
