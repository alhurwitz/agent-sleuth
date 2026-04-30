from pydantic import BaseModel

from sleuth._agent import Sleuth
from sleuth.backends.base import Capability
from sleuth.events import DoneEvent, RouteEvent, SearchEvent, TokenEvent
from sleuth.llm.stub import StubLLM
from sleuth.memory.session import Session
from sleuth.types import Chunk, Result, Source


class FakeBackend:
    name = "fake"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [Chunk(text="result text", source=Source(kind="url", location="https://a.com"))]


async def test_aask_yields_route_event():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()], cache=None)
    events = [e async for e in agent.aask("who is guido?")]
    route_events = [e for e in events if isinstance(e, RouteEvent)]
    assert len(route_events) == 1


async def test_aask_yields_search_event():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()], cache=None)
    events = [e async for e in agent.aask("who is guido?")]
    search_events = [e for e in events if isinstance(e, SearchEvent)]
    assert len(search_events) == 1
    assert search_events[0].backend == "fake"


async def test_aask_yields_token_events():
    agent = Sleuth(llm=StubLLM(["hello"]), backends=[FakeBackend()], cache=None)
    events = [e async for e in agent.aask("q?")]
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert any(e.text == "hello" for e in token_events)


async def test_aask_yields_done_event_last():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()], cache=None)
    events = [e async for e in agent.aask("q?")]
    assert isinstance(events[-1], DoneEvent)


async def test_aask_depth_fast_skips_deep():
    """Fast depth: route event must show 'fast'."""
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()], cache=None)
    events = [e async for e in agent.aask("q?", depth="fast")]
    route_event = next(e for e in events if isinstance(e, RouteEvent))
    assert route_event.depth == "fast"


def test_ask_returns_result():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()], cache=None)
    result = agent.ask("q?")
    assert isinstance(result, Result)
    assert result.text == "answer"


def test_ask_result_has_citations():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()], cache=None)
    result = agent.ask("q?")
    assert len(result.citations) >= 1


async def test_aask_with_cache_none_still_works():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()], cache=None)
    events = [e async for e in agent.aask("q?")]
    assert any(isinstance(e, DoneEvent) for e in events)


async def test_aask_with_session_adds_turn():
    session = Session()
    agent = Sleuth(llm=StubLLM(["a1", "a2"]), backends=[FakeBackend()], cache=None)
    async for _ in agent.aask("q1?", session=session):
        pass
    assert len(session.turns) == 1
    assert session.turns[0].query == "q1?"


# ---------------------------------------------------------------------------
# Cache-hit replay (spec §8: cache hits replay through the event stream)
# ---------------------------------------------------------------------------


class CountingFakeBackend:
    """FakeBackend that records how many times search() was called."""

    name = "fake"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    def __init__(self) -> None:
        self.call_count = 0

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        self.call_count += 1
        return [
            Chunk(
                text=f"result for {query}",
                source=Source(kind="url", location="https://example.com/a"),
            )
        ]


async def test_aask_cache_miss_then_hit_replays_through_event_stream(tmp_path):
    """Second call with the same query hits the cache and emits CacheHitEvent."""
    from sleuth.events import CacheHitEvent, CitationEvent
    from sleuth.memory.cache import SqliteCache

    backend = CountingFakeBackend()
    cache = SqliteCache(base_path=tmp_path / "test")
    agent = Sleuth(llm=StubLLM(["original answer"]), backends=[backend], cache=cache)

    # First call — cache miss, backend called.
    events1 = [e async for e in agent.aask("what is python?")]
    assert backend.call_count == 1
    assert not any(isinstance(e, CacheHitEvent) for e in events1)
    assert any(isinstance(e, DoneEvent) for e in events1)

    # Second call (same query) — cache hit, backend NOT called.
    events2 = [e async for e in agent.aask("what is python?")]
    assert backend.call_count == 1, "Backend should not be re-invoked on cache hit"

    cache_hits = [e for e in events2 if isinstance(e, CacheHitEvent)]
    assert len(cache_hits) == 1
    assert cache_hits[0].kind == "query"
    assert len(cache_hits[0].key) == 64  # sha256 hex

    # The replay still terminates with a DoneEvent and yields citations.
    done_events = [e for e in events2 if isinstance(e, DoneEvent)]
    citations = [e for e in events2 if isinstance(e, CitationEvent)]
    tokens = [e for e in events2 if isinstance(e, TokenEvent)]
    assert len(done_events) == 1
    assert len(citations) >= 1
    assert len(tokens) >= 1
    # Stats reflect the cache hit.
    assert done_events[0].stats.cache_hits.get("query") == 1
    assert done_events[0].stats.first_token_ms is None  # null on hit per spec §6


async def test_aask_cache_skipped_when_schema_passed(tmp_path):
    """Schema results aren't cached (round-trip not implemented yet)."""
    from sleuth.memory.cache import SqliteCache

    class _S(BaseModel):
        v: str

    backend = CountingFakeBackend()
    cache = SqliteCache(base_path=tmp_path / "test_schema")
    agent = Sleuth(llm=StubLLM(['{"v":"x"}']), backends=[backend], cache=cache)

    async for _ in agent.aask("q?", schema=_S):
        pass
    async for _ in agent.aask("q?", schema=_S):
        pass

    assert backend.call_count == 2  # both calls hit the backend, no caching


async def test_aask_cache_disabled_emits_no_cache_hit():
    """When cache=None, no cache lookup or write happens."""
    from sleuth.events import CacheHitEvent

    backend = CountingFakeBackend()
    agent = Sleuth(llm=StubLLM(["a"]), backends=[backend], cache=None)
    [e async for e in agent.aask("q?")]
    events = [e async for e in agent.aask("q?")]
    assert backend.call_count == 2
    assert not any(isinstance(e, CacheHitEvent) for e in events)
