from pydantic import TypeAdapter

from sleuth.events import (
    CacheHitEvent,
    CitationEvent,
    DoneEvent,
    Event,
    FetchEvent,
    PlanEvent,
    RouteEvent,
    SearchEvent,
    ThinkingEvent,
    TokenEvent,
)
from sleuth.types import RunStats, Source


def _stats() -> RunStats:
    return RunStats(
        latency_ms=100,
        first_token_ms=80,
        tokens_in=5,
        tokens_out=10,
        cache_hits={},
        backends_called=["stub"],
    )


def _source() -> Source:
    return Source(kind="url", location="https://example.com")


def test_route_event_discriminator():
    e = RouteEvent(type="route", depth="fast", reason="short query")
    assert e.type == "route"
    assert e.depth == "fast"


def test_plan_event():
    e = PlanEvent(type="plan", steps=[])
    assert e.type == "plan"


def test_search_event_no_error():
    e = SearchEvent(type="search", backend="tavily", query="foo")
    assert e.error is None


def test_search_event_with_error():
    e = SearchEvent(type="search", backend="tavily", query="foo", error="timeout")
    assert e.error == "timeout"


def test_fetch_event():
    e = FetchEvent(type="fetch", url="https://x.com", status=200)
    assert e.status == 200


def test_thinking_event():
    e = ThinkingEvent(type="thinking", text="reasoning...")
    assert e.text == "reasoning..."


def test_token_event():
    e = TokenEvent(type="token", text="hello")
    assert e.text == "hello"


def test_citation_event():
    e = CitationEvent(type="citation", index=0, source=_source())
    assert e.index == 0


def test_cache_hit_event():
    e = CacheHitEvent(type="cache_hit", kind="query", key="abc123")
    assert e.kind == "query"


def test_done_event():
    e = DoneEvent(type="done", stats=_stats())
    assert e.stats.latency_ms == 100


def test_event_union_roundtrip():
    """Discriminated union parses each event type correctly."""
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    payloads = [
        {"type": "route", "depth": "auto", "reason": "heuristic"},
        {"type": "plan", "steps": []},
        {"type": "search", "backend": "tavily", "query": "q"},
        {"type": "fetch", "url": "https://x.com", "status": 200},
        {"type": "thinking", "text": "hmm"},
        {"type": "token", "text": "tok"},
        {"type": "citation", "index": 0, "source": {"kind": "url", "location": "https://x.com"}},
        {"type": "cache_hit", "kind": "query", "key": "k"},
        {
            "type": "done",
            "stats": {
                "latency_ms": 1,
                "first_token_ms": None,
                "tokens_in": 0,
                "tokens_out": 0,
                "cache_hits": {},
                "backends_called": [],
            },
        },
    ]
    for p in payloads:
        event = adapter.validate_python(p)
        assert event.type == p["type"]  # type: ignore[index]
