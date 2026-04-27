import pytest

from sleuth.engine.router import Router, _is_deep, route
from sleuth.events import RouteEvent


def test_fast_passthrough():
    r = Router()
    event = r.route("anything", depth="fast")
    assert event.depth == "fast"
    assert event.type == "route"


def test_deep_passthrough():
    r = Router()
    event = r.route("anything", depth="deep")
    assert event.depth == "deep"


def test_auto_short_query_routes_fast():
    r = Router()
    event = r.route("what is python?", depth="auto")
    assert event.depth == "fast"


def test_auto_simple_factual_routes_fast():
    r = Router()
    for query in [
        "who invented Python?",
        "when was Python created?",
        "what does GIL stand for?",
    ]:
        event = r.route(query, depth="auto")
        assert event.depth == "fast", f"Expected fast for: {query!r}"


def test_auto_complex_query_routes_deep():
    r = Router()
    for query in [
        "compare the tradeoffs of async vs threading for IO-bound vs CPU-bound tasks in Python",
        "explain the design rationale for Python's memory model"
        " and how it affects multi-core performance",
        "what are all the breaking changes between Python 3.10 and 3.12"
        " and how do they affect our codebase?",
    ]:
        event = r.route(query, depth="auto")
        assert event.depth == "deep", f"Expected deep for: {query!r}"


def test_route_event_has_reason():
    r = Router()
    event = r.route("simple question?", depth="auto")
    assert isinstance(event.reason, str)
    assert len(event.reason) > 0


def test_route_event_type_is_route():
    r = Router()
    event = r.route("q", depth="fast")
    assert isinstance(event, RouteEvent)


# ---------------------------------------------------------------------------
# Phase 3: _is_deep heuristic tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        # should be deep
        ("compare Redis and Memcached for session storage", True),
        ("what are the differences between OAuth and OIDC?", True),
        ("how does our auth flow handle refresh tokens and what changed recently?", True),
        ("give me a comprehensive analysis of our caching strategy", True),
        ("research the best approaches for rate limiting", True),
        ("explain the tradeoffs between A and B in detail", True),
        ("what are all the ways X can fail and how do we handle each?", True),
        # should NOT be deep
        ("what is OAuth?", False),
        ("who maintains the auth middleware?", False),
        ("list all endpoints", False),
        ("define refresh token", False),
        ("show me the login function", False),
    ],
)
def test_is_deep_heuristic(query: str, expected: bool) -> None:
    assert _is_deep(query) is expected


# ---------------------------------------------------------------------------
# Phase 3: module-level route() async-generator tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_auto_deep_emits_deep_route_event() -> None:
    events = []
    async for e in route("compare OAuth vs OIDC in detail for enterprise use", depth="auto"):
        events.append(e)
    route_event = events[0]
    assert isinstance(route_event, RouteEvent)
    assert route_event.depth == "deep"


@pytest.mark.asyncio
async def test_route_explicit_deep_emits_deep() -> None:
    events = []
    async for e in route("anything", depth="deep"):
        events.append(e)
    assert events[0].depth == "deep"


@pytest.mark.asyncio
async def test_route_explicit_fast_not_reclassified() -> None:
    """depth='fast' is never upgraded to 'deep' by heuristic."""
    events = []
    async for e in route("compare everything in full detail", depth="fast"):
        events.append(e)
    assert events[0].depth == "fast"
