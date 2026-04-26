from sleuth.engine.router import Router
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
