"""Verify the public API surface re-exported from the top-level sleuth package."""


def test_sleuth_importable():
    from sleuth import Sleuth

    assert Sleuth is not None


def test_session_importable():
    from sleuth import Session

    assert Session is not None


def test_result_importable():
    from sleuth import Result

    assert Result is not None


def test_source_chunk_importable():
    from sleuth import Chunk, Source

    assert Source is not None
    assert Chunk is not None


def test_all_event_types_importable():
    from sleuth import (
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

    for sym in (
        RouteEvent,
        PlanEvent,
        SearchEvent,
        FetchEvent,
        ThinkingEvent,
        TokenEvent,
        CitationEvent,
        CacheHitEvent,
        DoneEvent,
        Event,
    ):
        assert sym is not None


def test_depth_length_importable():
    from sleuth import Depth, Length

    assert Depth is not None
    assert Length is not None


def test_backend_importable():
    from sleuth.backends import Tavily

    assert Tavily is not None
