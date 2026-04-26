from typing import Any

from sleuth.types import Chunk, Depth, Length, Result, RunStats, Source


def test_source_kinds():
    for kind in ("url", "file", "code"):
        s = Source(kind=kind, location="x")
        assert s.kind == kind


def test_source_optional_fields():
    s = Source(kind="url", location="https://example.com")
    assert s.title is None
    assert s.fetched_at is None


def test_chunk_defaults():
    s = Source(kind="file", location="/tmp/foo.md")
    c = Chunk(text="hello", source=s)
    assert c.score is None
    assert c.metadata == {}


def test_runstats_required_fields():
    stats = RunStats(
        latency_ms=200,
        first_token_ms=150,
        tokens_in=10,
        tokens_out=5,
        cache_hits={"query": 0},
        backends_called=["tavily"],
    )
    assert stats.latency_ms == 200


def test_runstats_first_token_ms_nullable():
    stats = RunStats(
        latency_ms=50,
        first_token_ms=None,
        tokens_in=0,
        tokens_out=0,
        cache_hits={},
        backends_called=[],
    )
    assert stats.first_token_ms is None


def test_result_no_schema():
    stats = RunStats(
        latency_ms=100,
        first_token_ms=90,
        tokens_in=5,
        tokens_out=10,
        cache_hits={},
        backends_called=[],
    )
    r: Result[Any] = Result(text="answer", citations=[], stats=stats)
    assert r.data is None


def test_result_generic_with_schema():
    from pydantic import BaseModel

    class MySchema(BaseModel):
        score: float

    stats = RunStats(
        latency_ms=100,
        first_token_ms=90,
        tokens_in=5,
        tokens_out=10,
        cache_hits={},
        backends_called=[],
    )
    r: Result[MySchema] = Result(text="ok", citations=[], data=MySchema(score=0.9), stats=stats)
    assert r.data is not None
    assert r.data.score == 0.9


def test_depth_literals():
    depths: list[Depth] = ["auto", "fast", "deep"]
    assert len(depths) == 3


def test_length_literals():
    lengths: list[Length] = ["brief", "standard", "thorough"]
    assert len(lengths) == 3
