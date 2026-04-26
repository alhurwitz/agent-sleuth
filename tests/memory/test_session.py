"""Tests for Session ring buffer (Phase 1 baseline) + persistence and flush (Phase 4)."""

import json
from pathlib import Path

import pytest

from sleuth.llm.base import Message
from sleuth.memory.session import Session
from sleuth.types import Result, RunStats, Source


def _result(text: str = "answer") -> "Result":  # type: ignore[type-arg]
    stats = RunStats(
        latency_ms=100,
        first_token_ms=90,
        tokens_in=5,
        tokens_out=10,
        cache_hits={},
        backends_called=[],
    )
    return Result(text=text, citations=[], stats=stats)


def _source(loc: str = "file:///test.md") -> Source:
    return Source(kind="file", location=loc, title="Test")


# ---------------------------------------------------------------------------
# Phase 1 baseline — ring buffer tests (must keep passing)
# ---------------------------------------------------------------------------


def test_session_starts_empty():
    s = Session()
    assert s.turns == []


def test_add_turn_and_retrieve():
    s = Session()
    s.add_turn("what is foo?", _result("foo is bar"), [])
    assert len(s.turns) == 1
    assert s.turns[0].query == "what is foo?"
    assert s.turns[0].result.text == "foo is bar"


def test_ring_buffer_respects_max_turns():
    s = Session(max_turns=3)
    for i in range(5):
        s.add_turn(f"q{i}", _result(f"a{i}"), [])
    assert len(s.turns) == 3
    # Oldest turns dropped; most recent kept
    assert s.turns[0].query == "q2"
    assert s.turns[-1].query == "q4"


def test_default_max_turns_is_20():
    s = Session()
    for i in range(25):
        s.add_turn(f"q{i}", _result(f"a{i}"), [])
    assert len(s.turns) == 20


def test_as_messages_returns_list():
    s = Session()
    s.add_turn("q1", _result("a1"), [])
    msgs = s.as_messages()
    assert all(isinstance(m, Message) for m in msgs)


def test_as_messages_interleaves_user_assistant():
    s = Session()
    s.add_turn("q1", _result("a1"), [])
    s.add_turn("q2", _result("a2"), [])
    msgs = s.as_messages()
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    assert msgs[2].role == "user"
    assert msgs[3].role == "assistant"


# ---------------------------------------------------------------------------
# Phase 4 persistence tests
# ---------------------------------------------------------------------------


def test_save_creates_json_file(tmp_path: Path) -> None:
    s = Session(max_turns=5)
    s.add_turn("what is X?", _result("X is Y."), [_source()])
    path = tmp_path / "session.json"
    s.save(path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["max_turns"] == 5
    assert len(data["turns"]) == 1
    assert data["turns"][0]["query"] == "what is X?"


def test_load_restores_session(tmp_path: Path) -> None:
    s = Session(max_turns=5)
    s.add_turn("q1", _result("a1"), [_source("file:///a.md")])
    path = tmp_path / "session.json"
    s.save(path)
    loaded = Session.load(path)
    assert loaded.max_turns == 5
    assert len(loaded.turns) == 1
    assert loaded.turns[0].query == "q1"
    assert loaded.turns[0].citations[0].location == "file:///a.md"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Session.load(tmp_path / "nonexistent.json")


def test_save_overwrites_existing(tmp_path: Path) -> None:
    s = Session(max_turns=5)
    path = tmp_path / "session.json"
    s.add_turn("q1", _result("a1"), [])
    s.save(path)
    s.add_turn("q2", _result("a2"), [])
    s.save(path)
    loaded = Session.load(path)
    assert len(loaded.turns) == 2


def test_roundtrip_preserves_ring_buffer_order(tmp_path: Path) -> None:
    s = Session(max_turns=3)
    for i in range(4):
        s.add_turn(f"q{i}", _result(f"a{i}"), [])
    # After overflow, first turn is q1
    path = tmp_path / "session.json"
    s.save(path)
    loaded = Session.load(path)
    assert [t.query for t in loaded.turns] == ["q1", "q2", "q3"]


def test_load_restores_result_text(tmp_path: Path) -> None:
    """Loaded session turns have result text preserved."""
    s = Session(max_turns=5)
    s.add_turn("q1", _result("the answer is 42"), [])
    path = tmp_path / "session.json"
    s.save(path)
    loaded = Session.load(path)
    assert loaded.turns[0].result.text == "the answer is 42"


async def test_flush_awaits_background_write(tmp_path: Path) -> None:
    """flush() completes the pending background write task (if any)."""
    s = Session(max_turns=5)
    s.add_turn("q1", _result("a1"), [])
    path = tmp_path / "session.json"
    # Schedule a background save
    s._schedule_background_save(path)
    # flush() must await the background task so the file exists after
    await s.flush()
    assert path.exists()


async def test_flush_noop_when_no_pending_write() -> None:
    """flush() with no pending write should return without error."""
    s = Session(max_turns=5)
    await s.flush()  # must not raise


async def test_flush_noop_after_sync_save(tmp_path: Path) -> None:
    """flush() after a synchronous save should not raise."""
    s = Session(max_turns=5)
    s.add_turn("q1", _result("a1"), [])
    path = tmp_path / "session.json"
    s.save(path)
    await s.flush()  # no pending task, should be a no-op
