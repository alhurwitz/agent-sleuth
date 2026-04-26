from sleuth.llm.base import Message
from sleuth.memory.session import Session
from sleuth.types import Result, RunStats


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
