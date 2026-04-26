from typing import Any

from sleuth.engine.synthesizer import Synthesizer, SynthEvent
from sleuth.events import CitationEvent, DoneEvent, ThinkingEvent, TokenEvent
from sleuth.llm.base import ReasoningDelta, Stop, TextDelta
from sleuth.llm.stub import StubLLM
from sleuth.types import Chunk, Source


def _chunk(url: str, text: str = "content") -> Chunk:
    return Chunk(text=text, source=Source(kind="url", location=url), score=0.9)


async def collect_events(synth: Synthesizer, **kwargs: Any) -> list[SynthEvent]:
    events = []
    async for event in synth.synthesize(**kwargs):
        events.append(event)
    return events


async def test_emits_token_events():
    stub = StubLLM([[TextDelta("hello"), TextDelta(" world"), Stop("end_turn")]])
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[_chunk("https://a.com")],
        history=[],
        stats_start_ms=0,
    )
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert len(token_events) >= 1
    assert any(e.text == "hello" for e in token_events)


async def test_emits_done_event():
    stub = StubLLM(["answer"])
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[_chunk("https://a.com")],
        history=[],
        stats_start_ms=0,
    )
    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    assert done_events[0].stats.tokens_out > 0


async def test_emits_citation_for_each_chunk():
    stub = StubLLM(["answer"])
    synth = Synthesizer(llm=stub)
    chunks = [_chunk("https://a.com"), _chunk("https://b.com")]
    events = await collect_events(
        synth,
        query="q",
        chunks=chunks,
        history=[],
        stats_start_ms=0,
    )
    citation_events = [e for e in events if isinstance(e, CitationEvent)]
    assert len(citation_events) == 2
    locations = {e.source.location for e in citation_events}
    assert "https://a.com" in locations
    assert "https://b.com" in locations


async def test_no_thinking_event_when_not_supported():
    stub = StubLLM(["answer"])
    assert stub.supports_reasoning is False
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[_chunk("https://a.com")],
        history=[],
        stats_start_ms=0,
    )
    thinking_events = [e for e in events if isinstance(e, ThinkingEvent)]
    assert thinking_events == []


async def test_thinking_event_when_supported():
    async def reasoner(messages):
        yield ReasoningDelta(text="thinking...")
        yield TextDelta(text="answer")
        yield Stop(reason="end_turn")

    class ReasoningStub(StubLLM):
        supports_reasoning = True

    stub = ReasoningStub(responses=reasoner)
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[_chunk("https://a.com")],
        history=[],
        stats_start_ms=0,
    )
    thinking_events = [e for e in events if isinstance(e, ThinkingEvent)]
    assert len(thinking_events) == 1
    assert thinking_events[0].text == "thinking..."


async def test_builds_result_text_from_token_events():
    stub = StubLLM([[TextDelta("hello"), TextDelta(" world"), Stop("end_turn")]])
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[],
        history=[],
        stats_start_ms=0,
    )
    token_texts = "".join(e.text for e in events if isinstance(e, TokenEvent))
    assert token_texts == "hello world"
