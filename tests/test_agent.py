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
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("who is guido?")]
    route_events = [e for e in events if isinstance(e, RouteEvent)]
    assert len(route_events) == 1


async def test_aask_yields_search_event():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("who is guido?")]
    search_events = [e for e in events if isinstance(e, SearchEvent)]
    assert len(search_events) == 1
    assert search_events[0].backend == "fake"


async def test_aask_yields_token_events():
    agent = Sleuth(llm=StubLLM(["hello"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("q?")]
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert any(e.text == "hello" for e in token_events)


async def test_aask_yields_done_event_last():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("q?")]
    assert isinstance(events[-1], DoneEvent)


async def test_aask_depth_fast_skips_deep():
    """Fast depth: route event must show 'fast'."""
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("q?", depth="fast")]
    route_event = next(e for e in events if isinstance(e, RouteEvent))
    assert route_event.depth == "fast"


def test_ask_returns_result():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()])
    result = agent.ask("q?")
    assert isinstance(result, Result)
    assert result.text == "answer"


def test_ask_result_has_citations():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()])
    result = agent.ask("q?")
    assert len(result.citations) >= 1


async def test_aask_with_cache_none_still_works():
    agent = Sleuth(llm=StubLLM(["answer"]), backends=[FakeBackend()], cache=None)
    events = [e async for e in agent.aask("q?")]
    assert any(isinstance(e, DoneEvent) for e in events)


async def test_aask_with_session_adds_turn():
    session = Session()
    agent = Sleuth(llm=StubLLM(["a1", "a2"]), backends=[FakeBackend()])
    async for _ in agent.aask("q1?", session=session):
        pass
    assert len(session.turns) == 1
    assert session.turns[0].query == "q1?"
