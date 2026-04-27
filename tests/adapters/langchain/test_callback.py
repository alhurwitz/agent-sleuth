"""Tests for SleuthCallbackHandler — bridges Sleuth events to LangChain callbacks."""

import pytest

pytest.importorskip("langchain_core", reason="langchain extra not installed")

from langchain_core.callbacks import BaseCallbackHandler

from sleuth import Sleuth
from sleuth.events import DoneEvent, SearchEvent
from sleuth.langchain import SleuthCallbackHandler
from sleuth.types import RunStats


@pytest.mark.adapter
def test_callback_handler_is_base_callback_handler() -> None:
    handler = SleuthCallbackHandler()
    assert isinstance(handler, BaseCallbackHandler)


@pytest.mark.adapter
def test_on_search_event_calls_on_tool_start() -> None:
    calls: list[dict] = []

    class SpyHandler(SleuthCallbackHandler):
        def on_tool_start(self, serialized, input_str, **kwargs):  # type: ignore[override]
            calls.append({"action": "tool_start", "input": input_str})

    handler = SpyHandler()
    event = SearchEvent(type="search", backend="fake", query="test query")
    handler.on_sleuth_event(event)
    assert len(calls) == 1
    assert calls[0]["input"] == "test query"


@pytest.mark.adapter
def test_on_done_event_calls_on_tool_end() -> None:
    calls: list[dict] = []

    class SpyHandler(SleuthCallbackHandler):
        def on_tool_end(self, output, **kwargs):  # type: ignore[override]
            calls.append({"action": "tool_end", "output": output})

    handler = SpyHandler()
    stats = RunStats(
        latency_ms=100,
        first_token_ms=50,
        tokens_in=10,
        tokens_out=20,
        cache_hits={},
        backends_called=["fake"],
    )
    event = DoneEvent(type="done", stats=stats)
    handler.on_sleuth_event(event)
    assert len(calls) == 1


@pytest.mark.adapter
async def test_sleuth_agent_with_callback_handler(sleuth_agent: Sleuth) -> None:
    """Run Sleuth with the callback handler attached; verify no exceptions."""
    handler = SleuthCallbackHandler()
    collected: list[str] = []
    async for event in sleuth_agent.aask("test query"):
        handler.on_sleuth_event(event)
        collected.append(event.type)
    assert "done" in collected
