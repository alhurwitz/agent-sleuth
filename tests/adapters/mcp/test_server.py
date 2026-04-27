"""Tests for sleuth.mcp.server — protocol compliance and event mapping."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from sleuth.events import (
    DoneEvent,
    FetchEvent,
    RouteEvent,
    SearchEvent,
    TokenEvent,
)
from sleuth.mcp.server import (
    SleuthMcpServer,
    build_search_tool,
    build_summarize_tool,
    events_to_notifications,
)
from sleuth.types import Result, RunStats, Source

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stats() -> RunStats:
    return RunStats(
        latency_ms=120,
        first_token_ms=80,
        tokens_in=10,
        tokens_out=20,
        cache_hits={},
        backends_called=["stub"],
    )


def _make_result(text: str = "answer") -> Result[Any]:
    return Result(
        text=text,
        citations=[Source(kind="url", location="https://example.com", title="Example")],
        data=None,
        stats=_make_stats(),
    )


# ---------------------------------------------------------------------------
# Tool schema tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_search_tool_schema() -> None:
    tool = build_search_tool()
    assert tool["name"] == "search"
    schema = tool["inputSchema"]
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "query" in props
    assert "depth" in props
    assert "schema" in props
    assert schema["required"] == ["query"]


@pytest.mark.unit
def test_summarize_tool_schema() -> None:
    tool = build_summarize_tool()
    assert tool["name"] == "summarize"
    schema = tool["inputSchema"]
    props = schema["properties"]
    assert "target" in props
    assert "length" in props
    assert schema["required"] == ["target"]


# ---------------------------------------------------------------------------
# Event → notification mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_events_to_notifications_search_event() -> None:
    """SearchEvent produces a progress notification dict."""
    ev = SearchEvent(type="search", backend="tavily", query="foo")
    notes = [n async for n in events_to_notifications([ev])]
    assert len(notes) == 1
    n = notes[0]
    assert n["type"] == "progress"
    assert "search" in n["message"].lower() or "tavily" in n["message"].lower()


@pytest.mark.unit
async def test_events_to_notifications_fetch_event() -> None:
    ev = FetchEvent(type="fetch", url="https://example.com", status=200)
    notes = [n async for n in events_to_notifications([ev])]
    assert len(notes) == 1
    assert notes[0]["type"] == "progress"


@pytest.mark.unit
async def test_events_to_notifications_token_event() -> None:
    ev = TokenEvent(type="token", text="Hello")
    notes = [n async for n in events_to_notifications([ev])]
    assert len(notes) == 1
    n = notes[0]
    assert n["type"] == "progress"
    assert "Hello" in n["message"]


@pytest.mark.unit
async def test_events_to_notifications_done_event_suppressed() -> None:
    """DoneEvent is NOT emitted as a progress notification — it becomes the final result."""
    ev = DoneEvent(type="done", stats=_make_stats())
    notes = [n async for n in events_to_notifications([ev])]
    assert notes == []


@pytest.mark.unit
async def test_events_to_notifications_route_event() -> None:
    ev = RouteEvent(type="route", depth="fast", reason="simple query")
    notes = [n async for n in events_to_notifications([ev])]
    assert len(notes) == 1
    assert notes[0]["type"] == "progress"


# ---------------------------------------------------------------------------
# SleuthMcpServer construction + tool listing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_server_tool_listing() -> None:
    mock_sleuth = MagicMock()
    server = SleuthMcpServer(sleuth=mock_sleuth)
    tools = server.list_tools()
    names = [t["name"] for t in tools]
    assert "search" in names
    assert "summarize" in names
    assert len(tools) == 2


# ---------------------------------------------------------------------------
# Tool call: search
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_search_tool_call_returns_result() -> None:
    """search() drains the event stream and returns a structured result."""
    mock_sleuth = MagicMock()
    mock_sleuth.ask = MagicMock(return_value=_make_result("The answer"))

    server = SleuthMcpServer(sleuth=mock_sleuth)
    result = await server.call_tool("search", {"query": "test query"})

    assert result["type"] == "result"
    assert "text" in result
    assert "citations" in result


@pytest.mark.unit
async def test_search_tool_call_with_depth() -> None:
    mock_sleuth = MagicMock()
    mock_sleuth.ask = MagicMock(return_value=_make_result("ans"))
    server = SleuthMcpServer(sleuth=mock_sleuth)
    result = await server.call_tool("search", {"query": "q", "depth": "deep"})
    assert result["type"] == "result"
    call_kwargs = mock_sleuth.ask.call_args
    assert call_kwargs.kwargs.get("depth") == "deep"


# ---------------------------------------------------------------------------
# Tool call: summarize
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_summarize_tool_call_returns_result() -> None:
    mock_sleuth = MagicMock()
    mock_sleuth.summarize = MagicMock(return_value=_make_result("summary"))
    server = SleuthMcpServer(sleuth=mock_sleuth)
    result = await server.call_tool("summarize", {"target": "https://example.com"})
    assert result["type"] == "result"
    assert result["text"] == "summary"


@pytest.mark.unit
async def test_summarize_tool_call_with_length() -> None:
    mock_sleuth = MagicMock()
    mock_sleuth.summarize = MagicMock(return_value=_make_result("s"))
    server = SleuthMcpServer(sleuth=mock_sleuth)
    await server.call_tool("summarize", {"target": "path/to/doc.md", "length": "brief"})
    call_kwargs = mock_sleuth.summarize.call_args
    assert call_kwargs.kwargs.get("length") == "brief"


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_unknown_tool_raises() -> None:
    mock_sleuth = MagicMock()
    server = SleuthMcpServer(sleuth=mock_sleuth)
    with pytest.raises(ValueError, match="Unknown tool"):
        await server.call_tool("explode", {})


# ---------------------------------------------------------------------------
# CLI / __main__ tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_help_exits_zero() -> None:
    """sleuth-mcp --help exits with code 0."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "sleuth.mcp", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--transport" in result.stdout
    assert "--host" in result.stdout
    assert "--port" in result.stdout


@pytest.mark.unit
def test_main_unknown_flag_exits_nonzero() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "sleuth.mcp", "--bogus-flag"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


@pytest.mark.unit
def test_main_missing_config_exits_nonzero(tmp_path: Any) -> None:
    """Pointing at a nonexistent config file exits with error."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "sleuth.mcp", "--config", str(tmp_path / "nope.toml")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr or "not found" in result.stdout


# ---------------------------------------------------------------------------
# MCP SDK protocol round-trip (requires mcp extra)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_list_tools_roundtrip() -> None:
    """SleuthMcpServer.list_tools() returns valid MCP tool descriptors."""
    mock_sleuth = MagicMock()
    server = SleuthMcpServer(sleuth=mock_sleuth)
    tools = server.list_tools()

    # Validate each tool has the required MCP tool descriptor fields
    for tool in tools:
        assert "name" in tool, f"Tool missing 'name': {tool}"
        assert "description" in tool, f"Tool missing 'description': {tool}"
        assert "inputSchema" in tool, f"Tool missing 'inputSchema': {tool}"
        schema = tool["inputSchema"]
        assert schema.get("type") == "object", f"inputSchema.type must be 'object': {schema}"
        assert "properties" in schema
        assert "required" in schema
        assert isinstance(schema["required"], list)


@pytest.mark.unit
async def test_search_result_structure() -> None:
    """search tool returns a dict conforming to the expected MCP result shape."""
    mock_sleuth = MagicMock()
    mock_sleuth.ask = MagicMock(return_value=_make_result("round-trip answer"))
    server = SleuthMcpServer(sleuth=mock_sleuth)
    result = await server.call_tool("search", {"query": "round trip"})

    # MCP result must be JSON-serialisable
    serialised = json.dumps(result)  # raises if not serialisable
    parsed = json.loads(serialised)

    assert parsed["type"] == "result"
    assert isinstance(parsed["text"], str)
    assert isinstance(parsed["citations"], list)
    for citation in parsed["citations"]:
        assert "location" in citation
        assert "kind" in citation


@pytest.mark.unit
async def test_summarize_result_structure() -> None:
    mock_sleuth = MagicMock()
    mock_sleuth.summarize = MagicMock(return_value=_make_result("summary text"))
    server = SleuthMcpServer(sleuth=mock_sleuth)
    result = await server.call_tool("summarize", {"target": "https://example.com"})

    parsed = json.loads(json.dumps(result))
    assert parsed["type"] == "result"
    assert parsed["text"] == "summary text"
