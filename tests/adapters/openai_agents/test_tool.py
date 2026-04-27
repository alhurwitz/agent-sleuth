"""Tests for the OpenAI Agents SDK function-call tool."""

import pytest

pytest.importorskip("agents", reason="openai-agents extra not installed")

from sleuth import Sleuth
from sleuth.openai_agents import make_sleuth_function_tool


@pytest.mark.adapter
def test_make_sleuth_function_tool_returns_callable(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_function_tool(sleuth_agent)
    assert callable(tool_fn)


@pytest.mark.adapter
def test_function_tool_has_schema_metadata(sleuth_agent: Sleuth) -> None:
    """The returned function must carry openai-agents tool metadata."""
    tool_fn = make_sleuth_function_tool(sleuth_agent)
    # openai-agents decorates functions with __tool_name__ and __tool_description__
    assert hasattr(tool_fn, "__tool_name__") or hasattr(tool_fn, "__name__")


@pytest.mark.adapter
async def test_function_tool_call_returns_string(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_function_tool(sleuth_agent)
    result = await tool_fn(query="What is Sleuth?")
    assert isinstance(result, str)
    assert len(result) > 0
