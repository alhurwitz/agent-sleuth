"""Tests for SleuthTool — the LangChain tool surface."""

import pytest

pytest.importorskip("langchain_core", reason="langchain extra not installed")

from langchain_core.tools import BaseTool

from sleuth import Sleuth
from sleuth.langchain import SleuthTool


@pytest.mark.adapter
def test_sleuth_tool_is_langchain_base_tool(sleuth_agent: Sleuth) -> None:
    tool = SleuthTool(agent=sleuth_agent)
    assert isinstance(tool, BaseTool)


@pytest.mark.adapter
def test_sleuth_tool_name_and_description(sleuth_agent: Sleuth) -> None:
    tool = SleuthTool(agent=sleuth_agent)
    assert tool.name == "sleuth_search"
    assert len(tool.description) > 10  # non-empty description


@pytest.mark.adapter
def test_sleuth_tool_run_returns_string(sleuth_agent: Sleuth) -> None:
    tool = SleuthTool(agent=sleuth_agent)
    result = tool.run("What is the capital of France?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
async def test_sleuth_tool_arun_returns_string(sleuth_agent: Sleuth) -> None:
    tool = SleuthTool(agent=sleuth_agent)
    result = await tool.arun("What is the capital of France?")
    assert isinstance(result, str)
    assert len(result) > 0
