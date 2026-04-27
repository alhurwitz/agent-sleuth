"""Tests for SleuthCrewAITool — CrewAI BaseTool subclass."""

import pytest

pytest.importorskip("crewai", reason="crewai extra not installed")

from crewai.tools import BaseTool

from sleuth import Sleuth
from sleuth.crewai import SleuthCrewAITool


@pytest.mark.adapter
def test_tool_is_crewai_base_tool(sleuth_agent: Sleuth) -> None:
    tool = SleuthCrewAITool(agent=sleuth_agent)
    assert isinstance(tool, BaseTool)


@pytest.mark.adapter
def test_tool_has_name_and_description(sleuth_agent: Sleuth) -> None:
    tool = SleuthCrewAITool(agent=sleuth_agent)
    assert tool.name == "sleuth_search"
    assert len(tool.description) > 10


@pytest.mark.adapter
def test_tool_run_returns_string(sleuth_agent: Sleuth) -> None:
    tool = SleuthCrewAITool(agent=sleuth_agent)
    result = tool._run("What is 42?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
async def test_tool_arun_returns_string(sleuth_agent: Sleuth) -> None:
    tool = SleuthCrewAITool(agent=sleuth_agent)
    result = await tool._arun("What is 42?")
    assert isinstance(result, str)


@pytest.mark.adapter
def test_tool_exposes_event_stream_via_on_event(sleuth_agent: Sleuth) -> None:
    """on_event callback receives Sleuth events during _run."""
    received: list[str] = []
    tool = SleuthCrewAITool(agent=sleuth_agent, on_event=lambda e: received.append(e.type))
    tool._run("test")
    assert "done" in received
