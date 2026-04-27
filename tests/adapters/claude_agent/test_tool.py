"""Tests for SleuthClaudeTool — Sleuth as a Claude Agent SDK tool."""

import pytest

from sleuth import Sleuth
from sleuth.claude_agent import SleuthClaudeTool


@pytest.mark.adapter
def test_sleuth_claude_tool_has_required_fields(sleuth_agent: Sleuth) -> None:
    tool = SleuthClaudeTool(agent=sleuth_agent)
    # Claude Agent SDK tools must expose: name, description, input_schema
    assert isinstance(tool.name, str) and len(tool.name) > 0
    assert isinstance(tool.description, str) and len(tool.description) > 0
    assert isinstance(tool.input_schema, dict)
    assert "properties" in tool.input_schema


@pytest.mark.adapter
def test_sleuth_claude_tool_input_schema_has_query(sleuth_agent: Sleuth) -> None:
    tool = SleuthClaudeTool(agent=sleuth_agent)
    props = tool.input_schema["properties"]
    assert "query" in props
    assert props["query"]["type"] == "string"


@pytest.mark.adapter
async def test_sleuth_claude_tool_call_returns_text(sleuth_agent: Sleuth) -> None:
    tool = SleuthClaudeTool(agent=sleuth_agent)
    result = await tool.call({"query": "What is Sleuth?"})
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
async def test_sleuth_claude_tool_emits_progress_blocks(sleuth_agent: Sleuth) -> None:
    """Progress blocks are emitted for search and token events."""
    tool = SleuthClaudeTool(agent=sleuth_agent)
    progress_blocks: list[dict] = []

    async def on_progress(block: dict) -> None:
        progress_blocks.append(block)

    await tool.call({"query": "test"}, on_progress=on_progress)
    # At minimum, a token or search progress block should have been emitted
    assert len(progress_blocks) >= 1
    block_types = {b.get("type") for b in progress_blocks}
    assert block_types & {"search_progress", "token_progress", "done_progress"}
