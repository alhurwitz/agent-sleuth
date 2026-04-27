"""Tests for Pydantic AI adapter — schema-validated tool."""

import pytest

pytest.importorskip("pydantic_ai", reason="pydantic-ai extra not installed")

from pydantic import BaseModel

from sleuth import Sleuth
from sleuth.pydantic_ai import SleuthInput, make_sleuth_tool


@pytest.mark.adapter
def test_sleuth_input_is_pydantic_model() -> None:
    """SleuthInput must be a Pydantic BaseModel for Pydantic AI schema inference."""
    assert issubclass(SleuthInput, BaseModel)
    # Must have query field
    assert "query" in SleuthInput.model_fields
    # Must have depth field
    assert "depth" in SleuthInput.model_fields


@pytest.mark.adapter
def test_make_sleuth_tool_returns_callable(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_tool(sleuth_agent)
    assert callable(tool_fn)


@pytest.mark.adapter
async def test_sleuth_tool_with_valid_input(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_tool(sleuth_agent)
    inputs = SleuthInput(query="What is 42?")
    result = await tool_fn(inputs)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
async def test_sleuth_tool_with_depth_override(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_tool(sleuth_agent)
    inputs = SleuthInput(query="deep question", depth="deep")
    result = await tool_fn(inputs)
    assert isinstance(result, str)
