"""Tests for AutoGen function-tool registration."""

import pytest

pytest.importorskip("autogen_agentchat", reason="autogen extra not installed")

from sleuth import Sleuth
from sleuth.autogen import make_sleuth_autogen_tool, register_sleuth_tool


@pytest.mark.adapter
def test_make_sleuth_autogen_tool_returns_callable(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_autogen_tool(sleuth_agent)
    assert callable(tool_fn)


@pytest.mark.adapter
def test_autogen_tool_has_name_and_docstring(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_autogen_tool(sleuth_agent)
    assert tool_fn.__name__ == "sleuth_search"
    assert tool_fn.__doc__ is not None and len(tool_fn.__doc__) > 10


@pytest.mark.adapter
def test_autogen_tool_sync_call_returns_string(sleuth_agent: Sleuth) -> None:
    """AutoGen function tools are called synchronously by the framework."""
    tool_fn = make_sleuth_autogen_tool(sleuth_agent)
    result = tool_fn(query="What is 42?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
def test_register_sleuth_tool_attaches_to_agent(sleuth_agent: Sleuth) -> None:
    """register_sleuth_tool returns a callable and works with duck-typed agents."""

    class FakeCaller:
        """Minimal duck-typed caller (autogen-agentchat v0.4 has no register_for_llm)."""

        pass

    class FakeExecutor:
        """Minimal duck-typed executor."""

        pass

    caller = FakeCaller()
    executor = FakeExecutor()

    # Should not raise even though duck-typed agents lack legacy registration methods
    tool_fn = register_sleuth_tool(sleuth_agent, caller=caller, executor=executor)
    assert callable(tool_fn)
