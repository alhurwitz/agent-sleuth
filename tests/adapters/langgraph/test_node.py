"""Tests for the LangGraph node factory."""

import pytest

pytest.importorskip("langgraph", reason="langgraph extra not installed")

from sleuth import Sleuth
from sleuth.langgraph import make_sleuth_node


@pytest.mark.adapter
def test_make_sleuth_node_returns_callable(sleuth_agent: Sleuth) -> None:
    node = make_sleuth_node(sleuth_agent)
    assert callable(node)


@pytest.mark.adapter
async def test_sleuth_node_reads_query_from_state(sleuth_agent: Sleuth) -> None:
    node = make_sleuth_node(sleuth_agent)
    state = {"query": "What is 42?", "messages": []}
    result = await node(state)
    # Node should return a dict with at least "answer" key
    assert isinstance(result, dict)
    assert "answer" in result
    assert isinstance(result["answer"], str)


@pytest.mark.adapter
async def test_sleuth_node_uses_messages_key_when_no_query(sleuth_agent: Sleuth) -> None:
    """When state has no 'query' key, fall back to last message content."""
    from langchain_core.messages import HumanMessage

    node = make_sleuth_node(sleuth_agent)
    state = {"messages": [HumanMessage(content="Explain auth flow")]}
    result = await node(state)
    assert "answer" in result


@pytest.mark.adapter
async def test_sleuth_node_custom_query_key(sleuth_agent: Sleuth) -> None:
    node = make_sleuth_node(sleuth_agent, query_key="search_input")
    state = {"search_input": "custom key query"}
    result = await node(state)
    assert "answer" in result
