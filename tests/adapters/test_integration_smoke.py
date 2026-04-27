"""Integration smoke test — one real Q&A round-trip per installed adapter.

Requires: SLEUTH_SMOKE_QUERY env var and at least one backend configured.
Marked integration; only runs nightly or with: pytest -m integration
"""

import os

import pytest

QUERY = os.getenv("SLEUTH_SMOKE_QUERY", "What is agent-sleuth?")


@pytest.mark.integration
def test_langchain_tool_round_trip(sleuth_agent):
    pytest.importorskip("langchain_core")
    from sleuth.langchain import SleuthTool

    tool = SleuthTool(agent=sleuth_agent)
    result = tool.run(QUERY)
    assert isinstance(result, str) and len(result) > 0


@pytest.mark.integration
async def test_llamaindex_query_engine_round_trip(sleuth_agent):
    pytest.importorskip("llama_index")
    from sleuth.llamaindex import SleuthQueryEngine

    engine = SleuthQueryEngine(agent=sleuth_agent)
    response = await engine.aquery(QUERY)
    assert response.response and len(response.response) > 0


@pytest.mark.integration
async def test_langgraph_node_round_trip(sleuth_agent):
    pytest.importorskip("langgraph")
    from sleuth.langgraph import make_sleuth_node

    node = make_sleuth_node(sleuth_agent)
    result = await node({"query": QUERY})
    assert result.get("answer")


@pytest.mark.integration
async def test_claude_agent_tool_round_trip(sleuth_agent):
    from sleuth.claude_agent import SleuthClaudeTool

    tool = SleuthClaudeTool(agent=sleuth_agent)
    result = await tool.call({"query": QUERY})
    assert isinstance(result, str) and len(result) > 0
