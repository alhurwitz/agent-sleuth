"""LangGraph node factory for Sleuth."""

from __future__ import annotations

from typing import Any

from sleuth import Sleuth
from sleuth.events import TokenEvent


def make_sleuth_node(
    agent: Sleuth,
    *,
    query_key: str = "query",
    answer_key: str = "answer",
) -> Any:
    """Return an async LangGraph node function backed by Sleuth search.

    The returned coroutine has signature ``async (state: dict) -> dict``.
    It reads the query from ``state[query_key]`` (or falls back to the last
    message's ``content`` if no such key exists), runs ``agent.aask``, and
    returns ``{answer_key: synthesized_text}``.

    Usage in a LangGraph graph::

        from langgraph.graph import StateGraph
        from sleuth.langgraph import make_sleuth_node

        graph = StateGraph(MyState)
        graph.add_node("search", make_sleuth_node(sleuth_agent))
    """

    async def _node(state: dict[str, Any]) -> dict[str, Any]:
        # Extract query from state
        if query_key in state:
            query: str = state[query_key]
        else:
            # Fall back to last message content (LangGraph messages pattern)
            messages = state.get("messages", [])
            if messages:
                last = messages[-1]
                query = last.content if hasattr(last, "content") else str(last)
            else:
                query = ""

        tokens: list[str] = []
        async for event in agent.aask(query):
            if isinstance(event, TokenEvent):
                tokens.append(event.text)

        return {answer_key: "".join(tokens)}

    return _node
