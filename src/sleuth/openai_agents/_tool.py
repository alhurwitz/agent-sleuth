"""OpenAI Agents SDK function-call tool for Sleuth."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from sleuth import Sleuth
from sleuth.events import TokenEvent


def make_sleuth_function_tool(
    agent: Sleuth,
    *,
    name: str = "sleuth_search",
    description: str = (
        "A reasoning-first search tool. Searches documents, code, and the web. "
        "Input: query (str), depth (str, optional: 'auto'|'fast'|'deep')."
    ),
) -> Callable[..., Awaitable[str]]:
    """Return an async callable suitable for registration as an OpenAI Agents SDK tool.

    The returned function signature is::

        async def sleuth_search(query: str, depth: str = "auto") -> str

    Usage::

        from agents import Agent
        from sleuth.openai_agents import make_sleuth_function_tool

        search_fn = make_sleuth_function_tool(sleuth_instance)
        agent = Agent(name="MyAgent", tools=[search_fn])
    """

    async def sleuth_search(query: str, depth: str = "auto") -> str:
        """Search with Sleuth. Returns the synthesized answer."""
        tokens: list[str] = []
        async for event in agent.aask(query, depth=depth):  # type: ignore[arg-type]
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
        return "".join(tokens)

    sleuth_search.__name__ = name
    sleuth_search.__doc__ = description
    # Attach metadata for OpenAI Agents SDK introspection
    sleuth_search.__tool_name__ = name  # type: ignore[attr-defined]
    sleuth_search.__tool_description__ = description  # type: ignore[attr-defined]

    return sleuth_search


__all__ = ["make_sleuth_function_tool"]
