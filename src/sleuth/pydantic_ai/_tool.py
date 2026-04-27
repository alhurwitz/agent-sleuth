"""Pydantic AI tool with schema validation for Sleuth."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, Field

from sleuth import Sleuth
from sleuth.events import TokenEvent


class SleuthInput(BaseModel):
    """Validated input schema for Sleuth tool — Pydantic AI infers this automatically."""

    query: str = Field(description="The natural-language search query.")
    depth: Literal["auto", "fast", "deep"] = Field(
        default="auto",
        description="Search depth: 'auto' lets Sleuth decide, 'fast' skips planning, "
        "'deep' uses full reflect loop.",
    )


def make_sleuth_tool(
    agent: Sleuth,
) -> Callable[[SleuthInput], Awaitable[str]]:
    """Return a Pydantic AI-compatible async tool function backed by Sleuth.

    Pydantic AI infers the JSON schema from ``SleuthInput``. Register the
    returned function with ``@pydantic_agent.tool`` or pass it to
    ``Agent(tools=[make_sleuth_tool(sleuth_instance)])``.

    Usage::

        from pydantic_ai import Agent
        from sleuth.pydantic_ai import make_sleuth_tool, SleuthInput

        tool = make_sleuth_tool(sleuth_instance)
        result = await tool(SleuthInput(query="What is Sleuth?"))
    """

    async def _sleuth_tool(inputs: SleuthInput) -> str:
        tokens: list[str] = []
        async for event in agent.aask(inputs.query, depth=inputs.depth):
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
        return "".join(tokens)

    _sleuth_tool.__name__ = "sleuth_search"
    _sleuth_tool.__doc__ = (
        "Search with Sleuth. Returns synthesized answer. Input: SleuthInput(query, depth)."
    )
    return _sleuth_tool


__all__ = ["SleuthInput", "make_sleuth_tool"]
