"""SleuthClaudeTool — Sleuth as a Claude Agent SDK tool with progress message blocks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from sleuth import Sleuth
from sleuth.events import (
    CitationEvent,
    DoneEvent,
    Event,
    SearchEvent,
    ThinkingEvent,
    TokenEvent,
)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


def _event_to_progress_block(event: Event) -> dict[str, Any] | None:
    """Map a Sleuth event to a Claude Agent SDK progress message block, or None."""
    if isinstance(event, SearchEvent):
        return {
            "type": "search_progress",
            "backend": event.backend,
            "query": event.query,
        }
    if isinstance(event, TokenEvent):
        return {"type": "token_progress", "text": event.text}
    if isinstance(event, ThinkingEvent):
        return {"type": "thinking_progress", "text": event.text}
    if isinstance(event, CitationEvent):
        return {
            "type": "citation_progress",
            "index": event.index,
            "source": event.source.model_dump(),
        }
    if isinstance(event, DoneEvent):
        return {
            "type": "done_progress",
            "latency_ms": event.stats.latency_ms,
            "backends_called": event.stats.backends_called,
        }
    # RouteEvent, PlanEvent, FetchEvent, CacheHitEvent — not surfaced as blocks
    return None


class SleuthClaudeTool:
    """Sleuth as a Claude Agent SDK tool.

    The Claude Agent SDK represents tool progress as typed message blocks
    streamed alongside the assistant response. Sleuth events map to these
    blocks so the agent can surface search progress in real time.

    Usage::

        from sleuth.claude_agent import SleuthClaudeTool
        tool = SleuthClaudeTool(agent=sleuth_instance)
        # Register with Claude Agent SDK agent:
        agent = ClaudeAgent(tools=[tool], ...)
    """

    name: str = "sleuth_search"
    description: str = (
        "A reasoning-first search tool that searches documents, code, and the web. "
        "Returns a synthesized answer with citations. Input: a natural-language query."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "depth": {
                "type": "string",
                "enum": ["auto", "fast", "deep"],
                "description": "Search depth. Defaults to 'auto'.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, agent: Sleuth) -> None:
        self._agent = agent

    async def call(
        self,
        inputs: dict[str, Any],
        *,
        on_progress: ProgressCallback = None,
    ) -> str:
        """Execute the search and optionally stream progress blocks.

        Args:
            inputs: Dict with at least ``query``. Optionally ``depth``.
            on_progress: Async callback receiving progress block dicts.

        Returns:
            The synthesized answer text.
        """
        query: str = inputs["query"]
        depth: str = inputs.get("depth", "auto")
        tokens: list[str] = []

        async for event in self._agent.aask(query, depth=depth):  # type: ignore[arg-type]
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
            if on_progress is not None:
                block = _event_to_progress_block(event)
                if block is not None:
                    await on_progress(block)

        return "".join(tokens)
