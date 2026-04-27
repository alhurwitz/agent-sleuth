"""SleuthCrewAITool — CrewAI BaseTool subclass backed by Sleuth."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

try:
    from crewai.tools import BaseTool
except ImportError as exc:  # pragma: no cover
    raise ImportError("CrewAI is not installed. Run: pip install agent-sleuth[crewai]") from exc

from pydantic import BaseModel, Field

from sleuth import Sleuth
from sleuth.events import Event, TokenEvent

OnEventCallback = Callable[[Event], None] | None


class _SleuthInput(BaseModel):
    """Input schema for SleuthCrewAITool."""

    query: str = Field(description="The natural-language search query.")


class SleuthCrewAITool(BaseTool):  # type: ignore[misc]
    """Sleuth as a CrewAI BaseTool.

    CrewAI has no native async callback surface. The ``on_event`` parameter
    exposes Sleuth's event stream via a sync callback for observability.

    Usage::

        from sleuth.crewai import SleuthCrewAITool
        tool = SleuthCrewAITool(agent=sleuth_instance)
        crew = Crew(agents=[...], tasks=[...], tools=[tool])
    """

    name: str = "sleuth_search"
    description: str = (
        "A reasoning-first search tool. Searches documents, code, and the web. "
        "Input: a natural-language query string. Returns a synthesized answer."
    )
    args_schema: type[BaseModel] = _SleuthInput

    def __init__(self, agent: Sleuth, on_event: OnEventCallback = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass pydantic validation for private attrs
        object.__setattr__(self, "_sleuth_agent", agent)
        object.__setattr__(self, "_on_event_cb", on_event)

    def _run(self, query: str, **kwargs: Any) -> str:
        """Synchronous implementation required by CrewAI BaseTool."""
        agent: Sleuth = object.__getattribute__(self, "_sleuth_agent")
        on_event: OnEventCallback = object.__getattribute__(self, "_on_event_cb")

        async def _collect() -> str:
            tokens: list[str] = []
            async for event in agent.aask(query):
                if isinstance(event, TokenEvent):
                    tokens.append(event.text)
                if on_event is not None:
                    on_event(event)
            return "".join(tokens)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _collect())
                return future.result()
        return asyncio.run(_collect())

    async def _arun(self, query: str, **kwargs: Any) -> str:
        """Async implementation for forward-compat."""
        agent: Sleuth = object.__getattribute__(self, "_sleuth_agent")
        on_event: OnEventCallback = object.__getattribute__(self, "_on_event_cb")
        tokens: list[str] = []
        async for event in agent.aask(query):
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
            if on_event is not None:
                on_event(event)
        return "".join(tokens)
