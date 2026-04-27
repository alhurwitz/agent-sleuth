"""AutoGen function-tool integration for Sleuth."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from sleuth import Sleuth
from sleuth.events import TokenEvent


def _run_in_new_loop(coro: Any) -> Any:
    """Run *coro* in a fresh event loop without disturbing the current loop policy."""
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def make_sleuth_autogen_tool(
    agent: Sleuth,
    *,
    name: str = "sleuth_search",
    description: str = (
        "A reasoning-first search tool that searches documents, code, and the web. "
        "Returns a synthesized answer with citations. "
        "Args: query (str) — natural-language question."
    ),
) -> Callable[..., str]:
    """Return a synchronous function suitable for AutoGen function-tool registration.

    AutoGen expects callable tools that run synchronously in the executor context.
    This wrapper drives Sleuth's async engine via ``asyncio.run``.

    Usage with pyautogen (autogen-agentchat >= 0.4)::

        from sleuth.autogen import make_sleuth_autogen_tool
        tool = make_sleuth_autogen_tool(sleuth_instance)
        # Register with an AssistantAgent's tools list:
        assistant = AssistantAgent(name="...", model_client=..., tools=[tool])
    """

    def sleuth_search(query: str) -> str:
        """Search with Sleuth and return the synthesized answer."""

        async def _collect() -> str:
            tokens: list[str] = []
            async for event in agent.aask(query):
                if isinstance(event, TokenEvent):
                    tokens.append(event.text)
            return "".join(tokens)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_in_new_loop, _collect())
                return future.result()
        return _run_in_new_loop(_collect())

    sleuth_search.__name__ = name
    sleuth_search.__doc__ = description
    return sleuth_search


def register_sleuth_tool(
    agent: Sleuth,
    *,
    caller: Any,
    executor: Any,
    name: str = "sleuth_search",
    description: str = (
        "A reasoning-first search tool that searches documents, code, and the web."
    ),
) -> Callable[..., str]:
    """Register Sleuth as a function tool on an AutoGen agent pair.

    Works with both the legacy autogen API (register_for_execution / register_for_llm)
    and the newer autogen-agentchat v0.4+ API (tools list attribute).

    Args:
        agent:       The Sleuth instance to back the tool.
        caller:      An AutoGen agent that calls the tool (AssistantAgent).
        executor:    An AutoGen agent that executes the tool (UserProxyAgent).
        name:        Tool function name (default: ``sleuth_search``).
        description: Tool description shown to the LLM.

    Returns:
        The registered tool function.
    """
    tool_fn = make_sleuth_autogen_tool(agent, name=name, description=description)

    # Legacy autogen registration pattern (v0.2/v0.3 API)
    if hasattr(executor, "register_for_execution"):
        executor.register_for_execution(name=name)(tool_fn)
    if hasattr(caller, "register_for_llm"):
        caller.register_for_llm(name=name, description=description)(tool_fn)

    return tool_fn


__all__ = ["make_sleuth_autogen_tool", "register_sleuth_tool"]
