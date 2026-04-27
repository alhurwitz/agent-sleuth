"""SleuthTool — Sleuth as a LangChain BaseTool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain is not installed. Run: pip install agent-sleuth[langchain]"
    ) from exc

from sleuth import Sleuth
from sleuth.events import TokenEvent


def _run_in_new_loop(coro: Any) -> Any:
    """Run *coro* in a fresh event loop without disturbing the current loop policy."""
    import asyncio

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun


class SleuthTool(BaseTool):  # type: ignore[misc]
    """Expose Sleuth search as a LangChain tool.

    Usage::

        from sleuth.langchain import SleuthTool
        tool = SleuthTool(agent=sleuth_instance)
        agent_executor = AgentExecutor(tools=[tool], ...)
    """

    name: str = "sleuth_search"
    description: str = (
        "A reasoning-first search tool. Use for questions that require searching "
        "documents, code, or the web. Input: a natural-language query string."
    )
    agent: Sleuth

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        import asyncio

        async def _collect() -> str:
            tokens: list[str] = []
            async for event in self.agent.aask(query):
                if isinstance(event, TokenEvent):
                    tokens.append(event.text)
            return "".join(tokens)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_in_new_loop, _collect())
                return future.result()
        return _run_in_new_loop(_collect())

    async def _arun(
        self,
        query: str,
        run_manager: Any = None,
    ) -> str:
        tokens: list[str] = []
        async for event in self.agent.aask(query):
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
        return "".join(tokens)
