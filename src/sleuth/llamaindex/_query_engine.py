"""SleuthQueryEngine — Sleuth as a LlamaIndex QueryEngine."""

from __future__ import annotations

from typing import Any

try:
    from llama_index.core.base.response.schema import Response
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.schema import QueryBundle
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LlamaIndex is not installed. Run: pip install agent-sleuth[llamaindex]"
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


class SleuthQueryEngine(BaseQueryEngine):  # type: ignore[misc]
    """Expose Sleuth as a LlamaIndex QueryEngine.

    Usage::

        from sleuth.llamaindex import SleuthQueryEngine
        engine = SleuthQueryEngine(agent=sleuth_instance)
        response = engine.query("How does auth work?")
    """

    def __init__(self, agent: Sleuth, **kwargs: Any) -> None:
        self._agent = agent
        # BaseQueryEngine requires callback_manager; default to None (no callbacks)
        kwargs.setdefault("callback_manager", None)
        super().__init__(**kwargs)

    def _query(self, query_bundle: QueryBundle) -> Response:
        import asyncio

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_in_new_loop, self._aquery(query_bundle))
                return future.result()
        return _run_in_new_loop(self._aquery(query_bundle))

    def _get_prompt_modules(self) -> dict[str, Any]:
        """Return empty prompt modules (Sleuth manages prompting internally)."""
        return {}

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        tokens: list[str] = []
        async for event in self._agent.aask(query_bundle.query_str):
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
        return Response(response="".join(tokens))
