"""SleuthRetriever — Sleuth as a LlamaIndex BaseRetriever."""

from __future__ import annotations

from typing import Any

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LlamaIndex is not installed. Run: pip install agent-sleuth[llamaindex]"
    ) from exc

from sleuth import Sleuth
from sleuth.types import Chunk


def _run_in_new_loop(coro: Any) -> Any:
    """Run *coro* in a fresh event loop without disturbing the current loop policy."""
    import asyncio

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def _chunk_to_node_with_score(chunk: Chunk) -> NodeWithScore:
    node = TextNode(
        text=chunk.text,
        metadata={
            "source": chunk.source.location,
            "kind": chunk.source.kind,
            "title": chunk.source.title or "",
        },
    )
    return NodeWithScore(node=node, score=chunk.score or 0.0)


class SleuthRetriever(BaseRetriever):  # type: ignore[misc]
    """Expose Sleuth backends as a LlamaIndex retriever.

    Usage::

        from sleuth.llamaindex import SleuthRetriever
        retriever = SleuthRetriever(agent=sleuth_instance)
    """

    def __init__(self, agent: Sleuth, **kwargs: Any) -> None:
        self._agent = agent
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        import asyncio

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_in_new_loop, self._aretrieve(query_bundle))
                return future.result()
        return _run_in_new_loop(self._aretrieve(query_bundle))

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes: list[NodeWithScore] = []
        for backend in self._agent._backends:
            chunks = await backend.search(query_bundle.query_str, k=10)
            nodes.extend(_chunk_to_node_with_score(c) for c in chunks)
        return nodes
