"""SleuthRetriever — Sleuth as a LangChain BaseRetriever."""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain is not installed. Run: pip install agent-sleuth[langchain]"
    ) from exc

from sleuth import Sleuth
from sleuth.types import Chunk


def _chunk_to_document(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "source": chunk.source.location,
            "kind": chunk.source.kind,
            "title": chunk.source.title or "",
            "score": chunk.score,
        },
    )


class SleuthRetriever(BaseRetriever):  # type: ignore[misc]
    """Expose Sleuth as a LangChain retriever.

    Returns raw chunks (as LangChain Documents) rather than a synthesized
    answer — suitable for use inside RetrievalQA chains.

    Usage::

        from sleuth.langchain import SleuthRetriever
        retriever = SleuthRetriever(agent=sleuth_instance)
        qa = RetrievalQA.from_chain_type(llm=..., retriever=retriever)
    """

    agent: Sleuth

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        import asyncio

        async def _collect() -> list[Document]:
            docs: list[Document] = []
            for backend in self.agent._backends:
                chunks = await backend.search(query, k=10)
                docs.extend(_chunk_to_document(c) for c in chunks)
            return docs

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _collect())
                    return future.result()
            return loop.run_until_complete(_collect())
        except RuntimeError:
            return asyncio.run(_collect())

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        docs: list[Document] = []
        for backend in self.agent._backends:
            chunks = await backend.search(query, k=10)
            docs.extend(_chunk_to_document(c) for c in chunks)
        return docs
