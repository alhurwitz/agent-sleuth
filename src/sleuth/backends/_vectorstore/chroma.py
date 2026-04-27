"""Chroma adapter for VectorStoreRAG.

Install the extra: pip install agent-sleuth[chroma]

chromadb is imported lazily; the adapter is synchronous internally
(chromadb's .query() is sync) but wrapped in asyncio.to_thread so the
event loop is never blocked.
"""

from __future__ import annotations

import asyncio
from typing import Any

from sleuth.backends.vectorstore import VectorMatch
from sleuth.types import Source


class ChromaAdapter:
    """Wraps a Chroma Collection object.

    Parameters
    ----------
    collection:
        A ``chromadb.Collection`` (sync or async) already obtained by the
        caller. Sleuth never creates, modifies, or deletes collections.
    source_key:
        Metadata field that contains the chunk URL / file path.
    """

    def __init__(
        self,
        collection: Any,
        *,
        source_key: str = "source",
    ) -> None:
        self._collection = collection
        self._source_key = source_key

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]:
        # chromadb collection.query is synchronous; run in thread to avoid blocking.
        result: dict[str, Any] = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs: list[str] = result["documents"][0] if result["documents"] else []
        metadatas: list[dict[str, Any]] = result["metadatas"][0] if result["metadatas"] else []
        distances: list[float] = result["distances"][0] if result["distances"] else []

        matches: list[VectorMatch] = []
        for doc, meta, dist in zip(docs, metadatas, distances, strict=False):
            # Convert L2 distance to a [0,1] similarity score (1 = identical).
            score = max(0.0, 1.0 - dist)
            location: str = meta.get(self._source_key, "")
            extra_meta = {mk: mv for mk, mv in meta.items() if mk != self._source_key}
            matches.append(
                VectorMatch(
                    text=doc,
                    score=score,
                    source=Source(kind="file", location=location, title=None),
                    metadata=extra_meta,
                )
            )
        return matches
