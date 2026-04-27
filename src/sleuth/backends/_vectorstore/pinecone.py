"""Pinecone adapter for VectorStoreRAG.

Install the extra: pip install agent-sleuth[pinecone]

The pinecone SDK is imported lazily so omitting the extra never causes
an ImportError in unrelated code paths.
"""

from __future__ import annotations

from typing import Any

from sleuth.backends.vectorstore import VectorMatch
from sleuth.types import Source


class PineconeAdapter:
    """Wraps a Pinecone Index object.

    Parameters
    ----------
    index:
        A ``pinecone.Index`` (or async-compatible equivalent) already
        initialised by the caller. Sleuth never creates or deletes indexes.
    text_key:
        Metadata field that contains the raw text of the chunk.
    source_key:
        Metadata field that contains the chunk's URL / file path.
    namespace:
        Optional Pinecone namespace to scope queries.
    """

    def __init__(
        self,
        index: Any,
        *,
        text_key: str = "text",
        source_key: str = "source",
        namespace: str | None = None,
    ) -> None:
        self._index = index
        self._text_key = text_key
        self._source_key = source_key
        self._namespace = namespace

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]:
        kwargs: dict[str, Any] = dict(
            vector=embedding,
            top_k=k,
            include_metadata=True,
        )
        if self._namespace is not None:
            kwargs["namespace"] = self._namespace

        response = await self._index.query(**kwargs)
        matches: list[VectorMatch] = []
        for m in response.matches:
            text: str = m.metadata[self._text_key]
            location: str = m.metadata.get(self._source_key, "")
            extra_meta = {
                mk: mv
                for mk, mv in m.metadata.items()
                if mk not in (self._text_key, self._source_key)
            }
            matches.append(
                VectorMatch(
                    text=text,
                    score=float(m.score),
                    source=Source(kind="file", location=location, title=None),
                    metadata=extra_meta,
                )
            )
        return matches
