"""Qdrant adapter for VectorStoreRAG.

Install the extra: pip install agent-sleuth[qdrant]

The qdrant-client SDK is imported lazily so omitting the extra never causes
an ImportError in unrelated code paths.
"""

from __future__ import annotations

from typing import Any

from sleuth.backends.vectorstore import VectorMatch
from sleuth.types import Source


class QdrantAdapter:
    """Wraps an async Qdrant client.

    Parameters
    ----------
    client:
        An ``AsyncQdrantClient`` already initialised by the caller.
    collection_name:
        Qdrant collection to query.
    text_key:
        Payload field that contains the chunk text.
    source_key:
        Payload field that contains the chunk URL / file path.
    """

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str,
        text_key: str = "text",
        source_key: str = "source",
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._text_key = text_key
        self._source_key = source_key

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]:
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=embedding,
            limit=k,
            with_payload=True,
        )
        matches: list[VectorMatch] = []
        for pt in results:
            payload: dict[str, Any] = pt.payload or {}
            text: str = payload[self._text_key]
            location: str = payload.get(self._source_key, "")
            extra_meta = {
                pk: pv for pk, pv in payload.items() if pk not in (self._text_key, self._source_key)
            }
            matches.append(
                VectorMatch(
                    text=text,
                    score=float(pt.score),
                    source=Source(kind="file", location=location, title=None),
                    metadata=extra_meta,
                )
            )
        return matches
