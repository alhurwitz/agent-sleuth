"""Weaviate adapter for VectorStoreRAG.

Install the extra: pip install agent-sleuth[weaviate]

weaviate-client v4 is imported lazily so omitting the extra never causes
an ImportError in unrelated code paths.
"""

from __future__ import annotations

from typing import Any

from sleuth.backends.vectorstore import VectorMatch
from sleuth.types import Source


class WeaviateAdapter:
    """Wraps a Weaviate v4 Collection object.

    Parameters
    ----------
    collection:
        A ``weaviate.collections.Collection`` already obtained from the
        connected client. Sleuth never creates or modifies collections.
    text_key:
        Property that holds the chunk text.
    source_key:
        Property that holds the chunk URL / file path.
    """

    def __init__(
        self,
        collection: Any,
        *,
        text_key: str = "text",
        source_key: str = "source",
    ) -> None:
        self._collection = collection
        self._text_key = text_key
        self._source_key = source_key

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]:
        response = await self._collection.query.near_vector(
            near_vector=embedding,
            limit=k,
            return_metadata=["certainty"],
        )
        matches: list[VectorMatch] = []
        for obj in response.objects:
            props: dict[str, Any] = obj.properties or {}
            text: str = props[self._text_key]
            location: str = props.get(self._source_key, "")
            # certainty is Weaviate's cosine similarity in [0, 1]
            score: float = float(getattr(obj.metadata, "certainty", 0.0) or 0.0)
            extra_meta = {
                pk: pv for pk, pv in props.items() if pk not in (self._text_key, self._source_key)
            }
            matches.append(
                VectorMatch(
                    text=text,
                    score=score,
                    source=Source(kind="file", location=location, title=None),
                    metadata=extra_meta,
                )
            )
        return matches
