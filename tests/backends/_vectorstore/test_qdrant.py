"""Unit tests for QdrantAdapter."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from sleuth.backends._vectorstore.qdrant import QdrantAdapter


def _qdrant_scored_point(text: str, url: str, score: float) -> MagicMock:
    pt = MagicMock()
    pt.score = score
    pt.payload = {"text": text, "source": url}
    return pt


@pytest.mark.unit
async def test_qdrant_adapter_query_returns_vector_matches() -> None:
    client = AsyncMock()
    client.search.return_value = [
        _qdrant_scored_point("qdrant chunk", "https://example.com/doc", 0.88),
    ]

    adapter = QdrantAdapter(
        client=client,
        collection_name="my-col",
        text_key="text",
        source_key="source",
    )
    matches = await adapter.query([0.1, 0.2], k=5)

    client.search.assert_called_once_with(
        collection_name="my-col",
        query_vector=[0.1, 0.2],
        limit=5,
        with_payload=True,
    )
    assert len(matches) == 1
    assert matches[0].text == "qdrant chunk"
    assert matches[0].score == 0.88
    assert matches[0].source.location == "https://example.com/doc"


@pytest.mark.unit
async def test_qdrant_adapter_empty_result() -> None:
    client = AsyncMock()
    client.search.return_value = []

    adapter = QdrantAdapter(client=client, collection_name="col", text_key="text", source_key="src")
    matches = await adapter.query([0.0], k=3)
    assert matches == []


@pytest.mark.integration
async def test_qdrant_real_index() -> None:
    """Requires QDRANT_URL env var (and optionally QDRANT_API_KEY, QDRANT_COLLECTION)."""
    qdrant_client = pytest.importorskip("qdrant_client")
    url = os.environ["QDRANT_URL"]
    api_key = os.environ.get("QDRANT_API_KEY")
    collection = os.environ.get("QDRANT_COLLECTION", "test")

    client = qdrant_client.AsyncQdrantClient(url=url, api_key=api_key)
    adapter = QdrantAdapter(
        client=client, collection_name=collection, text_key="text", source_key="source"
    )
    matches = await adapter.query([0.0] * 384, k=3)
    assert isinstance(matches, list)
