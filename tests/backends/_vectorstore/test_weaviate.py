"""Unit tests for WeaviateAdapter."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from sleuth.backends._vectorstore.weaviate import WeaviateAdapter


def _weaviate_object(text: str, url: str, certainty: float) -> MagicMock:
    obj = MagicMock()
    obj.properties = {"text": text, "source": url}
    obj.metadata = MagicMock()
    obj.metadata.certainty = certainty
    return obj


@pytest.mark.unit
async def test_weaviate_adapter_query_returns_matches() -> None:
    fake_collection = MagicMock()
    fake_query = MagicMock()
    fake_collection.query = fake_query
    fake_query.near_vector = AsyncMock(
        return_value=MagicMock(
            objects=[
                _weaviate_object("weaviate chunk", "https://wiki.example.com/page", 0.93),
            ]
        )
    )

    adapter = WeaviateAdapter(
        collection=fake_collection,
        text_key="text",
        source_key="source",
    )
    matches = await adapter.query([0.5, 0.5], k=3)

    fake_query.near_vector.assert_called_once_with(
        near_vector=[0.5, 0.5],
        limit=3,
        return_metadata=["certainty"],
    )
    assert len(matches) == 1
    assert matches[0].text == "weaviate chunk"
    assert matches[0].score == 0.93
    assert matches[0].source.location == "https://wiki.example.com/page"


@pytest.mark.unit
async def test_weaviate_adapter_empty_result() -> None:
    fake_collection = MagicMock()
    fake_query = MagicMock()
    fake_collection.query = fake_query
    fake_query.near_vector = AsyncMock(return_value=MagicMock(objects=[]))

    adapter = WeaviateAdapter(collection=fake_collection, text_key="text", source_key="src")
    matches = await adapter.query([0.0], k=5)
    assert matches == []


@pytest.mark.integration
async def test_weaviate_real_collection() -> None:
    """Requires WEAVIATE_URL, WEAVIATE_CLASS, optional WEAVIATE_API_KEY."""
    weaviate = pytest.importorskip("weaviate")
    url = os.environ["WEAVIATE_URL"]
    class_name = os.environ["WEAVIATE_CLASS"]
    api_key = os.environ.get("WEAVIATE_API_KEY")

    auth = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
    client = await weaviate.connect_to_custom(http_host=url, auth_credentials=auth)
    collection = client.collections.get(class_name)

    adapter = WeaviateAdapter(collection=collection, text_key="text", source_key="source")
    matches = await adapter.query([0.0] * 384, k=3)
    assert isinstance(matches, list)
    await client.close()
