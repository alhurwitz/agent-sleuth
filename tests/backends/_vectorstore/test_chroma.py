"""Unit tests for ChromaAdapter."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from sleuth.backends._vectorstore.chroma import ChromaAdapter


def _chroma_query_result(
    docs: list[str],
    sources: list[str],
    distances: list[float],
) -> dict[str, list[object]]:
    """Shape matches chromadb QueryResult dict."""
    return {
        "documents": [docs],  # outer list = n_results batches
        "metadatas": [[{"source": s} for s in sources]],
        "distances": [distances],
        "ids": [[f"id-{i}" for i in range(len(docs))]],
    }


@pytest.mark.unit
async def test_chroma_adapter_query_returns_matches() -> None:
    fake_collection = MagicMock()
    fake_collection.query.return_value = _chroma_query_result(
        docs=["chroma chunk A", "chroma chunk B"],
        sources=["gs://bucket/a.txt", "gs://bucket/b.txt"],
        distances=[0.05, 0.12],  # Chroma returns L2 distances; lower = better
    )

    adapter = ChromaAdapter(collection=fake_collection, source_key="source")
    matches = await adapter.query([0.1, 0.2, 0.3], k=2)

    fake_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    assert len(matches) == 2
    assert matches[0].text == "chroma chunk A"
    # score = 1 - distance (normalised so higher = more similar)
    assert abs(matches[0].score - 0.95) < 1e-6
    assert matches[0].source.location == "gs://bucket/a.txt"


@pytest.mark.unit
async def test_chroma_adapter_empty_returns_empty() -> None:
    fake_collection = MagicMock()
    fake_collection.query.return_value = _chroma_query_result([], [], [])

    adapter = ChromaAdapter(collection=fake_collection, source_key="source")
    matches = await adapter.query([0.0], k=5)
    assert matches == []


@pytest.mark.integration
async def test_chroma_real_collection() -> None:
    """Requires CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION env vars."""
    chromadb = pytest.importorskip("chromadb")
    host = os.environ["CHROMA_HOST"]
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    collection_name = os.environ["CHROMA_COLLECTION"]

    client = chromadb.AsyncHttpClient(host=host, port=port)
    collection = await client.get_collection(collection_name)
    adapter = ChromaAdapter(collection=collection, source_key="source")
    matches = await adapter.query([0.0] * 384, k=3)
    assert isinstance(matches, list)
