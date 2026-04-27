"""Unit tests for PineconeAdapter.

The adapter is tested with a fake Pinecone Index object so no network is needed.
Integration tests require PINECONE_API_KEY and are gated by @pytest.mark.integration.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from sleuth.backends._vectorstore.pinecone import PineconeAdapter
from sleuth.backends.vectorstore import VectorMatch

# ---------------------------------------------------------------------------
# Fake Pinecone index
# ---------------------------------------------------------------------------


def _pinecone_match(id_: str, score: float, text: str, url: str) -> MagicMock:
    """Build a mock object shaped like pinecone QueryResponse match."""
    m = MagicMock()
    m.score = score
    m.metadata = {"text": text, "source": url}
    m.id = id_
    return m


def _pinecone_query_response(matches: list[MagicMock]) -> MagicMock:
    resp = MagicMock()
    resp.matches = matches
    return resp


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_pinecone_adapter_query_returns_vector_matches() -> None:
    fake_index = AsyncMock()
    fake_index.query.return_value = _pinecone_query_response(
        [
            _pinecone_match("id-1", 0.92, "chunk alpha", "s3://bucket/doc.pdf"),
        ]
    )

    adapter = PineconeAdapter(index=fake_index, text_key="text", source_key="source")
    matches = await adapter.query([0.1, 0.2, 0.3], k=1)

    fake_index.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        top_k=1,
        include_metadata=True,
    )
    assert len(matches) == 1
    assert isinstance(matches[0], VectorMatch)
    assert matches[0].text == "chunk alpha"
    assert matches[0].score == 0.92
    assert matches[0].source.location == "s3://bucket/doc.pdf"


@pytest.mark.unit
async def test_pinecone_adapter_missing_text_key_raises() -> None:
    fake_index = AsyncMock()
    fake_index.query.return_value = _pinecone_query_response(
        [
            _pinecone_match("id-1", 0.9, "irrelevant", "loc"),
        ]
    )
    # text_key points at a non-existent metadata field
    adapter = PineconeAdapter(index=fake_index, text_key="body", source_key="source")
    # Override the match metadata to be missing "body"
    fake_index.query.return_value.matches[0].metadata = {"source": "loc"}

    with pytest.raises(KeyError):
        await adapter.query([0.0], k=1)


@pytest.mark.unit
async def test_pinecone_adapter_respects_namespace() -> None:
    fake_index = AsyncMock()
    fake_index.query.return_value = _pinecone_query_response([])

    adapter = PineconeAdapter(index=fake_index, text_key="text", source_key="src", namespace="ns1")
    await adapter.query([0.0, 0.0], k=5)

    call_kwargs = fake_index.query.call_args.kwargs
    assert call_kwargs.get("namespace") == "ns1"


@pytest.mark.integration
async def test_pinecone_real_index() -> None:
    """Requires PINECONE_API_KEY and PINECONE_INDEX_NAME env vars."""
    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.environ["PINECONE_INDEX_NAME"]

    pinecone = pytest.importorskip("pinecone")
    pc = pinecone.Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    adapter = PineconeAdapter(index=index, text_key="text", source_key="source")
    # A zero vector always returns _something_ from a populated index
    matches = await adapter.query([0.0] * 1536, k=3)
    assert isinstance(matches, list)
    assert all(isinstance(m, VectorMatch) for m in matches)
