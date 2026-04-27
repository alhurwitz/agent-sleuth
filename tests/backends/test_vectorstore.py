"""Unit tests for VectorStoreRAG and its supporting protocols."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from sleuth.backends.base import Capability
from sleuth.backends.vectorstore import (
    VectorMatch,
    VectorStore,
    VectorStoreRAG,
)
from sleuth.memory.semantic import Embedder
from sleuth.types import Source

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeEmbedder:
    """Always returns a fixed-length zero vector.

    Implements the canonical Embedder protocol from sleuth.memory.semantic:
      async def embed(self, texts: Sequence[str]) -> list[list[float]]
    """

    name: str = "fake"
    dim: int = 4

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


@dataclass
class FakeVectorStore:
    """Records calls; returns pre-configured matches."""

    results: list[VectorMatch] = field(default_factory=list)
    calls: list[tuple[list[float], int]] = field(default_factory=list)

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]:
        self.calls.append((embedding, k))
        return self.results[:k]


def _make_match(text: str, score: float = 0.9, url: str = "file:///doc.md") -> VectorMatch:
    return VectorMatch(
        text=text,
        score=score,
        source=Source(kind="file", location=url, title=None),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Protocol conformance — Embedder
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_fake_embedder_satisfies_protocol() -> None:
    embedder: Embedder = FakeEmbedder(dim=4)
    result = await embedder.embed(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 4
    assert all(isinstance(v, float) for v in result[0])


# ---------------------------------------------------------------------------
# Protocol conformance — VectorStore
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_fake_vectorstore_satisfies_protocol() -> None:
    store: VectorStore = FakeVectorStore(results=[_make_match("chunk A")])
    matches = await store.query([0.0, 0.0, 0.0, 0.0], k=1)
    assert len(matches) == 1
    assert matches[0].text == "chunk A"


# ---------------------------------------------------------------------------
# VectorStoreRAG.search behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_search_embeds_query_and_returns_chunks() -> None:
    match = _make_match("relevant text", score=0.95)
    store = FakeVectorStore(results=[match])
    embedder = FakeEmbedder(dim=4)
    backend = VectorStoreRAG(store=store, embedder=embedder, name="test-vs")

    chunks = await backend.search("what is auth?", k=1)

    # embedding was called
    assert store.calls == [([0.0, 0.0, 0.0, 0.0], 1)]
    # result mapped to Chunk
    assert len(chunks) == 1
    assert chunks[0].text == "relevant text"
    assert chunks[0].score == 0.95
    assert chunks[0].source.location == "file:///doc.md"


@pytest.mark.unit
async def test_search_respects_k() -> None:
    matches = [_make_match(f"doc {i}", score=float(i) / 10) for i in range(10)]
    store = FakeVectorStore(results=matches)
    backend = VectorStoreRAG(store=store, embedder=FakeEmbedder())

    chunks = await backend.search("query", k=3)

    assert len(chunks) == 3
    assert store.calls[-1][1] == 3  # k was forwarded correctly


@pytest.mark.unit
async def test_search_propagates_metadata() -> None:
    match = VectorMatch(
        text="code snippet",
        score=0.8,
        source=Source(kind="code", location="repo/main.py", title="main"),
        metadata={"language": "python"},
    )
    store = FakeVectorStore(results=[match])
    backend = VectorStoreRAG(store=store, embedder=FakeEmbedder())

    chunks = await backend.search("python function", k=1)

    assert chunks[0].metadata == {"language": "python"}
    assert chunks[0].source.kind == "code"


@pytest.mark.unit
def test_backend_has_required_attributes() -> None:
    backend = VectorStoreRAG(store=FakeVectorStore(), embedder=FakeEmbedder())
    assert isinstance(backend.name, str)
    assert isinstance(backend.capabilities, frozenset)
    assert Capability.DOCS in backend.capabilities


@pytest.mark.unit
def test_custom_capabilities_override() -> None:
    caps = frozenset({Capability.PRIVATE})
    backend = VectorStoreRAG(
        store=FakeVectorStore(),
        embedder=FakeEmbedder(),
        capabilities=caps,
    )
    assert backend.capabilities == caps


@pytest.mark.unit
async def test_search_empty_store_returns_empty_list() -> None:
    store = FakeVectorStore(results=[])
    backend = VectorStoreRAG(store=store, embedder=FakeEmbedder())

    chunks = await backend.search("anything", k=10)

    assert chunks == []


# ---------------------------------------------------------------------------
# BackendTestKit compliance
# ---------------------------------------------------------------------------

from tests.contract.test_backend_protocol import BackendTestKit  # noqa: E402


class TestVectorStoreRAGContractCompliance(BackendTestKit):
    """Run the shared Backend protocol suite against VectorStoreRAG."""

    @pytest.fixture
    def backend(self) -> VectorStoreRAG:
        matches = [
            _make_match("result A", score=0.9),
            _make_match("result B", score=0.8),
        ]
        return VectorStoreRAG(
            store=FakeVectorStore(results=matches),
            embedder=FakeEmbedder(),
        )
