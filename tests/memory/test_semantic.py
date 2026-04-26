"""Tests for SemanticCache and Embedder protocol (Phase 4)."""

import math

import numpy as np
import pytest

from sleuth.events import CacheHitEvent
from sleuth.memory.cache import MemoryCache
from sleuth.memory.semantic import Embedder, SemanticCache, StubEmbedder

# ---------------------------------------------------------------------------
# StubEmbedder
# ---------------------------------------------------------------------------


async def test_stub_embedder_returns_unit_vector() -> None:
    emb = StubEmbedder()
    result = await emb.embed(["hello world"])
    assert len(result) == 1
    vec = result[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5


async def test_stub_embedder_same_text_same_vector() -> None:
    emb = StubEmbedder()
    vecs = await emb.embed(["cat", "cat"])
    # Identical inputs → identical vectors → cosine = 1.0
    dot = sum(a * b for a, b in zip(vecs[0], vecs[1], strict=True))
    assert dot > 0.99


async def test_stub_embedder_batch() -> None:
    emb = StubEmbedder()
    result = await emb.embed(["a", "b", "c"])
    assert len(result) == 3


def test_stub_embedder_has_name_and_dim() -> None:
    emb = StubEmbedder()
    assert isinstance(emb.name, str) and len(emb.name) > 0
    assert isinstance(emb.dim, int) and emb.dim > 0


def test_stub_embedder_is_embedder_protocol() -> None:
    """StubEmbedder structurally satisfies the Embedder protocol."""
    emb = StubEmbedder()
    assert isinstance(emb, Embedder)


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


@pytest.fixture
def backing() -> MemoryCache:
    return MemoryCache()


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def sc(backing: MemoryCache, embedder: StubEmbedder) -> SemanticCache:
    return SemanticCache(
        cache=backing,
        embedder=embedder,
        threshold=0.92,
        window_s=600,
    )


async def test_semantic_cache_miss_on_empty(sc: SemanticCache) -> None:
    result, event = await sc.lookup("what is python?")
    assert result is None
    assert event is None


async def test_semantic_cache_hit_on_identical_query(sc: SemanticCache) -> None:
    payload = {"text": "Python is a programming language."}
    await sc.store("what is python?", payload)
    result, event = await sc.lookup("what is python?")
    assert result == payload
    assert isinstance(event, CacheHitEvent)
    assert event.kind == "semantic"


async def test_semantic_cache_event_fields(sc: SemanticCache) -> None:
    await sc.store("foo query", {"text": "bar"})
    _result, event = await sc.lookup("foo query")
    assert event is not None
    assert event.type == "cache_hit"
    assert event.kind == "semantic"
    assert isinstance(event.key, str) and len(event.key) > 0


async def test_semantic_cache_miss_when_orthogonal(backing: MemoryCache) -> None:
    """Embedder that returns orthogonal vectors → always a miss."""
    from collections.abc import Sequence

    _call_count = 0

    class OrthogonalEmbedder:
        name = "orthogonal"
        dim = 4

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            nonlocal _call_count
            results = []
            for _ in texts:
                vec = [0.0, 0.0, 0.0, 0.0]
                vec[_call_count % 4] = 1.0
                _call_count += 1
                results.append(vec)
            return results

    sc = SemanticCache(
        cache=backing,
        embedder=OrthogonalEmbedder(),
        threshold=0.92,
        window_s=600,
    )
    await sc.store("what is python?", {"text": "something"})
    result, event = await sc.lookup("totally different query")
    assert result is None
    assert event is None


async def test_semantic_cache_expired_entry_is_miss(
    backing: MemoryCache, embedder: StubEmbedder
) -> None:
    import asyncio

    sc = SemanticCache(
        cache=backing,
        embedder=embedder,
        threshold=0.92,
        window_s=1,  # 1 second window
    )
    await sc.store("what is python?", {"text": "answer"})
    await asyncio.sleep(1.1)
    result, _event = await sc.lookup("what is python?")
    assert result is None


async def test_semantic_cache_store_and_lookup_different_keys(sc: SemanticCache) -> None:
    """Multiple entries can be stored and retrieved independently."""
    await sc.store("query one", {"text": "answer one"})
    await sc.store("query two", {"text": "answer two"})
    r1, _ = await sc.lookup("query one")
    r2, _ = await sc.lookup("query two")
    assert r1 == {"text": "answer one"}
    assert r2 == {"text": "answer two"}


async def test_numpy_cosine_works_with_float_lists() -> None:
    """Confirm cosine similarity works on list[float] (not just np.ndarray)."""
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    cos = float(np.dot(arr_a, arr_b))
    assert abs(cos - 1.0) < 1e-6
