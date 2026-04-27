"""SemanticCache — opt-in embedding-similarity cache layer.

Usage::

    from sleuth.memory.semantic import SemanticCache, FastembedEmbedder
    from sleuth.memory.cache import SqliteCache

    sc = SemanticCache(
        cache=SqliteCache(),
        embedder=FastembedEmbedder(),   # requires agent-sleuth[semantic]
        threshold=0.92,
        window_s=600,
    )

The ``Embedder`` protocol (canonical per conventions §5.6) is intentionally
small so users can swap in any embedding provider without pulling in fastembed.

Phase 6 (VectorStoreRAG) imports ``Embedder`` from ``sleuth.memory.semantic``
— **do not redefine it in backends/vectorstore.py**.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from sleuth.events import CacheHitEvent
from sleuth.memory.cache import Cache

if TYPE_CHECKING:
    pass


def _require_numpy() -> Any:
    """Lazy-import numpy so this module loads without `agent-sleuth[semantic]`.

    SemanticCache and FastembedEmbedder both need numpy at runtime; calling
    them without the `semantic` extra installed raises a clear ImportError.
    """
    try:
        import numpy as np

        return np
    except ImportError as e:
        raise ImportError(
            "SemanticCache / FastembedEmbedder require numpy. "
            "Install with: pip install 'agent-sleuth[semantic]'"
        ) from e


# ---------------------------------------------------------------------------
# Embedder protocol  (conventions §5.6 — canonical shape)
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Minimal async embedder interface.

    Implementations must expose ``name`` (str) and ``dim`` (int) as class or
    instance attributes, and an async ``embed`` method returning unit-norm
    float vectors.
    """

    name: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one unit-norm embedding per input text.

        Args:
            texts: A non-empty sequence of strings to embed.

        Returns:
            A list of ``dim``-dimensional float lists; order matches *texts*.
        """
        ...


# ---------------------------------------------------------------------------
# StubEmbedder — deterministic, zero-dependency, for tests
# ---------------------------------------------------------------------------


class StubEmbedder:
    """Deterministic embedder for tests.

    Uses a simple character-frequency hash to produce a consistent unit-norm
    vector without any ML library.  Compatible with the ``Embedder`` protocol.
    """

    name: str = "stub"
    dim: int = 64

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for ch in text:
                vec[ord(ch) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            else:
                vec[0] = 1.0  # fallback for empty string
            results.append(vec)
        return results


# ---------------------------------------------------------------------------
# FastembedEmbedder — requires agent-sleuth[semantic]
# ---------------------------------------------------------------------------


class FastembedEmbedder:
    """Fastembed BGE-small embedder.

    Lazy-imports ``fastembed`` so the class can be imported without the extra
    installed; it raises ``ImportError`` only when first used.

    Args:
        model_name: fastembed model identifier.  Defaults to BGE-small-en-v1.5.
    """

    name: str
    dim: int

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.name = model_name
        self.dim = 384  # BGE-small-en-v1.5 embedding dimension
        self._model: Any = None  # loaded lazily

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "fastembed is required for FastembedEmbedder. "
                    "Install with: pip install 'agent-sleuth[semantic]'"
                ) from exc
            self._model = TextEmbedding(model_name=self.name)
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import asyncio

        np = _require_numpy()
        model = self._load()
        loop = asyncio.get_event_loop()
        # fastembed is synchronous; offload to thread pool
        raw: list[Any] = await loop.run_in_executor(None, lambda: list(model.embed(list(texts))))
        results: list[list[float]] = []
        for vec in raw:
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            results.append(arr.tolist())
        return results


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

_SEMANTIC_NS = "semantic"
_INDEX_KEY = "__index__"


def _entry_key(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity of two unit-norm float lists."""
    np = _require_numpy()
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    return float(np.dot(arr_a, arr_b))


class SemanticCache:
    """Embedding-similarity cache layer in front of any ``Cache`` implementation.

    On ``lookup``:
      1. Embed the query.
      2. Compare against stored entries within *window_s*.
      3. Return the best match if its cosine similarity ≥ *threshold*, plus a
         ``CacheHitEvent(kind="semantic", ...)``.

    On ``store``:
      - Persist the result alongside its embedding vector and timestamp.

    Args:
        cache: The backing ``Cache`` instance (must satisfy the Cache Protocol).
        embedder: An ``Embedder`` instance.  Defaults to ``StubEmbedder``.
            Use ``FastembedEmbedder`` in production.
        threshold: Cosine similarity threshold.  Defaults to 0.92.
        window_s: Entries older than this (seconds) are ignored.  Defaults to
            600 (10 minutes).
    """

    def __init__(
        self,
        cache: Cache,
        embedder: Embedder | None = None,
        *,
        threshold: float = 0.92,
        window_s: int = 600,
    ) -> None:
        self._cache = cache
        self._embedder: Embedder = embedder if embedder is not None else StubEmbedder()
        self._threshold = threshold
        self._window_s = window_s

    async def lookup(
        self,
        query: str,
    ) -> tuple[Any | None, CacheHitEvent | None]:
        """Search for a semantically similar cached result.

        Returns:
            A ``(result, CacheHitEvent)`` pair.  Both elements are ``None``
            on a miss; both are populated on a hit.
        """
        index_raw = await self._cache.get(_SEMANTIC_NS, _INDEX_KEY)
        if index_raw is None:
            return None, None

        index: list[dict[str, Any]] = index_raw
        query_vec = (await self._embedder.embed([query]))[0]
        now = time.time()
        cutoff = now - self._window_s

        best_score = -1.0
        best_entry: dict[str, Any] | None = None

        for entry in index:
            if entry.get("ts", 0.0) < cutoff:
                continue
            stored_vec: list[float] = entry["vec"]
            score = _cosine(query_vec, stored_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self._threshold:
            return None, None

        entry_key: str = best_entry["key"]
        result = await self._cache.get(_SEMANTIC_NS, entry_key)
        if result is None:
            return None, None

        event = CacheHitEvent(type="cache_hit", kind="semantic", key=entry_key)
        return result, event

    async def store(self, query: str, result: Any) -> None:
        """Store *result* under an embedding of *query*."""
        query_vec = (await self._embedder.embed([query]))[0]
        entry_key = _entry_key(query)

        # Persist the result
        await self._cache.set(_SEMANTIC_NS, entry_key, result, ttl_s=self._window_s)

        # Update the index
        index_raw = await self._cache.get(_SEMANTIC_NS, _INDEX_KEY)
        index: list[dict[str, Any]] = index_raw if index_raw is not None else []

        # Remove stale entry for the same key if present
        index = [e for e in index if e.get("key") != entry_key]
        index.append(
            {
                "key": entry_key,
                "vec": query_vec,
                "ts": time.time(),
            }
        )
        await self._cache.set(_SEMANTIC_NS, _INDEX_KEY, index, ttl_s=self._window_s)
