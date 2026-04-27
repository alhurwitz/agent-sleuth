"""VectorStoreRAG — opt-in adapter wrapping an existing vector index.

Spec reference: §7.5.

This module owns:
  - VectorMatch dataclass
  - VectorStore protocol  (each vendor adapter implements this)
  - VectorStoreRAG class  (implements Backend protocol from §5.2)

The Embedder protocol is owned by Phase 4 (sleuth.memory.semantic, §5.6).
Import it from there — do not redefine it here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sleuth.backends.base import Capability
from sleuth.memory.semantic import Embedder
from sleuth.types import Chunk, Source

logger = logging.getLogger("sleuth.backends.vectorstore")


# ---------------------------------------------------------------------------
# VectorMatch — the atomic result returned by a VectorStore adapter
# ---------------------------------------------------------------------------


@dataclass
class VectorMatch:
    """One result returned from a vector index query."""

    text: str
    score: float
    source: Source
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VectorStore protocol
# Each vendor adapter (Pinecone, Qdrant, Chroma, Weaviate) implements this.
# upsert is intentionally absent — Sleuth never writes to the user's index.
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Read-only interface to a pre-existing vector index.

    Sleuth only queries existing indexes; there is no upsert/write path
    per spec §7.5.
    """

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]: ...


# ---------------------------------------------------------------------------
# VectorStoreRAG — Backend implementation
# ---------------------------------------------------------------------------


class VectorStoreRAG:
    """Backend adapter that queries an existing vector store.

    Embeds the query text using the provided ``Embedder`` (which takes a
    Sequence[str] and returns list[list[float]] per conventions §5.6), then
    delegates to ``store.query()`` and maps results to ``Chunk`` objects.

    Parameters
    ----------
    store:
        A VectorStore adapter (e.g. PineconeAdapter, QdrantAdapter).
    embedder:
        Converts query text → embedding vector for the store.
        Must implement the Embedder protocol from sleuth.memory.semantic.
    name:
        Human-readable backend name surfaced in SearchEvent.backend.
    capabilities:
        Defaults to {Capability.DOCS}; override for private corpora.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        name: str = "vectorstore",
        capabilities: frozenset[Capability] | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self.name = name
        self.capabilities: frozenset[Capability] = (
            capabilities if capabilities is not None else frozenset({Capability.DOCS})
        )

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        """Embed *query* then call the underlying store, returning Chunks.

        The canonical Embedder.embed(texts: Sequence[str]) -> list[list[float]]
        is called with a single-item list; we take the first (and only) result.
        """
        logger.debug("vectorstore search: query=%r k=%d backend=%s", query, k, self.name)
        embeddings: list[list[float]] = await self._embedder.embed([query])
        embedding: list[float] = embeddings[0]
        matches: list[VectorMatch] = await self._store.query(embedding, k)
        return [
            Chunk(
                text=match.text,
                source=match.source,
                score=match.score,
                metadata=match.metadata,
            )
            for match in matches
        ]


__all__ = ["VectorMatch", "VectorStore", "VectorStoreRAG"]
