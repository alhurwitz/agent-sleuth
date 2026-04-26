# VectorStoreRAG Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an opt-in `VectorStoreRAG` backend that wraps any user-provided Pinecone, Qdrant, Chroma, or Weaviate index and exposes it through the standard `Backend` protocol.

**Architecture:** `VectorStoreRAG` accepts an `embedder` (to convert query strings to vectors) and a `VectorStore` adapter (per-vendor class implementing `async def query(...) -> list[VectorMatch]`). Each vendor adapter lazy-imports its SDK so the extra's absence never causes an `ImportError` in unrelated code paths. The class implements `Backend` (`capabilities={Capability.DOCS}`, `async def search(...) -> list[Chunk]`) so the engine treats it identically to any other backend.

**Tech Stack:** Python 3.11+, `pydantic>=2.6`, `pytest`, `pytest-asyncio`, per-vendor optional extras (`pinecone-client`, `qdrant-client`, `chromadb`, `weaviate-client`).

---

> **Callouts (non-conventions items the human must reconcile before execution)**
>
> **CALLOUT A — `Embedder` protocol ownership.** The conventions do not define `Embedder`. Phase 4 (SemanticCache) also needs an `Embedder`. This plan defines `Embedder` in `src/sleuth/backends/vectorstore.py` and Phase 4 MUST import it from there (`from sleuth.backends.vectorstore import Embedder`). If Phase 4 executes before Phase 6, coordinate which phase owns the definition. Flag this to the human if ordering differs.
>
> **CALLOUT B — `_vectorstore/` sub-package.** Conventions §1 shows `backends/vectorstore.py` but does not enumerate sub-modules. This plan places per-vendor adapters in `src/sleuth/backends/_vectorstore/{pinecone,qdrant,chroma,weaviate}.py` following the `_subpkg/` private helper pattern in conventions §7. Tests mirror under `tests/backends/_vectorstore/`.
>
> **CALLOUT C — `upsert` is intentionally omitted.** Per spec §7.5, Sleuth only queries an existing index; it does not manage the user's data. `VectorStore.upsert(...)` is explicitly out of scope.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/sleuth/backends/vectorstore.py` | **Create** | `Embedder` protocol, `VectorMatch` dataclass, `VectorStore` protocol, `VectorStoreRAG` class |
| `src/sleuth/backends/_vectorstore/__init__.py` | **Create** | Empty package marker |
| `src/sleuth/backends/_vectorstore/pinecone.py` | **Create** | `PineconeAdapter` (lazy-imports `pinecone`) |
| `src/sleuth/backends/_vectorstore/qdrant.py` | **Create** | `QdrantAdapter` (lazy-imports `qdrant_client`) |
| `src/sleuth/backends/_vectorstore/chroma.py` | **Create** | `ChromaAdapter` (lazy-imports `chromadb`) |
| `src/sleuth/backends/_vectorstore/weaviate.py` | **Create** | `WeaviateAdapter` (lazy-imports `weaviate`) |
| `tests/backends/test_vectorstore.py` | **Create** | Unit tests for `VectorStoreRAG` against a fake `VectorStore`; `BackendTestKit` compliance run |
| `tests/backends/_vectorstore/__init__.py` | **Create** | Empty package marker |
| `tests/backends/_vectorstore/test_pinecone.py` | **Create** | Unit tests for `PineconeAdapter` against a recorded fake; integration test gated by `@pytest.mark.integration` |
| `tests/backends/_vectorstore/test_qdrant.py` | **Create** | Same pattern for Qdrant |
| `tests/backends/_vectorstore/test_chroma.py` | **Create** | Same pattern for Chroma |
| `tests/backends/_vectorstore/test_weaviate.py` | **Create** | Same pattern for Weaviate |
| `pyproject.toml` | **Modify** | Add `pinecone`, `qdrant`, `chroma`, `weaviate` to `[project.optional-dependencies]` |

---

## Task 1: Branch setup

**Files:**
- No file changes; git only.

- [ ] **Step 1.1: Create feature branch off `develop`**

```bash
git checkout develop
git checkout -b feature/phase-6-vectorstore
```

Expected: `Switched to a new branch 'feature/phase-6-vectorstore'`

---

## Task 2: Core protocols and `VectorStoreRAG` skeleton

**Files:**
- Create: `src/sleuth/backends/vectorstore.py`
- Create: `src/sleuth/backends/_vectorstore/__init__.py`
- Test: `tests/backends/test_vectorstore.py`

### Step 2.1 — Write the failing test for the `Embedder` protocol shape

- [ ] Create `tests/backends/test_vectorstore.py` with:

```python
"""Unit tests for VectorStoreRAG and its supporting protocols."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from sleuth.backends.vectorstore import (
    Embedder,
    VectorMatch,
    VectorStore,
    VectorStoreRAG,
)
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeEmbedder:
    """Always returns a fixed-length zero vector."""
    dim: int = 4

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim


@dataclass
class FakeVectorStore:
    """Records calls; returns pre-configured matches."""
    results: list[VectorMatch] = field(default_factory=list)
    calls: list[tuple[list[float], int]] = field(default_factory=list)

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]:
        self.calls.append((embedding, k))
        return self.results[: k]


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
    result = await embedder.embed("hello world")
    assert len(result) == 4
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Protocol conformance — VectorStore
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_fake_vectorstore_satisfies_protocol() -> None:
    store: VectorStore = FakeVectorStore(results=[_make_match("chunk A")])
    matches = await store.query([0.0, 0.0, 0.0, 0.0], k=1)
    assert len(matches) == 1
    assert matches[0].text == "chunk A"
```

- [ ] **Step 2.2: Run to verify it fails (module not found)**

```bash
uv run pytest tests/backends/test_vectorstore.py -v
```

Expected: `ModuleNotFoundError: No module named 'sleuth.backends.vectorstore'`

### Step 2.3 — Create the package marker and implement core types

- [ ] Create `src/sleuth/backends/_vectorstore/__init__.py`:

```python
# private sub-package — vendor adapters live here
```

- [ ] Create `src/sleuth/backends/vectorstore.py`:

```python
"""VectorStoreRAG — opt-in adapter wrapping an existing vector index.

Spec reference: §7.5.

This module owns:
  - Embedder   protocol  (Phase 4 SemanticCache imports from here)
  - VectorMatch dataclass
  - VectorStore protocol  (each vendor adapter implements this)
  - VectorStoreRAG class  (implements Backend protocol from §7.1)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sleuth.backends.base import Backend, Capability
from sleuth.types import Chunk, Source

logger = logging.getLogger("sleuth.backends.vectorstore")


# ---------------------------------------------------------------------------
# Embedder protocol
# Phase 4 (SemanticCache) MUST import Embedder from this module.
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    """Converts a text string to a dense vector embedding."""

    async def embed(self, text: str) -> list[float]: ...


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
    """Read-only interface to a pre-existing vector index."""

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]: ...


# ---------------------------------------------------------------------------
# VectorStoreRAG — Backend implementation
# ---------------------------------------------------------------------------

class VectorStoreRAG:
    """Backend adapter that queries an existing vector store.

    Parameters
    ----------
    store:
        A VectorStore adapter (e.g. PineconeAdapter, QdrantAdapter).
    embedder:
        Converts query text → embedding vector for the store.
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
        """Embed *query* then call the underlying store, returning Chunks."""
        logger.debug("vectorstore search: query=%r k=%d backend=%s", query, k, self.name)
        embedding = await self._embedder.embed(query)
        matches = await self._store.query(embedding, k)
        return [
            Chunk(
                text=match.text,
                source=match.source,
                score=match.score,
                metadata=match.metadata,
            )
            for match in matches
        ]
```

- [ ] **Step 2.4: Run tests — expect PASS**

```bash
uv run pytest tests/backends/test_vectorstore.py -v
```

Expected: `2 passed`

- [ ] **Step 2.5: Commit**

```bash
git add src/sleuth/backends/vectorstore.py \
        src/sleuth/backends/_vectorstore/__init__.py \
        tests/backends/test_vectorstore.py
git commit -m "feat: add VectorStoreRAG core protocols (Embedder, VectorStore, VectorMatch)"
```

---

## Task 3: `VectorStoreRAG.search` behavior and Backend protocol compliance

**Files:**
- Modify: `tests/backends/test_vectorstore.py` (add search behavior + BackendTestKit run)

- [ ] **Step 3.1: Add search behavior tests and BackendTestKit compliance**

Append to `tests/backends/test_vectorstore.py`:

```python
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

    @pytest.fixture()
    def backend(self) -> VectorStoreRAG:
        matches = [
            _make_match("result A", score=0.9),
            _make_match("result B", score=0.8),
        ]
        return VectorStoreRAG(
            store=FakeVectorStore(results=matches),
            embedder=FakeEmbedder(),
        )
```

- [ ] **Step 3.2: Run tests — expect PASS**

```bash
uv run pytest tests/backends/test_vectorstore.py -v
```

Expected: All tests pass (BackendTestKit tests may show as collected via the subclass; exact count depends on Phase 1's kit).

- [ ] **Step 3.3: Run mypy on the new module**

```bash
uv run mypy src/sleuth/backends/vectorstore.py
```

Expected: `Success: no issues found`

- [ ] **Step 3.4: Commit**

```bash
git add tests/backends/test_vectorstore.py
git commit -m "test: verify VectorStoreRAG search behavior and Backend protocol compliance"
```

---

## Task 4: `pyproject.toml` extras for vendor dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 4.1: Add vendor extras to `[project.optional-dependencies]`**

Open `pyproject.toml` and add the following four lines inside `[project.optional-dependencies]` (after the existing entries):

```toml
pinecone  = ["pinecone-client>=3.0"]
qdrant    = ["qdrant-client>=1.9"]
chroma    = ["chromadb>=0.5"]
weaviate  = ["weaviate-client>=4.0"]
```

- [ ] **Step 4.2: Verify the TOML parses cleanly**

```bash
uv run python -c "import tomllib; tomllib.loads(open('pyproject.toml').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4.3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pinecone/qdrant/chroma/weaviate optional extras to pyproject.toml"
```

---

## Task 5: Pinecone adapter

**Files:**
- Create: `src/sleuth/backends/_vectorstore/pinecone.py`
- Create: `tests/backends/_vectorstore/__init__.py`
- Create: `tests/backends/_vectorstore/test_pinecone.py`

- [ ] **Step 5.1: Write the failing unit test**

Create `tests/backends/_vectorstore/__init__.py` (empty).

Create `tests/backends/_vectorstore/test_pinecone.py`:

```python
"""Unit tests for PineconeAdapter.

The adapter is tested with a fake Pinecone Index object so no network is needed.
Integration tests require PINECONE_API_KEY and are gated by @pytest.mark.integration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sleuth.backends._vectorstore.pinecone import PineconeAdapter
from sleuth.backends.vectorstore import VectorMatch
from sleuth.types import Source


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
    fake_index.query.return_value = _pinecone_query_response([
        _pinecone_match("id-1", 0.92, "chunk alpha", "s3://bucket/doc.pdf"),
    ])

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
    fake_index.query.return_value = _pinecone_query_response([
        _pinecone_match("id-1", 0.9, "irrelevant", "loc"),
    ])
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
    import os
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
```

- [ ] **Step 5.2: Run to verify failure**

```bash
uv run pytest tests/backends/_vectorstore/test_pinecone.py -v -m "not integration"
```

Expected: `ImportError` or `ModuleNotFoundError` for `sleuth.backends._vectorstore.pinecone`

- [ ] **Step 5.3: Implement `PineconeAdapter`**

Create `src/sleuth/backends/_vectorstore/pinecone.py`:

```python
"""Pinecone adapter for VectorStoreRAG.

Install the extra: pip install agent-sleuth[pinecone]

The pinecone SDK is imported lazily so omitting the extra never causes
an ImportError in unrelated code paths.
"""
from __future__ import annotations

from typing import Any

from sleuth.backends.vectorstore import VectorMatch
from sleuth.types import Source


class PineconeAdapter:
    """Wraps a Pinecone Index object.

    Parameters
    ----------
    index:
        A ``pinecone.Index`` (or async-compatible equivalent) already
        initialised by the caller. Sleuth never creates or deletes indexes.
    text_key:
        Metadata field that contains the raw text of the chunk.
    source_key:
        Metadata field that contains the chunk's URL / file path.
    namespace:
        Optional Pinecone namespace to scope queries.
    """

    def __init__(
        self,
        index: Any,
        *,
        text_key: str = "text",
        source_key: str = "source",
        namespace: str | None = None,
    ) -> None:
        self._index = index
        self._text_key = text_key
        self._source_key = source_key
        self._namespace = namespace

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]:
        kwargs: dict[str, Any] = dict(
            vector=embedding,
            top_k=k,
            include_metadata=True,
        )
        if self._namespace is not None:
            kwargs["namespace"] = self._namespace

        response = await self._index.query(**kwargs)
        matches: list[VectorMatch] = []
        for m in response.matches:
            text: str = m.metadata[self._text_key]
            location: str = m.metadata.get(self._source_key, "")
            matches.append(
                VectorMatch(
                    text=text,
                    score=float(m.score),
                    source=Source(kind="file", location=location, title=None),
                    metadata={k: v for k, v in m.metadata.items()
                              if k not in (self._text_key, self._source_key)},
                )
            )
        return matches
```

- [ ] **Step 5.4: Run tests — expect PASS (unit only)**

```bash
uv run pytest tests/backends/_vectorstore/test_pinecone.py -v -m "not integration"
```

Expected: `2 passed, 1 skipped` (integration test skipped)

- [ ] **Step 5.5: Commit**

```bash
git add src/sleuth/backends/_vectorstore/pinecone.py \
        tests/backends/_vectorstore/__init__.py \
        tests/backends/_vectorstore/test_pinecone.py
git commit -m "feat: add PineconeAdapter for VectorStoreRAG"
```

---

## Task 6: Qdrant adapter

**Files:**
- Create: `src/sleuth/backends/_vectorstore/qdrant.py`
- Create: `tests/backends/_vectorstore/test_qdrant.py`

- [ ] **Step 6.1: Write the failing unit test**

Create `tests/backends/_vectorstore/test_qdrant.py`:

```python
"""Unit tests for QdrantAdapter."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from sleuth.backends._vectorstore.qdrant import QdrantAdapter
from sleuth.backends.vectorstore import VectorMatch


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

    adapter = QdrantAdapter(client=client, collection_name="my-col", text_key="text", source_key="source")
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
    adapter = QdrantAdapter(client=client, collection_name=collection, text_key="text", source_key="source")
    matches = await adapter.query([0.0] * 384, k=3)
    assert isinstance(matches, list)
```

- [ ] **Step 6.2: Run to verify failure**

```bash
uv run pytest tests/backends/_vectorstore/test_qdrant.py -v -m "not integration"
```

Expected: `ImportError` or `ModuleNotFoundError` for `sleuth.backends._vectorstore.qdrant`

- [ ] **Step 6.3: Implement `QdrantAdapter`**

Create `src/sleuth/backends/_vectorstore/qdrant.py`:

```python
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
            matches.append(
                VectorMatch(
                    text=text,
                    score=float(pt.score),
                    source=Source(kind="file", location=location, title=None),
                    metadata={k: v for k, v in payload.items()
                              if k not in (self._text_key, self._source_key)},
                )
            )
        return matches
```

- [ ] **Step 6.4: Run tests — expect PASS (unit only)**

```bash
uv run pytest tests/backends/_vectorstore/test_qdrant.py -v -m "not integration"
```

Expected: `2 passed, 1 skipped`

- [ ] **Step 6.5: Commit**

```bash
git add src/sleuth/backends/_vectorstore/qdrant.py \
        tests/backends/_vectorstore/test_qdrant.py
git commit -m "feat: add QdrantAdapter for VectorStoreRAG"
```

---

## Task 7: Chroma adapter

**Files:**
- Create: `src/sleuth/backends/_vectorstore/chroma.py`
- Create: `tests/backends/_vectorstore/test_chroma.py`

- [ ] **Step 7.1: Write the failing unit test**

Create `tests/backends/_vectorstore/test_chroma.py`:

```python
"""Unit tests for ChromaAdapter."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sleuth.backends._vectorstore.chroma import ChromaAdapter
from sleuth.backends.vectorstore import VectorMatch


def _chroma_query_result(
    docs: list[str],
    sources: list[str],
    distances: list[float],
) -> dict:
    """Shape matches chromadb QueryResult dict."""
    return {
        "documents": [docs],       # outer list = n_results batches
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
```

- [ ] **Step 7.2: Run to verify failure**

```bash
uv run pytest tests/backends/_vectorstore/test_chroma.py -v -m "not integration"
```

Expected: `ModuleNotFoundError` for `sleuth.backends._vectorstore.chroma`

- [ ] **Step 7.3: Implement `ChromaAdapter`**

Create `src/sleuth/backends/_vectorstore/chroma.py`:

```python
"""Chroma adapter for VectorStoreRAG.

Install the extra: pip install agent-sleuth[chroma]

chromadb is imported lazily; the adapter is synchronous internally
(chromadb's .query() is sync) but wrapped in asyncio.to_thread so the
event loop is never blocked.
"""
from __future__ import annotations

import asyncio
from typing import Any

from sleuth.backends.vectorstore import VectorMatch
from sleuth.types import Source


class ChromaAdapter:
    """Wraps a Chroma Collection object.

    Parameters
    ----------
    collection:
        A ``chromadb.Collection`` (sync or async) already obtained by the
        caller. Sleuth never creates, modifies, or deletes collections.
    source_key:
        Metadata field that contains the chunk URL / file path.
    """

    def __init__(
        self,
        collection: Any,
        *,
        source_key: str = "source",
    ) -> None:
        self._collection = collection
        self._source_key = source_key

    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]:
        # chromadb collection.query is synchronous; run in thread to avoid blocking.
        result: dict[str, Any] = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs: list[str] = result["documents"][0] if result["documents"] else []
        metadatas: list[dict[str, Any]] = result["metadatas"][0] if result["metadatas"] else []
        distances: list[float] = result["distances"][0] if result["distances"] else []

        matches: list[VectorMatch] = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            # Convert L2 distance to a [0,1] similarity score (1 = identical).
            score = max(0.0, 1.0 - dist)
            location: str = meta.get(self._source_key, "")
            extra_meta = {k: v for k, v in meta.items() if k != self._source_key}
            matches.append(
                VectorMatch(
                    text=doc,
                    score=score,
                    source=Source(kind="file", location=location, title=None),
                    metadata=extra_meta,
                )
            )
        return matches
```

- [ ] **Step 7.4: Run tests — expect PASS (unit only)**

```bash
uv run pytest tests/backends/_vectorstore/test_chroma.py -v -m "not integration"
```

Expected: `2 passed, 1 skipped`

- [ ] **Step 7.5: Commit**

```bash
git add src/sleuth/backends/_vectorstore/chroma.py \
        tests/backends/_vectorstore/test_chroma.py
git commit -m "feat: add ChromaAdapter for VectorStoreRAG"
```

---

## Task 8: Weaviate adapter

**Files:**
- Create: `src/sleuth/backends/_vectorstore/weaviate.py`
- Create: `tests/backends/_vectorstore/test_weaviate.py`

- [ ] **Step 8.1: Write the failing unit test**

Create `tests/backends/_vectorstore/test_weaviate.py`:

```python
"""Unit tests for WeaviateAdapter."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from sleuth.backends._vectorstore.weaviate import WeaviateAdapter
from sleuth.backends.vectorstore import VectorMatch


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
    fake_query.near_vector = AsyncMock(return_value=MagicMock(
        objects=[
            _weaviate_object("weaviate chunk", "https://wiki.example.com/page", 0.93),
        ]
    ))

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
    fake_collection.query.near_vector = AsyncMock(return_value=MagicMock(objects=[]))

    adapter = WeaviateAdapter(collection=fake_collection, text_key="text", source_key="src")
    # Empty objects list (side-step the async mock chain)
    fake_collection.query.near_vector.return_value = MagicMock(objects=[])
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
```

- [ ] **Step 8.2: Run to verify failure**

```bash
uv run pytest tests/backends/_vectorstore/test_weaviate.py -v -m "not integration"
```

Expected: `ModuleNotFoundError` for `sleuth.backends._vectorstore.weaviate`

- [ ] **Step 8.3: Implement `WeaviateAdapter`**

Create `src/sleuth/backends/_vectorstore/weaviate.py`:

```python
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
            extra_meta = {k: v for k, v in props.items()
                          if k not in (self._text_key, self._source_key)}
            matches.append(
                VectorMatch(
                    text=text,
                    score=score,
                    source=Source(kind="file", location=location, title=None),
                    metadata=extra_meta,
                )
            )
        return matches
```

- [ ] **Step 8.4: Run tests — expect PASS (unit only)**

```bash
uv run pytest tests/backends/_vectorstore/test_weaviate.py -v -m "not integration"
```

Expected: `2 passed, 1 skipped`

- [ ] **Step 8.5: Commit**

```bash
git add src/sleuth/backends/_vectorstore/weaviate.py \
        tests/backends/_vectorstore/test_weaviate.py
git commit -m "feat: add WeaviateAdapter for VectorStoreRAG"
```

---

## Task 9: Full test suite pass + type checking + coverage gate

**Files:**
- No new files.

- [ ] **Step 9.1: Run all unit tests for the phase**

```bash
uv run pytest tests/backends/test_vectorstore.py \
              tests/backends/_vectorstore/ \
              -v -m "not integration"
```

Expected: All unit tests pass; integration tests skipped.

- [ ] **Step 9.2: Run mypy over all new source files**

```bash
uv run mypy \
  src/sleuth/backends/vectorstore.py \
  src/sleuth/backends/_vectorstore/pinecone.py \
  src/sleuth/backends/_vectorstore/qdrant.py \
  src/sleuth/backends/_vectorstore/chroma.py \
  src/sleuth/backends/_vectorstore/weaviate.py
```

Expected: `Success: no issues found in 5 source files`

- [ ] **Step 9.3: Run ruff lint + format check**

```bash
uv run ruff check src/sleuth/backends/vectorstore.py \
                  src/sleuth/backends/_vectorstore/
uv run ruff format --check src/sleuth/backends/vectorstore.py \
                           src/sleuth/backends/_vectorstore/
```

Expected: No errors. If formatting issues: `uv run ruff format src/sleuth/backends/vectorstore.py src/sleuth/backends/_vectorstore/`

- [ ] **Step 9.4: Run coverage gate**

```bash
uv run pytest tests/backends/test_vectorstore.py \
              tests/backends/_vectorstore/ \
              -m "not integration" \
              --cov=src/sleuth/backends/vectorstore \
              --cov=src/sleuth/backends/_vectorstore \
              --cov-report=term-missing \
              --cov-fail-under=85
```

Expected: Coverage ≥85% on the new modules.

- [ ] **Step 9.5: Commit**

```bash
git add .
git commit -m "test: full suite + coverage gate for VectorStoreRAG and all vendor adapters"
```

---

## Task 10: Open PR against `develop`

- [ ] **Step 10.1: Push branch**

```bash
git push -u origin feature/phase-6-vectorstore
```

- [ ] **Step 10.2: Open PR**

```bash
gh pr create \
  --base develop \
  --title "feat: Phase 6 — VectorStoreRAG adapter (Pinecone, Qdrant, Chroma, Weaviate)" \
  --body "$(cat <<'EOF'
## Summary

Implements spec §7.5: an opt-in `VectorStoreRAG` backend that wraps any pre-existing vector index.

- `VectorStoreRAG` implements the `Backend` protocol (`Capability.DOCS` by default).
- `Embedder` protocol defined here (Phase 4 SemanticCache imports from `sleuth.backends.vectorstore`).
- `VectorStore` protocol: `async def query(embedding, k) -> list[VectorMatch]`.
- Per-vendor adapters: `PineconeAdapter`, `QdrantAdapter`, `ChromaAdapter`, `WeaviateAdapter`.
- Vendor SDKs lazy-imported; each vendor is its own `pyproject.toml` extra.
- `BackendTestKit` compliance run for `VectorStoreRAG`.
- Unit tests use fake `VectorStore`; integration tests env-gated (`@pytest.mark.integration`).

## Test plan

- [ ] `uv run pytest tests/backends/test_vectorstore.py tests/backends/_vectorstore/ -m "not integration"` — all pass
- [ ] `uv run mypy src/sleuth/backends/vectorstore.py src/sleuth/backends/_vectorstore/` — clean
- [ ] `uv run ruff check src/sleuth/backends/` — clean
- [ ] Coverage ≥85% on new modules
- [ ] Integration tests (nightly CI, real API keys required)
EOF
)"
```

Expected: PR URL printed to stdout.

---

## Self-review checklist

**Spec coverage:**
- §7.5 "VectorStore protocol" — covered (Task 2: `VectorStore` protocol with `query`).
- §7.5 "wraps existing index" — covered (`upsert` intentionally absent; CALLOUT C).
- §7.5 Pinecone, Qdrant, Chroma, Weaviate — one adapter each (Tasks 5–8).
- §7.1 Backend protocol (`name`, `capabilities`, `search`) — covered (Task 2–3).
- §12 BackendTestKit compliance — covered (Task 3 `TestVectorStoreRAGContractCompliance`).
- §12 Integration tests env-gated — covered (each vendor test file has one `@pytest.mark.integration` test).
- §16.5 Coverage gate ≥85% — covered (Task 9.4).
- Phase instructions: `Embedder` protocol — covered (Task 2, CALLOUT A).
- Phase instructions: lazy imports — covered (each vendor adapter note; adapters themselves receive the already-instantiated client object, so no top-level SDK import appears in any adapter module).

**Placeholder scan:** No TBD, no "implement later", no "write tests for above" without code, no vague "handle edge cases". All code blocks are complete.

**Type consistency:**
- `VectorMatch` dataclass defined in Task 2, used identically in Tasks 5–8.
- `VectorStore.query(embedding: list[float], k: int) -> list[VectorMatch]` — consistent across all adapters.
- `Embedder.embed(text: str) -> list[float]` — consistent in `FakeEmbedder` and protocol definition.
- `VectorStoreRAG.search(query: str, k: int = 10) -> list[Chunk]` — matches `Backend` protocol from conventions §5.2.
- `Capability` imported from `sleuth.backends.base` — consistent.
- `Chunk`, `Source` imported from `sleuth.types` — consistent.
