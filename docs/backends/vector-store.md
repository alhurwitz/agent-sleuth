# Vector stores

`VectorStoreRAG` is an opt-in backend adapter that queries an existing vector index you already manage. Sleuth never writes to your index — it only reads.

---

## Overview

```python
from sleuth.backends.vectorstore import VectorStoreRAG

backend = VectorStoreRAG(
    store=your_adapter,         # any object implementing VectorStore protocol
    embedder=your_embedder,     # any object implementing Embedder protocol
    name="my-vectors",          # surfaced in SearchEvent.backend
    capabilities=None,          # defaults to frozenset({Capability.DOCS})
)
```

At query time, `VectorStoreRAG`:
1. Embeds the query string using `embedder.embed([query])`.
2. Calls `store.query(embedding, k)` to retrieve the top-k `VectorMatch` objects.
3. Maps each `VectorMatch` to a `Chunk` and returns the list.

---

## `VectorStore` protocol

Your store adapter must implement one async method:

```python
from sleuth.backends.vectorstore import VectorStore, VectorMatch

class VectorStore(Protocol):
    async def query(self, embedding: list[float], k: int) -> list[VectorMatch]: ...
```

`VectorMatch` is a dataclass:

```python
@dataclass
class VectorMatch:
    text: str
    score: float
    source: Source
    metadata: dict[str, Any]
```

---

## `Embedder` protocol

Imported from `sleuth.memory.semantic`:

```python
from sleuth.memory.semantic import Embedder

class Embedder(Protocol):
    name: str
    dim: int
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
```

Use `FastembedEmbedder` (requires `agent-sleuth[semantic]`) or bring your own.

---

## Vendor adapters

### Pinecone

Install: `pip install 'agent-sleuth[pinecone]'`

```python
import pinecone
from sleuth.backends._vectorstore.pinecone import PineconeAdapter

pc = pinecone.Pinecone(api_key="...")
index = pc.Index("my-index")

adapter = PineconeAdapter(
    index=index,
    text_key="text",         # metadata field containing chunk text
    source_key="source",     # metadata field containing URL / file path
    namespace=None,          # optional Pinecone namespace
)
```

### Qdrant

Install: `pip install 'agent-sleuth[qdrant]'`

```python
from qdrant_client import AsyncQdrantClient
from sleuth.backends._vectorstore.qdrant import QdrantAdapter

client = AsyncQdrantClient(url="http://localhost:6333")

adapter = QdrantAdapter(
    client=client,
    collection_name="my-collection",
    text_key="text",
    source_key="source",
)
```

### Chroma

Install: `pip install 'agent-sleuth[chroma]'`

```python
import chromadb
from sleuth.backends._vectorstore.chroma import ChromaAdapter

client = chromadb.Client()
collection = client.get_collection("my-collection")

adapter = ChromaAdapter(
    collection=collection,
    source_key="source",
)
```

Chroma's synchronous `collection.query()` is wrapped in `asyncio.to_thread` to avoid blocking the event loop.

### Weaviate

Install: `pip install 'agent-sleuth[weaviate]'`

```python
import weaviate
from sleuth.backends._vectorstore.weaviate import WeaviateAdapter

client = weaviate.connect_to_local()
collection = client.collections.get("MyCollection")

adapter = WeaviateAdapter(
    collection=collection,
    text_key="text",
    source_key="source",
)
```

Uses Weaviate v4 (`weaviate-client>=4`). Scores are taken from `metadata.certainty` (cosine similarity in `[0, 1]`).

---

## Full example (Pinecone)

```python
import os
import pinecone
from sleuth import Sleuth
from sleuth.backends.vectorstore import VectorStoreRAG
from sleuth.backends._vectorstore.pinecone import PineconeAdapter
from sleuth.memory.semantic import FastembedEmbedder
from sleuth.llm.anthropic import Anthropic

pc = pinecone.Pinecone(api_key=os.environ["PINECONE_KEY"])
index = pc.Index("my-docs")

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[
        VectorStoreRAG(
            store=PineconeAdapter(index, text_key="text", source_key="url"),
            embedder=FastembedEmbedder(),
            name="pinecone-docs",
        )
    ],
)

result = agent.ask("How does session expiry work?")
print(result.text)
```

!!! note "Embedder must match index"
    The embedder you pass to `VectorStoreRAG` must produce vectors of the same dimension and metric as your index. If your index was built with OpenAI `text-embedding-3-small` (1536-dim), you must use the same model at query time.
