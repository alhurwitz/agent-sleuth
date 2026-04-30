# Backends reference

Protocol definition, capability flags, and all four built-in backend classes.

See [Backends](../backends/index.md) for usage guides and comparison tables.

---

## Protocol & base types

::: sleuth.backends.base.Capability
    options:
      show_root_heading: true

::: sleuth.backends.base.Backend
    options:
      show_root_heading: true

---

## `LocalFiles`

::: sleuth.backends.localfiles.LocalFiles
    options:
      show_root_heading: true
      members:
        - __init__
        - search
        - warm_index

---

## `CodeSearch`

::: sleuth.backends.codesearch.CodeSearch
    options:
      show_root_heading: true
      members:
        - __init__
        - search

---

## `WebBackend` (factory)

::: sleuth.backends.web.WebBackend
    options:
      show_root_heading: true

### Per-provider classes

::: sleuth.backends._web.tavily.TavilyBackend
    options:
      show_root_heading: true
      members:
        - __init__
        - search

::: sleuth.backends._web.exa.ExaBackend
    options:
      show_root_heading: true
      members:
        - __init__
        - search

::: sleuth.backends._web.brave.BraveBackend
    options:
      show_root_heading: true
      members:
        - __init__
        - search

::: sleuth.backends._web.serpapi.SerpAPIBackend
    options:
      show_root_heading: true
      members:
        - __init__
        - search

---

## `VectorStoreRAG`

::: sleuth.backends.vectorstore.VectorStore
    options:
      show_root_heading: true

::: sleuth.backends.vectorstore.VectorMatch
    options:
      show_root_heading: true

::: sleuth.backends.vectorstore.VectorStoreRAG
    options:
      show_root_heading: true
      members:
        - __init__
        - search

### Vendor adapters

::: sleuth.backends._vectorstore.pinecone.PineconeAdapter
    options:
      show_root_heading: true
      members:
        - __init__
        - query

::: sleuth.backends._vectorstore.qdrant.QdrantAdapter
    options:
      show_root_heading: true
      members:
        - __init__
        - query

::: sleuth.backends._vectorstore.chroma.ChromaAdapter
    options:
      show_root_heading: true
      members:
        - __init__
        - query

::: sleuth.backends._vectorstore.weaviate.WeaviateAdapter
    options:
      show_root_heading: true
      members:
        - __init__
        - query
