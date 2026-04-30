# Memory reference

Cache protocol, implementations, session ring buffer, and semantic similarity cache.

See [Caching & memory](../concepts/caching.md) for design explanations and TTL tables.

---

## Cache layer

::: sleuth.memory.cache.Cache
    options:
      show_root_heading: true

::: sleuth.memory.cache.MemoryCache
    options:
      show_root_heading: true
      members:
        - get
        - set
        - delete
        - clear

::: sleuth.memory.cache.SqliteCache
    options:
      show_root_heading: true
      members:
        - __init__
        - get
        - set
        - delete
        - clear

---

## Session

::: sleuth.memory.session.Session
    options:
      show_root_heading: true
      members:
        - __init__
        - max_turns
        - turns
        - add_turn
        - as_messages
        - save
        - load
        - flush

---

## Semantic cache

::: sleuth.memory.semantic.Embedder
    options:
      show_root_heading: true

::: sleuth.memory.semantic.StubEmbedder
    options:
      show_root_heading: true
      members:
        - embed

::: sleuth.memory.semantic.FastembedEmbedder
    options:
      show_root_heading: true
      members:
        - __init__
        - embed

::: sleuth.memory.semantic.SemanticCache
    options:
      show_root_heading: true
      members:
        - __init__
        - lookup
        - store
