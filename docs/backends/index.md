# Backends

A backend is anything that implements one async method: `search(query, k) -> list[Chunk]`. Sleuth ships four built-in backends; you can write your own in under 30 lines.

---

## Comparison

| Backend | Best for | `Capability` flags | Requires |
| --- | --- | --- | --- |
| [LocalFiles](local-files.md) | Markdown, PDF, HTML, code corpora on disk | `DOCS` | `agent-sleuth[localfiles]` for PDFs; LLM for indexing/navigation |
| [CodeSearch](code-search.md) | Source code: symbol lookup, function context | `CODE` | `rg` binary on PATH; `agent-sleuth[code]` for tree-sitter |
| [WebBackend](web.md) | Live web search, news, status pages | `WEB`, `FRESH` | API key for chosen provider; `agent-sleuth[exa]` for Exa |
| [VectorStoreRAG](vector-store.md) | Existing vector indexes (Pinecone, Qdrant, Chroma, Weaviate) | `DOCS` (default) | `agent-sleuth[pinecone|qdrant|chroma|weaviate]` |

All backends are used together in parallel. Pass as many as you need:

```python
from sleuth import Sleuth
from sleuth.backends import WebBackend, CodeSearch
from sleuth.backends.localfiles import LocalFiles
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[
        WebBackend(provider="tavily", api_key="..."),
        LocalFiles(path="./docs"),
        CodeSearch(path="./src"),
    ],
)
```

The Executor fans every query out to all backends simultaneously, applies per-backend timeouts (8 s for web, 4 s for local), and merges results by deduplicating on `source.location`.

---

## `Capability` flags and routing

In deep mode, the Planner can restrict which backends handle a given sub-query by specifying `backends: ["web", "fresh"]` in its JSON output. The Executor checks each backend's `capabilities` frozenset before dispatching. In fast mode, all backends receive every query.

```python
from sleuth.backends.base import Capability

# Check what a backend advertises:
print(backend.capabilities)
# frozenset({<Capability.WEB: 'web'>, <Capability.FRESH: 'fresh'>})
```

---

## Custom backends

Any object satisfying the `Backend` protocol works. See [Custom backends](custom.md) for a step-by-step tutorial including error handling, cancellation safety, and `BackendTestKit` validation.
