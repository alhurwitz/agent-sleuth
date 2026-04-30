# API reference

Auto-generated from source docstrings via `mkdocstrings`. These pages document every public class, method, and attribute with the exact signatures and types from the code.

---

## Modules

| Page | What's documented |
| --- | --- |
| [Sleuth](sleuth.md) | The `Sleuth` agent class — constructor, `aask`, `ask`, `asummarize`, `summarize`, `warm_index` |
| [Events & types](events-types.md) | All 9 event classes, the `Event` union, `Source`, `Chunk`, `RunStats`, `Result[T]`, `Depth`, `Length` |
| [Backends](backends.md) | `Backend` protocol, `Capability` enum, all four built-in backend classes |
| [LLM clients](llm.md) | `LLMClient` protocol, `LLMChunk` union, `Message`, `Tool`, `Anthropic`, `OpenAI`, `StubLLM` |
| [Memory](memory.md) | `Cache` protocol, `MemoryCache`, `SqliteCache`, `Session`, `Embedder`, `SemanticCache` |

---

## Reading the reference

Signatures are rendered with full type annotations. Methods prefixed with `_` are internal and not documented here. For usage context and examples, refer to the [Concepts](../concepts/index.md), [Backends](../backends/index.md), and [Recipes](../recipes/index.md) sections.
