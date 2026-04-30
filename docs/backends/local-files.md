# Local files

`LocalFiles` gives you LLM-navigated search over your local document corpus without an embeddings pipeline. It builds a hierarchical tree-of-contents (PageIndex-style) on first use and persists it alongside the data.

---

## How it works

Instead of chunking every document into vectors, `LocalFiles` builds a two-level tree per corpus:

1. **Indexing:** each document is parsed into sections; the indexer LLM writes a one-sentence summary for each branch node. The tree is serialized to `<corpus>/.sleuth/index/<version_hash>.json`.
2. **Navigation:** at query time, the navigator LLM receives the compact table-of-contents and selects which branches to descend into. Only those branches' leaf chunks are returned.

This approach is borrowed from the [PageIndex](https://github.com/VectifyAI/PageIndex) technique (VectifyAI, 2024).

### Why not vector RAG?

| Approach | Strengths | Weaknesses |
| --- | --- | --- |
| Vector RAG | Fast at query time, no LLM needed | Needs embeddings infra, poor at structural queries |
| Agentic file-read | No index required | Very slow (reads every file per query) |
| **Hierarchical (LocalFiles)** | **No embeddings infra, fast at query time, structural-aware** | **First-run indexing cost (one-time, cheap)** |

---

## Constructor

```python
from sleuth.backends.localfiles import LocalFiles

backend = LocalFiles(
    path="./docs",                    # corpus root (resolved to absolute path)
    indexer_llm=None,                 # LLM used to write branch summaries
    navigator_llm=None,               # LLM used to navigate the tree at query time
    include=None,                     # glob patterns; default ["**/*"]
    exclude=None,                     # gitignore-style excludes; see defaults below
    max_branch_descent=3,             # max depth to descend into the tree
    rebuild="mtime",                  # invalidation strategy: "mtime" | "hash" | "always"
)
```

**`path`** — the root directory of your corpus. Resolved to an absolute path at construction time.

**`indexer_llm`** — the LLM used to write branch summaries during indexing. If `None`, falls back to `navigator_llm`. At least one must be provided; `LocalFiles` raises `RuntimeError` at search time if neither is set.

**`navigator_llm`** — the LLM used to navigate the tree and select branches at query time. If `None`, falls back to `indexer_llm`. You can use a fast/cheap model for navigation and a capable model for synthesis:

```python
from sleuth.llm.anthropic import Anthropic

backend = LocalFiles(
    path="./docs",
    indexer_llm=Anthropic(model="claude-sonnet-4-6"),
    navigator_llm=Anthropic(model="claude-haiku-4-5"),  # faster/cheaper for navigation
)
```

**`include`** — list of glob patterns (gitignore syntax). Defaults to `["**/*"]`. Only files matching at least one include pattern are indexed.

**`exclude`** — list of gitignore-style patterns to skip. Defaults to:
```python
[".git/**", ".sleuth/**", ".venv/**", "node_modules/**",
 "dist/**", "build/**", "__pycache__/**", "*.pyc", "*.pyo", ".DS_Store"]
```

**`max_branch_descent`** — maximum tree depth the navigator descends. Default `3`. Increase for deep directory hierarchies; decrease to trade recall for speed.

**`rebuild`** — controls when the index is regenerated:
- `"mtime"` (default): rebuild when any file's modification time changes.
- `"hash"`: rebuild when the SHA-256 hash of file contents changes (more accurate, slower to check).
- `"always"`: rebuild on every process start.

---

## Supported formats

| Extension | Parser |
| --- | --- |
| `.md`, `.markdown`, `.txt`, `.rst` | Built-in text parser |
| `.html`, `.htm` | HTML parser |
| `.pdf` | `pymupdf` (requires `agent-sleuth[localfiles]`) |
| `.py`, `.js`, `.mjs`, `.cjs`, `.ts`, `.tsx` | Source code (tree-sitter sections) |

---

## Index persistence

The index is stored at `<corpus>/.sleuth/index/<version_hash>.json`. Each re-index creates a new versioned file; stale versions are pruned automatically. The `.sleuth/` directory is excluded from indexing by default.

!!! tip "Index before your first query"
    Pre-warm the index to avoid a slow first query:
    ```python
    import asyncio
    asyncio.run(agent.warm_index())
    ```
    `Sleuth.warm_index()` calls `LocalFiles.warm_index()` on every configured backend that supports it.

---

## Cost honesty

Indexing costs LLM tokens (one call per document branch to generate summaries). For a typical technical doc corpus:

- ~200 pages → a few minutes + ~$0.10–0.50 depending on model choice
- Index is reused across all queries until files change
- Navigation at query time costs 1–2 LLM calls regardless of corpus size

Using `claude-haiku-4-5` or `gpt-4o-mini` for navigation significantly reduces per-query cost.

---

## Summarization integration

When `Sleuth.summarize(target=path, length=...)` is called with a path that exists on disk, Sleuth delegates to `LocalFiles._get_summary()` instead of running the full engine:

```python
# Brief: root node summary (one sentence from the index)
result = agent.summarize("./docs/auth.md", length="brief")

# Standard: root + level-1 children summaries
result = agent.summarize("./docs/auth.md", length="standard")

# Thorough: full tree walk (all node summaries concatenated)
result = agent.summarize("./docs/auth.md", length="thorough")
```

This is fast because it reads from the already-built index, not the LLM.

---

## Example

```python
import asyncio
from sleuth import Sleuth
from sleuth.backends.localfiles import LocalFiles
from sleuth.llm.anthropic import Anthropic

llm = Anthropic(model="claude-sonnet-4-6")
fast = Anthropic(model="claude-haiku-4-5")

agent = Sleuth(
    llm=llm,
    backends=[
        LocalFiles(
            path="./docs",
            indexer_llm=llm,
            navigator_llm=fast,
            exclude=[".git/**", ".sleuth/**", "node_modules/**", "*.pyc"],
            max_branch_descent=3,
            rebuild="mtime",
        )
    ],
)

# Pre-warm (optional but recommended)
asyncio.run(agent.warm_index())

# Query
result = agent.ask("How does the rate limiter work?")
print(result.text)
for src in result.citations:
    print(" -", src.location)
```
