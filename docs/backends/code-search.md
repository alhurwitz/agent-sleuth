# Code search

`CodeSearch` gives you symbol-aware, context-rich retrieval over source code using a two-phase pipeline: ripgrep for lexical hits, tree-sitter to expand each hit to its enclosing function or class.

---

## How it works

### Phase 1 — ripgrep lexical search

[ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) is called with the query as a search pattern. It returns matching lines with file path and line number. Ripgrep respects `.gitignore` automatically.

### Phase 2 — tree-sitter expansion

Each line-level hit is expanded to its **enclosing function, class, or module-level block** using tree-sitter. This gives the synthesizer meaningful code context rather than isolated lines. Languages without tree-sitter support fall back to the raw hit line.

### Symbol index shortcut

For "where is X defined" queries (matching patterns like "where is `foo` defined", "definition of `bar`", "locate `Baz`"), `CodeSearch` skips ripgrep entirely and queries a SQLite symbol index at `<corpus>/.sleuth/symbols.db`. This provides near-instant definition lookup.

### Optional embedding re-rank

When `rerank=True`, the top results from phase 1+2 are re-ranked by cosine similarity to the query using `fastembed` BGE-small. This improves precision for ambiguous queries at the cost of the embedding call.

---

## Constructor

```python
from sleuth.backends.codesearch import CodeSearch

backend = CodeSearch(
    path="./src",                         # repository root
    rerank=False,                         # enable embedding re-rank (requires code-embed extra)
    rerank_model="BAAI/bge-small-en-v1.5",# fastembed model for re-rank
    rebuild="mtime",                      # "mtime" | "always" for symbol index
)
```

**`path`** — root directory of the repository. `.gitignore` rules are respected automatically.

**`rerank`** — when `True`, results are re-ranked by embedding similarity. Requires `pip install 'agent-sleuth[code-embed]'`. Default `False`.

**`rerank_model`** — fastembed model identifier for re-ranking. Default: `"BAAI/bge-small-en-v1.5"`.

**`rebuild`** — when to regenerate the symbol index:
- `"mtime"` (default): rebuild when any tracked file's mtime changes.
- `"always"`: rebuild on every process start.

---

## Runtime requirement: `rg`

`CodeSearch` requires `rg` (ripgrep) on the system PATH.

```bash
# macOS
brew install ripgrep

# Ubuntu / Debian
apt-get install ripgrep

# Cargo
cargo install ripgrep
```

---

## Installing tree-sitter support

```bash
pip install 'agent-sleuth[code]'
```

This installs the tree-sitter parsers for Python, JavaScript, and TypeScript. Without the extra, phase 2 is skipped and hits fall back to raw line text.

### Supported languages

| Language | Extensions |
| --- | --- |
| Python | `.py` |
| JavaScript | `.js`, `.mjs`, `.cjs` |
| TypeScript | `.ts`, `.tsx` |

Other file types are indexed by ripgrep (lexical) but not expanded to function context.

---

## Symbol index

The symbol index at `<corpus>/.sleuth/symbols.db` stores definitions (functions, classes, methods) keyed by name. It is built incrementally from tree-sitter parse results.

Queries matching the definition-lookup pattern use this index:

- "where is `authenticate` defined"
- "definition of `TokenBucket`"
- "locate `build_tree`"
- "find definition of `Planner`"

When the symbol is found, its source file and line number are returned with a snippet of context lines.

---

## Enabling embedding re-rank

```bash
pip install 'agent-sleuth[code-embed]'
```

```python
backend = CodeSearch(path="./src", rerank=True)
```

Re-ranking adds one embedding call per search but substantially improves precision for multi-word queries or queries where multiple functions have similar names.

---

## Example

```python
from sleuth import Sleuth
from sleuth.backends.codesearch import CodeSearch
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[
        CodeSearch(
            path="./src",
            rerank=False,       # set True + install code-embed for better precision
            rebuild="mtime",
        )
    ],
)

# Symbol lookup (fast path)
result = agent.ask("where is authenticate defined")
print(result.text)
for src in result.citations:
    print(" -", src.location)  # e.g. src/auth.py:L42-L58

# Context-rich search
result = agent.ask("How does the token bucket rate limiter work?")
print(result.text)
```
