---
hide:
  - navigation
  - toc
---

# agent-sleuth

> **Plug-and-play agentic search with reasoning, planning, citations, and observability — for any Python LLM stack.**

`agent-sleuth` (PyPI: `agent-sleuth`, import: `sleuth`) gives any agentic Python application a fast, well-cited search capability in under five lines of code. Bring your own LLM, point it at one or more **backends** — local files, source code, web search APIs, vector stores, or your own — and call `agent.aask("...")` to stream a fully-cited answer through a typed event stream.

```python
from sleuth import Sleuth
from sleuth.backends import LocalFiles, WebBackend
from sleuth.llm import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[
        WebBackend(provider="tavily", api_key="..."),
        LocalFiles(path="./docs"),
    ],
)

async for event in agent.aask("How does our auth flow handle refresh tokens?"):
    print(event.type, event)
```

[**Get started in 5 minutes →**](quickstart.md){ .md-button .md-button--primary }
[**Read the architecture →**](concepts/architecture.md){ .md-button }

---

## Why agent-sleuth

<div class="grid cards" markdown>

- :material-magnify:{ .lg .middle } **One unified Q&A pipeline**

    ---

    Heuristic Router → optional Planner → parallel Executor → streaming Synthesizer. A single typed event stream is the primary surface — no callbacks, no tracers required.

    [→ Architecture](concepts/architecture.md)

- :material-file-tree:{ .lg .middle } **PageIndex local search by default**

    ---

    `LocalFiles` builds a hierarchical tree-of-contents per document and navigates it with the LLM at query time. **No embeddings infrastructure required.**

    [→ LocalFiles backend](backends/local-files.md)

- :material-rocket-launch:{ .lg .middle } **Speculative pre-fetch**

    ---

    When the planner runs, backend search starts on the first emitted sub-query while the planner is still streaming the rest. Hides planner latency behind search latency.

    [→ Deep mode](recipes/deep-mode.md)

- :material-key-variant:{ .lg .middle } **BYOK LLM, BYO backends**

    ---

    `LLMClient` and `Backend` are 30-line protocols. The package never imports a model SDK as a hard dep — Anthropic and OpenAI shims live behind extras.

    [→ BYOK & protocols](concepts/byok.md)

- :material-puzzle:{ .lg .middle } **Eight framework adapters**

    ---

    Drop into LangChain, LangGraph, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, Pydantic AI, CrewAI, or AutoGen. Plus `sleuth-mcp` for any MCP-compatible client.

    [→ Adapters](adapters/index.md)

- :material-replay:{ .lg .middle } **Cache-hit replay**

    ---

    Cached query results replay through the same event stream prefixed with `CacheHitEvent` — consumers never need a special cache-aware code path.

    [→ Caching & memory](concepts/caching.md)

</div>

---

## Install

```bash
# Core (no LLM/framework deps)
pip install agent-sleuth

# Add an LLM
pip install 'agent-sleuth[anthropic]'

# Add a framework
pip install 'agent-sleuth[langchain]'

# Add a backend
pip install 'agent-sleuth[localfiles]'

# Run as an MCP server
pip install 'agent-sleuth[mcp]'
sleuth-mcp --transport stdio
```

[**See all extras →**](quickstart.md#install)

---

## Quality

- **465+ unit tests**, **8 perf benchmarks**, env-gated integration smoke tests
- **Coverage gate ≥85 %** (currently 88–95 % depending on module)
- **Strict mypy** across `src/sleuth/`
- **Performance gate** (`perf.yml`): median `first_token_ms > 1500 ms` fails; p50/p95 regression > 10 % vs the develop baseline fails
- **CI matrix:** Python 3.11 / 3.12 / 3.13 × ubuntu-latest / macos-latest

---

## Acknowledgements

The hierarchical local-files backend builds on the [PageIndex](https://github.com/VectifyAI/PageIndex) technique (VectifyAI, 2024) — reasoning-based document indexing that replaces vector chunking with an LLM-navigated table of contents.
