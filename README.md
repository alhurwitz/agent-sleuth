<div align="center">

# agent-sleuth

**Plug-and-play agentic search with reasoning, planning, citations, and observability — for any Python LLM stack.**

[![CI](https://github.com/alhurwitz/agent-sleuth/actions/workflows/ci.yml/badge.svg)](https://github.com/alhurwitz/agent-sleuth/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://pypi.org/project/agent-sleuth/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[**Documentation**](https://alhurwitz.github.io/agent-sleuth/) ·
[**Quickstart**](https://alhurwitz.github.io/agent-sleuth/quickstart/) ·
[**Release notes**](docs/release-notes/v0.1.0.md) ·
[**Design spec**](docs/superpowers/specs/2026-04-25-sleuth-design.md)

</div>

---

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

Use it as a Python SDK, behind any of **eight popular agent-framework adapters** (LangChain, LangGraph, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, Pydantic AI, CrewAI, AutoGen), or via the **`sleuth-mcp`** server (MCP over stdio + HTTP).

## Why agent-sleuth

| If you want… | Reach for… |
| --- | --- |
| One agent search call across web + your docs + your code | `Sleuth(backends=[WebBackend(...), LocalFiles(...), CodeSearch(...)])` |
| Citations to real sources (file paths, URLs, line ranges) | Every `Chunk` carries a `Source`; the synthesizer emits `CitationEvent`s |
| Streaming tokens + structured Pydantic output in one call | `agent.aask(query, schema=YourModel)` — get tokens AND `result.data` |
| To see *how* the agent reached its answer | Subscribe to the typed event stream — `Route → Plan → Search → Token → Citation → Done` |
| To not be locked into one LLM vendor | BYOK — pass any `LLMClient`. Anthropic/OpenAI shims behind extras. |
| To slot search into LangChain/CrewAI/etc. without re-architecting | `pip install agent-sleuth[langchain]` and use `SleuthTool` |
| MCP-native search for any compatible client (Claude Desktop, etc.) | `pip install agent-sleuth[mcp]` and run `sleuth-mcp --transport stdio` |
| Hierarchical local search without an embeddings pipeline | `LocalFiles(path=...)` builds a per-document tree-of-contents (PageIndex-style) — no vectors |

## Highlights

- **One unified Q&A pipeline** — heuristic Router → optional Planner → parallel Executor → streaming Synthesizer. A single typed event stream is the primary observability surface (no callbacks, no tracers required).
- **PageIndex-style local search by default** — `LocalFiles` builds a hierarchical tree-of-contents per document (markdown, PDF, HTML, code) and navigates it with the LLM at query time. **No embeddings infrastructure required.** Vector RAG is opt-in via `VectorStoreRAG` for teams who already have one.
- **Speculative pre-fetch** — when the planner runs, backend search starts on the first sub-query while the planner is still streaming the rest, hiding planner latency behind search latency.
- **Cache-hit replay** — cached query results replay through the same event stream prefixed with `CacheHitEvent`, so consumers never need a special cache-aware code path.
- **Cited and structured** — every answer carries citations; pass `schema=YourPydanticModel` for typed structured output without changing the rest of the pipeline.
- **BYOK LLM, BYO backends** — `LLMClient` and `Backend` are 30-line protocols; the package never imports a model SDK as a hard dep.
- **Performance gates in CI** — `perf.yml` fails on `first_token_ms > 1500 ms` (median) or p50/p95 regression > 10 % vs the develop baseline.

## Install

```bash
# Core (no LLM/framework deps)
pip install agent-sleuth

# Add an LLM
pip install 'agent-sleuth[anthropic]'      # Anthropic SDK shim
pip install 'agent-sleuth[openai]'         # OpenAI SDK shim

# Add a framework adapter
pip install 'agent-sleuth[langchain]'
pip install 'agent-sleuth[claude-agent]'
pip install 'agent-sleuth[langgraph]'
pip install 'agent-sleuth[llamaindex]'
pip install 'agent-sleuth[openai-agents]'
pip install 'agent-sleuth[pydantic-ai]'
pip install 'agent-sleuth[crewai]'
pip install 'agent-sleuth[autogen]'

# Add backend deps
pip install 'agent-sleuth[localfiles]'     # PDFs, code parsers
pip install 'agent-sleuth[code]'           # CodeSearch (tree-sitter)
pip install 'agent-sleuth[code-embed]'     # Optional embedding re-rank for code
pip install 'agent-sleuth[exa]'            # Exa web search
pip install 'agent-sleuth[web-fetch]'      # fetch=True mode (trafilatura, tiktoken)

# Add vector-store vendors
pip install 'agent-sleuth[pinecone]'
pip install 'agent-sleuth[qdrant]'
pip install 'agent-sleuth[chroma]'
pip install 'agent-sleuth[weaviate]'

# Memory
pip install 'agent-sleuth[semantic]'       # SemanticCache (fastembed)

# MCP server
pip install 'agent-sleuth[mcp]'
sleuth-mcp --transport stdio
sleuth-mcp --transport http --host 127.0.0.1 --port 8765
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Frontends                                                 │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  Native API  │  │  Framework      │  │  MCP server   │  │
│  │  Sleuth(...)  │  │  adapters (×8)  │  │  sleuth-mcp   │  │
│  └──────┬───────┘  └────────┬────────┘  └───────┬───────┘  │
└─────────┴───────────────────┴────────────────────┴─────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│  Engine                                                    │
│   Router  →  Planner  →  Executor  →  Synthesizer          │
│              │             │             │                 │
│              ▼             ▼             ▼                 │
│         Cache layer  /  Session memory                     │
│                                                            │
│  Emits one typed event stream throughout                   │
└─────────────────────────────┬─────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│  Backends (parallel, async — implement Backend protocol)   │
│   • WebBackend (Tavily / Exa / Brave / SerpAPI factory)    │
│   • LocalFiles  (hierarchical PageIndex; default)          │
│   • CodeSearch  (ripgrep + tree-sitter, optional embed)    │
│   • VectorStoreRAG (Pinecone / Qdrant / Chroma / Weaviate) │
│   • Custom (your code, no inheritance required)            │
└────────────────────────────────────────────────────────────┘
```

[Full design spec →](docs/superpowers/specs/2026-04-25-sleuth-design.md)

## What's in the box

### Engine

- **Heuristic Router** — depth (`auto` / `fast` / `deep`) chosen without any LLM call.
- **Planner + reflect loop** (`depth="deep"`) — decomposes multi-part queries, runs sub-searches in parallel, reflects on results, bails when the planner says "done" or `max_iterations` is reached.
- **Speculative prefetch** — search on first emitted sub-query starts while the planner streams the rest.
- **Streaming Synthesizer** — emits tokens as the LLM generates them; citations resolve in parallel.
- **Per-backend timeouts** — `asyncio.wait_for` wrap; defaults 8 s web / 4 s local. Failures surface as `SearchEvent(error=...)` and the run continues.

### Backends

| Backend | What it does |
| --- | --- |
| **`LocalFiles`** | PageIndex-style tree-of-contents (markdown, PDF via `pymupdf`, HTML, code via tree-sitter). Persists per-corpus at `<dir>/.sleuth/index/`. |
| **`CodeSearch`** | Two-phase retrieval: ripgrep lexical hits → tree-sitter expand to enclosing function/class. SQLite symbol index. Optional embedding re-rank. |
| **`WebBackend`** | Factory + per-provider classes for Tavily, Exa, Brave, SerpAPI. Per-domain rate limit + exponential backoff. Optional `fetch=True` mode (parallel-fetches top-N pages, chunks via `trafilatura` + `tiktoken`). |
| **`VectorStoreRAG`** | Opt-in adapter wrapping a `VectorStore` protocol. Vendor adapters for Pinecone, Qdrant, Chroma, Weaviate. |
| **Custom** | Implement `Backend` (one async `search(query, k) -> list[Chunk]`) and pass it. `BackendTestKit` validates protocol compliance. |

### Memory

| Layer | Default | Notes |
| --- | --- | --- |
| **`SqliteCache`** | On at `~/.sleuth/cache/{query,fetch,plan}.sqlite` | Per-namespace TTL: query=10 min · fetch=24 h · plan=1 h |
| **`IndexCache`** | Per-corpus at `<dir>/.sleuth/index/` | Invalidated by file mtime/hash |
| **`SemanticCache`** | Off (opt-in via `semantic` extra) | Fastembed BGE-small, 0.92 cosine, 10 min window — pluggable embedder |
| **`Session`** | In-memory ring buffer | `save(path)` / `Session.load(path)` / `await session.flush()` for persistence |

### LLM shims (BYOK)

- **`sleuth.llm.Anthropic`** (`anthropic` extra) — `supports_reasoning=True` for Claude extended-thinking models when `thinking=True`.
- **`sleuth.llm.OpenAI`** (`openai` extra) — `supports_reasoning=True` for o-series. Both shims lazy-import their SDK.
- **`sleuth.llm.StubLLM`** — deterministic test double; cycles scripted responses per call.

### Frontends

| Frontend | Module | Surface |
| --- | --- | --- |
| Native Python SDK | `sleuth` | `Sleuth.ask` / `aask` / `summarize` / `asummarize` |
| LangChain (Tier 1) | `sleuth.langchain` | `SleuthTool`, `SleuthRetriever`, `SleuthCallbackHandler` |
| Claude Agent SDK (Tier 1) | `sleuth.claude_agent` | `SleuthClaudeTool` (events → progress blocks) |
| LangGraph | `sleuth.langgraph` | `make_sleuth_node` factory |
| LlamaIndex | `sleuth.llamaindex` | `SleuthQueryEngine`, `SleuthRetriever` |
| OpenAI Agents SDK | `sleuth.openai_agents` | `make_sleuth_function_tool` |
| Pydantic AI | `sleuth.pydantic_ai` | `SleuthInput`, `make_sleuth_tool` |
| CrewAI | `sleuth.crewai` | `SleuthCrewAITool` (with `on_event` callback) |
| AutoGen | `sleuth.autogen` | `make_sleuth_autogen_tool`, `register_sleuth_tool` |
| MCP server | `sleuth-mcp` | `search` / `summarize` over stdio + HTTP |

## Quickstart

### 1. One backend + Tavily web search

```python
import asyncio
from sleuth import Sleuth
from sleuth.backends import WebBackend
from sleuth.llm import Anthropic

async def main():
    agent = Sleuth(
        llm=Anthropic(model="claude-sonnet-4-6"),
        backends=[WebBackend(provider="tavily", api_key="...")],
    )
    result = await agent.aask("Who maintains Anthropic's claude-agent-sdk?")
    async for event in result:
        if event.type == "token":
            print(event.text, end="", flush=True)
        elif event.type == "citation":
            print(f"\n[cite {event.index}] {event.source.location}")

asyncio.run(main())
```

### 2. Search your local docs (no embeddings)

```python
from sleuth import Sleuth
from sleuth.backends import LocalFiles
from sleuth.llm import Anthropic

llm = Anthropic(model="claude-sonnet-4-6")
agent = Sleuth(
    llm=llm,
    fast_llm=Anthropic(model="claude-haiku-4-5"),  # used for indexing/navigation
    backends=[LocalFiles(path="./docs")],
)

# First call indexes the corpus once (a few cents and a few minutes
# for ~200 docs); subsequent calls reuse the persisted tree.
result = agent.ask("Where do we explain refresh-token rotation?")
print(result.text)
for src in result.citations:
    print(" -", src.location)
```

### 3. Structured output

```python
from pydantic import BaseModel

class Verdict(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

result = agent.ask("Is the deploy script idempotent?", schema=Verdict)
print(result.data.answer, result.data.confidence)
```

### 4. Multi-turn

```python
from sleuth import Session

session = Session()
agent.ask("Who maintains the auth middleware?", session=session)
agent.ask("What changed in their last commit?", session=session)
```

### 5. Deep mode (multi-step reasoning)

```python
events = agent.aask(
    "Compare OAuth and OIDC for our use case",
    depth="deep",          # router would have picked this anyway
    max_iterations=4,
)
```

### 6. Use it as a LangChain tool

```python
from sleuth.langchain import SleuthTool

tool = SleuthTool(agent=agent)
# Pass `tool` to any LangChain agent / chain that accepts `BaseTool`.
```

### 7. Run as an MCP server

```toml
# ~/.config/sleuth/mcp.toml
[llm]
name = "anthropic:claude-sonnet-4-6"

[[backends]]
type = "web"
provider = "tavily"
api_key_env = "TAVILY_API_KEY"

[[backends]]
type = "localfiles"
path = "/var/data/docs"
```

```bash
sleuth-mcp --transport stdio
# or
sleuth-mcp --transport http --host 127.0.0.1 --port 8765
```

## Quality

- **465+ unit tests**, **8 perf benchmarks**, env-gated integration smoke tests
- **Coverage gate ≥85 %** (currently 88–95 % depending on module)
- **Strict mypy** across `src/sleuth/`
- **Performance gate** (`perf.yml`): median `first_token_ms > 1500 ms` fails; p50/p95 regression > 10 % vs develop fails
- **CI matrix:** Python 3.11 / 3.12 / 3.13 × ubuntu-latest / macos-latest

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow. Quick version:

```bash
git clone https://github.com/alhurwitz/agent-sleuth
git checkout develop
uv sync --frozen --all-extras --group dev
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

# Branch off develop, conventional commits, PR to develop
git checkout -b feature/my-thing
# ... work ...
uv run ruff check . && uv run ruff format --check .
uv run mypy src/sleuth
uv run pytest -m "not integration"
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

The hierarchical local-files backend builds on the [PageIndex](https://github.com/VectifyAI/PageIndex) technique (VectifyAI, 2024) — reasoning-based document indexing that replaces vector chunking with an LLM-navigated table of contents.
