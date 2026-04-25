# Sleuth — Design Specification

**Date:** 2026-04-25
**Status:** Draft, pending review
**Repo:** `agent-sleuth`
**Import name:** `sleuth`

---

## 1. Overview

Sleuth is a Python package that gives any agentic application a fast, plug-and-play search capability with built-in reasoning, planning, and observability. It targets two delivery surfaces:

- A native Python SDK with thin adapters for the popular agent frameworks (LangChain/LangGraph, LlamaIndex, Pydantic AI, OpenAI Agents SDK, Claude Agent SDK, CrewAI, AutoGen).
- An MCP server, so any MCP-speaking client gets the same capability without a Python integration.

### Goals

1. **Plug-and-play.** A user with an LLM client and a directory of docs can ask questions in under five lines of code, with sensible defaults and zero infrastructure.
2. **Fast.** Architecture-driven speed: parallel-everything, heuristic routing that skips planning when planning is not needed, speculative pre-fetch, streaming all the way down. Sub-second first token on the fast path with cache hits.
3. **Reasoning-first local search.** Instead of vector RAG, the default local backend builds a hierarchical "tree of contents" per document (PageIndex-style) and navigates it with the LLM at query time.
4. **Observable.** Every internal step emits a typed event — route decisions, plan, search calls, fetches, LLM reasoning tokens (when supported), answer tokens, citations. Users render whatever they want.
5. **Cited and structured.** Every answer carries citations to real sources. Optional Pydantic schema for structured output.
6. **BYOK LLM.** No model SDK is a hard dependency. The user passes any LLM client conforming to the `LLMClient` protocol.

### Non-goals (initial release)

- Owning a web crawler, web index, or competing with Tavily/Exa on web search quality.
- Long-term vector memory of conversations.
- Built-in fine-tuning, training, or evaluation pipelines.
- A hosted service. Sleuth is a library; the MCP server is meant to be run by the user.

---

## 2. Naming conventions

| Surface | Name |
| --- | --- |
| GitHub repo / distribution | `agent-sleuth` |
| PyPI package | `agent-sleuth` |
| Python import | `sleuth` |
| MCP server binary | `sleuth-mcp` |

Rationale: `sleuth` is taken on PyPI by an unrelated abandoned package; the `agent-` prefix disambiguates while keeping the import clean (`from sleuth import Sleuth`).

---

## 3. Architecture

```
┌───────────────────────────────────────────────────────────┐
│  Frontends                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │
│  │ Native API   │ │ Framework    │ │ MCP server       │   │
│  │ Sleuth class │ │ adapters     │ │ (stdio + HTTP)   │   │
│  │              │ │ (langchain,  │ │                  │   │
│  │              │ │ llamaindex,  │ │                  │   │
│  │              │ │ openai_      │ │                  │   │
│  │              │ │ agents,      │ │                  │   │
│  │              │ │ claude_      │ │                  │   │
│  │              │ │ agent,       │ │                  │   │
│  │              │ │ pydantic_ai, │ │                  │   │
│  │              │ │ crewai,      │ │                  │   │
│  │              │ │ autogen)     │ │                  │   │
│  └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘   │
│         └────────────────┴────────────────────┘           │
└────────────────────────┬──────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────┐
│  Engine                                                   │
│   Router → Planner → Executor → Synthesizer               │
│              │           │           │                    │
│              ▼           ▼           ▼                    │
│        Cache layer / Session memory                       │
│                                                           │
│  Emits a single typed event stream throughout             │
└────────────────────────┬──────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────┐
│  Backend adapters (parallel, async)                       │
│  WebBackend (Tavily, Exa, Brave, SerpAPI)                 │
│  LocalFiles (hierarchical PageIndex-style; default)       │
│  CodeSearch (ripgrep + tree-sitter, optional embeddings)  │
│  VectorStoreRAG (opt-in adapter for Pinecone/Qdrant/etc.) │
│  Custom (user-implemented Backend protocol)               │
└───────────────────────────────────────────────────────────┘
```

### Five hard rules

1. **BYOK LLM.** A `LLMClient` protocol with thin shims for Anthropic and OpenAI (more later). The package never imports a model SDK as a hard dep.
2. **Backend protocol.** Every backend implements `async def search(query, k) -> list[Chunk]`. Web, local-files, and code are three implementations of the same contract.
3. **Async-first, sync-wrapped.** Every public coroutine has a sync twin (`agent.ask` and `agent.aask` — see API).
4. **Streaming all the way down.** The synthesizer yields tokens; citations stream as sources resolve; cache hits replay through the same stream.
5. **No hidden global state.** All caches and sessions are explicit objects passed by the user.

---

## 4. Public API

```python
from sleuth import Sleuth, Session
from sleuth.backends import Tavily, LocalFiles, CodeSearch
from sleuth.llm import Anthropic, OpenAI    # thin BYOK shims

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    fast_llm=Anthropic(model="claude-haiku-4-5"),  # optional, used for routing/picking
    backends=[
        Tavily(api_key=...),
        LocalFiles(path="./docs"),     # auto-indexes on first use
        CodeSearch(path="./repo"),
    ],
    cache="default",                   # in-memory + on-disk; pass Cache(...) to customize
    semantic_cache=False,              # opt-in
)

# One-shot Q&A — async event stream
async for event in agent.aask("How does our auth flow handle refresh tokens?"):
    handle(event)  # see Section 5 for event types

# Sync mirror
result = agent.ask("...")              # blocks; returns Result(text, citations, data, stats)

# Summarize — same engine, summary-flavored synthesizer
async for event in agent.asummarize(
    target="https://example.com/article",   # URL, file path, or query
    length="brief",                          # "brief" | "standard" | "thorough"
): ...

# Structured output (opt-in)
class Verdict(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

result = agent.ask("...", schema=Verdict)   # result.data is a Verdict

# Multi-turn session
session = Session()
agent.ask("Who is the CTO of Anthropic?", session=session)
agent.ask("What did they work on before?", session=session)

# Depth override
agent.ask("...", depth="fast")              # skip planning, single fan-out
agent.ask("...", depth="deep", max_iterations=4)
agent.ask("...", depth="auto")              # default; router decides
```

### Design choices baked in

- **One agent, many calls.** `Sleuth(...)` is constructed once with backends/LLM/cache; `ask` and `summarize` are stateless unless `session=` is passed.
- **Events, not callbacks.** Async generators throughout — composable with framework adapters and easy to bridge to MCP progress notifications.
- **`ask` is the sync wrapper, `aask` is the async primary.** Avoids duplication; sync still streams via a generator that drives the event loop internally.

---

## 5. Event stream

Every run emits a typed, ordered stream of events. Cached runs emit the same events as live runs (with a `CacheHitEvent` marker), so consumers never need a special path.

```python
class RouteEvent(BaseModel):    type: Literal["route"]; depth: Depth; reason: str
class PlanEvent(BaseModel):     type: Literal["plan"]; steps: list[PlanStep]
class SearchEvent(BaseModel):   type: Literal["search"]; backend: str; query: str
class FetchEvent(BaseModel):    type: Literal["fetch"]; url: str; status: int
class ThinkingEvent(BaseModel): type: Literal["thinking"]; text: str
class TokenEvent(BaseModel):    type: Literal["token"]; text: str
class CitationEvent(BaseModel): type: Literal["citation"]; index: int; source: Source
class CacheHitEvent(BaseModel): type: Literal["cache_hit"]; kind: str; key: str
class DoneEvent(BaseModel):     type: Literal["done"]; stats: RunStats

Event = Annotated[
    Union[RouteEvent, PlanEvent, SearchEvent, FetchEvent,
          ThinkingEvent, TokenEvent, CitationEvent, CacheHitEvent, DoneEvent],
    Field(discriminator="type"),
]
```

`ThinkingEvent` is emitted only when the underlying LLM exposes reasoning content (Claude extended thinking, OpenAI o-series reasoning). The `LLMClient` protocol carries a `supports_reasoning` flag; when absent, no `ThinkingEvent` is emitted.

Framework adapters map events to native callback hooks (LangChain `BaseCallbackHandler`, LlamaIndex `CallbackManager`, etc.) so existing observability tooling Just Works.

---

## 6. Data shapes (Pydantic)

```python
class Source(BaseModel):
    kind: Literal["url", "file", "code"]
    location: str            # url, file path, or repo:path:line_range
    title: str | None = None
    fetched_at: datetime | None = None

class Chunk(BaseModel):
    text: str
    source: Source
    score: float | None = None
    metadata: dict[str, Any] = {}

class RunStats(BaseModel):
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cache_hits: dict[str, int]
    backends_called: list[str]

class Result(BaseModel):
    text: str
    citations: list[Source]
    data: BaseModel | None = None     # populated when schema= is passed
    stats: RunStats
```

Public + serialized data shapes are Pydantic v2. Internal-only structs (planner state, executor scratch) stay as `@dataclass`/`attrs` to keep hot paths cheap.

---

## 7. Backends

### 7.1 Backend protocol

```python
class Backend(Protocol):
    name: str
    capabilities: frozenset[Capability]   # {WEB, DOCS, CODE, FRESH, ...}

    async def search(self, query: str, k: int = 10) -> list[Chunk]: ...
```

Capabilities tell the planner which backends are eligible for a given sub-query. Multiple matching backends fan out in parallel via `asyncio.gather`; results are merged and de-duplicated by source.

### 7.2 WebBackend

- Adapters for Tavily, Exa, Brave, SerpAPI, plus a `WebBackend(provider=...)` factory.
- Optional `fetch=True` mode: parallel-fetches top-N pages and chunks them via `trafilatura` + token-aware splitter. Off by default — most APIs return adequate snippets.
- Per-domain rate limit and exponential backoff baked in.

### 7.3 LocalFiles (hierarchical, default local backend — no vectors)

PageIndex-style: build a tree-of-contents per document during indexing; navigate the tree with the LLM at query time.

```
Indexing (one-time + incremental on file change)
  walk dir → parse each doc's native structure
                 (markdown headings, PDF TOC/headings,
                  HTML h1-h6, code modules via tree-sitter)
           → build node tree:
                 root → section → subsection → leaf chunk
           → LLM writes a short summary at each non-leaf node
                 (one fast-LLM call per doc, roughly)
           → persist to .sleuth/index/<hash>.json + .sqlite

Query (fast, cited)
  receive query
    → load tree-of-contents (compact: 1-5 KB even for big corpora)
    → LLM picks which branches to descend (one structured-output call)
    → recurse into picked branches; LLM may prune further at each level
    → return matched leaf chunks to synthesizer
    → each citation is a real path: "report.pdf § 3.2.1: 'Refresh Tokens'"
```

```python
class LocalFiles(Backend):
    def __init__(
        self,
        path: str | Path,
        indexer_llm: LLMClient | None = None,    # for tree summaries
        navigator_llm: LLMClient | None = None,  # for branch selection
        include: list[str] = ["**/*"],
        exclude: list[str] = [".git/**", "node_modules/**", ".sleuth/**", ...],
        max_branch_descent: int = 3,
        rebuild: Literal["mtime", "hash", "always"] = "mtime",
    ): ...
```

#### Why hierarchical beats vectors and naive agentic reading

| Property | Vectors | Agentic file-read | Hierarchical |
| --- | --- | --- | --- |
| Index latency (per query) | ms | seconds | sub-second |
| LLM calls per query | 0 | 1–3 | 1–2 |
| Handles "summarize this doc" | poor | ok | great (root node) |
| Citations point to real structure | chunk IDs | file:line | doc § section |
| No embeddings infra | needs them | none | none |
| Works on a fresh dir, no setup | bad UX | ok | great |

#### Summarization gets it for free

Each node has an LLM-written summary baked in during indexing. `agent.summarize(target=path, length="brief")` returns the root summary; `length="standard"` walks one level deep; `length="thorough"` walks the full tree.

#### Cost honesty

- First indexing of a 200-document corpus: a few cents and a few minutes with a fast model. Lazy by default; eager via `agent.warm_index()`.
- Incremental updates: only changed files re-summarize.
- Cache: tree-of-contents pinned in memory after first load; per-file content cache keyed by mtime; navigator decisions cached by `(query_hash, tree_version)`.

### 7.4 CodeSearch

- Two-phase retrieval: ripgrep for lexical hits → tree-sitter to expand each hit to its enclosing function/class → optional embedding re-rank when query is non-lexical.
- Symbol-aware: separate index of definitions for "where is X defined" queries.
- Same hierarchical summaries (via tree-sitter) as LocalFiles, applied at module/class level.
- Respects `.gitignore`. Re-index only when `(mtime, content hash)` changes.

### 7.5 VectorStoreRAG (opt-in)

For users with an existing Pinecone/Qdrant/Chroma/Weaviate index. Wraps a `VectorStore` protocol. Not the default — kept available for migration paths and team-scale corpora that already have embeddings.

### 7.6 Custom backends

Users implement the `Backend` protocol; the engine treats them identically. Common cases: internal Slack/Notion/Confluence/Linear adapters. We ship `BackendTestKit` to validate custom implementations against the contract.

---

## 8. Memory and cache

Three explicit, swappable layers. None are global; every cache is an object.

```
┌─────────────────────────────────────────────────────────┐
│  Session (multi-turn coherence)                         │
│   - Ring buffer of (query, answer, citations) tuples    │
│   - User-managed lifetime: one Session per chat thread  │
│   - Persistable: session.save(path) / Session.load(path)│
└─────────────────────────────────────────────────────────┘
                         ↓ uses
┌─────────────────────────────────────────────────────────┐
│  Cache (transparent dedup, on by default)               │
│   - QueryCache:  (query_hash, backend_set, depth) → Result │
│   - FetchCache:  url/file → parsed content              │
│   - IndexCache:  hierarchical tree per LocalFiles dir   │
│   - PlanCache:   (query_hash, tree_version) → plan      │
└─────────────────────────────────────────────────────────┘
                         ↓ optional layer in front
┌─────────────────────────────────────────────────────────┐
│  SemanticCache (opt-in, off by default)                 │
│   - Embedding-similarity match against recent queries   │
│   - Threshold-gated; every hit emits CacheHitEvent      │
│   - Pluggable embedder (default: fastembed BGE-small)   │
└─────────────────────────────────────────────────────────┘
```

### Async + streaming guarantees

- Every cache lookup is async (`async def get/set` on the `Cache` protocol).
- Cache hits replay through the same event stream, prefixed with `CacheHitEvent`.
- Sessions update at end-of-turn; the `DoneEvent` is emitted first, the session write happens in a background task and is awaited only on the next `agent.ask`.

### Defaults

- `Cache`: SQLite-backed, on disk at `~/.sleuth/cache/`, per-namespace TTL (`query=10min`, `fetch=24h`, `index=∞`).
- `SemanticCache`: off. When enabled, threshold defaults to 0.92 cosine similarity, 10-minute window.
- `Session`: in-memory ring buffer, default 20 turns. Not vector-backed — that's a future feature.

---

## 9. Branching, versioning, and release

### Branching model (Gitflow)

| Branch | Purpose |
| --- | --- |
| `main` | Released versions only. Each merge here is a tagged release. Protected. |
| `develop` | Integration branch for the next release. Feature branches merge here. Protected. |
| `feature/<name>` | Branched from `develop`, merged back via PR. One per implementation chunk. |
| `release/<version>` | Stabilization branch off `develop`. Merges into `main` (tagged) and `develop`. |
| `hotfix/<version>` | Off `main` for emergencies. Merges into `main` (tagged) and `develop`. |

### Conventions

- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`, `breaking!:`.
- **Versioning:** SemVer. Pre-1.0 (`0.x.y`): any minor may break. From 1.0 forward: strict semver.
- **PRs:** All work goes through PR. Required checks: ruff, mypy, pytest, backend contract suite. No direct commits to `main` or `develop`.
- **CHANGELOG:** Auto-generated at release time from Conventional Commits (e.g. `git-cliff`).
- **Tags:** Signed tags on `main` for each release.

### Initial setup actions

1. Create `develop` branch off `main`.
2. Initialize `uv` project: `pyproject.toml`, `.python-version`, `uv.lock`. (See §16.)
3. Add `.pre-commit-config.yaml` with `commitizen` at `commit-msg` stage to bounce bad commit messages locally. (See §16.4.)
4. Add `.github/workflows/{ci,integration,perf,release}.yml`. (See §16.6.)
5. Configure branch protection on `main` and `develop` (PR + green CI required).
6. Add `CONTRIBUTING.md` documenting the flow.
7. Configure PyPI Trusted Publisher (OIDC) for release. (See §16.8.)

---

## 10. Frontends

### 10.1 Native Python API

`Sleuth` class as defined in Section 4. Distributed as `agent-sleuth` on PyPI; imported as `sleuth`.

### 10.2 Framework adapters

Each is its own optional install: `pip install agent-sleuth[langchain]`, etc. Core has zero framework deps.

| Adapter | Module | Surface |
| --- | --- | --- |
| LangChain | `sleuth.langchain` | `SleuthTool`, `SleuthRetriever`, `SleuthCallbackHandler` |
| LangGraph | `sleuth.langgraph` | Node factory for graph state machines |
| LlamaIndex | `sleuth.llamaindex` | `SleuthQueryEngine`, `SleuthRetriever` |
| OpenAI Agents SDK | `sleuth.openai_agents` | Function-call tool |
| Claude Agent SDK | `sleuth.claude_agent` | Tool with progress mapped to message blocks |
| Pydantic AI | `sleuth.pydantic_ai` | Tool with schema validation |
| CrewAI | `sleuth.crewai` | `BaseTool` subclass |
| AutoGen | `sleuth.autogen` | Function-tool registration |

Adapter tiers (decides maintenance burden):

- **Tier 1 (full support):** LangChain, MCP, Claude Agent SDK.
- **Tier 2 (best-effort):** the rest.

### 10.3 MCP server

- Binary: `sleuth-mcp` (stdio + HTTP transports).
- Tools exposed: `search(query, depth?, schema?)` and `summarize(target, length?)`.
- Events stream as MCP progress notifications; final result is structured (`text + citations + data`).
- Backends configured server-side via YAML or env vars; clients see only the two tools.

---

## 11. Speed levers

Speed budget comes entirely from architecture (LLM is BYOK). The package keeps the LLM on the critical path as little as possible.

1. **Parallel-everything.** Backends, web fetches, file reads — all `asyncio.gather`. Per-backend timeouts so a slow API can't drag the run.
2. **Heuristic router skips planning.** A regex/keyword classifier (no LLM call) decides `auto → fast | deep`. Most "what is X" / "who did Y" queries skip the planner entirely → one LLM call total (synthesis).
3. **Speculative pre-fetch.** When the planner *is* used, backend search starts on the first emitted sub-query while the planner is still streaming the rest. Hides planner latency behind search latency.
4. **Streaming synthesizer.** Tokens emit as generated; citations resolve in parallel. First-token target: <1.5s on cache miss for the fast path.
5. **Cache before LLM.** `QueryCache` and `FetchCache` checked before any LLM call.
6. **Tree-of-contents is small.** Even on huge corpora, the navigator LLM sees a 1–5 KB outline.
7. **Optional `fast_llm` slot.** User can supply a small model for routing/picking; defaults to the main LLM if absent.

---

## 12. Testing strategy

- **Backend contract tests.** Every backend (built-in or user) runs through `BackendTestKit`: protocol compliance, error shapes, timeout behavior, cancellation safety.
- **Engine unit tests.** Deterministic LLM stubs (`StubLLM` replays scripted responses) drive the engine through every router/planner branch.
- **Snapshot tests for event streams.** Given stub LLM + stub backends, the event sequence is deterministic and snapshot-tested.
- **Integration tests.** Env-gated against real APIs. CI runs nightly, not on every push.
- **Adapter tests.** Each framework adapter spins up its host framework and runs one Q&A. Catches version drift.
- **Performance regression suite.** Fixed corpus + fixed stub LLM; tracks p50/p95 latency and event-stream timing per release. CI fails on regression beyond threshold.
- **MCP conformance.** `mcp` SDK validator runs against the server; asserts protocol compliance.

---

## 13. Risks and trade-offs

1. **PageIndex indexing has up-front LLM cost.** First-time index of a 200-doc corpus is minutes and cents. Mitigation: lazy by default, `warm_index()` for eager, persistent cache.
2. **Adapter maintenance is real work.** Eight frameworks × their version churn = ongoing burden. Mitigation: thin adapters, contract tests, tier them (Tier 1 full support, Tier 2 best-effort).
3. **"Extremely fast" is bounded by the user's LLM choice.** If a user passes a slow model for synthesis, we cannot make synthesis fast. Mitigation: clear docs, recommended-defaults table, latency budgets in `DoneEvent.stats`.
4. **Local file reading on huge corpora.** Tree-of-contents can become large itself. Mitigation: hierarchical compaction (summarize sibling sub-trees if a level expands too far); navigator gets a paginated tree view.
5. **Web search costs money.** Tavily/Exa/etc. are paid APIs. Mitigation: aggressive `FetchCache`; documented cost notes; `WebBackend(rate_limit=...)` to cap.
6. **Semantic cache foot-gun.** Off by default; every hit logged via `CacheHitEvent`.

---

## 14. Out of scope (future work)

- Vector-backed `Session` memory for multi-thousand-turn conversations.
- A built-in evaluation harness (correctness, faithfulness, citation accuracy).
- Fine-tuning the navigator LLM on user-specific corpora.
- A hosted SaaS frontend.
- Multi-modal (image/audio/video) document indexing.

---

## 15. Open questions to resolve before / during implementation

These were not fully nailed down in brainstorming and should be settled in the implementation plan or first PR:

1. Project layout: single package `sleuth/` vs. namespace `sleuth.core` + adapter packages. Recommendation: single package with optional-dep adapters; revisit if the install matrix gets unwieldy.
2. PDF parser choice for LocalFiles indexing (`pypdf`, `pdfplumber`, `pymupdf`). Need to benchmark for both speed and TOC fidelity.
3. Default `fast_llm` recommendation: do we ship `Anthropic("claude-haiku-4-5")` as a literal default, or just document the recommendation? Leaning toward documentation-only to keep BYOK pure.
4. Whether `WebBackend` should ship a single unified provider with `provider="tavily"` etc., or one class per provider. Recommendation: factory function + per-provider class for power users.
5. MCP server config format: YAML, TOML, or both. Recommendation: TOML to match `pyproject.toml`.

---

## 16. Tooling and dev environment

This section is referenced from §9.2 ("Initial setup actions") — implementation lands here.

### 16.1 Package and environment manager: `uv`

`uv` is the only supported tool for dependency management and virtual environments.

- `pyproject.toml` declares dependencies, optional-dep groups (one per framework adapter: `langchain`, `llamaindex`, `openai-agents`, `claude-agent`, `pydantic-ai`, `crewai`, `autogen`, `mcp`), and a `dev` dependency group.
- `uv.lock` is checked in for dev-environment reproducibility. Library consumers get flexible version ranges from `pyproject.toml`; the lockfile is dev-only.
- `.python-version` pins the development Python version (managed by `uv python pin`).
- Common commands:
  - `uv sync --all-extras --group dev` — install everything for development.
  - `uv run <cmd>` — run a command in the project venv.
  - `uv add <pkg>` / `uv remove <pkg>` — manage deps.
  - `uv build` — build wheel + sdist.
- All CI uses `uv sync --frozen` to enforce lockfile parity.

### 16.2 Lint and format

- `ruff` for both linting and formatting (replaces black, isort, flake8, pyupgrade).
- Configuration in `pyproject.toml` under `[tool.ruff]`.
- Runs via pre-commit on staged files and via CI on the full repo.

### 16.3 Type checking

- `mypy --strict` for `src/sleuth/`. Tests are looser (`disallow_untyped_defs = false` under `[tool.mypy]` overrides for `tests/`).
- Pydantic v2 mypy plugin enabled.
- Runs via pre-commit (changed files) and CI (full repo).

### 16.4 Pre-commit hooks

`pre-commit` framework, configured via `.pre-commit-config.yaml`. Required hooks:

- `ruff` — lint + auto-fix.
- `ruff-format` — format.
- `mypy` — type-check changed files.
- `commitizen` (Python) at the `commit-msg` stage — enforces Conventional Commits.
- `check-yaml`, `check-toml`, `check-merge-conflict`.
- `end-of-file-fixer`, `trailing-whitespace`.
- `detect-secrets` — block accidental commit of credentials.

Installation: `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg`. Wrapped behind a `make setup` (or `just setup`) target.

### 16.5 Testing

- `pytest` as the test runner.
- `pytest-asyncio` with `asyncio_mode = "auto"` so `async def test_*` is collected automatically.
- `pytest-cov` for coverage; gate at **85%** for `src/sleuth/`. Adapters may dip below since they need integration envs.
- `syrupy` for event-stream snapshot tests.
- `pytest-xdist` (`-n auto`) for parallel execution in CI.
- `hypothesis` for property-based testing of the `Backend` protocol.
- `pytest-benchmark` for the perf regression suite.

Test markers:

- `@pytest.mark.unit` — default; runs on every push.
- `@pytest.mark.integration` — env-gated, nightly only.
- `@pytest.mark.perf` — regression suite, separate workflow.
- `@pytest.mark.adapter` — per-framework smoke; runs on PR when an adapter file changes.

### 16.6 CI/CD (GitHub Actions)

- **`ci.yml`** — runs on every push and PR. Steps: `uv sync --frozen --all-extras --group dev` → ruff → mypy → unit + contract tests → coverage upload. Matrix: Python 3.11, 3.12, 3.13 on `ubuntu-latest` + `macos-latest`.
- **`integration.yml`** — nightly cron + manual dispatch. Runs `pytest -m integration` against real APIs using repo secrets.
- **`perf.yml`** — runs on PR. Fails if p50 or p95 regress >10% vs `develop` baseline.
- **`release.yml`** — triggered on `v*` tag pushed to `main`. Builds via `uv build`, publishes to PyPI via Trusted Publisher (OIDC, no long-lived tokens), creates GitHub release with auto-generated CHANGELOG.

### 16.7 Dependency hygiene and security

- **Dependabot** weekly, for Python deps, GitHub Actions, and pre-commit hooks. Patch updates auto-merge if CI passes.
- **`pip-audit`** (or `uv pip audit` once stable) in CI for known CVEs.
- **GitHub Secret Scanning** + push protection enabled at the repo level.

### 16.8 Release automation

- `git-cliff` generates `CHANGELOG.md` from Conventional Commits at release time.
- Tags are signed (`git tag -s`) by maintainers.
- PyPI publish uses Trusted Publisher (OIDC) — no long-lived API tokens stored anywhere.

### 16.9 Reproducibility checklist

- `uv.lock` checked in.
- `.python-version` checked in.
- All CI uses `uv sync --frozen`.
- Determinism in tests: `StubLLM` for engine tests; `responses`/`respx` for HTTP mocking; fixed seeds where randomness is allowed.
- Library consumers get flexible version ranges in `pyproject.toml`; we never hard-pin transitive deps for them.
