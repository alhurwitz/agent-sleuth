# agent-sleuth — Implementation conventions

Shared reference for all phase-implementation plans. Each plan should **reference, not redefine** anything in this file. If a phase needs a new file/path/name not listed here, raise a callout at the top of the plan and the human reconciles before execution.

> **For plan authors:** read this end-to-end before writing your phase plan. Anything you'd want to "decide on the fly" probably has an answer here. If it doesn't, flag it.

---

## 1. Package layout (locked, spec §15 #1)

`src/sleuth/` is the single root package. All adapters and shims live as submodules under it; framework deps are gated as optional extras.

```
src/sleuth/
├── __init__.py            # public re-exports: Sleuth, Session, Result, Source, Chunk, event types
├── _version.py            # version string
├── _agent.py              # the Sleuth class
├── events.py              # Pydantic event types + discriminated Event union
├── types.py               # Source, Chunk, RunStats, Result[T], plus Depth Literal
├── errors.py              # SleuthError hierarchy (see §6)
├── logging.py             # `sleuth` logger setup (no handlers attached by default)
├── engine/
│   ├── __init__.py
│   ├── router.py          # heuristic depth router (no LLM calls)
│   ├── planner.py         # LLM planner (deep mode)
│   ├── executor.py        # parallel backend fan-out, timeouts, cancellation
│   └── synthesizer.py     # streaming token + citation emitter
├── backends/
│   ├── __init__.py
│   ├── base.py            # Backend Protocol, Capability StrEnum, BackendError re-export
│   ├── web.py             # WebBackend factory + per-provider classes
│   ├── localfiles.py      # PageIndex-style hierarchical
│   ├── codesearch.py      # ripgrep + tree-sitter
│   └── vectorstore.py     # opt-in adapter
├── memory/
│   ├── __init__.py
│   ├── cache.py           # Cache Protocol + MemoryCache + SqliteCache
│   ├── semantic.py        # SemanticCache (opt-in)
│   └── session.py         # Session ring buffer + persistence
├── llm/
│   ├── __init__.py
│   ├── base.py            # LLMClient Protocol, LLMChunk union, Message, Tool
│   ├── stub.py            # StubLLM for tests (always available; no extra)
│   ├── anthropic.py       # extras=[anthropic]
│   └── openai.py          # extras=[openai]
├── langchain/             # extras=[langchain]
├── langgraph/             # extras=[langgraph]
├── llamaindex/            # extras=[llamaindex]
├── openai_agents/         # extras=[openai-agents]
├── claude_agent/          # extras=[claude-agent]
├── pydantic_ai/           # extras=[pydantic-ai]
├── crewai/                # extras=[crewai]
├── autogen/               # extras=[autogen]
└── mcp/                   # extras=[mcp], includes sleuth-mcp entrypoint
```

Tests mirror this:

```
tests/
├── conftest.py            # cross-cutting fixtures
├── contract/              # Backend-protocol contract suite
├── engine/
├── backends/
├── memory/
├── llm/
├── snapshots/             # syrupy event-stream snapshots
├── adapters/              # one subdir per framework
├── integration/           # env-gated, real APIs
└── perf/                  # pytest-benchmark regression suite
```

---

## 2. File-creation ownership per phase

A file is **owned** by the phase that creates it. Later phases **modify** (never re-create) it. If your plan needs to create a file owned by another phase, flag it at the top of your plan.

| Phase | Owns / creates |
| --- | --- |
| **0 Bootstrap** | `pyproject.toml`, `uv.lock`, `.python-version`, `.pre-commit-config.yaml`, `.github/workflows/{ci,integration,perf,release}.yml`, `CONTRIBUTING.md`, the empty `src/sleuth/__init__.py` and every package `__init__.py` stub, `tests/conftest.py`, `tests/<area>/__init__.py` stubs for every test area in conventions §1 (engine, backends, memory, llm, contract, snapshots, adapters, integration, perf), `.gitignore` additions for `.sleuth/` and `.venv/` |
| **1 Core MVP** | `events.py`, `types.py`, `errors.py`, `logging.py`, `_agent.py`, `llm/base.py`, `llm/stub.py`, `backends/base.py`, `engine/router.py`, `engine/synthesizer.py`, `engine/executor.py` (single-backend fan-out), `memory/cache.py` (in-memory `MemoryCache` only), `memory/session.py` (in-memory ring buffer; persistence comes in Phase 4), `backends/web.py` with **Tavily-only** smoke implementation, `tests/contract/test_backend_protocol.py`, `tests/engine/`, `tests/snapshots/` baseline |
| **2 LocalFiles** | `backends/localfiles.py`, indexer/navigator helpers under `backends/_localfiles/`, `tests/backends/test_localfiles.py`. Resolves spec §15 #2 (PDF parser) inside the plan |
| **3 Planner + deep mode** | `engine/planner.py`. Modifies `engine/executor.py` for speculative prefetch and `engine/router.py` to choose `deep` |
| **4 Memory** | Replaces `memory/cache.py` with `SqliteCache`, adds `memory/semantic.py`, expands `memory/session.py` with `save()/load()/flush()` |
| **5 CodeSearch** | `backends/codesearch.py`, `backends/_codesearch/` helpers |
| **6 VectorStoreRAG** | `backends/vectorstore.py`, vector-adapter sub-modules |
| **7 Framework adapters** | `langchain/`, `langgraph/`, `llamaindex/`, `openai_agents/`, `claude_agent/`, `pydantic_ai/`, `crewai/`, `autogen/` (each is a self-contained subpackage with its own tests under `tests/adapters/<framework>/`) |
| **8 MCP server** | `mcp/server.py`, `mcp/__main__.py`, MCP-specific config loader. Modifies `pyproject.toml` to register `sleuth-mcp = "sleuth.mcp.__main__:main"`. Resolves spec §15 #5 |
| **9 Web provider shims** | Expands `backends/web.py` with Exa, Brave, SerpAPI on top of Phase 1's Tavily shim. Resolves spec §15 #4 |
| **10 LLM shims** | `llm/anthropic.py`, `llm/openai.py`. Resolves spec §15 #3 inside the plan |
| **11 Perf hardening** | Modifies `engine/executor.py` (per-backend timeouts), `.github/workflows/perf.yml` (gates), creates `tests/perf/` |

---

## 3. `pyproject.toml` shape (Phase 0 lands; later phases append)

Phase 0 creates this skeleton. Later phases that need a new optional dep just add a row to `[project.optional-dependencies]` and (if applicable) register a script in `[project.scripts]`.

```toml
[project]
name = "agent-sleuth"
version = "0.1.0"
description = "Plug-and-play agentic search with reasoning, planning, citations, and observability."
requires-python = ">=3.11"
license = "MIT"
authors = [{name = "agent-sleuth maintainers"}]
dependencies = [
    "pydantic>=2.6",
    "httpx>=0.27",
    "anyio>=4.3",
    "aiosqlite>=0.19",      # Cache backend (Phase 4)
]

[project.optional-dependencies]
# LLM shims (Phase 10)
anthropic     = ["anthropic>=0.40"]
openai        = ["openai>=1.40"]

# Framework adapters (Phase 7)
langchain     = ["langchain-core>=0.3"]
langgraph     = ["langgraph>=0.2"]
llamaindex    = ["llama-index-core>=0.11"]
openai-agents = ["openai-agents>=0.1"]
claude-agent  = ["claude-agent-sdk>=0.1"]
pydantic-ai   = ["pydantic-ai>=0.0.13"]
crewai        = ["crewai>=0.80"]
autogen       = ["pyautogen>=0.3"]

# MCP frontend (Phase 8)
mcp           = ["mcp>=1.0", "uvicorn>=0.30"]

# Backends
localfiles    = ["pymupdf>=1.24", "pathspec>=0.12", "xxhash>=3.4"]    # Phase 2
code          = ["tree-sitter>=0.22", "pathspec>=0.12"]                # Phase 5
code-embed    = ["fastembed>=0.3", "numpy>=1.26"]                      # Phase 5 optional re-rank
exa           = ["exa-py>=1.0"]                                        # Phase 9
web-fetch     = ["trafilatura>=1.7", "tiktoken>=0.7"]                  # Phase 9 fetch=True mode

# Vector store adapters (Phase 6) — each behind its own extra
pinecone      = ["pinecone-client>=4.0"]
qdrant        = ["qdrant-client>=1.10"]
chroma        = ["chromadb>=0.5"]
weaviate      = ["weaviate-client>=4.6"]

# Memory layer (Phase 4)
semantic      = ["fastembed>=0.3", "numpy>=1.26"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "pytest-xdist>=3.5",
    "pytest-benchmark>=4.0",
    "syrupy>=4.6",
    "hypothesis>=6.100",
    "respx>=0.21",
    "ruff>=0.6",
    "mypy>=1.11",
    "pre-commit>=3.7",
    "commitizen>=3.27",
    "detect-secrets>=1.5",
    "pip-audit>=2.7",
]

[project.scripts]
# Phase 8 adds: sleuth-mcp = "sleuth.mcp.__main__:main"

[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sleuth"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "RUF", "ASYNC"]

[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]
mypy_path = "src"

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "-ra --strict-markers --strict-config"
testpaths = ["tests"]
markers = [
    "unit: default — runs every push",
    "integration: env-gated, nightly",
    "perf: regression suite",
    "adapter: per-framework smoke",
]

[tool.coverage.run]
source = ["src/sleuth"]
branch = true

[tool.coverage.report]
fail_under = 85
exclude_also = ["if TYPE_CHECKING:", "@overload"]
```

---

## 4. Public API surface (Phase 1 owns; later phases extend)

These signatures are frozen. Any later phase that wants to change them must update this section first.

```python
# sleuth/_agent.py
class Sleuth:
    def __init__(
        self,
        llm: LLMClient,
        backends: list[Backend],
        *,
        fast_llm: LLMClient | None = None,
        cache: Cache | Literal["default"] | None = "default",
        semantic_cache: SemanticCache | bool = False,
        session: Session | None = None,
    ): ...

    def aask(
        self,
        query: str,
        *,
        depth: Depth = "auto",
        max_iterations: int = 4,
        schema: type[BaseModel] | None = None,
        session: Session | None = None,
    ) -> AsyncIterator[Event]: ...

    def ask(
        self,
        query: str,
        *,
        depth: Depth = "auto",
        max_iterations: int = 4,
        schema: type[BaseModel] | None = None,
        session: Session | None = None,
    ) -> Result[T]: ...

    def asummarize(
        self,
        target: str,                                # URL, file path, or query
        *,
        length: Literal["brief", "standard", "thorough"] = "standard",
        schema: type[BaseModel] | None = None,
    ) -> AsyncIterator[Event]: ...

    def summarize(self, target: str, *, length: Length = "standard", schema: type[BaseModel] | None = None) -> Result[T]: ...

    async def warm_index(self) -> None: ...    # eager index of LocalFiles backends


Depth = Literal["auto", "fast", "deep"]
Length = Literal["brief", "standard", "thorough"]
```

`__init__.py` re-exports: `Sleuth`, `Session`, `Result`, `Source`, `Chunk`, `RouteEvent`, `PlanEvent`, `SearchEvent`, `FetchEvent`, `ThinkingEvent`, `TokenEvent`, `CitationEvent`, `CacheHitEvent`, `DoneEvent`, `Event`, `Depth`, `Length`.

---

## 5. Frozen protocols and types

Phase 1 creates these. Later phases use them as-is. (Source of truth: spec §6 + §7.1 — copy here for plan-author convenience, do not diverge.)

### 5.1 LLMClient (`sleuth/llm/base.py`)

```python
@dataclass
class TextDelta: text: str
@dataclass
class ReasoningDelta: text: str
@dataclass
class ToolCall: id: str; name: str; arguments: dict[str, Any]
@dataclass
class Stop: reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "error"]

LLMChunk = TextDelta | ReasoningDelta | ToolCall | Stop

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None

class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]

class LLMClient(Protocol):
    name: str
    supports_reasoning: bool
    supports_structured_output: bool

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]: ...
```

### 5.2 Backend (`sleuth/backends/base.py`)

```python
class Capability(StrEnum):
    WEB = "web"
    DOCS = "docs"
    CODE = "code"
    FRESH = "fresh"
    PRIVATE = "private"

class Backend(Protocol):
    name: str
    capabilities: frozenset[Capability]

    async def search(self, query: str, k: int = 10) -> list[Chunk]: ...
```

### 5.3 Cache (`sleuth/memory/cache.py`)

```python
class Cache(Protocol):
    async def get(self, namespace: str, key: str) -> Any | None: ...
    async def set(self, namespace: str, key: str, value: Any, *, ttl_s: int | None = None) -> None: ...
    async def delete(self, namespace: str, key: str) -> None: ...
    async def clear(self, namespace: str | None = None) -> None: ...
```

Namespaces: `"query"`, `"fetch"`, `"plan"`, `"index"` (Phase 2 only — see spec §8).

### 5.4 Event types (`sleuth/events.py`)

Verbatim from spec §5. Discriminated `Event` union. Do not rename or reorder fields. Note these specific shapes (consumed by multiple phases):

```python
class PlanStep(BaseModel):
    query: str
    backends: list[str] | None = None    # backend names to try; None = router decides
    done: bool = False                   # end-of-iteration sentinel (deep mode)

class SearchEvent(BaseModel):
    type: Literal["search"]
    backend: str
    query: str
    error: str | None = None             # set when the backend timed out or raised

class FetchEvent(BaseModel):
    type: Literal["fetch"]
    url: str
    status: int
    error: str | None = None
```

Phase 1 owns `PlanStep` in `events.py` (Pydantic, public). Phase 3's `Planner` imports it — **do not redefine as `@dataclass` in `engine/planner.py`**.

### 5.6 Embedder protocol (`sleuth/memory/semantic.py`, Phase 4)

```python
class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
```

Phase 4 owns the protocol and ships `StubEmbedder` (test default) + `FastembedEmbedder` (behind `agent-sleuth[semantic]`). Phase 6 (`VectorStoreRAG`) imports `Embedder` from `sleuth.memory.semantic` — **do not redefine in `backends/vectorstore.py`**.

### 5.5 Result / Source / Chunk / RunStats (`sleuth/types.py`)

Verbatim from spec §6 (with `Result[T]` generic and `RunStats.first_token_ms`).

---

## 6. Error hierarchy (`sleuth/errors.py`, Phase 1)

```python
class SleuthError(Exception): ...
class BackendError(SleuthError): ...
class BackendTimeoutError(BackendError): ...
class LLMError(SleuthError): ...
class CacheError(SleuthError): ...
class ConfigError(SleuthError): ...
```

Backends raise `BackendError` (or `BackendTimeoutError`); the engine catches them per spec §7.1 "Failure handling" and turns them into `SearchEvent(error=...)`.

---

## 7. Naming

- **Files:** `lower_snake_case.py`. Private helpers live as `_underscore_prefix.py` or in `_subpkg/`.
- **Classes:** `PascalCase`. Protocols are noun-form (`Backend`, `LLMClient`, `Cache`) — never `IBackend`.
- **Async functions with sync twins:** async gets the `a` prefix (`aask`/`ask`, `asummarize`/`summarize`). Internal-only async functions don't need the prefix.
- **Test files:** `tests/<area>/test_<thing>.py`. Class-based grouping only when shared fixtures justify it.
- **Fixtures:** cross-cutting in `tests/conftest.py`; area-specific in `tests/<area>/conftest.py`.
- **Settings/env:** `SLEUTH_*` prefix (e.g. `SLEUTH_CACHE_DIR`, `SLEUTH_LOG_LEVEL`).
- **Cache directory:** `~/.sleuth/cache/` (user-home) for query/fetch/plan; `<corpus>/.sleuth/index/` (repo-local) for `IndexCache`.

---

## 8. Test infrastructure (Phase 0 lands; everyone else uses)

| Concern | Tool |
| --- | --- |
| Async test runner | `pytest-asyncio` (auto mode) |
| HTTP mocking | `respx` (async-first, sits in front of `httpx`) |
| Snapshot tests | `syrupy` for event-stream sequences |
| Property tests | `hypothesis` for `Backend` protocol invariants |
| Determinism | `StubLLM` (Phase 1) replays scripted responses |
| Coverage gate | `pytest-cov` ≥85% on `src/sleuth/`; adapters may dip lower |

`StubLLM` interface (Phase 1 owns — do not redefine):

```python
class StubLLM(LLMClient):
    name = "stub"
    supports_reasoning = False
    supports_structured_output = True

    def __init__(
        self,
        responses: Sequence[str | LLMChunk] | Callable[[list[Message]], AsyncIterator[LLMChunk]],
    ) -> None: ...

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]: ...
```

If you pass a list of strings, `StubLLM` emits each as a `TextDelta` then a `Stop("end_turn")`. If you pass a callable, it owns the full response.

`BackendTestKit` (Phase 1 owns — Phase 2/5/6/7/9 reuse): a parametrized pytest harness that asserts protocol compliance, error shapes, timeout behavior, cancellation safety. Plans for new backends MUST run their backend through `BackendTestKit`; they don't write equivalent tests by hand.

---

## 9. Branch + commit conventions

Per spec §9. Plan agents:

- Target `develop` (Phase 0 creates it; later phases assume it exists).
- Branch name: `feature/phase-<N>-<short-name>` (e.g. `feature/phase-1-core-mvp`).
- Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `perf:`.
- Each plan task ends in a commit step with the exact message.
- No `--no-verify`. If pre-commit fails, fix the underlying issue.

---

## 10. Cross-plan rules

- **Reference, don't reproduce.** If your plan needs a Phase-1 type, write `from sleuth.types import Result` — don't redefine `Result` in your plan.
- **Don't add to other phases' files in your plan.** If you need to extend a file owned by another phase, escalate at the top of your plan.
- **Defer deps.** If a phase later than yours owns a dep you'd want, mock it (`StubLLM`, fake backend) instead.
- **Self-review before handoff.** Run the writing-plans skill self-review checklist on your plan before saying done.
