# CLAUDE.md — agent-sleuth

Repo memory for Claude Code. Read this first before doing anything in this repo.

## What this project is

**Sleuth** is a Python package that gives any agentic application a fast, plug-and-play search capability with built-in reasoning, planning, observability, citations, and structured output. It targets the popular Python agent frameworks via thin native adapters and any other client via an MCP server.

- GitHub repo / distribution: **`agent-sleuth`**
- Python import name: **`sleuth`**
- MCP server binary: **`sleuth-mcp`**

## Current status

**Design phase, no code yet.**

- Brainstorming complete. Design spec written and committed.
- Spec location: [`docs/superpowers/specs/2026-04-25-sleuth-design.md`](docs/superpowers/specs/2026-04-25-sleuth-design.md)
- Implementation plan: **not yet written.** Next step.

## How to resume work

When you pick this up in a new session:

1. **Read the spec first** — `docs/superpowers/specs/2026-04-25-sleuth-design.md`. It is the single source of truth for the design. Do not relitigate decisions already in there; if something needs to change, edit the spec deliberately and note why.
2. **Review §15 of the spec** ("Open questions to resolve before / during implementation"). Those are the gaps the implementation plan needs to close.
3. **If the user is ready to implement:** invoke the `superpowers:writing-plans` skill on the spec doc to generate the implementation plan. Do not start writing code without an approved plan.
4. **If the user wants to revise the design first:** edit the spec, run a self-review pass (placeholders, contradictions, scope, ambiguity), then proceed to writing-plans.

## Key decisions baked into the design

These are settled. Treat them as constraints, not options to revisit casually.

| Area | Decision |
| --- | --- |
| Language | Python, async-first with sync wrappers (`aask` async / `ask` sync) |
| LLM strategy | **BYOK.** Package never imports a model SDK as a hard dep. Thin optional shims for Anthropic/OpenAI under `sleuth.llm`. |
| Search backends | Hybrid: wrap web search APIs (Tavily/Exa/Brave/SerpAPI), own local backends. |
| **Default local backend** | **PageIndex-style hierarchical tree-of-contents per document. NOT vector RAG.** Vector retrieval is opt-in via `VectorStoreRAG`. |
| Reasoning depth | Configurable: `auto` (router decides) / `fast` (single-shot fan-out) / `deep` (full reflect loop). Default `auto`. |
| Memory | Three explicit layers: `Session` (multi-turn), `Cache` (transparent dedup, on by default), `SemanticCache` (opt-in, off by default). |
| Output | Streaming token + citations + optional Pydantic structured output, all in one event stream. |
| Observability | Single typed event stream — `RouteEvent`, `PlanEvent`, `SearchEvent`, `FetchEvent`, `ThinkingEvent`, `TokenEvent`, `CitationEvent`, `CacheHitEvent`, `DoneEvent`. |
| Data shapes | Public + serialized = Pydantic v2. Internal hot paths = `@dataclass`/`attrs`. |
| Frontends | Native Python SDK + thin adapters for LangChain, LangGraph, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, Pydantic AI, CrewAI, AutoGen. Plus an MCP server. |
| Adapter tiers | Tier 1 (full support): LangChain, MCP, Claude Agent SDK. Tier 2 (best-effort): the rest. |
| Optional deps | Each framework adapter is its own extra: `pip install agent-sleuth[langchain]`, etc. Core has zero framework deps. |

## Repo conventions

### Branching (Gitflow)

| Branch | Purpose |
| --- | --- |
| `main` | Released versions only. Each merge here is a tagged release. Protected. |
| `develop` | Integration branch for the next release. Feature branches merge here. Protected. |
| `feature/<name>` | Branched from `develop`, merged back via PR. One per implementation chunk. |
| `release/<version>` | Stabilization branch off `develop`. Merges into `main` (tagged) and `develop`. |
| `hotfix/<version>` | Off `main` for emergencies. Merges into `main` (tagged) and `develop`. |

- All work goes through PR. **No direct commits to `main` or `develop`.**
- Required PR checks (once configured): `ruff`, `mypy`, `pytest`, backend contract suite.

### Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`, `breaking!:`. CHANGELOG is auto-generated at release time.

### Versioning

SemVer. While `0.x.y`, any minor may break. From `1.0.0` forward, strict semver.

## Tooling and dev environment

Full detail in spec **§16**. Key points to honor:

| Concern | Choice |
| --- | --- |
| Package / venv manager | **`uv`** (only). `uv.lock` checked in. `.python-version` checked in. |
| Lint + format | `ruff` (replaces black/isort/flake8/pyupgrade). |
| Type check | `mypy --strict` for `src/sleuth/`; looser for `tests/`. Pydantic v2 plugin on. |
| Pre-commit | `ruff`, `ruff-format`, `mypy`, `commitizen` (commit-msg), `detect-secrets`, standard hygiene hooks. |
| Test runner | `pytest` + `pytest-asyncio` (auto mode), `pytest-cov` (≥85% gate), `syrupy` (snapshot), `pytest-xdist`, `hypothesis`, `pytest-benchmark`. |
| Test markers | `unit` (default) / `integration` (nightly) / `perf` / `adapter`. |
| CI | GitHub Actions: `ci.yml` (every push), `integration.yml` (nightly), `perf.yml` (PR), `release.yml` (tag on `main`). Matrix: Python 3.11/3.12/3.13 × ubuntu/macos. |
| Release | PyPI via Trusted Publisher (OIDC) on signed tag push. Auto-CHANGELOG via `git-cliff`. |
| Security | Dependabot weekly + auto-merge of patches; `pip-audit` in CI; GitHub Secret Scanning enabled. |
| Reproducibility | All CI uses `uv sync --frozen`. `StubLLM` + `respx` for deterministic tests. |

Common commands during development:

```bash
uv sync --all-extras --group dev          # initial setup
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run pytest -m "not integration"        # fast tests
uv run pytest -m integration              # nightly tests (needs API keys)
uv run ruff check . && uv run ruff format .
uv run mypy src/sleuth
uv build                                   # wheel + sdist
```

## Coding constraints (please honor when implementing)

1. **Async-first.** Every public coroutine has a sync twin. Never block the event loop in core paths.
2. **Pydantic v2 for public data shapes**, plain dataclasses/attrs for internal-only hot-path structs.
3. **No hidden global state.** Caches, sessions, and clients are all explicit objects passed by the user.
4. **Streaming is non-negotiable.** The synthesizer yields tokens; the event stream is the primary output. Even cached runs replay through the event stream (prefixed with `CacheHitEvent`).
5. **Backend protocol is the universal contract.** Web, local, code, and custom backends all implement `async def search(query, k) -> list[Chunk]`. Don't special-case backends in the engine.
6. **Don't import a model SDK as a hard dep.** Anthropic/OpenAI shims live behind optional extras and import lazily.

## Initial setup not yet done

When implementation starts, the first feature branch should land (full detail in spec §9.2 and §16):

1. `develop` branch created off `main`.
2. `uv` project initialized: `pyproject.toml` (with `agent-sleuth` distribution name + optional-dep groups), `.python-version`, `uv.lock`.
3. `.pre-commit-config.yaml` with ruff, ruff-format, mypy, commitizen (commit-msg), detect-secrets, and standard hygiene hooks.
4. `.github/workflows/{ci,integration,perf,release}.yml`.
5. Branch protection on `main` and `develop`.
6. `CONTRIBUTING.md` documenting Gitflow + Conventional Commits + uv workflow.
7. Empty package skeleton at `src/sleuth/__init__.py` plus the directory layout below.
8. PyPI Trusted Publisher (OIDC) configured for the release workflow.

## Design open questions (from spec §15)

These need to be resolved during the implementation plan or the first PR:

1. Project layout — single package vs. namespace package.
2. PDF parser choice (`pypdf` / `pdfplumber` / `pymupdf`) — need a benchmark.
3. Whether to ship a literal default `fast_llm`, or document the recommendation only.
4. `WebBackend` factory shape — single class with `provider="..."` vs. one class per provider.
5. MCP server config format — TOML recommended (matches `pyproject.toml`).

## Where things will live (planned)

```
agent-sleuth/
├── pyproject.toml
├── README.md
├── CLAUDE.md                   ← this file
├── CHANGELOG.md                ← auto-generated
├── CONTRIBUTING.md
├── LICENSE
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-25-sleuth-design.md
├── src/
│   └── sleuth/
│       ├── __init__.py
│       ├── engine/             ← router, planner, executor, synthesizer
│       ├── backends/
│       │   ├── web.py
│       │   ├── localfiles.py   ← hierarchical tree-of-contents
│       │   ├── codesearch.py
│       │   └── vectorstore.py  ← opt-in adapter
│       ├── memory/             ← Cache, SemanticCache, Session
│       ├── llm/                ← LLMClient protocol + Anthropic/OpenAI shims
│       ├── events.py           ← Pydantic event types
│       ├── langchain/          ← optional adapter (extras=langchain)
│       ├── llamaindex/
│       ├── openai_agents/
│       ├── claude_agent/
│       ├── pydantic_ai/
│       ├── crewai/
│       ├── autogen/
│       └── mcp/                ← MCP server (sleuth-mcp entrypoint)
└── tests/
    ├── contract/               ← BackendTestKit suite
    ├── engine/                 ← stub-LLM unit tests
    ├── snapshots/              ← event-stream snapshot tests
    ├── integration/            ← env-gated, real APIs
    ├── adapters/               ← per-framework smoke tests
    └── perf/                   ← regression suite
```

This layout is a recommendation, not yet decided — see open question #1.
