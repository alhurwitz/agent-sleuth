# CLAUDE.md — agent-sleuth

Repo memory for Claude Code. Read this first; the design spec is the source of truth.

**What this is.** Sleuth is a Python package that gives any agentic application a fast, plug-and-play search capability with built-in reasoning, planning, observability, citations, and structured output. It targets the popular Python agent frameworks via thin native adapters, and any other client via an MCP server.

- Distribution: `agent-sleuth` (PyPI) · Import: `sleuth` · MCP binary: `sleuth-mcp`

## Current status

**Design phase, no code yet.**

- Spec: [docs/superpowers/specs/2026-04-25-sleuth-design.md](docs/superpowers/specs/2026-04-25-sleuth-design.md). Single source of truth — if this file disagrees with the spec, the spec wins.
- Implementation plan: not yet written. Next step.

## How to resume work

1. **Read the spec first.** Do not relitigate decisions already in there; if something needs to change, edit the spec deliberately and note why.
2. **Review spec §15** ("Open questions to resolve before / during implementation"). Those are the gaps the implementation plan needs to close.
3. **If the user is ready to implement:** invoke the `superpowers:writing-plans` skill on the spec to generate the implementation plan. Do not start writing code without an approved plan.
4. **If the user wants to revise the design first:** edit the spec, run a self-review pass (placeholders, contradictions, scope, ambiguity), then proceed to writing-plans.

## Key decisions (do not relitigate casually)

| Area | Decision |
| --- | --- |
| Language | Python, async-first with sync wrappers (`aask` async / `ask` sync) |
| LLM strategy | **BYOK.** Package never imports a model SDK as a hard dep. Thin optional shims for Anthropic/OpenAI under `sleuth.llm`. |
| Search backends | Hybrid: wrap web search APIs (Tavily/Exa/Brave/SerpAPI), own local backends. |
| **Default local backend** | **PageIndex-style hierarchical tree-of-contents per document. NOT vector RAG.** Vector retrieval is opt-in via `VectorStoreRAG`. |
| Reasoning depth | Configurable: `auto` (router decides) / `fast` (single-shot fan-out) / `deep` (full reflect loop). Default `auto`. |
| Memory | Three explicit layers: `Session` (multi-turn), `Cache` (transparent dedup, on by default), `SemanticCache` (opt-in, off by default). |
| Output | Streaming token + citations + optional Pydantic structured output, all in one event stream. |
| Observability | Single typed event stream (`Route`, `Plan`, `Search`, `Fetch`, `Thinking`, `Token`, `Citation`, `CacheHit`, `Done`). Stdlib `logging` under the `sleuth` namespace as a secondary channel. |
| Data shapes | Public + serialized = Pydantic v2 (`Result[T]` is generic). Internal hot paths = `@dataclass`/`attrs`. |
| Frontends | Native Python SDK + thin adapters for LangChain, LangGraph, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, Pydantic AI, CrewAI, AutoGen. Plus an MCP server. |
| Adapter tiers | Tier 1 (full): LangChain, Claude Agent SDK, MCP frontend. Tier 2 (best-effort): LangGraph, LlamaIndex, OpenAI Agents SDK, Pydantic AI, CrewAI, AutoGen. |
| Optional deps | Each framework adapter is its own extra: `pip install agent-sleuth[langchain]`, etc. Core has zero framework deps. |

## Coding constraints

1. **Async-first.** Every public coroutine has a sync twin. Never block the event loop in core paths.
2. **Pydantic v2 for public data shapes**, plain dataclasses/attrs for internal-only hot-path structs.
3. **No hidden global state.** Caches, sessions, and clients are all explicit objects passed by the user.
4. **Streaming is non-negotiable.** The synthesizer yields tokens; the event stream is the primary output. Even cached runs replay through the event stream (prefixed with `CacheHitEvent`).
5. **Backend protocol is the universal contract.** Web, local, code, and custom backends all implement `async def search(query, k) -> list[Chunk]`. Don't special-case backends in the engine.
6. **Don't import a model SDK as a hard dep.** Anthropic/OpenAI shims live behind optional extras and import lazily.

## Where everything else lives

The spec is structured so this file doesn't need to repeat it. When you need detail:

| Topic | See |
| --- | --- |
| Architecture and hard rules | spec §3 |
| Public API, event stream, data shapes | spec §4–6 |
| Backends (web, local-files, code, vector) | spec §7 |
| Memory and cache layers (locations, TTLs) | spec §8 |
| Branching, commits, versioning, release | spec §9 |
| Frontends and adapter tiers | spec §10 |
| Speed levers | spec §11 |
| Testing strategy and CI gates | spec §12 + §16.5–16.6 |
| Risks and mitigations | spec §13 |
| Non-goals (never) vs deferred to v2+ | spec §1 + §14 |
| Open questions to resolve | spec §15 |
| Tooling, dev environment, release automation | spec §16 |
| Bootstrap PR scope (first feature branch) | spec §9 "Initial setup actions" + §16 |
| Repo layout | single package `src/sleuth/` with optional-dep adapters (spec §15 #1, decided 2026-04-25) |

## Common dev commands (once bootstrapped — `pyproject.toml` doesn't exist yet)

```bash
uv sync --all-extras --group dev          # initial setup
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run pytest -m "not integration"        # fast tests
uv run pytest -m integration              # nightly tests (needs API keys)
uv run ruff check . && uv run ruff format .
uv run mypy src/sleuth
uv build                                   # wheel + sdist
```
