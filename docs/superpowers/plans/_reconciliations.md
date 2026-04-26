# Plan reconciliations

The 12 phase plans were authored in parallel by 12 agents. Each agent saw the spec and the conventions doc but not the other plans. After cross-review, a handful of cross-plan deltas surfaced. Apply these when executing each plan.

> **Order of authority:** spec > `_conventions.md` (latest) > this file > individual plan. If a plan disagrees with `_conventions.md`, follow `_conventions.md`.

---

## Phase 0 — Bootstrap

**Update `pyproject.toml` to use the full extras list from `_conventions.md` §3** (not what's in the plan body — the plan was written before the extras consolidation pass).

Add these to `[project.dependencies]` (the plan may have only `pydantic / httpx / anyio`):

```
"aiosqlite>=0.19",
```

Add these `[project.optional-dependencies]` rows beyond the framework-adapter set the plan already has:

```
localfiles    = ["pymupdf>=1.24", "pathspec>=0.12", "xxhash>=3.4"]
code          = ["tree-sitter>=0.22", "pathspec>=0.12"]
code-embed    = ["fastembed>=0.3", "numpy>=1.26"]
exa           = ["exa-py>=1.0"]
web-fetch     = ["trafilatura>=1.7", "tiktoken>=0.7"]
pinecone      = ["pinecone-client>=4.0"]
qdrant        = ["qdrant-client>=1.10"]
chroma        = ["chromadb>=0.5"]
weaviate      = ["weaviate-client>=4.6"]
semantic      = ["fastembed>=0.3", "numpy>=1.26"]
mcp           = ["mcp>=1.0", "uvicorn>=0.30"]   # uvicorn was missing
```

**Also create `tests/<area>/__init__.py` stubs** for every test area listed in `_conventions.md` §1: `engine, backends, memory, llm, contract, snapshots, adapters, integration, perf`. This avoids the Phase 7 / Phase 8 race over `tests/adapters/__init__.py`.

---

## Phase 1 — Core MVP

The plan already gets `PlanStep` and `SearchEvent.error` right (Phase 1 is canonical). No reconciliation needed.

**Note for executors:** Phase 1's `_agent.summarize()` builds `query = f"summarize: {target} (length={length})"` and runs the engine. Phase 2 will extend this with file-path branching to `LocalFiles._get_summary()` — that work belongs to Phase 2's plan, not Phase 1's.

---

## Phase 2 — LocalFiles

The plan correctly resolves spec §15 #2 by picking **pymupdf**.

**Add a step at the end of the plan** (before the final lint/PR task) that modifies `_agent.py.asummarize()` to route file-path targets through `LocalFiles._get_summary(target, length)` when a `LocalFiles` backend is present in the agent's backend list. The plan documents the integration point at Task 12 but doesn't ship the wiring — add it.

---

## Phase 3 — Planner + deep mode

**Do not redefine `PlanStep`.** The plan defines it as `@dataclass class PlanStep` in `engine/planner.py`. Replace with:

```python
from sleuth.events import PlanStep
```

The canonical Pydantic shape lives in Phase 1's `events.py` per `_conventions.md` §5.4:

```python
class PlanStep(BaseModel):
    query: str
    backends: list[str] | None = None
    done: bool = False
```

Phase 3's plan uses `PlanStep(query=..., done=...)` and `PlanStep(query=..., backends=...)` — those constructions still work with the Pydantic shape, just import from `sleuth.events`.

**`_agent.py` modification for `depth="deep"` dispatch:** the plan flags this as an inline NOTE. Make it an explicit task that adds the ~5 lines to `_agent.py`. This is a legitimate cross-phase modification to a Phase 1 file.

---

## Phase 4 — Memory layer

The plan correctly owns `Embedder` in `src/sleuth/memory/semantic.py` per `_conventions.md` §5.6. No reconciliation needed beyond `pyproject.toml` (handled by Phase 0 reconciliation above).

---

## Phase 5 — CodeSearch

The `code` and `code-embed` extras now live in the canonical `pyproject.toml` (Phase 0). Drop the plan's task that adds them — verify they're already present instead.

---

## Phase 6 — VectorStoreRAG

**Do not redefine `Embedder`.** The plan defines it in `src/sleuth/backends/vectorstore.py`. Replace with:

```python
from sleuth.memory.semantic import Embedder
```

Per `_conventions.md` §5.6, Phase 4 owns the `Embedder` protocol. The shape and ownership were not nailed down when the plan was written — they are now.

The vendor extras (`pinecone`, `qdrant`, `chroma`, `weaviate`) now live in canonical `pyproject.toml` (Phase 0). Drop the plan's tasks that add them.

---

## Phase 7 — Framework adapters

`tests/adapters/__init__.py` is now Phase 0's responsibility (see Phase 0 reconciliation). Drop any task that creates it.

---

## Phase 8 — MCP server

`uvicorn` is now part of the `mcp` extra (`_conventions.md` §3) and lands in Phase 0's `pyproject.toml`. The plan's runtime-import-with-error-message workaround can be removed; the dep is just present. Drop the related tasks/steps.

`tests/adapters/__init__.py` and `tests/adapters/mcp/__init__.py` parent stub: Phase 0 owns the former (see Phase 0 reconciliation). Phase 8 still creates `tests/adapters/mcp/__init__.py` as needed.

---

## Phase 9 — Web providers

`exa` and `web-fetch` extras now live in the canonical `pyproject.toml` (Phase 0). Drop the plan's task that adds them.

---

## Phase 10 — LLM shims

No reconciliation needed.

---

## Phase 11 — Perf hardening

`Backend.timeout_s` is **not** in the frozen `Backend` protocol per `_conventions.md` §5.2. The plan's `getattr(backend, "timeout_s", None)` duck-typed lookup is the correct pattern — keep as-is. We're deliberately not adding `timeout_s` to the protocol because most backends won't define it and we don't want to force them to.

`scripts/perf-baseline.py` is owned by Phase 11 (no convention update needed — `scripts/` is whose-creates-it-owns-it).

---

## Spec edit applied

`docs/superpowers/specs/2026-04-25-sleuth-design.md` §5 was updated alongside the plan-writing pass to:

- Add `error: str | None = None` to `SearchEvent` (referenced from §7.1 "Failure handling" but missing from §5; introduced when §7.1 was added).
- Add `error: str | None = None` to `FetchEvent` for the same reason.
- Inline the `PlanStep` definition next to `PlanEvent` so it's not an undefined symbol.

If you read the spec before reading this file: the relevant section is §5 "Event stream".
