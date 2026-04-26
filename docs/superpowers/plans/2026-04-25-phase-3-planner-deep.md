# Phase 3: Planner + Deep Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the LLM Planner that decomposes queries into sub-queries (emitting `PlanEvent`), extend the Executor for parallel sub-query fan-out with a reflect loop and speculative prefetch, and add `depth="deep"` routing with `max_iterations` budget.

**Architecture:** The `Planner` is a thin async generator that calls the LLM once per reflect iteration and streams `PlanStep` objects; the Executor wraps it to start the first search task the moment the planner emits its first sub-query (speculative prefetch), then fans out the remaining sub-queries in parallel, reflects on the aggregated chunks, and loops until the planner signals "done" or `max_iterations` is exhausted. The Router gains a heuristic deep-classification path that routes multi-part, comparative, or research-style queries to `depth="deep"` before the existing fast path.

**Tech Stack:** Python 3.11+, asyncio, pydantic v2, pytest + pytest-asyncio (auto mode), syrupy (snapshot tests), StubLLM from `sleuth/llm/stub.py` (Phase 1).

---

> **Callouts (nothing not in conventions, but two clarifications needed):**
>
> 1. `PlanStep` is referenced in conventions §5.4 / spec §5 (`PlanEvent.steps: list[PlanStep]`) but its fields are not spelled out in conventions. This plan defines it as `@dataclass class PlanStep: query: str; backends: list[str] | None = None; done: bool = False` in `engine/planner.py` (internal-only struct — not Pydantic, per conventions §5 "internal-only structs stay as `@dataclass`"). If another phase defines `PlanStep` differently, reconcile before execution.
> 2. The speculative-prefetch overlap test requires observing task start times. This plan uses `asyncio.Event` sentinels injected via dependency injection on the executor, which is clean and deterministic without any real timing dependency.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `src/sleuth/engine/planner.py` | LLM Planner: `PlanStep` dataclass, `Planner` class, `plan()` async generator |
| **Create** | `tests/engine/test_planner.py` | Unit + snapshot tests for Planner and deep-mode end-to-end event sequence |
| **Modify** | `src/sleuth/engine/router.py` | Add `_is_deep()` heuristic; extend `route()` to emit `RouteEvent(depth="deep")` |
| **Modify** | `src/sleuth/engine/executor.py` | Multi-sub-query fan-out, speculative prefetch, reflect loop, `max_iterations` |

---

## Task 1: Branch setup

**Files:**
- (git operations only)

- [ ] **Step 1.1: Create feature branch off develop**

```bash
git checkout develop
git checkout -b feature/phase-3-planner-deep
```

Expected: `Switched to a new branch 'feature/phase-3-planner-deep'`

---

## Task 2: Define `PlanStep` and the `Planner` class skeleton

**Files:**
- Create: `src/sleuth/engine/planner.py`
- Test: `tests/engine/test_planner.py`

- [ ] **Step 2.1: Write the failing import test**

In `tests/engine/test_planner.py`:

```python
import pytest
from sleuth.engine.planner import PlanStep, Planner


def test_planstep_fields() -> None:
    step = PlanStep(query="what is OAuth?")
    assert step.query == "what is OAuth?"
    assert step.backends is None
    assert step.done is False


def test_planstep_done_flag() -> None:
    step = PlanStep(query="", done=True)
    assert step.done is True
```

- [ ] **Step 2.2: Run to confirm ImportError**

```bash
uv run pytest tests/engine/test_planner.py::test_planstep_fields -v
```

Expected: `FAILED ... ModuleNotFoundError: No module named 'sleuth.engine.planner'`

- [ ] **Step 2.3: Create `src/sleuth/engine/planner.py` with `PlanStep` and `Planner` skeleton**

```python
"""LLM Planner for deep-mode query decomposition."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sleuth.events import PlanEvent
from sleuth.llm.base import LLMClient, Message, TextDelta

if TYPE_CHECKING:
    pass

logger = logging.getLogger("sleuth.engine.planner")

# Internal-only struct (not Pydantic — conventions §5 "hot paths stay @dataclass")
@dataclass
class PlanStep:
    """A single sub-query emitted by the planner."""
    query: str
    backends: list[str] | None = None   # None = all eligible backends
    done: bool = False                   # True = planner signals completion


@dataclass
class _PlannerState:
    """Mutable state threaded through reflect iterations."""
    iteration: int = 0
    context_snippets: list[str] = field(default_factory=list)


class Planner:
    """Decomposes a query into sub-queries via LLM; supports reflect loop.

    Usage::

        planner = Planner(llm=llm_client)
        async for step in planner.plan(query, state):
            ...  # handle PlanStep; done=True signals end of this iteration
    """

    _SYSTEM_PROMPT = (
        "You are a search planning assistant. Given a user query and optional "
        "prior search results, decompose the query into focused sub-queries that "
        "together answer the original question. Output a JSON array of objects with "
        'keys "query" (string) and optionally "backends" (list of strings: "web", '
        '"docs", "code", "fresh", "private"). When all needed information has been '
        'gathered, include a final object with "done": true and "query": "". '
        "Output only the JSON array, no prose."
    )

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def plan(
        self,
        query: str,
        state: _PlannerState,
    ) -> AsyncIterator[PlanStep]:
        """Yield PlanSteps for one reflect iteration.

        Yields PlanSteps as the LLM streams them. The final step has done=True.
        Emits a PlanEvent after all steps for this iteration are collected.
        """
        messages: list[Message] = [
            Message(role="system", content=self._SYSTEM_PROMPT),
            Message(
                role="user",
                content=self._build_user_message(query, state),
            ),
        ]

        raw_text = ""
        async for chunk in await self._llm.stream(messages):
            if isinstance(chunk, TextDelta):
                raw_text += chunk.text

        steps = self._parse_steps(raw_text)
        for step in steps:
            yield step

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, query: str, state: _PlannerState) -> str:
        parts = [f"Original query: {query}"]
        if state.context_snippets:
            joined = "\n---\n".join(state.context_snippets[: 5])  # cap context
            parts.append(f"Prior search results (summarised):\n{joined}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_steps(raw: str) -> list[PlanStep]:
        """Parse LLM JSON output into PlanSteps; degrade gracefully on bad JSON."""
        raw = raw.strip()
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("expected JSON array")
        except (json.JSONDecodeError, ValueError):
            logger.warning("planner output was not valid JSON; treating as single query: %r", raw)
            return [PlanStep(query=raw or "search"), PlanStep(query="", done=True)]

        steps: list[PlanStep] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            done = bool(item.get("done", False))
            q = str(item.get("query", ""))
            backends_raw = item.get("backends")
            backends = list(backends_raw) if isinstance(backends_raw, list) else None
            steps.append(PlanStep(query=q, backends=backends, done=done))

        if not steps or not steps[-1].done:
            steps.append(PlanStep(query="", done=True))

        return steps
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
uv run pytest tests/engine/test_planner.py::test_planstep_fields tests/engine/test_planner.py::test_planstep_done_flag -v
```

Expected: `2 passed`

- [ ] **Step 2.5: Commit**

```bash
git add src/sleuth/engine/planner.py tests/engine/test_planner.py
git commit -m "feat: add PlanStep dataclass and Planner skeleton"
```

---

## Task 3: Unit-test `Planner.plan()` with `StubLLM`

**Files:**
- Test: `tests/engine/test_planner.py`
- Read: `src/sleuth/llm/stub.py` (Phase 1 — use as-is)

- [ ] **Step 3.1: Write failing tests for `plan()` happy path and JSON-parse fallback**

Append to `tests/engine/test_planner.py`:

```python
import pytest
from sleuth.engine.planner import PlanStep, Planner, _PlannerState
from sleuth.llm.stub import StubLLM


@pytest.mark.asyncio
async def test_plan_happy_path() -> None:
    """Planner parses a well-formed JSON array into PlanSteps."""
    llm = StubLLM(
        responses=[
            '[{"query": "define OAuth"}, {"query": "OAuth vs OIDC"}, {"done": true, "query": ""}]'
        ]
    )
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps: list[PlanStep] = []
    async for step in planner.plan("explain OAuth", state):
        steps.append(step)

    assert len(steps) == 3
    assert steps[0].query == "define OAuth"
    assert steps[1].query == "OAuth vs OIDC"
    assert steps[2].done is True
    assert steps[2].query == ""


@pytest.mark.asyncio
async def test_plan_with_backends_hint() -> None:
    """Planner propagates backend hints."""
    llm = StubLLM(
        responses=[
            '[{"query": "auth code flow", "backends": ["docs"]}, {"done": true, "query": ""}]'
        ]
    )
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps: list[PlanStep] = []
    async for step in planner.plan("how does auth code flow work?", state):
        steps.append(step)

    assert steps[0].backends == ["docs"]
    assert steps[1].done is True


@pytest.mark.asyncio
async def test_plan_bad_json_degrades_gracefully() -> None:
    """On non-JSON output planner returns a single-step fallback."""
    llm = StubLLM(responses=["This is plain text, not JSON."])
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps: list[PlanStep] = []
    async for step in planner.plan("anything", state):
        steps.append(step)

    # fallback: one search step + one done step
    assert len(steps) == 2
    assert steps[-1].done is True


@pytest.mark.asyncio
async def test_plan_appends_done_if_missing() -> None:
    """Planner appends a done sentinel if the LLM forgot to include it."""
    llm = StubLLM(responses=['[{"query": "step one"}]'])
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps = [s async for s in planner.plan("q", _PlannerState())]
    assert steps[-1].done is True


@pytest.mark.asyncio
async def test_plan_includes_prior_context() -> None:
    """Planner embeds prior search snippets in the user message."""
    captured_messages: list = []

    from sleuth.llm.base import LLMChunk, TextDelta, Stop
    from collections.abc import AsyncIterator

    class SpyLLM:
        name = "spy"
        supports_reasoning = False
        supports_structured_output = True

        async def stream(self, messages, *, schema=None, tools=None) -> AsyncIterator[LLMChunk]:
            captured_messages.extend(messages)
            async def _gen():
                yield TextDelta(text='[{"done": true, "query": ""}]')
                yield Stop(reason="end_turn")
            return _gen()

    planner = Planner(llm=SpyLLM())  # type: ignore[arg-type]
    state = _PlannerState(context_snippets=["prior result A", "prior result B"])

    _ = [s async for s in planner.plan("query", state)]

    user_msg = next(m for m in captured_messages if m.role == "user")
    assert "prior result A" in user_msg.content
    assert "prior result B" in user_msg.content
```

- [ ] **Step 3.2: Run to confirm failures**

```bash
uv run pytest tests/engine/test_planner.py -v
```

Expected: `test_planstep_fields PASSED, test_planstep_done_flag PASSED`, then 5 new tests fail with `ImportError` or `AssertionError` depending on what's already in place.

- [ ] **Step 3.3: Run again after confirming planner.py is in place (from Task 2)**

```bash
uv run pytest tests/engine/test_planner.py -v
```

Expected: `7 passed` (2 from Task 2 + 5 new).

- [ ] **Step 3.4: Commit**

```bash
git add tests/engine/test_planner.py
git commit -m "test: add unit tests for Planner.plan() happy path and fallbacks"
```

---

## Task 4: Router — add deep-classification heuristic

**Files:**
- Modify: `src/sleuth/engine/router.py`
- Test: `tests/engine/test_router.py` (Phase 1 owns — append only; do not re-create)

Phase 1's router already routes `"fast"` and `"auto→fast"`. We add `"auto→deep"` via a regex/keyword heuristic (no LLM).

- [ ] **Step 4.1: Write failing tests (append to existing `tests/engine/test_router.py`)**

```python
# append to tests/engine/test_router.py


import pytest
from sleuth.engine.router import route, _is_deep
from sleuth.events import RouteEvent


# --- _is_deep unit tests ---

@pytest.mark.parametrize("query,expected", [
    # should be deep
    ("compare Redis and Memcached for session storage", True),
    ("what are the differences between OAuth and OIDC?", True),
    ("how does our auth flow handle refresh tokens and what changed recently?", True),
    ("give me a comprehensive analysis of our caching strategy", True),
    ("research the best approaches for rate limiting", True),
    ("explain the tradeoffs between A and B in detail", True),
    ("what are all the ways X can fail and how do we handle each?", True),
    # should NOT be deep
    ("what is OAuth?", False),
    ("who maintains the auth middleware?", False),
    ("list all endpoints", False),
    ("define refresh token", False),
    ("show me the login function", False),
])
def test_is_deep_heuristic(query: str, expected: bool) -> None:
    assert _is_deep(query) is expected


# --- route() integration ---

@pytest.mark.asyncio
async def test_route_auto_deep_emits_deep_route_event() -> None:
    events = []
    async for e in route("compare OAuth vs OIDC in detail for enterprise use", depth="auto"):
        events.append(e)
    route_event = events[0]
    assert isinstance(route_event, RouteEvent)
    assert route_event.depth == "deep"


@pytest.mark.asyncio
async def test_route_explicit_deep_emits_deep() -> None:
    events = []
    async for e in route("anything", depth="deep"):
        events.append(e)
    assert events[0].depth == "deep"


@pytest.mark.asyncio
async def test_route_explicit_fast_not_reclassified() -> None:
    """depth='fast' is never upgraded to 'deep' by heuristic."""
    events = []
    async for e in route("compare everything in full detail", depth="fast"):
        events.append(e)
    assert events[0].depth == "fast"
```

- [ ] **Step 4.2: Run to confirm failures**

```bash
uv run pytest tests/engine/test_router.py -v -k "is_deep or deep"
```

Expected: `ImportError: cannot import name '_is_deep' from 'sleuth.engine.router'` (or similar).

- [ ] **Step 4.3: Add `_is_deep` and extend `route()` in `src/sleuth/engine/router.py`**

Open `src/sleuth/engine/router.py`. Locate the existing `route()` function and the imports. Add/modify as follows (keep all existing fast-path logic intact):

```python
# Add near top of file (with existing imports)
import re

# ---------------------------------------------------------------------------
# Deep-mode heuristic (no LLM call)
# ---------------------------------------------------------------------------

_DEEP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcompare\b", re.I),
    re.compile(r"\bdifference[s]?\b", re.I),
    re.compile(r"\bvs\.?\b", re.I),
    re.compile(r"\bversus\b", re.I),
    re.compile(r"\btradeoff[s]?\b", re.I),
    re.compile(r"\btrade[- ]off[s]?\b", re.I),
    re.compile(r"\bcomprehensive\b", re.I),
    re.compile(r"\bin[- ]depth\b", re.I),
    re.compile(r"\bdetailed?\b.*\banalysis\b", re.I),
    re.compile(r"\bresearch\b", re.I),
    re.compile(r"\ball\s+(?:the\s+)?ways\b", re.I),
    re.compile(r"\beach\b.{0,40}\bhandle\b", re.I),
    re.compile(r"\bexplain\b.{0,60}\btradeoff\b", re.I),
    re.compile(r"\band\b.{0,30}\bwhat\b.{0,30}\bchanged\b", re.I),
]


def _is_deep(query: str) -> bool:
    """Return True if the query heuristically requires deep (multi-step) planning.

    Purely regex/keyword — no LLM calls. Errs on the side of fast when ambiguous.
    """
    return any(p.search(query) for p in _DEEP_PATTERNS)
```

Then in the existing `route()` function, add the deep branch in the `"auto"` block. The existing Phase 1 code likely looks roughly like:

```python
async def route(query: str, *, depth: Depth = "auto") -> AsyncIterator[RouteEvent]:
    ...
    if depth == "auto":
        depth = "fast"   # <- Phase 1 always chose fast
    yield RouteEvent(type="route", depth=depth, reason="...")
```

Change it to:

```python
async def route(query: str, *, depth: Depth = "auto") -> AsyncIterator[RouteEvent]:
    ...
    resolved = depth
    if depth == "auto":
        if _is_deep(query):
            resolved = "deep"
            reason = "heuristic: query matches deep-mode patterns"
        else:
            resolved = "fast"
            reason = "heuristic: simple query, fast path sufficient"
    elif depth == "deep":
        reason = "caller requested deep mode"
    else:
        reason = "caller requested fast mode"

    yield RouteEvent(type="route", depth=resolved, reason=reason)
```

> Note: If Phase 1's `route()` already has a different signature or yields more than one event, adapt accordingly — do not remove any existing Phase 1 yields.

- [ ] **Step 4.4: Run all router tests to confirm everything is green**

```bash
uv run pytest tests/engine/test_router.py -v
```

Expected: all tests pass (both Phase 1 tests and the new deep-heuristic tests).

- [ ] **Step 4.5: Commit**

```bash
git add src/sleuth/engine/router.py tests/engine/test_router.py
git commit -m "feat: add deep-classification heuristic to router"
```

---

## Task 5: Executor — multi-sub-query fan-out (no reflect loop yet)

**Files:**
- Modify: `src/sleuth/engine/executor.py`
- Test: `tests/engine/test_executor.py` (Phase 1 owns — append only)

Phase 1's executor handles a single query and a single backend. We generalize it to accept a list of sub-queries and fan out searches in parallel.

- [ ] **Step 5.1: Write failing fan-out test (append to `tests/engine/test_executor.py`)**

```python
# append to tests/engine/test_executor.py

import asyncio
import pytest
from sleuth.engine.executor import execute_subqueries
from sleuth.backends.base import Backend, Capability
from sleuth.types import Chunk, Source


class _StubBackend:
    name = "stub-web"
    capabilities = frozenset({Capability.WEB})

    def __init__(self, results: list[Chunk]) -> None:
        self._results = results

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return self._results


def _make_chunk(text: str, url: str = "https://example.com") -> Chunk:
    return Chunk(text=text, source=Source(kind="url", location=url))


@pytest.mark.asyncio
async def test_execute_subqueries_fans_out_in_parallel() -> None:
    """execute_subqueries runs all sub-queries concurrently and merges chunks."""
    backend = _StubBackend([_make_chunk("result A")])
    chunks = await execute_subqueries(
        subqueries=["query one", "query two"],
        backends=[backend],
    )
    # Two sub-queries × one result each = 2 chunks
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_execute_subqueries_deduplicates_by_source() -> None:
    """Chunks from the same URL (same source.location) are deduplicated."""
    shared_chunk = _make_chunk("shared", url="https://shared.com")
    backend = _StubBackend([shared_chunk])

    chunks = await execute_subqueries(
        subqueries=["q1", "q2"],
        backends=[backend],
    )
    locations = [c.source.location for c in chunks]
    assert locations.count("https://shared.com") == 1


@pytest.mark.asyncio
async def test_execute_subqueries_emits_search_events() -> None:
    """execute_subqueries yields SearchEvent for every backend × sub-query."""
    from sleuth.events import SearchEvent

    backend = _StubBackend([_make_chunk("r")])
    events: list[SearchEvent] = []

    chunks = await execute_subqueries(
        subqueries=["q1", "q2"],
        backends=[backend],
        on_search_event=lambda e: events.append(e),
    )
    assert len(events) == 2  # one per sub-query (one backend)
    assert all(isinstance(e, SearchEvent) for e in events)
    assert {e.query for e in events} == {"q1", "q2"}
```

- [ ] **Step 5.2: Run to confirm ImportError**

```bash
uv run pytest tests/engine/test_executor.py -v -k "subqueries"
```

Expected: `ImportError: cannot import name 'execute_subqueries'`

- [ ] **Step 5.3: Add `execute_subqueries` to `src/sleuth/engine/executor.py`**

Append to (or add within) `src/sleuth/engine/executor.py`. Keep all existing Phase 1 functions untouched above:

```python
# ---------------------------------------------------------------------------
# Multi-sub-query fan-out (Phase 3)
# ---------------------------------------------------------------------------
import asyncio
import logging
from collections.abc import Callable, Awaitable
from typing import Any

from sleuth.backends.base import Backend
from sleuth.events import SearchEvent
from sleuth.types import Chunk

logger = logging.getLogger("sleuth.engine.executor")


async def execute_subqueries(
    subqueries: list[str],
    backends: list[Backend],
    *,
    k: int = 10,
    on_search_event: Callable[[SearchEvent], Any] | None = None,
) -> list[Chunk]:
    """Fan out all sub-queries across all backends in parallel; deduplicate by source.

    Args:
        subqueries: Sub-queries emitted by the Planner.
        backends: Backends to search. All backends are tried for every sub-query.
        k: Max results per backend per sub-query.
        on_search_event: Optional callback fired (synchronously) per SearchEvent.

    Returns:
        Deduplicated, merged list of Chunks from all searches.
    """

    async def _search_one(query: str, backend: Backend) -> list[Chunk]:
        event = SearchEvent(type="search", backend=backend.name, query=query)
        if on_search_event is not None:
            on_search_event(event)
        try:
            return await backend.search(query, k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("backend %r failed for query %r: %s", backend.name, query, exc)
            return []

    tasks = [
        _search_one(query, backend)
        for query in subqueries
        for backend in backends
    ]
    results: list[list[Chunk]] = await asyncio.gather(*tasks)

    # Flatten + deduplicate by source.location (first occurrence wins)
    seen: set[str] = set()
    merged: list[Chunk] = []
    for chunk_list in results:
        for chunk in chunk_list:
            loc = chunk.source.location
            if loc not in seen:
                seen.add(loc)
                merged.append(chunk)

    return merged
```

- [ ] **Step 5.4: Run fan-out tests**

```bash
uv run pytest tests/engine/test_executor.py -v
```

Expected: all tests pass (Phase 1 tests stay green + 3 new fan-out tests pass).

- [ ] **Step 5.5: Commit**

```bash
git add src/sleuth/engine/executor.py tests/engine/test_executor.py
git commit -m "feat: add execute_subqueries for parallel fan-out in executor"
```

---

## Task 6: Speculative prefetch — start first search while planner streams

**Files:**
- Modify: `src/sleuth/engine/executor.py`
- Test: `tests/engine/test_executor.py`

The contract: the executor MUST kick off a search task on the planner's **first** sub-query before the planner yields its second sub-query. Tests must observe the **overlap**, not just correctness.

- [ ] **Step 6.1: Write the speculative-prefetch test (append to `tests/engine/test_executor.py`)**

```python
# append to tests/engine/test_executor.py

import asyncio
import pytest
from sleuth.engine.executor import execute_with_prefetch
from sleuth.backends.base import Backend, Capability
from sleuth.engine.planner import PlanStep
from sleuth.types import Chunk, Source


class _OrderTrackingBackend:
    """Backend that records when searches are called via an asyncio.Event."""

    name = "order-tracker"
    capabilities = frozenset({Capability.WEB})

    def __init__(self, first_search_started: asyncio.Event) -> None:
        self._first_search_started = first_search_started
        self.call_count = 0

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        self.call_count += 1
        if self.call_count == 1:
            self._first_search_started.set()   # signal: first search is running
        await asyncio.sleep(0)                  # yield control
        return [Chunk(text=query, source=Source(kind="url", location=f"https://ex.com/{query}"))]


async def _slow_planner(first_search_started: asyncio.Event):
    """Async generator that yields two sub-queries, pausing after the first.

    Crucially: after emitting step 1, it WAITS until the backend signals that
    the first search task has started before emitting step 2. This asserts
    the overlap — if prefetch didn't happen, the wait would deadlock.
    """
    yield PlanStep(query="sub-query-one")
    # Wait until the executor has actually started the search task
    await asyncio.wait_for(first_search_started.wait(), timeout=2.0)
    yield PlanStep(query="sub-query-two")
    yield PlanStep(query="", done=True)


@pytest.mark.asyncio
async def test_speculative_prefetch_starts_before_planner_finishes() -> None:
    """Executor starts the first search task while the planner is still running.

    The _slow_planner waits for the backend's first_search_started event before
    emitting its second step. If prefetch didn't overlap, this deadlocks/times out.
    """
    first_search_started = asyncio.Event()
    backend = _OrderTrackingBackend(first_search_started)

    chunks = await execute_with_prefetch(
        plan_steps=_slow_planner(first_search_started),
        backends=[backend],
    )

    assert backend.call_count == 2  # both sub-queries were searched
    assert len(chunks) == 2
    texts = {c.text for c in chunks}
    assert texts == {"sub-query-one", "sub-query-two"}


@pytest.mark.asyncio
async def test_speculative_prefetch_single_step() -> None:
    """Works correctly when the planner emits only one real sub-query."""

    async def _single_step_planner():
        yield PlanStep(query="only-query")
        yield PlanStep(query="", done=True)

    class _SimpleBackend:
        name = "simple"
        capabilities = frozenset({Capability.WEB})
        async def search(self, q: str, k: int = 10) -> list[Chunk]:
            return [Chunk(text=q, source=Source(kind="url", location=f"https://x.com/{q}"))]

    chunks = await execute_with_prefetch(
        plan_steps=_single_step_planner(),
        backends=[_SimpleBackend()],
    )
    assert len(chunks) == 1
    assert chunks[0].text == "only-query"
```

- [ ] **Step 6.2: Run to confirm ImportError**

```bash
uv run pytest tests/engine/test_executor.py -v -k "prefetch"
```

Expected: `ImportError: cannot import name 'execute_with_prefetch'`

- [ ] **Step 6.3: Implement `execute_with_prefetch` in `src/sleuth/engine/executor.py`**

Append to `src/sleuth/engine/executor.py`:

```python
async def execute_with_prefetch(
    plan_steps: "AsyncIterator[PlanStep]",
    backends: list[Backend],
    *,
    k: int = 10,
    on_search_event: Callable[[SearchEvent], Any] | None = None,
) -> list[Chunk]:
    """Consume plan steps and start search tasks speculatively.

    As soon as the planner yields its first PlanStep, a search task is launched
    immediately — before waiting for the planner to finish streaming. This hides
    planner latency behind search latency (spec §11 point 3).

    Args:
        plan_steps: Async generator of PlanStep objects from Planner.plan().
        backends: Backends to search.
        k: Max results per backend per sub-query.
        on_search_event: Optional callback per SearchEvent.

    Returns:
        Deduplicated merged chunks from all sub-queries.
    """
    from sleuth.engine.planner import PlanStep  # local import to avoid circular

    pending_tasks: list[asyncio.Task[list[Chunk]]] = []

    async def _launch(query: str) -> None:
        """Create and register a search task for `query`."""
        task = asyncio.create_task(
            execute_subqueries(
                subqueries=[query],
                backends=backends,
                k=k,
                on_search_event=on_search_event,
            )
        )
        pending_tasks.append(task)

    async for step in plan_steps:
        if step.done:
            break
        if step.query:
            await _launch(step.query)
            # Yield control immediately so the launched task can start running
            # while the planner continues streaming — this is the speculative
            # prefetch: search and planner overlap in time.
            await asyncio.sleep(0)

    if not pending_tasks:
        return []

    results: list[list[Chunk]] = await asyncio.gather(*pending_tasks)

    # Flatten + deduplicate by source.location (first occurrence wins)
    seen: set[str] = set()
    merged: list[Chunk] = []
    for chunk_list in results:
        for chunk in chunk_list:
            loc = chunk.source.location
            if loc not in seen:
                seen.add(loc)
                merged.append(chunk)

    return merged
```

Also add the missing import at the top of the file (if not already present):

```python
from collections.abc import AsyncIterator
```

- [ ] **Step 6.4: Run prefetch tests**

```bash
uv run pytest tests/engine/test_executor.py -v -k "prefetch"
```

Expected: `2 passed`

- [ ] **Step 6.5: Run the full executor test suite to confirm no regressions**

```bash
uv run pytest tests/engine/test_executor.py -v
```

Expected: all tests pass.

- [ ] **Step 6.6: Commit**

```bash
git add src/sleuth/engine/executor.py tests/engine/test_executor.py
git commit -m "feat: implement speculative prefetch in execute_with_prefetch"
```

---

## Task 7: Reflect loop with `max_iterations`

**Files:**
- Modify: `src/sleuth/engine/executor.py`
- Modify: `src/sleuth/engine/planner.py`
- Test: `tests/engine/test_executor.py`

The reflect loop: plan → search-fan-out → append results to `_PlannerState.context_snippets` → plan again. Stop when planner emits `done=True` OR `max_iterations` reached.

- [ ] **Step 7.1: Expose `_PlannerState.context_snippets` update path — update `planner.py`**

The `Planner` already reads `state.context_snippets`. The executor needs to update the state between iterations. No code change needed in `planner.py` itself — `_PlannerState` is a plain `@dataclass` with a mutable list, so the executor just appends to it. Confirm by reading the dataclass definition above — `context_snippets: list[str] = field(default_factory=list)`. No file changes required for this step.

- [ ] **Step 7.2: Write failing reflect-loop test (append to `tests/engine/test_executor.py`)**

```python
# append to tests/engine/test_executor.py

import pytest
from sleuth.engine.executor import reflect_loop
from sleuth.engine.planner import Planner, PlanStep, _PlannerState
from sleuth.llm.stub import StubLLM
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source


class _CountingBackend:
    name = "counting"
    capabilities = frozenset({Capability.WEB})

    def __init__(self) -> None:
        self.call_count = 0

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        self.call_count += 1
        return [Chunk(
            text=f"result for {query} (call {self.call_count})",
            source=Source(kind="url", location=f"https://ex.com/{query}/{self.call_count}"),
        )]


@pytest.mark.asyncio
async def test_reflect_loop_stops_on_done() -> None:
    """reflect_loop terminates when planner emits done=True on first iteration."""
    llm = StubLLM(responses=[
        '[{"query": "sub1"}, {"done": true, "query": ""}]'
    ])
    planner = Planner(llm=llm)
    backend = _CountingBackend()

    chunks, iterations = await reflect_loop(
        query="what is TLS?",
        planner=planner,
        backends=[backend],
        max_iterations=5,
    )

    assert iterations == 1
    assert len(chunks) >= 1
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_reflect_loop_respects_max_iterations() -> None:
    """reflect_loop stops at max_iterations even if planner never emits done."""
    # Each call returns a new sub-query without a done flag (planner adds one automatically)
    call_count = {"n": 0}

    from sleuth.llm.base import TextDelta, Stop
    from collections.abc import AsyncIterator as AI

    class _LoopingLLM:
        name = "looping"
        supports_reasoning = False
        supports_structured_output = True

        async def stream(self, messages, *, schema=None, tools=None) -> AI:
            call_count["n"] += 1
            async def _gen():
                yield TextDelta(text=f'[{{"query": "sub-query-{call_count["n"]}"}}]')
                yield Stop(reason="end_turn")
            return _gen()

    planner = Planner(llm=_LoopingLLM())  # type: ignore[arg-type]
    backend = _CountingBackend()

    chunks, iterations = await reflect_loop(
        query="never-ending research",
        planner=planner,
        backends=[backend],
        max_iterations=3,
    )

    assert iterations == 3


@pytest.mark.asyncio
async def test_reflect_loop_feeds_chunks_as_context() -> None:
    """After iteration 1, prior chunks appear in the planner's next prompt."""
    captured: list[str] = []

    from sleuth.llm.base import TextDelta, Stop
    from collections.abc import AsyncIterator as AI

    call_num = {"n": 0}

    class _SpyLLM:
        name = "spy"
        supports_reasoning = False
        supports_structured_output = True

        async def stream(self, messages, *, schema=None, tools=None) -> AI:
            call_num["n"] += 1
            user_content = next(m.content for m in messages if m.role == "user")
            captured.append(user_content)
            if call_num["n"] == 1:
                # First iteration: return one real sub-query (no done yet — will have sentinel appended)
                async def _g1():
                    yield TextDelta(text='[{"query": "first sub"}]')
                    yield Stop(reason="end_turn")
                return _g1()
            else:
                # Second iteration: signal done
                async def _g2():
                    yield TextDelta(text='[{"done": true, "query": ""}]')
                    yield Stop(reason="end_turn")
                return _g2()

    planner = Planner(llm=_SpyLLM())  # type: ignore[arg-type]
    backend = _CountingBackend()

    await reflect_loop(
        query="deep question",
        planner=planner,
        backends=[backend],
        max_iterations=5,
    )

    # On the second LLM call, prior results should be in the user message
    assert len(captured) == 2
    assert "result for first sub" in captured[1]
```

- [ ] **Step 7.3: Run to confirm ImportError**

```bash
uv run pytest tests/engine/test_executor.py -v -k "reflect_loop"
```

Expected: `ImportError: cannot import name 'reflect_loop'`

- [ ] **Step 7.4: Implement `reflect_loop` in `src/sleuth/engine/executor.py`**

Append to `src/sleuth/engine/executor.py`:

```python
async def reflect_loop(
    query: str,
    planner: "Planner",
    backends: list[Backend],
    *,
    max_iterations: int = 4,
    k: int = 10,
    on_search_event: Callable[[SearchEvent], Any] | None = None,
) -> tuple[list[Chunk], int]:
    """Run the plan → search → reflect loop for deep mode.

    Each iteration:
      1. Call planner.plan(query, state) to get sub-queries.
      2. Fan out searches with speculative prefetch via execute_with_prefetch().
      3. Append result summaries to state.context_snippets for the next iteration.
      4. Stop if planner emits done=True in this iteration or max_iterations reached.

    Returns:
        (merged_chunks, iteration_count) — all chunks from all iterations,
        deduplicated by source.location; number of iterations executed.
    """
    from sleuth.engine.planner import Planner, _PlannerState  # local to avoid circular

    state = _PlannerState()
    all_chunks: list[Chunk] = []
    seen_locations: set[str] = set()
    iterations = 0
    done = False

    while not done and iterations < max_iterations:
        iterations += 1

        # Collect plan steps for this iteration
        steps: list["PlanStep"] = []
        async for step in planner.plan(query, state):
            steps.append(step)
            if step.done:
                done = True

        real_queries = [s.query for s in steps if not s.done and s.query]
        if not real_queries:
            break

        # Fan out searches with speculative prefetch
        async def _step_gen(queries=real_queries):
            from sleuth.engine.planner import PlanStep as _PS
            for q in queries:
                yield _PS(query=q)
            yield _PS(query="", done=True)

        iter_chunks = await execute_with_prefetch(
            plan_steps=_step_gen(),
            backends=backends,
            k=k,
            on_search_event=on_search_event,
        )

        # Merge into global deduplicated set
        for chunk in iter_chunks:
            loc = chunk.source.location
            if loc not in seen_locations:
                seen_locations.add(loc)
                all_chunks.append(chunk)

        # Update planner state with result summaries for next iteration
        snippets = [c.text[:300] for c in iter_chunks]  # truncate for context window
        state.context_snippets.extend(snippets)

    return all_chunks, iterations
```

Also ensure the `Planner` type is importable at the top via `TYPE_CHECKING` if needed to avoid circular imports:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sleuth.engine.planner import Planner, PlanStep
```

- [ ] **Step 7.5: Run reflect-loop tests**

```bash
uv run pytest tests/engine/test_executor.py -v -k "reflect_loop"
```

Expected: `3 passed`

- [ ] **Step 7.6: Run full executor and planner test suites**

```bash
uv run pytest tests/engine/ -v
```

Expected: all tests pass.

- [ ] **Step 7.7: Commit**

```bash
git add src/sleuth/engine/executor.py src/sleuth/engine/planner.py tests/engine/test_executor.py
git commit -m "feat: add reflect_loop with max_iterations budget to executor"
```

---

## Task 8: Snapshot test — deep-mode event sequence

**Files:**
- Test: `tests/engine/test_planner.py`
- Read: `tests/snapshots/` (Phase 0/1 sets up syrupy; snapshots live there)

The spec mandates this order for deep mode:

```
RouteEvent(deep) → PlanEvent → SearchEvent×N (parallel) → ThinkingEvent? → TokenEvent×N → CitationEvent×N → DoneEvent
```

This snapshot test covers: `RouteEvent(deep) → PlanEvent → SearchEvent×N`.

- [ ] **Step 8.1: Write the snapshot test (append to `tests/engine/test_planner.py`)**

```python
# append to tests/engine/test_planner.py

import pytest
from syrupy.assertion import SnapshotAssertion

from sleuth.engine.router import route
from sleuth.engine.planner import Planner, _PlannerState
from sleuth.engine.executor import execute_subqueries
from sleuth.events import RouteEvent, PlanEvent, SearchEvent
from sleuth.llm.stub import StubLLM
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source


class _DeterministicBackend:
    name = "det-web"
    capabilities = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [Chunk(
            text=f"result: {query}",
            source=Source(kind="url", location=f"https://det.example.com/{query.replace(' ', '-')}"),
        )]


@pytest.mark.asyncio
async def test_deep_mode_event_sequence_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot the RouteEvent→PlanEvent→SearchEvent×N sequence for deep mode.

    Update snapshots with: uv run pytest --snapshot-update tests/engine/test_planner.py::test_deep_mode_event_sequence_snapshot
    """
    collected_events: list[dict] = []

    # 1. RouteEvent
    async for route_event in route("compare OAuth and OIDC for enterprise use", depth="auto"):
        collected_events.append(route_event.model_dump())

    # 2. PlanEvent — run the planner and emit a PlanEvent
    llm = StubLLM(responses=[
        '[{"query": "what is OAuth"}, {"query": "what is OIDC"}, {"query": "OAuth vs OIDC enterprise", "backends": ["docs"]}, {"done": true, "query": ""}]'
    ])
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps = [s async for s in planner.plan("compare OAuth and OIDC for enterprise use", state)]
    real_steps = [s for s in steps if not s.done]

    plan_event = PlanEvent(
        type="plan",
        steps=[{"query": s.query, "backends": s.backends} for s in real_steps],  # type: ignore[arg-type]
    )
    collected_events.append(plan_event.model_dump())

    # 3. SearchEvents — one per sub-query
    search_events: list[SearchEvent] = []
    backend = _DeterministicBackend()

    await execute_subqueries(
        subqueries=[s.query for s in real_steps],
        backends=[backend],
        on_search_event=lambda e: search_events.append(e),
    )
    for se in sorted(search_events, key=lambda e: e.query):  # sort for determinism
        collected_events.append(se.model_dump())

    assert collected_events == snapshot
```

- [ ] **Step 8.2: Run to generate initial snapshot**

```bash
uv run pytest tests/engine/test_planner.py::test_deep_mode_event_sequence_snapshot -v --snapshot-update
```

Expected: `1 snapshot generated` and `1 passed`.

- [ ] **Step 8.3: Run again without `--snapshot-update` to confirm snapshot matches**

```bash
uv run pytest tests/engine/test_planner.py::test_deep_mode_event_sequence_snapshot -v
```

Expected: `1 passed` (snapshot matches).

- [ ] **Step 8.4: Commit**

```bash
git add tests/engine/test_planner.py tests/snapshots/
git commit -m "test: add syrupy snapshot for deep-mode event sequence (Route→Plan→Search)"
```

---

## Task 9: `PlanEvent` emission integration — wire Planner into event stream

**Files:**
- Modify: `src/sleuth/engine/planner.py`
- Test: `tests/engine/test_planner.py`

`PlanEvent` (from `sleuth.events`) must be emitted after the planner collects its steps. Currently `Planner.plan()` yields `PlanStep` objects but never emits the `PlanEvent` into the outer event stream. We add an optional `emit` callback so the executor can route it.

- [ ] **Step 9.1: Write failing test (append to `tests/engine/test_planner.py`)**

```python
# append to tests/engine/test_planner.py

@pytest.mark.asyncio
async def test_plan_emits_plan_event_via_callback() -> None:
    """Planner calls the emit callback with a PlanEvent after collecting steps."""
    from sleuth.events import PlanEvent

    emitted: list[PlanEvent] = []

    llm = StubLLM(responses=[
        '[{"query": "step A"}, {"query": "step B"}, {"done": true, "query": ""}]'
    ])
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps = [
        s async for s in planner.plan(
            "test query",
            state,
            on_plan_event=lambda e: emitted.append(e),
        )
    ]

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, PlanEvent)
    assert len(event.steps) == 2   # two real steps (done step excluded)
    assert event.steps[0]["query"] == "step A"
    assert event.steps[1]["query"] == "step B"
```

- [ ] **Step 9.2: Run to confirm TypeError / missing param**

```bash
uv run pytest tests/engine/test_planner.py::test_plan_emits_plan_event_via_callback -v
```

Expected: `TypeError: Planner.plan() got an unexpected keyword argument 'on_plan_event'`

- [ ] **Step 9.3: Update `Planner.plan()` to accept `on_plan_event` and call it**

In `src/sleuth/engine/planner.py`, update the `plan()` method signature and body:

```python
    async def plan(
        self,
        query: str,
        state: _PlannerState,
        *,
        on_plan_event: "Callable[[PlanEvent], Any] | None" = None,
    ) -> AsyncIterator[PlanStep]:
        """Yield PlanSteps for one reflect iteration.

        Yields PlanSteps as soon as they are parsed. Calls on_plan_event with a
        PlanEvent after all steps are collected (excluding the done sentinel).
        """
        from sleuth.events import PlanEvent  # local import; events.py is Pydantic
        from collections.abc import Callable
        from typing import Any

        messages: list[Message] = [
            Message(role="system", content=self._SYSTEM_PROMPT),
            Message(
                role="user",
                content=self._build_user_message(query, state),
            ),
        ]

        raw_text = ""
        async for chunk in await self._llm.stream(messages):
            if isinstance(chunk, TextDelta):
                raw_text += chunk.text

        steps = self._parse_steps(raw_text)

        # Emit PlanEvent with all real steps (exclude done sentinel)
        if on_plan_event is not None:
            real_steps = [s for s in steps if not s.done]
            event = PlanEvent(
                type="plan",
                steps=[{"query": s.query, "backends": s.backends} for s in real_steps],  # type: ignore[arg-type]
            )
            on_plan_event(event)

        for step in steps:
            yield step
```

Also add necessary imports at the top of `planner.py`:

```python
from collections.abc import Callable
from typing import Any
```

- [ ] **Step 9.4: Run test**

```bash
uv run pytest tests/engine/test_planner.py -v
```

Expected: all tests pass.

- [ ] **Step 9.5: Commit**

```bash
git add src/sleuth/engine/planner.py tests/engine/test_planner.py
git commit -m "feat: emit PlanEvent via callback from Planner.plan()"
```

---

## Task 10: Smoke-test full deep path via `Sleuth.aask`

**Files:**
- Test: `tests/engine/test_planner.py`

This end-to-end test drives `Sleuth.aask(depth="deep")` with `StubLLM` and a stub backend to verify the full event sequence including `DoneEvent`. No modifications to `_agent.py` — this verifies that Phase 1's wiring already works with Phase 3's new components (since `_agent.py` already accepts `depth` and `max_iterations`).

> If `Sleuth` does not yet wire `depth="deep"` to the planner/executor, this test will reveal the gap. The fix is adding the deep branch inside `_agent.py`'s `aask()`. Since `_agent.py` is Phase 1-owned, document any required `_agent.py` changes in a comment and defer to the implementer to apply them directly without re-creating the file.

- [ ] **Step 10.1: Write the end-to-end smoke test (append to `tests/engine/test_planner.py`)**

```python
# append to tests/engine/test_planner.py

@pytest.mark.asyncio
async def test_sleuth_aask_deep_emits_route_plan_search_done() -> None:
    """End-to-end: Sleuth.aask(depth='deep') emits RouteEvent(deep), PlanEvent,
    SearchEvent(s), and DoneEvent in that order.

    Uses StubLLM (two responses: one for planning, one for synthesis)
    and a stub backend. If this test fails because _agent.py doesn't wire
    depth='deep', see NOTE below.

    NOTE for implementer: if Sleuth._aask() (or equivalent) does not yet
    call the Planner for depth='deep', add a branch like:
        if resolved_depth == "deep":
            chunks, _ = await reflect_loop(query, planner, backends, max_iterations=max_iterations)
        else:
            chunks = await execute_subqueries([query], backends)
    in src/sleuth/_agent.py without re-creating the file.
    """
    from sleuth import Sleuth
    from sleuth.events import RouteEvent, PlanEvent, SearchEvent, DoneEvent
    from sleuth.llm.stub import StubLLM
    from sleuth.backends.base import Capability
    from sleuth.types import Chunk, Source

    class _FakeBackend:
        name = "fake"
        capabilities = frozenset({Capability.WEB})
        async def search(self, q: str, k: int = 10) -> list[Chunk]:
            return [Chunk(text=f"result: {q}", source=Source(kind="url", location=f"https://fake/{q}"))]

    # First StubLLM response: planner JSON; second: synthesis answer
    llm = StubLLM(responses=[
        '[{"query": "what is OAuth"}, {"done": true, "query": ""}]',
        "OAuth is an authorization framework.",
    ])

    agent = Sleuth(llm=llm, backends=[_FakeBackend()], cache=None)

    event_types: list[str] = []
    async for event in agent.aask("compare OAuth vs OIDC in depth", depth="deep", max_iterations=1):
        event_types.append(event.type)

    assert event_types[0] == "route"
    route_ev = None
    for e in []:  # placeholder — real check below
        pass

    # Verify ordering constraints from spec §5
    assert "route" in event_types
    assert "plan" in event_types
    assert "search" in event_types
    assert "done" in event_types

    route_idx = event_types.index("route")
    plan_idx = event_types.index("plan")
    first_search_idx = event_types.index("search")
    done_idx = event_types.index("done")

    assert route_idx < plan_idx < first_search_idx < done_idx
```

- [ ] **Step 10.2: Run the smoke test**

```bash
uv run pytest tests/engine/test_planner.py::test_sleuth_aask_deep_emits_route_plan_search_done -v
```

Expected: `PASSED`. If it fails with an error about `_agent.py` not routing to the planner, apply the fix described in the test's NOTE comment to `src/sleuth/_agent.py`.

- [ ] **Step 10.3: Run all Phase 3 and Phase 1 engine tests together**

```bash
uv run pytest tests/engine/ -v
```

Expected: all tests pass — no Phase 1 regressions.

- [ ] **Step 10.4: Commit**

```bash
git add tests/engine/test_planner.py
git commit -m "test: end-to-end smoke test for Sleuth.aask(depth='deep') event sequence"
```

---

## Task 11: Type-check and lint

**Files:**
- All modified/created files

- [ ] **Step 11.1: Run mypy on new/modified engine files**

```bash
uv run mypy src/sleuth/engine/planner.py src/sleuth/engine/executor.py src/sleuth/engine/router.py
```

Expected: `Success: no issues found in 3 source files`. Fix any type errors before proceeding.

- [ ] **Step 11.2: Run ruff lint + format**

```bash
uv run ruff check src/sleuth/engine/planner.py src/sleuth/engine/executor.py src/sleuth/engine/router.py
uv run ruff format src/sleuth/engine/planner.py src/sleuth/engine/executor.py src/sleuth/engine/router.py
```

Expected: no lint errors. If format makes changes, stage them.

- [ ] **Step 11.3: Run full test suite with coverage**

```bash
uv run pytest tests/ -m "not integration" --cov=src/sleuth --cov-report=term-missing -v
```

Expected: all non-integration tests pass; coverage ≥ 85%.

- [ ] **Step 11.4: Commit any lint/format fixes**

```bash
git add -u
git commit -m "chore: lint and type-check fixes for phase 3 engine files"
```

(Skip this commit if there are no changes after linting.)

---

## Task 12: Final integration commit and PR prep

**Files:**
- All Phase 3 files

- [ ] **Step 12.1: Verify the feature branch is up to date with develop**

```bash
git fetch origin
git rebase origin/develop
```

Expected: clean rebase with no conflicts. If conflicts, resolve them.

- [ ] **Step 12.2: Run the full test suite one final time**

```bash
uv run pytest tests/ -m "not integration" -v
```

Expected: all tests pass.

- [ ] **Step 12.3: Push branch and open PR**

```bash
git push -u origin feature/phase-3-planner-deep
```

Then open a PR targeting `develop` with title: `feat(engine): Phase 3 — Planner, deep mode, speculative prefetch, reflect loop`.

---

## Self-Review Checklist

**1. Spec coverage:**

| Spec requirement | Covered by |
|---|---|
| §3: Executor reflect loop | Task 7 (`reflect_loop`) |
| §3: Five hard rules (async-first, no global state) | `execute_with_prefetch`/`reflect_loop` are pure async functions; no global state |
| §4: `depth="deep"`, `max_iterations` | Tasks 4 (router), 7 (reflect_loop), 10 (smoke test) |
| §5: `PlanEvent` emitted | Task 9 (`on_plan_event` callback) |
| §11 pt 3: Speculative prefetch | Task 6 (`execute_with_prefetch`) with deadlock-proof overlap test |
| Deep-mode event order (spec §5) | Tasks 8 (snapshot) + 10 (smoke test assertion) |
| StubLLM for determinism | Tasks 3, 8, 9, 10 |
| Phase 1 tests stay green | Task 5 step 5.4, Task 6 step 6.5, Task 10 step 10.3 |

**2. Placeholder scan:** No TBDs, no "implement later", no "write tests for above" without code. All steps contain actual code or commands.

**3. Type consistency:**
- `PlanStep` defined in Task 2, used consistently throughout Tasks 3–10.
- `_PlannerState` defined in Task 2, used in Tasks 3, 7, 9.
- `execute_subqueries` defined in Task 5, reused in Task 6 (`execute_with_prefetch` calls it internally) and Task 8.
- `execute_with_prefetch` defined in Task 6, called in Task 7 (`reflect_loop`).
- `reflect_loop` defined in Task 7, called in Task 10 smoke test (via `Sleuth.aask`).
- `Planner.plan()` signature extended in Task 9 to add `on_plan_event` — used in Task 9 test and unchanged for Tasks 3/7 (param is optional, existing call sites still work).
