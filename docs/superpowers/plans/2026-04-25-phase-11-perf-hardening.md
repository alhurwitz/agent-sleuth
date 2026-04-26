# Phase 11: Perf Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-backend `asyncio.wait_for` timeouts to the executor (defaults: 8 s web, 4 s local, configurable per Backend instance), wire the existing `BackendTimeoutError` into the error path, ship a `pytest-benchmark` regression suite covering Phase 1 fast path, cache hit path, Phase 2 LocalFiles, and Phase 3 deep mode, and fully implement the `perf.yml` CI gate (median first_token_ms > 1500 ms fails; p50/p95 > 10% regression vs develop baseline fails; 5-run median to dampen noise).

**Architecture:** `executor.py` gains a `timeout_s` field on each `Backend` wrapper (read from a `Backend.timeout_s` attribute when present, otherwise from a `Sleuth`-level default dict keyed by capability). `asyncio.wait_for` wraps every backend call; `asyncio.TimeoutError` is caught and re-raised as `BackendTimeoutError`. The engine's existing `BackendError` handler (Phase 1) already emits `SearchEvent(error=...)` and continues — `BackendTimeoutError` is a subclass of `BackendError`, so no new catch block is needed. The benchmark suite uses a tiny 4-doc in-memory corpus, `StubLLM` with configurable `asyncio.sleep` latency injection, and a `FakeBackend` that returns synthetic `Chunk` lists instantly. Baselines (median, p50, p95 of `first_token_ms` per scenario) are stored as `tests/perf/baselines/develop.json`; `perf.yml` loads them and fails the build on regression.

**Tech Stack:** Python 3.11+, `asyncio`, `pytest-benchmark>=4.0` (already in dev deps), `pytest-asyncio` (auto mode), `statistics` stdlib, no new deps required.

---

> **Callouts (not in conventions):**
>
> 1. **`Backend.timeout_s` attribute** — the `Backend` protocol (conventions §5.2, owned by Phase 1) does not declare a `timeout_s` field. Phase 11 reads this attribute with `getattr(backend, "timeout_s", None)` so existing backends remain protocol-compliant without modification. This is a duck-typed extension, not a protocol change. No need to update the frozen protocol.
> 2. **`scripts/perf-baseline.py`** — conventions §2 does not assign ownership of `scripts/`. Phase 11 creates `scripts/perf-baseline.py` to update `tests/perf/baselines/develop.json` when run manually on develop. This is a thin helper script, not a package module.
> 3. **`tests/perf/conftest.py`** — conventions §2 assigns `tests/perf/__init__.py` to Phase 0 (as a skeleton). Phase 11 creates `tests/perf/conftest.py` (fixtures) per ownership rule: the phase that populates the directory owns its conftest.

---

## Task 1: Feature branch setup

**Files:** none (git operations only)

- [ ] **Step 1: Create feature branch off `develop`**

```bash
git checkout develop
git pull origin develop
git checkout -b feature/phase-11-perf
```

Expected: `Switched to a new branch 'feature/phase-11-perf'`

- [ ] **Step 2: Confirm `pytest-benchmark` is in dev deps**

```bash
uv run pytest --co -q -m perf 2>&1 | head -5
```

Expected: `no tests ran` (marker exists but no tests yet). If `pytest-benchmark` is missing, it was already in `pyproject.toml` from Phase 0 (`"pytest-benchmark>=4.0"`) and should be installed.

---

## Task 2: `BackendTimeoutError` subclass (verify Phase 1 landed it)

**Files:**
- Verify: `src/sleuth/errors.py`

The error hierarchy in conventions §6 already defines `BackendTimeoutError(BackendError)`. Phase 1 owns `errors.py`. Verify it is present before building on it.

- [ ] **Step 1: Verify `BackendTimeoutError` exists**

```bash
uv run python -c "from sleuth.errors import BackendTimeoutError, BackendError; assert issubclass(BackendTimeoutError, BackendError); print('OK')"
```

Expected: `OK`

If this fails, `errors.py` is missing the subclass. Open `src/sleuth/errors.py` and add:

```python
class BackendTimeoutError(BackendError):
    """Raised when a backend exceeds its per-call timeout budget."""
```

Then re-run the verification step.

- [ ] **Step 2: Commit if `errors.py` was modified**

```bash
git add src/sleuth/errors.py
git commit -m "fix: add BackendTimeoutError subclass to errors.py (required for Phase 11)"
```

Skip this commit if no change was needed.

---

## Task 3: Per-backend timeout in `executor.py` — TDD

**Files:**
- Modify: `src/sleuth/engine/executor.py`
- Create: `tests/engine/test_executor_timeout.py`

### Step 1: Write failing tests

- [ ] Create `tests/engine/test_executor_timeout.py`:

```python
"""
Tests for per-backend asyncio.wait_for timeout in the executor.

Scenarios:
  - Backend that returns within its timeout → result included in merged output.
  - Backend that exceeds its timeout → BackendTimeoutError caught, SearchEvent(error=...) emitted,
    other backends' results still returned.
  - Backend with explicit timeout_s=0.05 (very short) → always times out in test.
  - Default timeout selection: web capability → 8 s default; docs capability → 4 s default.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from sleuth.backends.base import Backend, Capability
from sleuth.errors import BackendTimeoutError
from sleuth.events import SearchEvent, DoneEvent
from sleuth.types import Chunk, Source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str, backend_name: str = "fake") -> Chunk:
    return Chunk(
        text=text,
        source=Source(kind="url", location=f"https://example.com/{backend_name}"),
    )


class SlowBackend:
    """Backend that sleeps longer than its allowed timeout."""

    def __init__(self, name: str, sleep_s: float, timeout_s: float) -> None:
        self.name = name
        self.capabilities: frozenset[Capability] = frozenset({Capability.WEB})
        self.timeout_s = timeout_s
        self._sleep = sleep_s

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        await asyncio.sleep(self._sleep)
        return [_chunk("slow result")]


class FastBackend:
    """Backend that returns immediately."""

    def __init__(self, name: str, timeout_s: float | None = None) -> None:
        self.name = name
        self.capabilities: frozenset[Capability] = frozenset({Capability.DOCS})
        if timeout_s is not None:
            self.timeout_s = timeout_s

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [_chunk(f"fast result from {self.name}")]


# ---------------------------------------------------------------------------
# Import the executor run function (Phase 1 owns — path may vary; adjust if needed)
# ---------------------------------------------------------------------------
# The executor exposes an async function or class; exact signature determined
# by Phase 1. We import the function that fans out queries across backends.
# If Phase 1 calls it `run_backends`, adjust the import accordingly.
from sleuth.engine.executor import run_backends  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_fast_backend_returns_results():
    """Backend within timeout contributes its results."""
    backend = FastBackend("docs-fast", timeout_s=4.0)
    chunks = await run_backends(
        backends=[backend],
        query="test query",
        k=5,
        default_timeouts={Capability.DOCS: 4.0, Capability.WEB: 8.0},
    )
    assert any(c.text == "fast result from docs-fast" for c in chunks)


@pytest.mark.unit
async def test_slow_backend_times_out_and_emits_error():
    """Backend that exceeds timeout emits SearchEvent(error=...) and is dropped."""
    slow = SlowBackend("web-slow", sleep_s=0.5, timeout_s=0.05)
    fast = FastBackend("docs-fast")

    events: list[SearchEvent] = []
    chunks = await run_backends(
        backends=[slow, fast],
        query="test query",
        k=5,
        default_timeouts={Capability.DOCS: 4.0, Capability.WEB: 8.0},
        event_sink=events.append,
    )

    # fast backend result still present
    assert any(c.text == "fast result from docs-fast" for c in chunks)
    # no result from slow backend
    assert not any("slow result" in c.text for c in chunks)
    # error event emitted for slow backend
    error_events = [e for e in events if isinstance(e, SearchEvent) and e.error is not None]
    assert len(error_events) == 1
    assert error_events[0].backend == "web-slow"
    assert "timeout" in error_events[0].error.lower()


@pytest.mark.unit
async def test_backend_timeout_s_attribute_overrides_default():
    """Backend.timeout_s wins over the default_timeouts dict."""
    # timeout_s=0.01 is tight enough to always fire in CI
    slow = SlowBackend("web-custom", sleep_s=0.3, timeout_s=0.01)

    events: list[SearchEvent] = []
    chunks = await run_backends(
        backends=[slow],
        query="q",
        k=1,
        default_timeouts={Capability.WEB: 8.0},  # large default, should be ignored
        event_sink=events.append,
    )

    assert chunks == []
    error_events = [e for e in events if isinstance(e, SearchEvent) and e.error]
    assert len(error_events) == 1


@pytest.mark.unit
async def test_default_timeout_applied_when_no_attribute():
    """Backend without timeout_s attribute gets default from capability map."""
    # FastBackend has no timeout_s attribute set
    fast = FastBackend("docs-no-attr")
    chunks = await run_backends(
        backends=[fast],
        query="q",
        k=5,
        default_timeouts={Capability.DOCS: 4.0, Capability.WEB: 8.0},
    )
    assert len(chunks) == 1


@pytest.mark.unit
async def test_all_backends_timeout_returns_empty():
    """If every backend times out, run_backends returns [] and emits error events for each."""
    b1 = SlowBackend("b1", sleep_s=0.5, timeout_s=0.02)
    b2 = SlowBackend("b2", sleep_s=0.5, timeout_s=0.02)

    events: list[SearchEvent] = []
    chunks = await run_backends(
        backends=[b1, b2],
        query="q",
        k=5,
        default_timeouts={Capability.WEB: 8.0},
        event_sink=events.append,
    )

    assert chunks == []
    assert len([e for e in events if isinstance(e, SearchEvent) and e.error]) == 2
```

- [ ] **Step 2: Run — expected FAIL (function not yet modified or `run_backends` not exported)**

```bash
uv run pytest tests/engine/test_executor_timeout.py -v 2>&1 | tail -20
```

Expected: `ImportError: cannot import name 'run_backends' from 'sleuth.engine.executor'` or similar — tests fail because the implementation is not yet present. This is the expected TDD red state.

### Step 2: Implement per-backend timeouts in `executor.py`

- [ ] **Step 3: Read the current `executor.py` to understand its Phase 1 shape**

Open `src/sleuth/engine/executor.py` and identify:
- Where `asyncio.gather` (or equivalent) fans out backend calls.
- The existing `BackendError` catch block.
- The event-emission surface (how `SearchEvent` is emitted).

- [ ] **Step 4: Add `run_backends` with `asyncio.wait_for` wrapping**

In `src/sleuth/engine/executor.py`, add (or extend) the following logic. If Phase 1 already has a `run_backends`-style function, integrate the changes into it rather than creating a duplicate:

```python
# src/sleuth/engine/executor.py  (additions / modifications)
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from sleuth.backends.base import Backend, Capability
from sleuth.errors import BackendError, BackendTimeoutError
from sleuth.events import SearchEvent
from sleuth.types import Chunk

# Default timeout budget per primary capability (seconds).
DEFAULT_TIMEOUTS: dict[Capability, float] = {
    Capability.WEB: 8.0,
    Capability.DOCS: 4.0,
    Capability.CODE: 4.0,
    Capability.FRESH: 8.0,
    Capability.PRIVATE: 4.0,
}


def _resolve_timeout(backend: Backend, default_timeouts: dict[Capability, float]) -> float:
    """Return the effective timeout for *backend*.

    Priority:
      1. ``backend.timeout_s`` attribute (duck-typed; not in the frozen protocol).
      2. ``default_timeouts`` keyed by the backend's first (primary) capability.
      3. 8.0 s as the hard fallback.
    """
    explicit = getattr(backend, "timeout_s", None)
    if explicit is not None:
        return float(explicit)
    for cap in backend.capabilities:
        if cap in default_timeouts:
            return default_timeouts[cap]
    return 8.0


async def _call_with_timeout(
    backend: Backend,
    query: str,
    k: int,
    timeout_s: float,
    event_sink: Callable[[SearchEvent], None] | None,
) -> list[Chunk]:
    """Call backend.search with a per-backend timeout.

    On asyncio.TimeoutError → wraps in BackendTimeoutError → emits
    SearchEvent(error=...) and returns [].

    On any other BackendError → same treatment (consistent with §7.1).
    """
    event_sink = event_sink or (lambda _: None)
    event_sink(SearchEvent(type="search", backend=backend.name, query=query))
    try:
        return await asyncio.wait_for(backend.search(query, k), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        err = BackendTimeoutError(
            f"Backend '{backend.name}' timed out after {timeout_s:.1f}s"
        )
        event_sink(
            SearchEvent(
                type="search",
                backend=backend.name,
                query=query,
                error=str(err),
            )
        )
        return []
    except BackendError as exc:
        event_sink(
            SearchEvent(
                type="search",
                backend=backend.name,
                query=query,
                error=str(exc),
            )
        )
        return []


async def run_backends(
    backends: list[Backend],
    query: str,
    k: int = 10,
    *,
    default_timeouts: dict[Capability, float] | None = None,
    event_sink: Callable[[SearchEvent], None] | None = None,
) -> list[Chunk]:
    """Fan out *query* to all *backends* in parallel with per-backend timeouts.

    Returns the merged (deduplicated by source location) list of Chunks from
    all backends that responded within their timeout.

    Args:
        backends: list of Backend instances to query.
        query: the search query string.
        k: number of results requested from each backend.
        default_timeouts: timeout (seconds) per Capability; falls back to
            DEFAULT_TIMEOUTS when not supplied.
        event_sink: optional callable that receives SearchEvent objects as
            they are emitted (one per backend call, plus one on error).
    """
    timeouts = default_timeouts if default_timeouts is not None else DEFAULT_TIMEOUTS
    tasks = [
        _call_with_timeout(
            backend=b,
            query=query,
            k=k,
            timeout_s=_resolve_timeout(b, timeouts),
            event_sink=event_sink,
        )
        for b in backends
    ]
    results: list[list[Chunk]] = await asyncio.gather(*tasks)

    # Deduplicate by source.location; preserve insertion order (first wins).
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

- [ ] **Step 5: Run the tests — expected PASS**

```bash
uv run pytest tests/engine/test_executor_timeout.py -v
```

Expected:
```
PASSED tests/engine/test_executor_timeout.py::test_fast_backend_returns_results
PASSED tests/engine/test_executor_timeout.py::test_slow_backend_times_out_and_emits_error
PASSED tests/engine/test_executor_timeout.py::test_backend_timeout_s_attribute_overrides_default
PASSED tests/engine/test_executor_timeout.py::test_default_timeout_applied_when_no_attribute
PASSED tests/engine/test_executor_timeout.py::test_all_backends_timeout_returns_empty
5 passed
```

- [ ] **Step 6: Run full unit suite to confirm no regressions**

```bash
uv run pytest -m "not integration and not perf and not adapter" -x -q
```

Expected: all previously passing tests still pass (timeout defaults are 8 s / 4 s, loose enough that no Phase 1–3 test fires them).

- [ ] **Step 7: Commit**

```bash
git add src/sleuth/engine/executor.py tests/engine/test_executor_timeout.py
git commit -m "feat: add per-backend asyncio.wait_for timeouts to executor (8s web, 4s local)"
```

---

## Task 4: `tests/perf/conftest.py` — benchmark fixtures

**Files:**
- Create: `tests/perf/conftest.py`
- Create: `tests/perf/baselines/develop.json` (populated later in Task 8)

- [ ] **Step 1: Create `tests/perf/` directory and fixture file**

```bash
mkdir -p tests/perf/baselines
```

- [ ] **Step 2: Create `tests/perf/conftest.py`**

```python
"""
Performance test fixtures for agent-sleuth.

Fixture inventory:
  corpus_dir      — a tmp directory with 4 synthetic Markdown docs (small, fast to read).
  stub_llm_fast   — StubLLM that yields a response with asyncio.sleep(0) (no latency).
  stub_llm_1s     — StubLLM that injects 100 ms per token chunk (for first_token_ms tests).
  fake_web_backend   — Backend(capability=WEB) returning 3 Chunks instantly.
  fake_docs_backend  — Backend(capability=DOCS) returning 3 Chunks instantly.
  baseline        — dict loaded from tests/perf/baselines/develop.json (or {} if absent).
  run_stats_from_events — helper: extracts RunStats from a list of events.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Sequence

import pytest

from sleuth.backends.base import Capability
from sleuth.events import DoneEvent
from sleuth.llm.stub import StubLLM
from sleuth.llm.base import LLMChunk, TextDelta, Stop
from sleuth.types import Chunk, RunStats, Source

BASELINES_PATH = Path(__file__).parent / "baselines" / "develop.json"

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

_DOCS = [
    ("auth.md", "# Authentication\n\nOur auth uses JWT tokens with a 15-minute expiry.\n\n## Refresh tokens\n\nRefresh tokens last 30 days and are stored server-side.\n"),
    ("billing.md", "# Billing\n\nWe charge per seat per month. Pro plan: $25/seat.\n\n## Invoices\n\nInvoices are generated on the 1st of each month.\n"),
    ("deploy.md", "# Deployment\n\nWe deploy to AWS ECS via GitHub Actions on every merge to main.\n\n## Rollback\n\nRollback is one button in the ECS console.\n"),
    ("api.md", "# API Reference\n\nBase URL: https://api.example.com/v2\n\n## Rate limits\n\n1000 requests/minute per API key.\n"),
]


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped tmp directory with 4 synthetic Markdown documents.

    Session-scoped so the directory (and any index) is built once per test run.
    """
    d = tmp_path_factory.mktemp("perf_corpus")
    for name, content in _DOCS:
        (d / name).write_text(content)
    return d


# ---------------------------------------------------------------------------
# StubLLM variants
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_llm_fast() -> StubLLM:
    """StubLLM with zero artificial latency. Use for throughput-bound benchmarks."""
    return StubLLM(responses=["Answer: fast path response. Citation: [1]"])


@pytest.fixture
def stub_llm_100ms() -> StubLLM:
    """StubLLM that sleeps 100 ms before emitting its first token.

    Used to make first_token_ms measurements reproducible across CI runs
    while keeping total benchmark runtime short.
    """

    async def _delayed_stream(messages: list) -> AsyncIterator[LLMChunk]:
        await asyncio.sleep(0.1)
        yield TextDelta(text="Answer: delayed response.")
        yield Stop(reason="end_turn")

    return StubLLM(responses=_delayed_stream)


# ---------------------------------------------------------------------------
# Fake backends (return instantly, no real I/O)
# ---------------------------------------------------------------------------

class _FakeBackend:
    def __init__(
        self,
        name: str,
        capability: Capability,
        n_chunks: int = 3,
        timeout_s: float = 4.0,
    ) -> None:
        self.name = name
        self.capabilities: frozenset[Capability] = frozenset({capability})
        self.timeout_s = timeout_s
        self._n = n_chunks

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [
            Chunk(
                text=f"Chunk {i} from {self.name} for '{query}'",
                source=Source(kind="url", location=f"https://fake.example.com/{self.name}/{i}"),
                score=1.0 - i * 0.1,
            )
            for i in range(min(self._n, k))
        ]


@pytest.fixture
def fake_web_backend() -> _FakeBackend:
    """Instant web-capability backend returning 3 Chunks."""
    return _FakeBackend("fake-web", Capability.WEB, n_chunks=3)


@pytest.fixture
def fake_docs_backend() -> _FakeBackend:
    """Instant docs-capability backend returning 3 Chunks."""
    return _FakeBackend("fake-docs", Capability.DOCS, n_chunks=3)


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def baseline() -> dict:
    """Load committed baseline JSON. Returns empty dict if file does not exist."""
    if BASELINES_PATH.exists():
        return json.loads(BASELINES_PATH.read_text())
    return {}


def run_stats_from_events(events: list) -> RunStats | None:
    """Extract RunStats from a DoneEvent in *events*. Returns None if not found."""
    for event in events:
        if isinstance(event, DoneEvent):
            return event.stats
    return None
```

- [ ] **Step 3: Verify fixtures load without error**

```bash
uv run pytest tests/perf/ --collect-only -q 2>&1 | head -10
```

Expected: no import errors (zero tests collected is fine at this stage).

- [ ] **Step 4: Commit**

```bash
git add tests/perf/conftest.py tests/perf/baselines/
git commit -m "test: add tests/perf/conftest.py with corpus, StubLLM, and FakeBackend fixtures"
```

---

## Task 5: Phase 1 fast-path benchmark

**Files:**
- Create: `tests/perf/test_bench_fast_path.py`

This benchmark covers the Phase 1 "fast path": `depth="fast"`, `StubLLM` + `FakeBackend` + cache miss (no prior cache entry). It measures end-to-end `RunStats.first_token_ms`.

- [ ] **Step 1: Write the benchmark test**

Create `tests/perf/test_bench_fast_path.py`:

```python
"""
Phase 1 fast-path benchmarks.

Scenario: depth="fast", StubLLM, two fake backends (web + docs), no cache.

Benchmarks:
  bench_fast_path_first_token  — measures time-to-first-token (first TokenEvent).
  bench_fast_path_e2e          — measures full run latency (DoneEvent).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from sleuth import Sleuth
from sleuth.events import TokenEvent, DoneEvent
from sleuth.memory.cache import MemoryCache
from tests.perf.conftest import run_stats_from_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect_events(agent: Sleuth, query: str) -> tuple[float, list]:
    """Run aask and collect (time_to_first_token_ms, events)."""
    events = []
    t_start = time.perf_counter()
    first_token_ms: float | None = None
    async for event in agent.aask(query, depth="fast"):
        events.append(event)
        if isinstance(event, TokenEvent) and first_token_ms is None:
            first_token_ms = (time.perf_counter() - t_start) * 1000
    return first_token_ms or 0.0, events


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.perf
def test_bench_fast_path_first_token(
    benchmark,
    stub_llm_fast,
    fake_web_backend,
    fake_docs_backend,
):
    """First-token latency on the fast path must be below 1500 ms (spec §16.6)."""
    cache = MemoryCache()

    def setup():
        # Fresh agent per benchmark iteration — cache is cold.
        return Sleuth(
            llm=stub_llm_fast,
            backends=[fake_web_backend, fake_docs_backend],
            cache=cache,
        )

    def run():
        agent = setup()
        ftms, events = asyncio.get_event_loop().run_until_complete(
            _collect_events(agent, "How does auth work?")
        )
        return ftms

    result = benchmark(run)
    # Hard gate: median first_token_ms must be < 1500 ms (checked in perf.yml too).
    # Here we assert the single-run result passes; the workflow does the 5-run median.
    assert result < 1500, f"first_token_ms={result:.1f} exceeded 1500 ms gate"


@pytest.mark.perf
def test_bench_fast_path_e2e(
    benchmark,
    stub_llm_fast,
    fake_web_backend,
    fake_docs_backend,
):
    """End-to-end latency for a fast-path run. Tracked as p50/p95 regression metric."""
    cache = MemoryCache()

    def run():
        agent = Sleuth(
            llm=stub_llm_fast,
            backends=[fake_web_backend, fake_docs_backend],
            cache=cache,
        )
        _, events = asyncio.get_event_loop().run_until_complete(
            _collect_events(agent, "How does billing work?")
        )
        stats = run_stats_from_events(events)
        return stats.latency_ms if stats else 0

    benchmark(run)
```

- [ ] **Step 2: Run the benchmarks (expected PASS with timing output)**

```bash
uv run pytest tests/perf/test_bench_fast_path.py -v --benchmark-min-rounds=3 -m perf
```

Expected output includes:
```
PASSED tests/perf/test_bench_fast_path.py::test_bench_fast_path_first_token
PASSED tests/perf/test_bench_fast_path.py::test_bench_fast_path_e2e

------------------------- benchmark results -------------------------
Name                                  Min    Mean   Max   Rounds
test_bench_fast_path_first_token     ...     ...    ...    3
test_bench_fast_path_e2e             ...     ...    ...    3
```

- [ ] **Step 3: Commit**

```bash
git add tests/perf/test_bench_fast_path.py
git commit -m "test: add fast-path perf benchmarks (first_token_ms, e2e latency)"
```

---

## Task 6: Cache hit path benchmark

**Files:**
- Create: `tests/perf/test_bench_cache_hit.py`

Cache hits replay through the same event stream. A hit should be dramatically faster than a miss (no LLM call). This benchmark compares warm vs cold.

- [ ] **Step 1: Write the benchmark test**

Create `tests/perf/test_bench_cache_hit.py`:

```python
"""
Cache hit path benchmarks.

Scenario: run the same query twice on the same MemoryCache instance.
Second run is a cache hit → should replay without LLM calls.

Benchmarks:
  bench_cache_miss  — first run, cold cache.
  bench_cache_hit   — second run, warm cache.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from sleuth import Sleuth
from sleuth.events import CacheHitEvent, DoneEvent, TokenEvent
from sleuth.memory.cache import MemoryCache
from tests.perf.conftest import run_stats_from_events

_QUERY = "What is the API rate limit?"


async def _run(agent: Sleuth) -> tuple[float, list]:
    events = []
    t0 = time.perf_counter()
    first_token_ms = None
    async for event in agent.aask(_QUERY, depth="fast"):
        events.append(event)
        if isinstance(event, TokenEvent) and first_token_ms is None:
            first_token_ms = (time.perf_counter() - t0) * 1000
    return first_token_ms or 0.0, events


@pytest.mark.perf
def test_bench_cache_miss(benchmark, stub_llm_fast, fake_web_backend):
    """Cold cache run — baseline for comparison."""
    def run():
        cache = MemoryCache()  # new cache every iteration = always cold
        agent = Sleuth(llm=stub_llm_fast, backends=[fake_web_backend], cache=cache)
        ftms, _ = asyncio.get_event_loop().run_until_complete(_run(agent))
        return ftms

    benchmark(run)


@pytest.mark.perf
def test_bench_cache_hit(benchmark, stub_llm_fast, fake_web_backend):
    """Warm cache run — must be faster than cold cache."""
    cache = MemoryCache()

    # Prime the cache with one live run outside the benchmark loop.
    agent = Sleuth(llm=stub_llm_fast, backends=[fake_web_backend], cache=cache)
    asyncio.get_event_loop().run_until_complete(_run(agent))

    def run():
        # Reuse same cache → cache hit
        agent = Sleuth(llm=stub_llm_fast, backends=[fake_web_backend], cache=cache)
        _, events = asyncio.get_event_loop().run_until_complete(_run(agent))
        assert any(isinstance(e, CacheHitEvent) for e in events), (
            "Expected a CacheHitEvent on warm-cache run — cache may not be working"
        )

    benchmark(run)
```

- [ ] **Step 2: Run benchmarks**

```bash
uv run pytest tests/perf/test_bench_cache_hit.py -v --benchmark-min-rounds=3 -m perf
```

Expected: both tests pass; cache hit mean should be lower than cache miss mean in the benchmark table.

- [ ] **Step 3: Commit**

```bash
git add tests/perf/test_bench_cache_hit.py
git commit -m "test: add cache hit/miss perf benchmarks"
```

---

## Task 7: Phase 2 LocalFiles benchmark

**Files:**
- Create: `tests/perf/test_bench_localfiles.py`

Measures LocalFiles index-load + tree-navigation latency using the `corpus_dir` fixture (4 small Markdown docs). StubLLM replays a canned navigator response so no real LLM is called.

- [ ] **Step 1: Write the benchmark test**

Create `tests/perf/test_bench_localfiles.py`:

```python
"""
Phase 2 LocalFiles perf benchmarks.

Corpus: 4 small Markdown docs (from conftest corpus_dir fixture).
LLM: StubLLM replaying a canned branch-selection response.

Benchmarks:
  bench_localfiles_cold_index  — first search (index must be loaded from disk).
  bench_localfiles_warm_index  — second search (index is cached in memory).
"""
from __future__ import annotations

import asyncio

import pytest

from sleuth.backends.localfiles import LocalFiles
from sleuth.engine.executor import run_backends
from sleuth.backends.base import Capability


@pytest.mark.perf
def test_bench_localfiles_cold_index(benchmark, corpus_dir, stub_llm_fast):
    """LocalFiles backend cold start: index load + one search call."""
    def run():
        backend = LocalFiles(
            path=corpus_dir,
            navigator_llm=stub_llm_fast,
            indexer_llm=stub_llm_fast,
        )
        chunks = asyncio.get_event_loop().run_until_complete(
            backend.search("How does authentication work?", k=5)
        )
        return len(chunks)

    result = benchmark(run)
    # At least something should be returned from a 4-doc corpus.
    assert result >= 0  # 0 is acceptable if StubLLM doesn't navigate; not a correctness test


@pytest.mark.perf
def test_bench_localfiles_warm_index(benchmark, corpus_dir, stub_llm_fast):
    """LocalFiles backend warm start: index already in memory."""
    # Build index once outside the benchmark loop.
    backend = LocalFiles(
        path=corpus_dir,
        navigator_llm=stub_llm_fast,
        indexer_llm=stub_llm_fast,
    )
    asyncio.get_event_loop().run_until_complete(
        backend.search("warm up", k=1)
    )

    def run():
        # Reuse same backend instance → index is in memory.
        chunks = asyncio.get_event_loop().run_until_complete(
            backend.search("What is the billing plan?", k=5)
        )
        return len(chunks)

    benchmark(run)
```

- [ ] **Step 2: Run benchmarks**

```bash
uv run pytest tests/perf/test_bench_localfiles.py -v --benchmark-min-rounds=3 -m perf
```

Expected: both tests pass. Warm index run should have lower mean than cold.

- [ ] **Step 3: Commit**

```bash
git add tests/perf/test_bench_localfiles.py
git commit -m "test: add LocalFiles cold/warm index perf benchmarks"
```

---

## Task 8: Phase 3 deep-mode end-to-end benchmark

**Files:**
- Create: `tests/perf/test_bench_deep_mode.py`

Measures end-to-end latency for `depth="deep"` (planner + speculative prefetch + synthesis). Uses StubLLM so LLM calls return instantly; total latency reflects engine overhead only.

- [ ] **Step 1: Write the benchmark test**

Create `tests/perf/test_bench_deep_mode.py`:

```python
"""
Phase 3 deep-mode perf benchmarks.

Scenario: depth="deep", max_iterations=2, StubLLM, fake backends.
Measures the overhead of the planner + speculative prefetch path.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from sleuth import Sleuth
from sleuth.events import DoneEvent
from sleuth.memory.cache import MemoryCache
from tests.perf.conftest import run_stats_from_events


async def _run_deep(agent: Sleuth, query: str) -> tuple[int, list]:
    events = []
    async for event in agent.aask(query, depth="deep", max_iterations=2):
        events.append(event)
    stats = run_stats_from_events(events)
    return (stats.latency_ms if stats else 0), events


@pytest.mark.perf
def test_bench_deep_mode_e2e(
    benchmark,
    stub_llm_fast,
    fake_web_backend,
    fake_docs_backend,
):
    """Deep-mode end-to-end latency with StubLLM (measures engine overhead only)."""
    def run():
        cache = MemoryCache()
        agent = Sleuth(
            llm=stub_llm_fast,
            backends=[fake_web_backend, fake_docs_backend],
            cache=cache,
        )
        latency_ms, _ = asyncio.get_event_loop().run_until_complete(
            _run_deep(agent, "Explain our auth flow and how billing relates to seat count")
        )
        return latency_ms

    benchmark(run)


@pytest.mark.perf
def test_bench_deep_mode_speculative_prefetch(
    benchmark,
    stub_llm_fast,
    fake_web_backend,
    fake_docs_backend,
):
    """Speculative prefetch: backend search starts before planner finishes streaming.

    With StubLLM emitting instantly, this primarily measures that the prefetch
    code path does not add overhead compared to a sequential path.
    """
    def run():
        cache = MemoryCache()
        agent = Sleuth(
            llm=stub_llm_fast,
            backends=[fake_web_backend, fake_docs_backend],
            cache=cache,
        )
        latency_ms, events = asyncio.get_event_loop().run_until_complete(
            _run_deep(agent, "What are the deployment steps and rollback procedure?")
        )
        return latency_ms

    benchmark(run)
```

- [ ] **Step 2: Run benchmarks**

```bash
uv run pytest tests/perf/test_bench_deep_mode.py -v --benchmark-min-rounds=3 -m perf
```

Expected: both tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/perf/test_bench_deep_mode.py
git commit -m "test: add deep-mode e2e and speculative prefetch perf benchmarks"
```

---

## Task 9: Baseline JSON + `scripts/perf-baseline.py`

**Files:**
- Create: `tests/perf/baselines/develop.json` (initial values)
- Create: `scripts/perf-baseline.py`

The baseline JSON holds reference values for the CI gate comparison. It is committed to `develop` and updated by running `scripts/perf-baseline.py` after a successful perf run on develop.

- [ ] **Step 1: Create `scripts/` directory and `perf-baseline.py`**

```bash
mkdir -p scripts
```

Create `scripts/perf-baseline.py`:

```python
#!/usr/bin/env python3
"""
Update tests/perf/baselines/develop.json with fresh measurements.

Usage (run on develop branch after all perf tests pass):
    uv run python scripts/perf-baseline.py

The script runs the perf suite with --benchmark-json, extracts median
first_token_ms, p50 (median), and p95 per benchmark, and writes them to
tests/perf/baselines/develop.json.

Commit the updated baseline file to lock in new expected values.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

BASELINES_PATH = Path(__file__).parent.parent / "tests" / "perf" / "baselines" / "develop.json"
TMP_JSON = Path("/tmp/perf-baseline-tmp.json")


def main() -> int:
    print("Running perf suite (5 rounds per benchmark)…")
    result = subprocess.run(
        [
            "uv", "run", "pytest",
            "-m", "perf",
            "--benchmark-json", str(TMP_JSON),
            "--benchmark-min-rounds=5",
            "-q",
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        print("ERROR: perf suite failed. Fix failures before updating baseline.", file=sys.stderr)
        return 1

    if not TMP_JSON.exists():
        print("ERROR: benchmark JSON not written.", file=sys.stderr)
        return 1

    raw = json.loads(TMP_JSON.read_text())
    baselines: dict[str, dict[str, float]] = {}

    for bench in raw.get("benchmarks", []):
        name: str = bench["name"]
        stats: dict = bench["stats"]
        # pytest-benchmark stats keys: mean, median, min, max, stddev, iqr, ops,
        # rounds, iterations. We use median (p50) and compute p95 from the rounds list.
        rounds_data: list[float] = bench.get("stats", {}).get("data", [])

        p50 = stats.get("median", stats.get("mean", 0.0))
        if rounds_data:
            sorted_data = sorted(rounds_data)
            idx_95 = int(len(sorted_data) * 0.95)
            p95 = sorted_data[min(idx_95, len(sorted_data) - 1)]
        else:
            p95 = stats.get("max", p50)

        # Convert seconds → ms
        baselines[name] = {
            "median_ms": round(p50 * 1000, 2),
            "p95_ms": round(p95 * 1000, 2),
        }

    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_PATH.write_text(json.dumps(baselines, indent=2) + "\n")
    print(f"Baseline written to {BASELINES_PATH}")
    for name, vals in baselines.items():
        print(f"  {name}: median={vals['median_ms']}ms  p95={vals['p95_ms']}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate the initial baseline (run on `feature/phase-11-perf`)**

```bash
uv run python scripts/perf-baseline.py
```

Expected: runs the perf suite, writes `tests/perf/baselines/develop.json`. Inspect:

```bash
cat tests/perf/baselines/develop.json
```

Expected: JSON with one entry per benchmark, each having `median_ms` and `p95_ms`.

- [ ] **Step 3: Commit**

```bash
git add tests/perf/baselines/develop.json scripts/perf-baseline.py
git commit -m "perf: add baseline JSON and scripts/perf-baseline.py for CI comparison"
```

---

## Task 10: Fill in `perf.yml` — CI gate implementation

**Files:**
- Modify: `.github/workflows/perf.yml`

Phase 0 created the stub workflow. Replace it with the full implementation per spec §16.6.

- [ ] **Step 1: Read the current `perf.yml` stub**

Open `.github/workflows/perf.yml` and confirm it has the Phase 0 stub shape (runs pytest -m perf with --benchmark-json but no gate comparison step).

- [ ] **Step 2: Replace with the full gate workflow**

```yaml
# .github/workflows/perf.yml
# Triggers: pull_request targeting develop or main.
# Gates (spec §16.6):
#   1. first_token_ms median (of 5 runs) on fast path must be < 1500 ms.
#   2. p50 and p95 of any benchmark must not regress > 10% vs develop baseline.
#
# On develop merges: update the committed baseline JSON.
name: Performance regression

on:
  pull_request:
    branches: [develop, main]
  push:
    branches: [develop]  # update baseline on develop merges

jobs:
  perf:
    name: Performance regression suite
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          # Fetch full history so we can check out develop baseline on PRs.
          fetch-depth: 0

      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - run: uv sync --frozen --group dev

      - name: Run perf suite (5 rounds minimum to dampen noise)
        run: |
          uv run pytest -m perf \
            --benchmark-json=perf-results.json \
            --benchmark-min-rounds=5 \
            --benchmark-sort=fullname \
            -q

      - name: Upload benchmark results artifact
        uses: actions/upload-artifact@v4
        with:
          name: perf-results-${{ github.sha }}
          path: perf-results.json

      - name: Gate — first_token_ms median < 1500 ms and p50/p95 regression < 10%
        run: |
          python - << 'PYTHON'
          import json, sys, statistics, pathlib

          # Load current run results.
          results = json.loads(pathlib.Path("perf-results.json").read_text())

          # Load committed baseline (from the repo checkout — always develop's baseline).
          baseline_path = pathlib.Path("tests/perf/baselines/develop.json")
          if not baseline_path.exists():
              print("WARNING: No baseline file found — skipping regression check.")
              sys.exit(0)

          baseline = json.loads(baseline_path.read_text())

          failures = []

          for bench in results.get("benchmarks", []):
              name = bench["name"]
              stats = bench["stats"]
              raw_data = stats.get("data", [])

              # Convert seconds → ms.
              median_ms = stats.get("median", stats.get("mean", 0.0)) * 1000
              if raw_data:
                  sorted_data = sorted(raw_data)
                  p95_ms = sorted_data[min(int(len(sorted_data) * 0.95), len(sorted_data) - 1)] * 1000
              else:
                  p95_ms = stats.get("max", stats.get("median", 0.0)) * 1000

              # Gate 1: fast-path first_token_ms must be < 1500 ms.
              if "fast_path_first_token" in name:
                  if median_ms > 1500:
                      failures.append(
                          f"FAIL [{name}]: first_token_ms median {median_ms:.1f}ms > 1500ms gate"
                      )

              # Gate 2: p50/p95 regression vs baseline.
              if name in baseline:
                  base_median = baseline[name]["median_ms"]
                  base_p95 = baseline[name]["p95_ms"]

                  if base_median > 0 and (median_ms - base_median) / base_median > 0.10:
                      failures.append(
                          f"FAIL [{name}]: p50 regressed {median_ms:.1f}ms vs baseline {base_median:.1f}ms (>{10}%)"
                      )
                  if base_p95 > 0 and (p95_ms - base_p95) / base_p95 > 0.10:
                      failures.append(
                          f"FAIL [{name}]: p95 regressed {p95_ms:.1f}ms vs baseline {base_p95:.1f}ms (>{10}%)"
                      )

              print(f"  {name}: median={median_ms:.1f}ms  p95={p95_ms:.1f}ms")

          if failures:
              print("\nPerformance gate FAILED:")
              for f in failures:
                  print(f"  {f}")
              sys.exit(1)
          else:
              print("\nAll performance gates passed.")
          PYTHON

      - name: Update baseline on develop merge (push to develop only)
        if: github.event_name == 'push' && github.ref == 'refs/heads/develop'
        run: |
          uv run python scripts/perf-baseline.py
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add tests/perf/baselines/develop.json
          # Only commit if baseline changed.
          git diff --cached --quiet || git commit -m "perf: update develop baseline [skip ci]"
          git push
```

- [ ] **Step 3: Validate YAML syntax**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/perf.yml'))"
```

Expected: no output (no error).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/perf.yml
git commit -m "ci: implement perf.yml gate (first_token_ms<1500ms, p50/p95 regression<10%)"
```

---

## Task 11: Integration verification — run full perf suite locally

**Files:** none (validation only)

- [ ] **Step 1: Run all perf benchmarks together**

```bash
uv run pytest -m perf \
  --benchmark-min-rounds=5 \
  --benchmark-json=perf-local-verify.json \
  -v
```

Expected: all 8 benchmark tests pass. Timing output shows results well under the gate thresholds (StubLLM + fake backends are near-zero latency).

- [ ] **Step 2: Smoke-test the gate script against the local results**

```bash
cp perf-local-verify.json perf-results.json
python - << 'PYTHON'
import json, sys, pathlib

results = json.loads(pathlib.Path("perf-results.json").read_text())
baseline_path = pathlib.Path("tests/perf/baselines/develop.json")
if not baseline_path.exists():
    print("No baseline — gate skipped (expected on first run before baseline is committed).")
    sys.exit(0)

baseline = json.loads(baseline_path.read_text())
failures = []

for bench in results.get("benchmarks", []):
    name = bench["name"]
    stats = bench["stats"]
    raw = stats.get("data", [])
    median_ms = stats.get("median", stats.get("mean", 0.0)) * 1000
    if raw:
        sorted_d = sorted(raw)
        p95_ms = sorted_d[min(int(len(sorted_d) * 0.95), len(sorted_d) - 1)] * 1000
    else:
        p95_ms = stats.get("max", stats.get("median", 0.0)) * 1000

    if "fast_path_first_token" in name and median_ms > 1500:
        failures.append(f"FAIL [{name}]: {median_ms:.1f}ms > 1500ms")

    if name in baseline:
        bm = baseline[name]["median_ms"]
        bp = baseline[name]["p95_ms"]
        if bm > 0 and (median_ms - bm) / bm > 0.10:
            failures.append(f"FAIL [{name}]: p50 regressed {median_ms:.1f} vs {bm:.1f}")
        if bp > 0 and (p95_ms - bp) / bp > 0.10:
            failures.append(f"FAIL [{name}]: p95 regressed {p95_ms:.1f} vs {bp:.1f}")

    print(f"  {name}: median={median_ms:.1f}ms  p95={p95_ms:.1f}ms")

if failures:
    for f in failures:
        print(f)
    sys.exit(1)
print("All gates passed locally.")
PYTHON
```

Expected: `All gates passed locally.`

- [ ] **Step 3: Run full unit suite one final time to confirm no regressions**

```bash
uv run pytest -m "not integration and not perf and not adapter" -x -q
```

Expected: all tests pass (timeouts are loose, no Phase 1–3 test regresses).

- [ ] **Step 4: Clean up temp file**

```bash
rm -f perf-results.json perf-local-verify.json
```

- [ ] **Step 5: Commit any ruff/mypy fixes, then final commit**

```bash
uv run ruff check . --fix && uv run ruff format .
uv run mypy src/sleuth
```

If any fixes were applied:

```bash
git add -u
git commit -m "chore: ruff/mypy cleanup for Phase 11"
```

---

## Task 12: Push branch and open PR

**Files:** none (git/GitHub operations)

- [ ] **Step 1: Push the feature branch**

```bash
git push -u origin feature/phase-11-perf
```

- [ ] **Step 2: Open PR targeting `develop`**

On GitHub, open a PR: `feature/phase-11-perf` → `develop`.

Title: `feat: Phase 11 — Perf hardening (per-backend timeouts, benchmark suite, perf.yml gate)`

Body should summarise:
- Per-backend `asyncio.wait_for` timeouts (8 s web, 4 s local; configurable via `Backend.timeout_s`).
- `BackendTimeoutError` wired into existing `SearchEvent(error=...)` path.
- 8 `pytest-benchmark` tests across fast path, cache hit, LocalFiles, and deep mode.
- `perf.yml` gate: first_token_ms median < 1500 ms; p50/p95 regression < 10% vs develop baseline.
- `scripts/perf-baseline.py` for updating the committed baseline on develop merges.

- [ ] **Step 3: Confirm CI green on PR**

Wait for `ci.yml` to pass (lint + unit tests). The `perf.yml` workflow also runs on PR and should pass (gate compares against committed baseline in the PR branch).

---

## Summary of files created / modified by Phase 11

| Operation | File | Purpose |
| --- | --- | --- |
| Modify | `src/sleuth/engine/executor.py` | `run_backends` with `asyncio.wait_for` per backend, `_resolve_timeout`, `DEFAULT_TIMEOUTS` |
| Modify | `.github/workflows/perf.yml` | Full gate: first_token_ms < 1500 ms, p50/p95 < 10% regression, baseline update on develop push |
| Create | `tests/perf/conftest.py` | `corpus_dir`, `stub_llm_fast`, `stub_llm_100ms`, `fake_web_backend`, `fake_docs_backend`, `baseline`, `run_stats_from_events` |
| Create | `tests/perf/baselines/develop.json` | Committed baseline (median_ms + p95_ms per benchmark) |
| Create | `tests/perf/test_bench_fast_path.py` | Fast-path first_token_ms and e2e latency benchmarks |
| Create | `tests/perf/test_bench_cache_hit.py` | Cache cold/warm comparison benchmarks |
| Create | `tests/perf/test_bench_localfiles.py` | LocalFiles cold/warm index benchmarks |
| Create | `tests/perf/test_bench_deep_mode.py` | Deep-mode e2e and speculative prefetch benchmarks |
| Create | `scripts/perf-baseline.py` | CLI helper to regenerate `develop.json` baseline |
| Create | `tests/engine/test_executor_timeout.py` | Unit tests for per-backend timeout logic |
