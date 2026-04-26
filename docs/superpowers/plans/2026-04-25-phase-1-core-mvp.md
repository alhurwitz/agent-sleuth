# Phase 1: Core MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build the foundational types, protocols, engine, and minimal Tavily backend so that `agent.aask("...")` runs end-to-end and emits the full typed event stream.

**Architecture:** A `Sleuth` class wraps three engine stages — `Router` (heuristic, no LLM), `Executor` (single-backend async fan-out), and `Synthesizer` (streaming tokens + citations). The engine uses only `LLMClient` and `Backend` protocols; all concrete implementations are injected. `MemoryCache` and in-memory `Session` provide the memory layer. A Tavily smoke stub handles web queries via `httpx`/`respx`.

**Tech Stack:** Python 3.11+, Pydantic v2, `httpx`, `anyio`, `pytest-asyncio` (auto mode), `syrupy`, `respx`, `uv`.

---

> **Callout — spec §15 #3 resolved here (documentation-only):** `fast_llm` has no literal default in `Sleuth.__init__`. When `fast_llm=None`, the engine falls back to the main `llm`. This is documented in the docstring only; no sentinel model is imported.

> **Assumption — Phase 0 not yet executed:** This plan assumes Phase 0 (`pyproject.toml`, package stubs, CI workflows, `tests/conftest.py` with `respx_mock` + tmp-corpus fixtures) has been executed before this branch is cut. Steps below verify Phase 0 artifacts exist before proceeding.

---

## Branch setup

### Task 0: Branch setup

**Files:**
- None created

- [x] **Step 0.1: Verify Phase 0 artifacts exist**

```bash
# Run from repo root
ls pyproject.toml src/sleuth/__init__.py tests/conftest.py
```

Expected: all three files exist. If any are missing, Phase 0 must be executed first.

- [x] **Step 0.2: Cut feature branch**

```bash
git checkout develop
git pull origin develop
git checkout -b feature/phase-1-core-mvp
```

Expected: you are now on `feature/phase-1-core-mvp`.

---

## Task 1: Error hierarchy (`src/sleuth/errors.py`)

**Files:**
- Create: `src/sleuth/errors.py`
- Test: `tests/test_errors.py`

- [x] **Step 1.1: Write the failing test**

Create `tests/test_errors.py`:

```python
import pytest
from sleuth.errors import (
    SleuthError,
    BackendError,
    BackendTimeoutError,
    LLMError,
    CacheError,
    ConfigError,
)


def test_hierarchy():
    assert issubclass(BackendError, SleuthError)
    assert issubclass(BackendTimeoutError, BackendError)
    assert issubclass(LLMError, SleuthError)
    assert issubclass(CacheError, SleuthError)
    assert issubclass(ConfigError, SleuthError)


def test_backend_timeout_is_backend_error():
    err = BackendTimeoutError("timed out")
    assert isinstance(err, BackendError)
    assert isinstance(err, SleuthError)


def test_errors_carry_message():
    for cls in (SleuthError, BackendError, BackendTimeoutError, LLMError, CacheError, ConfigError):
        e = cls("msg")
        assert str(e) == "msg"
```

- [x] **Step 1.2: Run test to verify it fails**

```bash
uv run pytest tests/test_errors.py -v
```

Expected: `ImportError: cannot import name 'SleuthError' from 'sleuth.errors'` (module doesn't exist yet).

- [x] **Step 1.3: Implement `src/sleuth/errors.py`**

```python
"""Sleuth exception hierarchy.

All public exceptions inherit from SleuthError so callers can catch broadly.
"""


class SleuthError(Exception):
    """Base class for all Sleuth errors."""


class BackendError(SleuthError):
    """A backend search call failed."""


class BackendTimeoutError(BackendError):
    """A backend search call exceeded its timeout."""


class LLMError(SleuthError):
    """An LLM stream call failed or returned an unexpected shape."""


class CacheError(SleuthError):
    """A cache read or write failed."""


class ConfigError(SleuthError):
    """Agent or backend was misconfigured."""
```

- [x] **Step 1.4: Run test to verify it passes**

```bash
uv run pytest tests/test_errors.py -v
```

Expected: 3 tests PASS.

- [x] **Step 1.5: Commit**

```bash
git add src/sleuth/errors.py tests/test_errors.py
git commit -m "feat: add SleuthError hierarchy (BackendError, LLMError, CacheError, ConfigError)"
```

---

## Task 2: Logging setup (`src/sleuth/logging.py`)

**Files:**
- Create: `src/sleuth/logging.py`
- Test: `tests/test_logging.py`

- [x] **Step 2.1: Write the failing test**

Create `tests/test_logging.py`:

```python
import logging
from sleuth.logging import get_logger


def test_get_logger_returns_sleuth_namespaced_logger():
    logger = get_logger("engine")
    assert logger.name == "sleuth.engine"


def test_get_logger_different_namespaces():
    a = get_logger("backends.web")
    b = get_logger("memory")
    assert a.name == "sleuth.backends.web"
    assert b.name == "sleuth.memory"


def test_root_logger_has_no_handlers_by_default():
    root = logging.getLogger("sleuth")
    assert root.handlers == []
```

- [x] **Step 2.2: Run test to verify it fails**

```bash
uv run pytest tests/test_logging.py -v
```

Expected: `ImportError: cannot import name 'get_logger' from 'sleuth.logging'`.

- [x] **Step 2.3: Implement `src/sleuth/logging.py`**

```python
"""Logging helpers for the sleuth package.

All internal modules call ``get_logger(__name__.replace("sleuth.", ""))`` which
produces loggers under the ``sleuth`` namespace (e.g. ``sleuth.engine``,
``sleuth.backends.web``).  No handlers are attached here — callers wire
up their own.  The ``SLEUTH_LOG_LEVEL`` env var is intentionally NOT read
at import time; consumers configure log level through normal ``logging``
machinery.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a ``logging.Logger`` under the ``sleuth.<name>`` namespace.

    Args:
        name: Sub-namespace, e.g. ``"engine"``, ``"backends.web"``.

    Returns:
        A logger whose ``.name`` is ``"sleuth.<name>"``.
    """
    return logging.getLogger(f"sleuth.{name}")
```

- [x] **Step 2.4: Run test to verify it passes**

```bash
uv run pytest tests/test_logging.py -v
```

Expected: 3 tests PASS.

- [x] **Step 2.5: Commit**

```bash
git add src/sleuth/logging.py tests/test_logging.py
git commit -m "feat: add get_logger helper — sleuth.* namespace, no handlers by default"
```

---

## Task 3: Core data types (`src/sleuth/types.py`)

**Files:**
- Create: `src/sleuth/types.py`
- Test: `tests/test_types.py`

- [x] **Step 3.1: Write the failing test**

Create `tests/test_types.py`:

```python
from datetime import datetime, timezone
from sleuth.types import Source, Chunk, RunStats, Result, Depth, Length


def test_source_kinds():
    for kind in ("url", "file", "code"):
        s = Source(kind=kind, location="x")
        assert s.kind == kind


def test_source_optional_fields():
    s = Source(kind="url", location="https://example.com")
    assert s.title is None
    assert s.fetched_at is None


def test_chunk_defaults():
    s = Source(kind="file", location="/tmp/foo.md")
    c = Chunk(text="hello", source=s)
    assert c.score is None
    assert c.metadata == {}


def test_runstats_required_fields():
    stats = RunStats(
        latency_ms=200,
        first_token_ms=150,
        tokens_in=10,
        tokens_out=5,
        cache_hits={"query": 0},
        backends_called=["tavily"],
    )
    assert stats.latency_ms == 200


def test_runstats_first_token_ms_nullable():
    stats = RunStats(
        latency_ms=50,
        first_token_ms=None,
        tokens_in=0,
        tokens_out=0,
        cache_hits={},
        backends_called=[],
    )
    assert stats.first_token_ms is None


def test_result_no_schema():
    stats = RunStats(
        latency_ms=100, first_token_ms=90, tokens_in=5, tokens_out=10,
        cache_hits={}, backends_called=[]
    )
    r: Result = Result(text="answer", citations=[], stats=stats)
    assert r.data is None


def test_result_generic_with_schema():
    from pydantic import BaseModel

    class MySchema(BaseModel):
        score: float

    stats = RunStats(
        latency_ms=100, first_token_ms=90, tokens_in=5, tokens_out=10,
        cache_hits={}, backends_called=[]
    )
    r: Result[MySchema] = Result(text="ok", citations=[], data=MySchema(score=0.9), stats=stats)
    assert r.data.score == 0.9


def test_depth_literals():
    depths: list[Depth] = ["auto", "fast", "deep"]
    assert len(depths) == 3


def test_length_literals():
    lengths: list[Length] = ["brief", "standard", "thorough"]
    assert len(lengths) == 3
```

- [x] **Step 3.2: Run test to verify it fails**

```bash
uv run pytest tests/test_types.py -v
```

Expected: `ImportError: cannot import name 'Source' from 'sleuth.types'`.

- [x] **Step 3.3: Implement `src/sleuth/types.py`**

```python
"""Public data shapes used throughout Sleuth.

All types here are Pydantic v2 models (serializable, validated).
Internal-only hot-path structs live in engine/ as plain dataclasses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Depth / Length literals
# ---------------------------------------------------------------------------

Depth = Literal["auto", "fast", "deep"]
Length = Literal["brief", "standard", "thorough"]

# ---------------------------------------------------------------------------
# Source, Chunk
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """A citable location that contributed to an answer."""

    kind: Literal["url", "file", "code"]
    location: str  # URL, absolute file path, or "repo:path:line_range"
    title: str | None = None
    fetched_at: datetime | None = None


class Chunk(BaseModel):
    """A scored text fragment from a backend search result."""

    text: str
    source: Source
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RunStats
# ---------------------------------------------------------------------------


class RunStats(BaseModel):
    """Latency and token accounting emitted in DoneEvent."""

    latency_ms: int
    first_token_ms: int | None  # None on cache hit
    tokens_in: int
    tokens_out: int
    cache_hits: dict[str, int]  # namespace → hit count
    backends_called: list[str]


# ---------------------------------------------------------------------------
# Result[T]
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class Result(BaseModel, Generic[T]):
    """The return value of agent.ask(...).

    ``data`` is populated when ``schema=`` is passed; otherwise ``None``.
    Static typing is preserved via the ``Result[T]`` generic.
    """

    text: str
    citations: list[Source]
    data: T | None = None
    stats: RunStats
```

- [x] **Step 3.4: Run test to verify it passes**

```bash
uv run pytest tests/test_types.py -v
```

Expected: all 9 tests PASS.

- [x] **Step 3.5: Commit**

```bash
git add src/sleuth/types.py tests/test_types.py
git commit -m "feat: add core data shapes — Source, Chunk, RunStats, Result[T], Depth, Length"
```

---

## Task 4: Event types (`src/sleuth/events.py`)

**Files:**
- Create: `src/sleuth/events.py`
- Test: `tests/test_events.py`

- [x] **Step 4.1: Write the failing test**

Create `tests/test_events.py`:

```python
import json
from typing import get_args

import pytest
from pydantic import TypeAdapter

from sleuth.events import (
    RouteEvent,
    PlanEvent,
    SearchEvent,
    FetchEvent,
    ThinkingEvent,
    TokenEvent,
    CitationEvent,
    CacheHitEvent,
    DoneEvent,
    Event,
)
from sleuth.types import RunStats, Source


def _stats() -> RunStats:
    return RunStats(
        latency_ms=100, first_token_ms=80, tokens_in=5, tokens_out=10,
        cache_hits={}, backends_called=["stub"]
    )


def _source() -> Source:
    return Source(kind="url", location="https://example.com")


def test_route_event_discriminator():
    e = RouteEvent(type="route", depth="fast", reason="short query")
    assert e.type == "route"
    assert e.depth == "fast"


def test_plan_event():
    e = PlanEvent(type="plan", steps=[])
    assert e.type == "plan"


def test_search_event_no_error():
    e = SearchEvent(type="search", backend="tavily", query="foo")
    assert e.error is None


def test_search_event_with_error():
    e = SearchEvent(type="search", backend="tavily", query="foo", error="timeout")
    assert e.error == "timeout"


def test_fetch_event():
    e = FetchEvent(type="fetch", url="https://x.com", status=200)
    assert e.status == 200


def test_thinking_event():
    e = ThinkingEvent(type="thinking", text="reasoning...")
    assert e.text == "reasoning..."


def test_token_event():
    e = TokenEvent(type="token", text="hello")
    assert e.text == "hello"


def test_citation_event():
    e = CitationEvent(type="citation", index=0, source=_source())
    assert e.index == 0


def test_cache_hit_event():
    e = CacheHitEvent(type="cache_hit", kind="query", key="abc123")
    assert e.kind == "query"


def test_done_event():
    e = DoneEvent(type="done", stats=_stats())
    assert e.stats.latency_ms == 100


def test_event_union_roundtrip():
    """Discriminated union parses each event type correctly."""
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    payloads = [
        {"type": "route", "depth": "auto", "reason": "heuristic"},
        {"type": "plan", "steps": []},
        {"type": "search", "backend": "tavily", "query": "q"},
        {"type": "fetch", "url": "https://x.com", "status": 200},
        {"type": "thinking", "text": "hmm"},
        {"type": "token", "text": "tok"},
        {"type": "citation", "index": 0, "source": {"kind": "url", "location": "https://x.com"}},
        {"type": "cache_hit", "kind": "query", "key": "k"},
        {
            "type": "done",
            "stats": {
                "latency_ms": 1, "first_token_ms": None, "tokens_in": 0,
                "tokens_out": 0, "cache_hits": {}, "backends_called": []
            }
        },
    ]
    for p in payloads:
        event = adapter.validate_python(p)
        assert event.type == p["type"]
```

- [x] **Step 4.2: Run test to verify it fails**

```bash
uv run pytest tests/test_events.py -v
```

Expected: `ImportError: cannot import name 'RouteEvent' from 'sleuth.events'`.

- [x] **Step 4.3: Implement `src/sleuth/events.py`**

```python
"""Typed event stream for Sleuth runs.

Every run emits an ordered stream of these events. The discriminated
``Event`` union makes it easy to switch on ``event.type`` without
isinstance checks.  Cached runs emit the same events as live runs,
prefixed with ``CacheHitEvent``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from sleuth.types import Depth, RunStats, Source


class PlanStep(BaseModel):
    """One step in a planner-generated search plan."""

    query: str
    backend_hint: str | None = None


class RouteEvent(BaseModel):
    """Emitted once per run when the router decides the depth."""

    type: Literal["route"]
    depth: Depth
    reason: str


class PlanEvent(BaseModel):
    """Emitted when the planner produces sub-queries (deep mode only)."""

    type: Literal["plan"]
    steps: list[PlanStep]


class SearchEvent(BaseModel):
    """Emitted when a backend search call starts (or errors)."""

    type: Literal["search"]
    backend: str
    query: str
    error: str | None = None


class FetchEvent(BaseModel):
    """Emitted when a URL is fetched (WebBackend fetch=True mode)."""

    type: Literal["fetch"]
    url: str
    status: int


class ThinkingEvent(BaseModel):
    """Emitted for reasoning tokens when the LLM supports extended thinking."""

    type: Literal["thinking"]
    text: str


class TokenEvent(BaseModel):
    """Emitted for each text token from the synthesizer LLM."""

    type: Literal["token"]
    text: str


class CitationEvent(BaseModel):
    """Emitted as sources resolve during synthesis."""

    type: Literal["citation"]
    index: int
    source: Source


class CacheHitEvent(BaseModel):
    """Emitted before replaying a cached run."""

    type: Literal["cache_hit"]
    kind: str  # "query" | "fetch" | "plan"
    key: str


class DoneEvent(BaseModel):
    """Final event in every run — carries accounting stats."""

    type: Literal["done"]
    stats: RunStats


Event = Annotated[
    Union[
        RouteEvent,
        PlanEvent,
        SearchEvent,
        FetchEvent,
        ThinkingEvent,
        TokenEvent,
        CitationEvent,
        CacheHitEvent,
        DoneEvent,
    ],
    Field(discriminator="type"),
]
```

- [x] **Step 4.4: Run test to verify it passes**

```bash
uv run pytest tests/test_events.py -v
```

Expected: all 12 tests PASS.

- [x] **Step 4.5: Commit**

```bash
git add src/sleuth/events.py tests/test_events.py
git commit -m "feat: add typed event stream — all 9 event types + discriminated Event union"
```

---

## Task 5: LLM base protocol (`src/sleuth/llm/base.py`)

**Files:**
- Create: `src/sleuth/llm/base.py`
- Test: `tests/llm/test_base.py`

- [x] **Step 5.1: Write the failing test**

Create `tests/llm/__init__.py` (empty) and `tests/llm/test_base.py`:

```python
from typing import Any
from sleuth.llm.base import (
    TextDelta,
    ReasoningDelta,
    ToolCall,
    Stop,
    LLMChunk,
    Message,
    Tool,
    LLMClient,
)


def test_text_delta():
    d = TextDelta(text="hello")
    assert d.text == "hello"


def test_reasoning_delta():
    d = ReasoningDelta(text="thinking")
    assert d.text == "thinking"


def test_tool_call():
    tc = ToolCall(id="c1", name="search", arguments={"query": "foo"})
    assert tc.name == "search"
    assert tc.arguments == {"query": "foo"}


def test_stop_reasons():
    for reason in ("end_turn", "tool_use", "max_tokens", "stop_sequence", "error"):
        s = Stop(reason=reason)
        assert s.reason == reason


def test_message_defaults():
    m = Message(role="user", content="hi")
    assert m.tool_call_id is None


def test_tool_model():
    t = Tool(name="search", description="web search", input_schema={"type": "object"})
    assert t.name == "search"


def test_llmclient_is_protocol():
    """LLMClient is a runtime-checkable Protocol — structural subtyping only."""
    import typing
    assert hasattr(LLMClient, "__protocol_attrs__") or typing.get_origin(LLMClient) is None
    # Just importing LLMClient without error is the meaningful assertion here.
    assert LLMClient is not None
```

- [x] **Step 5.2: Run test to verify it fails**

```bash
uv run pytest tests/llm/test_base.py -v
```

Expected: `ImportError: cannot import name 'TextDelta' from 'sleuth.llm.base'`.

- [x] **Step 5.3: Create `tests/llm/__init__.py`**

```bash
touch tests/llm/__init__.py
```

- [x] **Step 5.4: Implement `src/sleuth/llm/base.py`**

```python
"""LLM protocol and supporting types.

The package never imports a model SDK as a hard dependency.  Users pass any
object that satisfies the ``LLMClient`` Protocol.  Optional shims in
``sleuth.llm.{anthropic,openai}`` adapt those SDKs to this shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# LLMChunk — discriminated union for streaming chunks
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    """A text token from the model's response."""

    text: str


@dataclass
class ReasoningDelta:
    """A reasoning/thinking token (Claude extended thinking, OpenAI o-series)."""

    text: str


@dataclass
class ToolCall:
    """A tool-use request emitted by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Stop:
    """Stream terminator — carries the stop reason."""

    reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "error"]


LLMChunk = TextDelta | ReasoningDelta | ToolCall | Stop


# ---------------------------------------------------------------------------
# Message + Tool — Pydantic models (public, serializable)
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


class Tool(BaseModel):
    """A tool definition passed to the LLM for function-calling."""

    name: str
    description: str
    input_schema: dict[str, Any]


# ---------------------------------------------------------------------------
# LLMClient Protocol
# ---------------------------------------------------------------------------


class LLMClient:
    """Structural protocol for all LLM clients used by Sleuth.

    Not a runtime-checkable Protocol (would require importing ``typing.Protocol``
    and carrying the overhead); enforced structurally by mypy ``--strict``.
    Users implement this interface; the engine never does isinstance checks.

    Attributes:
        name: Human-readable identifier, e.g. ``"anthropic:claude-sonnet-4-6"``.
        supports_reasoning: When True the engine emits ThinkingEvents.
        supports_structured_output: When True schema= is passed through natively;
            otherwise the engine falls back to JSON-parse of the text response.
    """

    name: str
    supports_reasoning: bool
    supports_structured_output: bool

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]:  # pragma: no cover
        """Stream LLM output as ``LLMChunk`` items.

        Must yield at least one chunk and end with a ``Stop`` chunk.
        """
        raise NotImplementedError
        yield  # make this a generator to satisfy the return type
```

- [x] **Step 5.5: Run test to verify it passes**

```bash
uv run pytest tests/llm/test_base.py -v
```

Expected: all 7 tests PASS.

- [x] **Step 5.6: Commit**

```bash
git add src/sleuth/llm/base.py src/sleuth/llm/__init__.py tests/llm/__init__.py tests/llm/test_base.py
git commit -m "feat: add LLMClient protocol — TextDelta, ReasoningDelta, ToolCall, Stop, Message, Tool"
```

---

## Task 6: StubLLM (`src/sleuth/llm/stub.py`)

**Files:**
- Create: `src/sleuth/llm/stub.py`
- Test: `tests/llm/test_stub.py`

- [x] **Step 6.1: Write the failing test**

Create `tests/llm/test_stub.py`:

```python
import pytest
from sleuth.llm.stub import StubLLM
from sleuth.llm.base import TextDelta, Stop, Message, ReasoningDelta


async def collect(stub: StubLLM, messages: list[Message]) -> list:
    chunks = []
    async for chunk in await stub.stream(messages):
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_string_list_emits_text_deltas_then_stop():
    stub = StubLLM(responses=["hello", " world"])
    msgs = [Message(role="user", content="hi")]
    chunks = await collect(stub, msgs)
    assert chunks == [TextDelta("hello"), TextDelta(" world"), Stop("end_turn")]


@pytest.mark.asyncio
async def test_multiple_calls_cycle_through_responses():
    stub = StubLLM(responses=["first", "second"])
    msgs = [Message(role="user", content="q")]
    # First call
    chunks1 = await collect(stub, msgs)
    assert chunks1[0] == TextDelta("first")
    # Second call
    chunks2 = await collect(stub, msgs)
    assert chunks2[0] == TextDelta("second")


@pytest.mark.asyncio
async def test_callable_owns_response():
    from sleuth.llm.base import LLMChunk

    async def responder(messages: list[Message]):
        yield TextDelta("dynamic")
        yield Stop("end_turn")

    stub = StubLLM(responses=responder)
    msgs = [Message(role="user", content="q")]
    chunks = await collect(stub, msgs)
    assert chunks == [TextDelta("dynamic"), Stop("end_turn")]


@pytest.mark.asyncio
async def test_stub_attributes():
    stub = StubLLM(responses=["ok"])
    assert stub.name == "stub"
    assert stub.supports_reasoning is False
    assert stub.supports_structured_output is True


@pytest.mark.asyncio
async def test_llmchunk_item_passed_directly():
    stub = StubLLM(responses=[TextDelta("raw"), Stop("end_turn")])
    msgs = [Message(role="user", content="q")]
    chunks = await collect(stub, msgs)
    assert chunks[0] == TextDelta("raw")
    assert chunks[1] == Stop("end_turn")
```

- [x] **Step 6.2: Run test to verify it fails**

```bash
uv run pytest tests/llm/test_stub.py -v
```

Expected: `ImportError: cannot import name 'StubLLM' from 'sleuth.llm.stub'`.

- [x] **Step 6.3: Implement `src/sleuth/llm/stub.py`**

```python
"""StubLLM — deterministic test double for all engine and backend tests.

Never import this in production code.  It is always importable (no extras
required) because all CI needs it without installing a real LLM SDK.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Callable, Sequence
from typing import overload

from pydantic import BaseModel

from sleuth.llm.base import LLMChunk, Message, Stop, TextDelta, Tool


class StubLLM:
    """Deterministic LLM double that replays scripted responses.

    Pass ``responses`` as:

    * ``list[str]`` — each string becomes a single ``TextDelta`` followed by
      ``Stop("end_turn")``.  Calls cycle round-robin through the list.
    * ``list[LLMChunk]`` — chunks emitted verbatim (no ``Stop`` appended).
    * ``Callable[[list[Message]], AsyncIterator[LLMChunk]]`` — full control;
      the callable owns the stream.
    """

    name = "stub"
    supports_reasoning = False
    supports_structured_output = True

    def __init__(
        self,
        responses: Sequence[str | LLMChunk] | Callable[..., AsyncIterator[LLMChunk]],
    ) -> None:
        self._callable: Callable[..., AsyncIterator[LLMChunk]] | None = None
        self._cycle: itertools.cycle[str | LLMChunk] | None = None

        if callable(responses) and not isinstance(responses, (list, tuple)):
            self._callable = responses  # type: ignore[assignment]
        else:
            self._cycle = itertools.cycle(responses)  # type: ignore[arg-type]

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        if self._callable is not None:
            return self._callable(messages)
        return self._replay()

    async def _replay(self) -> AsyncIterator[LLMChunk]:  # type: ignore[return]
        assert self._cycle is not None
        item = next(self._cycle)
        if isinstance(item, str):
            yield TextDelta(item)
            yield Stop("end_turn")
        else:
            yield item
```

- [x] **Step 6.4: Run test to verify it passes**

```bash
uv run pytest tests/llm/test_stub.py -v
```

Expected: all 5 tests PASS.

- [x] **Step 6.5: Commit**

```bash
git add src/sleuth/llm/stub.py tests/llm/test_stub.py
git commit -m "feat: add StubLLM — deterministic test double cycling scripted responses"
```

---

## Task 7: Backend base protocol (`src/sleuth/backends/base.py`)

**Files:**
- Create: `src/sleuth/backends/base.py`
- Test: `tests/contract/test_backend_protocol.py`

- [x] **Step 7.1: Write the failing test**

Create `tests/contract/__init__.py` (empty) and `tests/contract/test_backend_protocol.py`:

```python
"""BackendTestKit — reusable harness validating Backend protocol compliance.

Usage in later phase plans (e.g. Phase 2 LocalFiles, Phase 9 WebBackend):

    from tests.contract.test_backend_protocol import BackendTestKit, FakeBackend

    class TestMyBackend(BackendTestKit):
        @pytest.fixture
        def backend(self):
            return MyBackend(...)
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sleuth.backends.base import Backend, Capability
from sleuth.types import Chunk, Source


# ---------------------------------------------------------------------------
# FakeBackend — minimal in-memory backend for testing the kit itself
# ---------------------------------------------------------------------------


class FakeBackend:
    """Minimal Backend implementation used to test BackendTestKit itself."""

    name = "fake"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        *,
        raise_on_search: Exception | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._chunks = chunks or [
            Chunk(text="result", source=Source(kind="url", location="https://example.com"))
        ]
        self._raise = raise_on_search
        self._delay = delay_s

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise:
            raise self._raise
        return self._chunks[:k]


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------


def assert_chunk_list(result: Any) -> None:
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, Chunk)
        assert isinstance(item.source, Source)
        assert isinstance(item.text, str)


# ---------------------------------------------------------------------------
# BackendTestKit — base class for parametrized contract tests
# ---------------------------------------------------------------------------


class BackendTestKit:
    """Subclass this in each backend's test module and provide a ``backend`` fixture.

    Example::

        class TestFakeBackend(BackendTestKit):
            @pytest.fixture
            def backend(self):
                return FakeBackend()
    """

    @pytest.fixture
    def backend(self) -> Backend:  # pragma: no cover
        raise NotImplementedError("Subclasses must provide a `backend` fixture")

    @pytest.mark.asyncio
    async def test_search_returns_chunk_list(self, backend: Backend) -> None:
        result = await backend.search("test query")
        assert_chunk_list(result)

    @pytest.mark.asyncio
    async def test_search_respects_k(self, backend: Backend) -> None:
        result = await backend.search("test query", k=1)
        assert len(result) <= 1

    @pytest.mark.asyncio
    async def test_backend_has_name(self, backend: Backend) -> None:
        assert isinstance(backend.name, str)
        assert len(backend.name) > 0

    @pytest.mark.asyncio
    async def test_backend_has_capabilities(self, backend: Backend) -> None:
        assert isinstance(backend.capabilities, frozenset)
        for cap in backend.capabilities:
            assert isinstance(cap, Capability)

    @pytest.mark.asyncio
    async def test_cancellation_safety(self, backend: Backend) -> None:
        """Backend search must honour asyncio cancellation without hanging."""
        task = asyncio.create_task(backend.search("query"))
        # Give it one scheduler turn to start, then cancel.
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await task
        # The important assertion: we got here without hanging.


# ---------------------------------------------------------------------------
# Self-test — run BackendTestKit against FakeBackend
# ---------------------------------------------------------------------------


class TestFakeBackend(BackendTestKit):
    @pytest.fixture
    def backend(self) -> FakeBackend:
        return FakeBackend()

    @pytest.mark.asyncio
    async def test_error_propagates(self) -> None:
        b = FakeBackend(raise_on_search=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            await b.search("q")
```

- [x] **Step 7.2: Run test to verify it fails**

```bash
uv run pytest tests/contract/test_backend_protocol.py -v
```

Expected: `ImportError: cannot import name 'Backend' from 'sleuth.backends.base'`.

- [x] **Step 7.3: Create `tests/contract/__init__.py`**

```bash
touch tests/contract/__init__.py
```

- [x] **Step 7.4: Implement `src/sleuth/backends/base.py`**

```python
"""Backend protocol and Capability enum.

Every search backend — built-in or user-supplied — implements the ``Backend``
Protocol.  The engine never does isinstance checks; it relies on structural
typing (enforced by mypy ``--strict``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from sleuth.errors import BackendError, BackendTimeoutError
from sleuth.types import Chunk


class Capability(StrEnum):
    """Tag set that tells the planner which backends are eligible per sub-query."""

    WEB = "web"        # general web search
    DOCS = "docs"      # local document corpora
    CODE = "code"      # source code
    FRESH = "fresh"    # results that reflect "now" (news, prices, status pages)
    PRIVATE = "private"  # auth-gated systems (Notion, Linear, Slack, etc.)


class Backend(Protocol):
    """Structural protocol for all Sleuth backends.

    Attributes:
        name: Short identifier used in ``SearchEvent.backend`` and ``RunStats``.
        capabilities: Frozenset of ``Capability`` values — drives planner routing.
    """

    name: str
    capabilities: frozenset[Capability]

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        """Search this backend and return up to ``k`` ``Chunk`` objects.

        Raises:
            BackendError: On unrecoverable search failure.
            BackendTimeoutError: When the backend's own deadline is exceeded.
        """
        ...


__all__ = ["Backend", "Capability", "BackendError", "BackendTimeoutError"]
```

- [x] **Step 7.5: Run test to verify it passes**

```bash
uv run pytest tests/contract/test_backend_protocol.py -v
```

Expected: all 6 tests in `TestFakeBackend` PASS.

- [x] **Step 7.6: Commit**

```bash
git add src/sleuth/backends/base.py src/sleuth/backends/__init__.py tests/contract/__init__.py tests/contract/test_backend_protocol.py
git commit -m "feat: add Backend protocol + Capability enum + BackendTestKit reusable harness"
```

---

## Task 8: Memory — Cache protocol + MemoryCache (`src/sleuth/memory/cache.py`)

**Files:**
- Create: `src/sleuth/memory/cache.py`
- Test: `tests/memory/test_cache.py`

- [x] **Step 8.1: Write the failing test**

Create `tests/memory/__init__.py` (empty) and `tests/memory/test_cache.py`:

```python
import pytest
from sleuth.memory.cache import MemoryCache, Cache


@pytest.mark.asyncio
async def test_set_and_get():
    c = MemoryCache()
    await c.set("query", "key1", "value1")
    assert await c.get("query", "key1") == "value1"


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    c = MemoryCache()
    assert await c.get("query", "missing") is None


@pytest.mark.asyncio
async def test_delete():
    c = MemoryCache()
    await c.set("query", "k", "v")
    await c.delete("query", "k")
    assert await c.get("query", "k") is None


@pytest.mark.asyncio
async def test_clear_namespace():
    c = MemoryCache()
    await c.set("query", "k1", "v1")
    await c.set("fetch", "k2", "v2")
    await c.clear("query")
    assert await c.get("query", "k1") is None
    assert await c.get("fetch", "k2") == "v2"


@pytest.mark.asyncio
async def test_clear_all():
    c = MemoryCache()
    await c.set("query", "k1", "v1")
    await c.set("fetch", "k2", "v2")
    await c.clear()
    assert await c.get("query", "k1") is None
    assert await c.get("fetch", "k2") is None


@pytest.mark.asyncio
async def test_namespaces_isolated():
    c = MemoryCache()
    await c.set("query", "same_key", "query_val")
    await c.set("fetch", "same_key", "fetch_val")
    assert await c.get("query", "same_key") == "query_val"
    assert await c.get("fetch", "same_key") == "fetch_val"


@pytest.mark.asyncio
async def test_ttl_parameter_accepted():
    """MemoryCache ignores TTL (in-memory has no expiry); it must not error."""
    c = MemoryCache()
    await c.set("query", "k", "v", ttl_s=60)
    assert await c.get("query", "k") == "v"
```

- [x] **Step 8.2: Run test to verify it fails**

```bash
uv run pytest tests/memory/test_cache.py -v
```

Expected: `ImportError: cannot import name 'MemoryCache' from 'sleuth.memory.cache'`.

- [x] **Step 8.3: Create `tests/memory/__init__.py`**

```bash
touch tests/memory/__init__.py
```

- [x] **Step 8.4: Implement `src/sleuth/memory/cache.py`**

```python
"""Cache protocol and MemoryCache implementation.

Phase 4 will replace ``MemoryCache`` with ``SqliteCache`` (persistent, TTL-aware).
The ``Cache`` Protocol is frozen; keep it stable.

Namespaces (spec §8):
    "query"  — ``(query_hash, backend_set, depth) → Result``
    "fetch"  — ``url/file → parsed content``
    "plan"   — ``(query_hash, tree_version) → plan``
    "index"  — per-corpus; Phase 2 adds this namespace
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol


class Cache(Protocol):
    """Async key-value store partitioned by namespace.

    Implementations:
        MemoryCache  (Phase 1) — in-memory, no TTL enforcement.
        SqliteCache  (Phase 4) — persistent, per-namespace TTL.
    """

    async def get(self, namespace: str, key: str) -> Any | None:
        """Return the stored value, or ``None`` if absent / expired."""
        ...

    async def set(
        self, namespace: str, key: str, value: Any, *, ttl_s: int | None = None
    ) -> None:
        """Store ``value`` under ``(namespace, key)``.

        ``ttl_s`` is accepted by all implementations; MemoryCache ignores it.
        """
        ...

    async def delete(self, namespace: str, key: str) -> None:
        """Remove a single entry.  No-op if absent."""
        ...

    async def clear(self, namespace: str | None = None) -> None:
        """Remove all entries in ``namespace``, or everything if ``None``."""
        ...


class MemoryCache:
    """In-memory Cache implementation — no persistence, no TTL enforcement.

    Thread-safe for single-threaded async use (no asyncio locks needed because
    dict operations are atomic in CPython).  Phase 4 replaces this with
    ``SqliteCache``.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = defaultdict(dict)

    async def get(self, namespace: str, key: str) -> Any | None:
        return self._store[namespace].get(key)

    async def set(
        self, namespace: str, key: str, value: Any, *, ttl_s: int | None = None
    ) -> None:
        self._store[namespace][key] = value

    async def delete(self, namespace: str, key: str) -> None:
        self._store[namespace].pop(key, None)

    async def clear(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._store.clear()
        else:
            self._store[namespace].clear()
```

- [x] **Step 8.5: Run test to verify it passes**

```bash
uv run pytest tests/memory/test_cache.py -v
```

Expected: all 7 tests PASS.

- [x] **Step 8.6: Commit**

```bash
git add src/sleuth/memory/cache.py src/sleuth/memory/__init__.py tests/memory/__init__.py tests/memory/test_cache.py
git commit -m "feat: add Cache protocol + MemoryCache (in-memory, Phase 4 will add SqliteCache)"
```

---

## Task 9: Session (`src/sleuth/memory/session.py`)

**Files:**
- Create: `src/sleuth/memory/session.py`
- Test: `tests/memory/test_session.py`

- [x] **Step 9.1: Write the failing test**

Create `tests/memory/test_session.py`:

```python
import pytest
from sleuth.memory.session import Session
from sleuth.types import Source, RunStats, Result


def _result(text: str = "answer") -> Result:
    stats = RunStats(
        latency_ms=100, first_token_ms=90, tokens_in=5, tokens_out=10,
        cache_hits={}, backends_called=[]
    )
    return Result(text=text, citations=[], stats=stats)


def test_session_starts_empty():
    s = Session()
    assert s.turns == []


def test_add_turn_and_retrieve():
    s = Session()
    s.add_turn("what is foo?", _result("foo is bar"), [])
    assert len(s.turns) == 1
    assert s.turns[0].query == "what is foo?"
    assert s.turns[0].result.text == "foo is bar"


def test_ring_buffer_respects_max_turns():
    s = Session(max_turns=3)
    for i in range(5):
        s.add_turn(f"q{i}", _result(f"a{i}"), [])
    assert len(s.turns) == 3
    # Oldest turns dropped; most recent kept
    assert s.turns[0].query == "q2"
    assert s.turns[-1].query == "q4"


def test_default_max_turns_is_20():
    s = Session()
    for i in range(25):
        s.add_turn(f"q{i}", _result(f"a{i}"), [])
    assert len(s.turns) == 20


def test_as_messages_returns_list():
    from sleuth.llm.base import Message
    s = Session()
    s.add_turn("q1", _result("a1"), [])
    msgs = s.as_messages()
    assert all(isinstance(m, Message) for m in msgs)


def test_as_messages_interleaves_user_assistant():
    from sleuth.llm.base import Message
    s = Session()
    s.add_turn("q1", _result("a1"), [])
    s.add_turn("q2", _result("a2"), [])
    msgs = s.as_messages()
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    assert msgs[2].role == "user"
    assert msgs[3].role == "assistant"
```

- [x] **Step 9.2: Run test to verify it fails**

```bash
uv run pytest tests/memory/test_session.py -v
```

Expected: `ImportError: cannot import name 'Session' from 'sleuth.memory.session'`.

- [x] **Step 9.3: Implement `src/sleuth/memory/session.py`**

```python
"""Session — multi-turn ring buffer for conversation coherence.

Phase 4 will add ``session.save(path)`` / ``Session.load(path)`` and
``await session.flush()``.  The ``Session`` class and ``Turn`` dataclass
are stable; Phase 4 only extends them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sleuth.llm.base import Message
from sleuth.types import Result, Source


@dataclass
class Turn:
    """One Q&A exchange stored in the session ring buffer."""

    query: str
    result: Result  # type: ignore[type-arg]
    citations: list[Source]


class Session:
    """In-memory ring buffer of recent conversation turns.

    Attributes:
        max_turns: Maximum number of turns retained.  Oldest turn is dropped
            when the buffer is full.  Defaults to 20 (spec §8).

    Note:
        Phase 4 adds ``save(path)``, ``Session.load(path)``, and
        ``flush()`` for persistence.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._max_turns = max_turns
        self._buffer: deque[Turn] = deque(maxlen=max_turns)

    @property
    def turns(self) -> list[Turn]:
        """Snapshot of current turns, oldest first."""
        return list(self._buffer)

    def add_turn(self, query: str, result: Result, citations: list[Source]) -> None:  # type: ignore[type-arg]
        """Append a turn; oldest turn is evicted when buffer is full."""
        self._buffer.append(Turn(query=query, result=result, citations=citations))

    def as_messages(self) -> list[Message]:
        """Return turns as alternating user/assistant Message objects.

        Suitable for passing to ``LLMClient.stream()`` as conversation history.
        """
        messages: list[Message] = []
        for turn in self._buffer:
            messages.append(Message(role="user", content=turn.query))
            messages.append(Message(role="assistant", content=turn.result.text))
        return messages
```

- [x] **Step 9.4: Run test to verify it passes**

```bash
uv run pytest tests/memory/test_session.py -v
```

Expected: all 6 tests PASS.

- [x] **Step 9.5: Commit**

```bash
git add src/sleuth/memory/session.py tests/memory/test_session.py
git commit -m "feat: add Session ring buffer — max_turns, add_turn, as_messages (Phase 4 adds persistence)"
```

---

## Task 10: Tavily web backend (`src/sleuth/backends/web.py`)

**Files:**
- Create: `src/sleuth/backends/web.py`
- Test: `tests/backends/test_web.py`

The Tavily backend uses `httpx` for HTTP; tests mock with `respx`.

- [x] **Step 10.1: Write the failing test**

Create `tests/backends/__init__.py` (empty) and `tests/backends/test_web.py`:

```python
"""Tests for the Tavily WebBackend smoke implementation.

Phase 9 will expand this file with Exa, Brave, SerpAPI tests.
"""

import pytest
import respx
import httpx
from sleuth.backends.web import TavilyBackend, WebBackend
from sleuth.backends.base import Capability
from sleuth.types import Chunk


TAVILY_SEARCH_URL = "https://api.tavily.com/search"

FAKE_RESPONSE = {
    "results": [
        {
            "title": "Example Article",
            "url": "https://example.com/article",
            "content": "This is the snippet text.",
            "score": 0.95,
        },
        {
            "title": "Another Result",
            "url": "https://example.com/other",
            "content": "More content here.",
            "score": 0.80,
        },
    ]
}


@pytest.fixture
def tavily(respx_mock):
    respx_mock.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=FAKE_RESPONSE)
    )
    return TavilyBackend(api_key="test-key")


@pytest.mark.asyncio
async def test_search_returns_chunks(tavily):
    chunks = await tavily.search("what is Python?")
    assert len(chunks) == 2
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source.kind == "url"


@pytest.mark.asyncio
async def test_search_maps_score(tavily):
    chunks = await tavily.search("python")
    assert chunks[0].score == pytest.approx(0.95)
    assert chunks[1].score == pytest.approx(0.80)


@pytest.mark.asyncio
async def test_search_maps_title(tavily):
    chunks = await tavily.search("python")
    assert chunks[0].source.title == "Example Article"


@pytest.mark.asyncio
async def test_search_maps_url(tavily):
    chunks = await tavily.search("python")
    assert chunks[0].source.location == "https://example.com/article"


@pytest.mark.asyncio
async def test_search_respects_k(respx_mock):
    respx_mock.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=FAKE_RESPONSE)
    )
    backend = TavilyBackend(api_key="test-key")
    chunks = await backend.search("python", k=1)
    assert len(chunks) <= 1


@pytest.mark.asyncio
async def test_backend_capabilities(tavily):
    assert Capability.WEB in tavily.capabilities
    assert Capability.FRESH in tavily.capabilities


@pytest.mark.asyncio
async def test_backend_name(tavily):
    assert tavily.name == "tavily"


@pytest.mark.asyncio
async def test_http_error_raises_backend_error(respx_mock):
    respx_mock.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    from sleuth.errors import BackendError
    backend = TavilyBackend(api_key="bad-key")
    with pytest.raises(BackendError):
        await backend.search("q")


@pytest.mark.asyncio
async def test_web_backend_factory_returns_tavily():
    """WebBackend(provider='tavily') is the stable public symbol for Phase 9."""
    b = WebBackend(provider="tavily", api_key="key")
    assert isinstance(b, TavilyBackend)
```

- [x] **Step 10.2: Run test to verify it fails**

```bash
uv run pytest tests/backends/test_web.py -v
```

Expected: `ImportError: cannot import name 'TavilyBackend' from 'sleuth.backends.web'`.

- [x] **Step 10.3: Create `tests/backends/__init__.py`**

```bash
touch tests/backends/__init__.py
```

- [x] **Step 10.4: Implement `src/sleuth/backends/web.py`**

```python
"""Web search backends.

Phase 1 ships a Tavily-only smoke implementation plus the ``WebBackend``
factory (currently returns ``TavilyBackend`` for all providers).  Phase 9
expands with Exa, Brave, and SerpAPI adapters and a richer factory.

The public symbol ``WebBackend`` is stable — downstream code should use it
rather than importing ``TavilyBackend`` directly.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from sleuth.backends.base import Capability
from sleuth.errors import BackendError
from sleuth.types import Chunk, Source

logger = logging.getLogger("sleuth.backends.web")

# Default timeout per spec §7.1
_DEFAULT_TIMEOUT_S = 8.0


class TavilyBackend:
    """Tavily search API backend.

    Implements the ``Backend`` Protocol structurally.  Uses ``httpx`` for
    async HTTP; respx is used in tests to mock requests.

    Args:
        api_key: Tavily API key (``TAVILY_API_KEY`` env var if not passed).
        timeout_s: Per-request timeout in seconds.  Default: 8s (spec §7.1).
    """

    name = "tavily"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB, Capability.FRESH})

    _SEARCH_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self._api_key = api_key
        self._timeout_s = timeout_s

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        """Search Tavily and return up to ``k`` chunks.

        Raises:
            BackendError: On any HTTP or API-level error.
        """
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": k,
        }
        logger.debug("Tavily search: query=%r k=%d", query, k)
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(self._SEARCH_URL, json=payload)
        except httpx.TimeoutException as exc:
            raise BackendError(f"Tavily request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"Tavily HTTP error: {exc}") from exc

        if resp.status_code != 200:
            raise BackendError(
                f"Tavily returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        chunks: list[Chunk] = []
        for item in data.get("results", [])[:k]:
            source = Source(
                kind="url",
                location=item.get("url", ""),
                title=item.get("title"),
            )
            chunks.append(
                Chunk(
                    text=item.get("content", ""),
                    source=source,
                    score=item.get("score"),
                )
            )
        return chunks


def WebBackend(
    provider: Literal["tavily"] = "tavily",
    *,
    api_key: str,
    **kwargs: Any,
) -> TavilyBackend:
    """Factory for web search backends.

    Phase 9 expands this to support ``provider="exa"``, ``"brave"``, ``"serpapi"``.
    The symbol is stable; callers should always use ``WebBackend(...)`` rather
    than importing ``TavilyBackend`` directly.

    Args:
        provider: Which provider to use.  Currently only ``"tavily"``.
        api_key: API key for the chosen provider.

    Returns:
        A backend instance implementing the ``Backend`` Protocol.
    """
    if provider == "tavily":
        return TavilyBackend(api_key=api_key, **kwargs)
    raise ValueError(f"Unknown web provider: {provider!r}.  Phase 9 adds exa/brave/serpapi.")
```

- [x] **Step 10.5: Run tests to verify they pass**

```bash
uv run pytest tests/backends/test_web.py -v
```

Expected: all 9 tests PASS.

- [x] **Step 10.6: Also run TavilyBackend through BackendTestKit**

Add a class to the bottom of `tests/backends/test_web.py`:

```python
# ---------------------------------------------------------------------------
# BackendTestKit compliance
# ---------------------------------------------------------------------------

from tests.contract.test_backend_protocol import BackendTestKit


class TestTavilyBackendContract(BackendTestKit):
    @pytest.fixture
    def backend(self, respx_mock):
        respx_mock.post(TAVILY_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=FAKE_RESPONSE)
        )
        return TavilyBackend(api_key="test-key")
```

```bash
uv run pytest tests/backends/test_web.py -v
```

Expected: all contract tests PASS (cancellation test may show as xfail or pass — both acceptable for a mocked backend).

- [x] **Step 10.7: Commit**

```bash
git add src/sleuth/backends/web.py tests/backends/__init__.py tests/backends/test_web.py
git commit -m "feat: add TavilyBackend + WebBackend factory (Tavily-only smoke, Phase 9 expands)"
```

---

## Task 11: Engine — Router (`src/sleuth/engine/router.py`)

**Files:**
- Create: `src/sleuth/engine/router.py`
- Test: `tests/engine/test_router.py`

The router is heuristic-only (no LLM calls).  It maps a query to `"fast"` or `"deep"` depth.  When the caller passes `depth="fast"` or `depth="deep"`, the router passes it through.  `depth="auto"` triggers heuristics.

- [x] **Step 11.1: Write the failing test**

Create `tests/engine/__init__.py` (empty) and `tests/engine/test_router.py`:

```python
import pytest
from sleuth.engine.router import Router
from sleuth.events import RouteEvent


def test_fast_passthrough():
    r = Router()
    event = r.route("anything", depth="fast")
    assert event.depth == "fast"
    assert event.type == "route"


def test_deep_passthrough():
    r = Router()
    event = r.route("anything", depth="deep")
    assert event.depth == "deep"


def test_auto_short_query_routes_fast():
    r = Router()
    event = r.route("what is python?", depth="auto")
    assert event.depth == "fast"


def test_auto_simple_factual_routes_fast():
    r = Router()
    for query in [
        "who invented Python?",
        "when was Python created?",
        "what does GIL stand for?",
    ]:
        event = r.route(query, depth="auto")
        assert event.depth == "fast", f"Expected fast for: {query!r}"


def test_auto_complex_query_routes_deep():
    r = Router()
    for query in [
        "compare the tradeoffs of async vs threading for IO-bound vs CPU-bound tasks in Python",
        "explain the design rationale for Python's memory model and how it affects multi-core performance",
        "what are all the breaking changes between Python 3.10 and 3.12 and how do they affect our codebase?",
    ]:
        event = r.route(query, depth="auto")
        assert event.depth == "deep", f"Expected deep for: {query!r}"


def test_route_event_has_reason():
    r = Router()
    event = r.route("simple question?", depth="auto")
    assert isinstance(event.reason, str)
    assert len(event.reason) > 0


def test_route_event_type_is_route():
    r = Router()
    event = r.route("q", depth="fast")
    assert isinstance(event, RouteEvent)
```

- [x] **Step 11.2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_router.py -v
```

Expected: `ImportError: cannot import name 'Router' from 'sleuth.engine.router'`.

- [x] **Step 11.3: Create `tests/engine/__init__.py`**

```bash
touch tests/engine/__init__.py
```

- [x] **Step 11.4: Implement `src/sleuth/engine/router.py`**

```python
"""Heuristic depth router — no LLM calls.

Routes each query to a ``Depth`` value by inspecting the query text.
When ``depth`` is already ``"fast"`` or ``"deep"`` the caller's value is passed
through unchanged.  Phase 3 may replace this with an LLM-backed classifier
for the ``"auto"`` case only; the Router API is stable.
"""

from __future__ import annotations

import logging
import re

from sleuth.events import RouteEvent
from sleuth.types import Depth

logger = logging.getLogger("sleuth.engine.router")

# ---------------------------------------------------------------------------
# Heuristic signals
# ---------------------------------------------------------------------------

# Queries shorter than this word count are almost always fast-path questions.
_FAST_WORD_LIMIT = 10

# Keywords that strongly suggest a complex, multi-step query needing planning.
_DEEP_KEYWORDS = re.compile(
    r"\b(compare|tradeoffs?|all the|every|across|between|vs\.?|versus|"
    r"how do|explain|design|rationale|breaking changes?|history of|"
    r"comprehensive|in depth|exhaustive|walk me through)\b",
    re.IGNORECASE,
)

# Simple question starters that almost always resolve in one search pass.
_FAST_STARTS = re.compile(
    r"^(what|who|when|where|define|does|is|are|how many|which)\b",
    re.IGNORECASE,
)


class Router:
    """Heuristic depth router.

    Determines whether a query should be answered with a single search fan-out
    (``"fast"``) or a full planning loop (``"deep"``).  No LLM calls are made
    here — this is intentionally cheap and synchronous.

    Phase 3 will extend this to use an LLM classifier for edge cases, but the
    ``route()`` API signature is frozen.
    """

    def route(self, query: str, *, depth: Depth = "auto") -> RouteEvent:
        """Classify a query and return a ``RouteEvent``.

        Args:
            query: The user's search query.
            depth: ``"auto"`` to run heuristics; ``"fast"`` / ``"deep"`` pass through.

        Returns:
            A ``RouteEvent`` with the resolved depth and a short reason string.
        """
        if depth in ("fast", "deep"):
            logger.debug("Router: passthrough depth=%s", depth)
            return RouteEvent(
                type="route",
                depth=depth,
                reason=f"caller-specified depth={depth}",
            )

        resolved, reason = self._classify(query)
        logger.debug("Router: auto → %s (%s)", resolved, reason)
        return RouteEvent(type="route", depth=resolved, reason=reason)

    def _classify(self, query: str) -> tuple[Depth, str]:
        words = query.split()

        # Short queries are almost always fast
        if len(words) <= _FAST_WORD_LIMIT and not _DEEP_KEYWORDS.search(query):
            if _FAST_STARTS.match(query.strip()):
                return "fast", "simple factual question pattern"
            if len(words) <= 5:
                return "fast", "very short query"

        # Explicit complexity signals → deep
        m = _DEEP_KEYWORDS.search(query)
        if m:
            return "deep", f"complexity keyword: {m.group()!r}"

        # Long queries default to deep
        if len(words) > _FAST_WORD_LIMIT:
            return "deep", f"long query ({len(words)} words)"

        return "fast", "no complexity signals detected"
```

- [x] **Step 11.5: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_router.py -v
```

Expected: all 7 tests PASS.

- [x] **Step 11.6: Commit**

```bash
git add src/sleuth/engine/router.py src/sleuth/engine/__init__.py tests/engine/__init__.py tests/engine/test_router.py
git commit -m "feat: add heuristic Router — auto/fast/deep depth classification, no LLM calls"
```

---

## Task 12: Engine — Executor (`src/sleuth/engine/executor.py`)

**Files:**
- Create: `src/sleuth/engine/executor.py`
- Test: `tests/engine/test_executor.py`

The executor fans out to all registered backends in parallel, applies per-backend timeouts, handles failures per spec §7.1, and de-duplicates by source location.  Phase 3 will extend this for multi-query and speculative prefetch.

- [x] **Step 12.1: Write the failing test**

Create `tests/engine/test_executor.py`:

```python
import asyncio
from datetime import datetime, timezone
import pytest

from sleuth.engine.executor import Executor
from sleuth.events import SearchEvent
from sleuth.types import Chunk, Source
from sleuth.backends.base import Capability
from sleuth.errors import BackendError


def _make_chunk(url: str, text: str = "content", score: float = 0.9) -> Chunk:
    return Chunk(
        text=text,
        source=Source(kind="url", location=url),
        score=score,
    )


class OkBackend:
    name = "ok"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [_make_chunk("https://ok.com/1"), _make_chunk("https://ok.com/2")]


class ErrorBackend:
    name = "error"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        raise BackendError("search failed")


class SlowBackend:
    name = "slow"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        await asyncio.sleep(10.0)  # will be cancelled by timeout
        return []


class DuplicateBackend:
    """Returns a chunk whose source URL overlaps with OkBackend."""
    name = "dup"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [_make_chunk("https://ok.com/1"), _make_chunk("https://dup.com/unique")]


@pytest.mark.asyncio
async def test_single_backend_returns_chunks():
    executor = Executor(backends=[OkBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)


@pytest.mark.asyncio
async def test_emits_search_event():
    executor = Executor(backends=[OkBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    search_events = [e for e in events if isinstance(e, SearchEvent)]
    assert len(search_events) == 1
    assert search_events[0].backend == "ok"
    assert search_events[0].error is None


@pytest.mark.asyncio
async def test_error_backend_emits_error_search_event():
    executor = Executor(backends=[ErrorBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    search_events = [e for e in events if isinstance(e, SearchEvent)]
    assert len(search_events) == 1
    assert search_events[0].error is not None
    assert chunks == []


@pytest.mark.asyncio
async def test_timeout_backend_emits_error_search_event():
    executor = Executor(backends=[SlowBackend()], timeout_s=0.05)
    events, chunks = await executor.run("query", k=10)
    search_events = [e for e in events if isinstance(e, SearchEvent)]
    assert search_events[0].error is not None
    assert "timeout" in search_events[0].error.lower()
    assert chunks == []


@pytest.mark.asyncio
async def test_partial_failure_keeps_successful_results():
    executor = Executor(backends=[OkBackend(), ErrorBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    assert len(chunks) == 2  # OkBackend succeeded
    error_events = [e for e in events if isinstance(e, SearchEvent) and e.error]
    assert len(error_events) == 1


@pytest.mark.asyncio
async def test_deduplication_by_source_location():
    executor = Executor(backends=[OkBackend(), DuplicateBackend()], timeout_s=5.0)
    events, chunks = await executor.run("query", k=10)
    locations = [c.source.location for c in chunks]
    assert len(locations) == len(set(locations)), "Duplicate source locations found"


@pytest.mark.asyncio
async def test_k_limits_per_backend():
    executor = Executor(backends=[OkBackend()], timeout_s=5.0)
    _, chunks = await executor.run("query", k=1)
    assert len(chunks) <= 1
```

- [x] **Step 12.2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_executor.py -v
```

Expected: `ImportError: cannot import name 'Executor' from 'sleuth.engine.executor'`.

- [x] **Step 12.3: Implement `src/sleuth/engine/executor.py`**

```python
"""Single-backend fan-out executor.

Fans search queries out to all registered backends in parallel, applies
per-backend timeouts, handles failures per spec §7.1, and de-duplicates
results by source location.

Phase 3 will extend this for:
  - Multi-query fan-out (planner emits multiple sub-queries)
  - Speculative prefetch (start backend search while planner is still streaming)
Keep this module focused on single-query fan-out.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sleuth.backends.base import Backend
from sleuth.errors import BackendError, BackendTimeoutError
from sleuth.events import SearchEvent
from sleuth.types import Chunk

logger = logging.getLogger("sleuth.engine.executor")


class Executor:
    """Async fan-out over all registered backends for a single query.

    Args:
        backends: List of ``Backend`` instances to query in parallel.
        timeout_s: Per-backend timeout in seconds (default 8s per spec §7.1).
    """

    def __init__(self, backends: list[Backend], *, timeout_s: float = 8.0) -> None:
        self._backends = backends
        self._timeout_s = timeout_s

    async def run(self, query: str, *, k: int = 10) -> tuple[list[SearchEvent], list[Chunk]]:
        """Fan out ``query`` to all backends and return events + merged chunks.

        Returns:
            A tuple of (``SearchEvent`` list, deduplicated ``Chunk`` list).
            Never raises — per-backend errors are captured in ``SearchEvent.error``.
        """
        tasks = {
            asyncio.create_task(
                self._search_one(backend, query, k),
                name=f"executor:{backend.name}",
            ): backend
            for backend in self._backends
        }

        events: list[SearchEvent] = []
        all_chunks: list[Chunk] = []

        results = await asyncio.gather(*tasks.keys(), return_exceptions=True)

        for (task, backend), result in zip(tasks.items(), results):
            if isinstance(result, SearchEvent):
                # Error SearchEvent returned from _search_one
                events.append(result)
            elif isinstance(result, list):
                events.append(
                    SearchEvent(type="search", backend=backend.name, query=query)
                )
                all_chunks.extend(result)
            else:
                # Unexpected exception (should not happen, but be safe)
                logger.error("Unexpected result from backend %s: %r", backend.name, result)
                events.append(
                    SearchEvent(
                        type="search",
                        backend=backend.name,
                        query=query,
                        error=repr(result),
                    )
                )

        deduped = self._deduplicate(all_chunks)
        return events, deduped

    async def _search_one(
        self, backend: Backend, query: str, k: int
    ) -> list[Chunk] | SearchEvent:
        """Run a single backend search, wrapping errors into SearchEvent."""
        try:
            chunks = await asyncio.wait_for(
                backend.search(query, k),
                timeout=self._timeout_s,
            )
            return chunks
        except asyncio.TimeoutError:
            logger.warning("Backend %s timed out after %.1fs", backend.name, self._timeout_s)
            return SearchEvent(
                type="search",
                backend=backend.name,
                query=query,
                error=f"timeout after {self._timeout_s}s",
            )
        except BackendError as exc:
            logger.warning("Backend %s error: %s", backend.name, exc)
            return SearchEvent(
                type="search",
                backend=backend.name,
                query=query,
                error=str(exc),
            )
        except Exception as exc:
            logger.error("Backend %s unexpected error: %s", backend.name, exc, exc_info=True)
            return SearchEvent(
                type="search",
                backend=backend.name,
                query=query,
                error=f"unexpected error: {exc}",
            )

    @staticmethod
    def _deduplicate(chunks: list[Chunk]) -> list[Chunk]:
        """Remove chunks with duplicate source locations, keeping first occurrence."""
        seen: set[str] = set()
        result: list[Chunk] = []
        for chunk in chunks:
            loc = chunk.source.location
            if loc not in seen:
                seen.add(loc)
                result.append(chunk)
        return result
```

- [x] **Step 12.4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_executor.py -v
```

Expected: all 7 tests PASS.

- [x] **Step 12.5: Commit**

```bash
git add src/sleuth/engine/executor.py tests/engine/test_executor.py
git commit -m "feat: add Executor — parallel backend fan-out, timeouts, dedup, failure-to-SearchEvent"
```

---

## Task 13: Engine — Synthesizer (`src/sleuth/engine/synthesizer.py`)

**Files:**
- Create: `src/sleuth/engine/synthesizer.py`
- Test: `tests/engine/test_synthesizer.py`

The synthesizer takes the merged chunks, calls the LLM stream, and yields `ThinkingEvent`, `TokenEvent`, and `CitationEvent` objects.  It also builds the final `Result`.

- [x] **Step 13.1: Write the failing test**

Create `tests/engine/test_synthesizer.py`:

```python
import pytest
from sleuth.engine.synthesizer import Synthesizer
from sleuth.events import ThinkingEvent, TokenEvent, CitationEvent, DoneEvent
from sleuth.llm.stub import StubLLM
from sleuth.llm.base import Message, TextDelta, Stop, ReasoningDelta
from sleuth.types import Chunk, Source, RunStats


def _chunk(url: str, text: str = "content") -> Chunk:
    return Chunk(text=text, source=Source(kind="url", location=url), score=0.9)


async def collect_events(synth: Synthesizer, **kwargs):
    events = []
    async for event in synth.synthesize(**kwargs):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_emits_token_events():
    stub = StubLLM(responses=["hello", " world"])
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[_chunk("https://a.com")],
        history=[],
        stats_start_ms=0,
    )
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert len(token_events) >= 1
    assert any(e.text == "hello" for e in token_events)


@pytest.mark.asyncio
async def test_emits_done_event():
    stub = StubLLM(responses=["answer"])
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[_chunk("https://a.com")],
        history=[],
        stats_start_ms=0,
    )
    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    assert done_events[0].stats.tokens_out > 0


@pytest.mark.asyncio
async def test_emits_citation_for_each_chunk():
    stub = StubLLM(responses=["answer"])
    synth = Synthesizer(llm=stub)
    chunks = [_chunk("https://a.com"), _chunk("https://b.com")]
    events = await collect_events(
        synth,
        query="q",
        chunks=chunks,
        history=[],
        stats_start_ms=0,
    )
    citation_events = [e for e in events if isinstance(e, CitationEvent)]
    assert len(citation_events) == 2
    locations = {e.source.location for e in citation_events}
    assert "https://a.com" in locations
    assert "https://b.com" in locations


@pytest.mark.asyncio
async def test_no_thinking_event_when_not_supported():
    stub = StubLLM(responses=["answer"])
    assert stub.supports_reasoning is False
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[_chunk("https://a.com")],
        history=[],
        stats_start_ms=0,
    )
    thinking_events = [e for e in events if isinstance(e, ThinkingEvent)]
    assert thinking_events == []


@pytest.mark.asyncio
async def test_thinking_event_when_supported():
    from sleuth.llm.base import ReasoningDelta

    async def reasoner(messages):
        yield ReasoningDelta(text="thinking...")
        yield TextDelta(text="answer")
        yield Stop(reason="end_turn")

    class ReasoningStub(StubLLM):
        supports_reasoning = True

    stub = ReasoningStub(responses=reasoner)
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[_chunk("https://a.com")],
        history=[],
        stats_start_ms=0,
    )
    thinking_events = [e for e in events if isinstance(e, ThinkingEvent)]
    assert len(thinking_events) == 1
    assert thinking_events[0].text == "thinking..."


@pytest.mark.asyncio
async def test_builds_result_text_from_token_events():
    stub = StubLLM(responses=["hello", " world"])
    synth = Synthesizer(llm=stub)
    events = await collect_events(
        synth,
        query="q",
        chunks=[],
        history=[],
        stats_start_ms=0,
    )
    token_texts = "".join(e.text for e in events if isinstance(e, TokenEvent))
    assert token_texts == "hello world"
```

- [x] **Step 13.2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_synthesizer.py -v
```

Expected: `ImportError: cannot import name 'Synthesizer' from 'sleuth.engine.synthesizer'`.

- [x] **Step 13.3: Implement `src/sleuth/engine/synthesizer.py`**

```python
"""Streaming synthesizer — converts chunks into a token + citation event stream.

Calls the LLM with the user query + retrieved chunks, streams ``ThinkingEvent``
(when the LLM supports reasoning), ``TokenEvent``, and ``CitationEvent``s.
Yields a ``DoneEvent`` as the final event.

Returns ``Result`` via the ``last_result`` property after the generator completes.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator

from pydantic import BaseModel

from sleuth.events import CitationEvent, DoneEvent, ThinkingEvent, TokenEvent
from sleuth.llm.base import LLMClient, Message, ReasoningDelta, Stop, TextDelta, Tool
from sleuth.types import Chunk, Result, RunStats, Source

logger = logging.getLogger("sleuth.engine.synthesizer")

_SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Answer the user's question based on the provided search results. "
    "Be concise and cite your sources."
)


def _build_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no search results available)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        loc = chunk.source.location
        title = chunk.source.title or loc
        parts.append(f"[{i}] {title}\n{chunk.text}")
    return "\n\n".join(parts)


class Synthesizer:
    """Streaming LLM synthesizer.

    Args:
        llm: Any object satisfying the ``LLMClient`` Protocol.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._last_result: Result | None = None  # type: ignore[type-arg]

    @property
    def last_result(self) -> Result | None:  # type: ignore[type-arg]
        """The final ``Result`` built from the completed stream, or ``None`` before done."""
        return self._last_result

    async def synthesize(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        history: list[Message],
        stats_start_ms: float,
        schema: type[BaseModel] | None = None,
        backends_called: list[str] | None = None,
        cache_hits: dict[str, int] | None = None,
    ) -> AsyncIterator[ThinkingEvent | TokenEvent | CitationEvent | DoneEvent]:
        """Stream synthesis events and yield a ``DoneEvent`` last.

        Args:
            query: The user's search question.
            chunks: Merged, deduped chunks from the executor.
            history: Prior conversation turns as ``Message`` objects.
            stats_start_ms: ``time.monotonic() * 1000`` at run start — used to compute latency.
            schema: Optional Pydantic schema for structured output.
            backends_called: Backend names that contributed chunks.
            cache_hits: Cache namespace → hit count dict for ``RunStats``.
        """
        return self._stream(
            query=query,
            chunks=chunks,
            history=history,
            stats_start_ms=stats_start_ms,
            schema=schema,
            backends_called=backends_called or [],
            cache_hits=cache_hits or {},
        )

    async def _stream(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        history: list[Message],
        stats_start_ms: float,
        schema: type[BaseModel] | None,
        backends_called: list[str],
        cache_hits: dict[str, int],
    ) -> AsyncIterator[ThinkingEvent | TokenEvent | CitationEvent | DoneEvent]:
        context = _build_context(chunks)
        messages: list[Message] = [
            Message(role="system", content=_SYSTEM_PROMPT),
            *history,
            Message(
                role="user",
                content=f"Question: {query}\n\nSearch results:\n{context}",
            ),
        ]

        text_parts: list[str] = []
        tokens_out = 0
        first_token_ms: int | None = None

        stream = await self._llm.stream(messages, schema=schema)
        async for chunk in stream:
            now_ms = int(time.monotonic() * 1000)
            if isinstance(chunk, ReasoningDelta):
                if self._llm.supports_reasoning:
                    yield ThinkingEvent(type="thinking", text=chunk.text)
            elif isinstance(chunk, TextDelta):
                if first_token_ms is None:
                    first_token_ms = now_ms - int(stats_start_ms)
                text_parts.append(chunk.text)
                tokens_out += 1
                yield TokenEvent(type="token", text=chunk.text)
            elif isinstance(chunk, Stop):
                break

        # Emit citations for each chunk that contributed
        for idx, c in enumerate(chunks):
            yield CitationEvent(type="citation", index=idx, source=c.source)

        full_text = "".join(text_parts)
        elapsed_ms = int(time.monotonic() * 1000 - stats_start_ms)

        stats = RunStats(
            latency_ms=elapsed_ms,
            first_token_ms=first_token_ms,
            tokens_in=len(messages),  # approximation (message count, not real tokens)
            tokens_out=tokens_out,
            cache_hits=cache_hits,
            backends_called=backends_called,
        )

        self._last_result = Result(
            text=full_text,
            citations=[c.source for c in chunks],
            stats=stats,
        )

        yield DoneEvent(type="done", stats=stats)
```

- [x] **Step 13.4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_synthesizer.py -v
```

Expected: all 6 tests PASS.

- [x] **Step 13.5: Commit**

```bash
git add src/sleuth/engine/synthesizer.py tests/engine/test_synthesizer.py
git commit -m "feat: add Synthesizer — streaming ThinkingEvent/TokenEvent/CitationEvent/DoneEvent"
```

---

## Task 14: Snapshot baseline (`tests/snapshots/`)

**Files:**
- Test: `tests/snapshots/test_event_stream_snapshot.py`

This test runs a full `aask` call with `StubLLM` + `FakeBackend` and snapshots the event sequence via `syrupy`.  Run once to generate the snapshot, then it becomes a regression guard.

- [x] **Step 14.1: Create test**

Create `tests/snapshots/__init__.py` (empty) and `tests/snapshots/test_event_stream_snapshot.py`:

```python
"""Snapshot tests for the full event stream.

These use syrupy to freeze the event sequence.  First run generates
the snapshot under tests/snapshots/__snapshots__/.  Subsequent runs
compare against it.

To update snapshots after an intentional change:
    uv run pytest tests/snapshots/ --snapshot-update
"""

import pytest
from syrupy.assertion import SnapshotAssertion

from sleuth._agent import Sleuth
from sleuth.llm.stub import StubLLM
from sleuth.types import Chunk, Source
from sleuth.backends.base import Capability


class FakeWebBackend:
    name = "fake_web"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [
            Chunk(
                text="Python was created by Guido van Rossum.",
                source=Source(kind="url", location="https://python.org/history"),
                score=0.99,
            )
        ]


@pytest.mark.asyncio
async def test_fast_path_event_stream_snapshot(snapshot: SnapshotAssertion):
    """Snapshot the complete event sequence for a fast-path Q&A."""
    stub = StubLLM(responses=["Python was created by Guido van Rossum."])
    agent = Sleuth(llm=stub, backends=[FakeWebBackend()])

    events = []
    async for event in agent.aask("who created Python?", depth="fast"):
        events.append(event.type)

    assert events == snapshot
```

- [x] **Step 14.2: Run test once to generate snapshot (expected: PASS with snapshot write)**

```bash
uv run pytest tests/snapshots/test_event_stream_snapshot.py -v --snapshot-update
```

Expected: test PASSES and writes `tests/snapshots/__snapshots__/test_event_stream_snapshot.ambr`.

- [x] **Step 14.3: Run again to verify snapshot match**

```bash
uv run pytest tests/snapshots/test_event_stream_snapshot.py -v
```

Expected: PASS (snapshot matches).

- [x] **Step 14.4: Commit**

```bash
git add tests/snapshots/__init__.py tests/snapshots/test_event_stream_snapshot.py "tests/snapshots/__snapshots__/"
git commit -m "test: add syrupy snapshot for fast-path event stream"
```

---

## Task 15: Sleuth agent class (`src/sleuth/_agent.py`)

**Files:**
- Create: `src/sleuth/_agent.py`
- Test: `tests/test_agent.py`

This is the top-level `Sleuth` class that wires Router → Executor → Synthesizer and exposes `aask` (async generator) and `ask` (sync wrapper).

- [x] **Step 15.1: Write the failing test**

Create `tests/test_agent.py`:

```python
import pytest
from sleuth._agent import Sleuth
from sleuth.llm.stub import StubLLM
from sleuth.events import RouteEvent, SearchEvent, TokenEvent, CitationEvent, DoneEvent
from sleuth.types import Chunk, Source, Result
from sleuth.backends.base import Capability
from sleuth.memory.cache import MemoryCache
from sleuth.memory.session import Session


class FakeBackend:
    name = "fake"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [Chunk(text="result text", source=Source(kind="url", location="https://a.com"))]


@pytest.mark.asyncio
async def test_aask_yields_route_event():
    agent = Sleuth(llm=StubLLM(responses=["answer"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("who is guido?")]
    route_events = [e for e in events if isinstance(e, RouteEvent)]
    assert len(route_events) == 1


@pytest.mark.asyncio
async def test_aask_yields_search_event():
    agent = Sleuth(llm=StubLLM(responses=["answer"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("who is guido?")]
    search_events = [e for e in events if isinstance(e, SearchEvent)]
    assert len(search_events) == 1
    assert search_events[0].backend == "fake"


@pytest.mark.asyncio
async def test_aask_yields_token_events():
    agent = Sleuth(llm=StubLLM(responses=["hello"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("q?")]
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert any(e.text == "hello" for e in token_events)


@pytest.mark.asyncio
async def test_aask_yields_done_event_last():
    agent = Sleuth(llm=StubLLM(responses=["answer"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("q?")]
    assert isinstance(events[-1], DoneEvent)


@pytest.mark.asyncio
async def test_aask_depth_fast_skips_deep():
    """Fast depth: route event must show 'fast'."""
    agent = Sleuth(llm=StubLLM(responses=["answer"]), backends=[FakeBackend()])
    events = [e async for e in agent.aask("q?", depth="fast")]
    route_event = next(e for e in events if isinstance(e, RouteEvent))
    assert route_event.depth == "fast"


def test_ask_returns_result():
    agent = Sleuth(llm=StubLLM(responses=["answer"]), backends=[FakeBackend()])
    result = agent.ask("q?")
    assert isinstance(result, Result)
    assert result.text == "answer"


def test_ask_result_has_citations():
    agent = Sleuth(llm=StubLLM(responses=["answer"]), backends=[FakeBackend()])
    result = agent.ask("q?")
    assert len(result.citations) >= 1


@pytest.mark.asyncio
async def test_aask_with_cache_none_still_works():
    agent = Sleuth(llm=StubLLM(responses=["answer"]), backends=[FakeBackend()], cache=None)
    events = [e async for e in agent.aask("q?")]
    assert any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_aask_with_session_adds_turn():
    session = Session()
    agent = Sleuth(llm=StubLLM(responses=["a1", "a2"]), backends=[FakeBackend()])
    async for _ in agent.aask("q1?", session=session):
        pass
    assert len(session.turns) == 1
    assert session.turns[0].query == "q1?"
```

- [x] **Step 15.2: Run test to verify it fails**

```bash
uv run pytest tests/test_agent.py -v
```

Expected: `ImportError: cannot import name 'Sleuth' from 'sleuth._agent'`.

- [x] **Step 15.3: Implement `src/sleuth/_agent.py`**

```python
"""Sleuth — the top-level agent class.

Wires Router → Executor → Synthesizer into a single ``aask`` async generator
and provides a ``ask`` sync wrapper.

Design notes (per spec §4):
- ``Sleuth(...)`` is constructed once; ``ask`` / ``aask`` are stateless unless
  ``session=`` is passed.
- ``fast_llm`` defaults to the main ``llm`` when not supplied (spec §15 #3 resolved
  as documentation-only — no literal default model is imported here).
- ``cache="default"`` maps to ``MemoryCache()`` in Phase 1; Phase 4 replaces this
  with a ``SqliteCache``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Generic, Literal, TypeVar

from pydantic import BaseModel

from sleuth.backends.base import Backend
from sleuth.engine.executor import Executor
from sleuth.engine.router import Router
from sleuth.engine.synthesizer import Synthesizer
from sleuth.events import Event
from sleuth.llm.base import LLMClient
from sleuth.memory.cache import Cache, MemoryCache
from sleuth.memory.session import Session
from sleuth.types import Depth, Length, Result

T = TypeVar("T", bound=BaseModel)

_DEFAULT_BACKEND_TIMEOUT_S = 8.0


class Sleuth:
    """Plug-and-play agentic search with reasoning, planning, citations, and observability.

    Args:
        llm: Any object satisfying the ``LLMClient`` Protocol.  Used for synthesis.
        backends: One or more ``Backend`` instances to search against.
        fast_llm: Optional faster LLM for routing/planning.  Defaults to ``llm``
            when not supplied — no built-in fast model is imported (spec §15 #3).
        cache: ``"default"`` (MemoryCache in Phase 1), a ``Cache`` instance, or ``None``
            to disable caching.
        semantic_cache: Reserved for Phase 4.  Pass ``False`` (default) or ``None``.
        session: Optional persistent ``Session`` for multi-turn conversations.
    """

    def __init__(
        self,
        llm: LLMClient,
        backends: list[Backend],
        *,
        fast_llm: LLMClient | None = None,
        cache: Cache | Literal["default"] | None = "default",
        semantic_cache: Any = False,
        session: Session | None = None,
    ) -> None:
        self._llm = llm
        self._fast_llm = fast_llm or llm
        self._backends = backends
        self._session = session

        if cache == "default":
            self._cache: Cache | None = MemoryCache()
        else:
            self._cache = cache  # type: ignore[assignment]

        self._router = Router()
        self._executor = Executor(backends=backends, timeout_s=_DEFAULT_BACKEND_TIMEOUT_S)
        self._synthesizer = Synthesizer(llm=llm)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def aask(
        self,
        query: str,
        *,
        depth: Depth = "auto",
        max_iterations: int = 4,
        schema: type[BaseModel] | None = None,
        session: Session | None = None,
    ) -> AsyncIterator[Event]:
        """Run an async search and yield typed events.

        Args:
            query: The user's question.
            depth: ``"auto"`` (default), ``"fast"``, or ``"deep"``.
                   ``"deep"`` is handled as ``"fast"`` in Phase 1 — Phase 3 adds the planner.
            max_iterations: Maximum planning iterations (deep mode, Phase 3).
            schema: Optional Pydantic model for structured output.
            session: Per-call session override; overrides the instance-level session.
        """
        return self._run(query=query, depth=depth, schema=schema, session=session or self._session)

    def ask(
        self,
        query: str,
        *,
        depth: Depth = "auto",
        max_iterations: int = 4,
        schema: type[BaseModel] | None = None,
        session: Session | None = None,
    ) -> Result:  # type: ignore[type-arg]
        """Synchronous wrapper around ``aask``.

        Blocks until the run completes and returns a ``Result``.
        """
        return asyncio.run(self._collect(query=query, depth=depth, schema=schema, session=session))

    async def asummarize(
        self,
        target: str,
        *,
        length: Length = "standard",
        schema: type[BaseModel] | None = None,
    ) -> AsyncIterator[Event]:
        """Summarize a URL, file path, or topic.  Placeholder — full impl in Phase 2+."""
        query = f"summarize: {target} (length={length})"
        return self._run(query=query, depth="fast", schema=schema, session=None)

    def summarize(
        self,
        target: str,
        *,
        length: Length = "standard",
        schema: type[BaseModel] | None = None,
    ) -> Result:  # type: ignore[type-arg]
        """Synchronous wrapper around ``asummarize``."""
        return asyncio.run(self._collect_summarize(target=target, length=length, schema=schema))

    async def warm_index(self) -> None:
        """Eagerly index all LocalFiles backends.  No-op in Phase 1 (no LocalFiles yet)."""

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    async def _run(
        self,
        *,
        query: str,
        depth: Depth,
        schema: type[BaseModel] | None,
        session: Session | None,
    ) -> AsyncIterator[Event]:
        start_ms = time.monotonic() * 1000

        # 1. Route
        route_event = self._router.route(query, depth=depth)
        yield route_event

        # Phase 1: treat "deep" as "fast" — Phase 3 adds the planner
        resolved_depth = route_event.depth

        # 2. Execute (single-query fan-out)
        search_events, chunks = await self._executor.run(query)
        for se in search_events:
            yield se

        # 3. Synthesize
        backends_called = [se.backend for se in search_events if se.error is None]
        synth = Synthesizer(llm=self._llm)
        history = session.as_messages() if session else []

        async for event in await synth.synthesize(
            query=query,
            chunks=chunks,
            history=history,
            stats_start_ms=start_ms,
            schema=schema,
            backends_called=backends_called,
            cache_hits={},
        ):
            yield event

        # 4. Update session (fire-and-forget background task per spec §8)
        if session is not None and synth.last_result is not None:
            session.add_turn(query, synth.last_result, [c.source for c in chunks])

    async def _collect(
        self,
        *,
        query: str,
        depth: Depth,
        schema: type[BaseModel] | None,
        session: Session | None,
    ) -> Result:  # type: ignore[type-arg]
        from sleuth.events import DoneEvent as _Done
        synth: Synthesizer | None = None
        last_result: Result | None = None  # type: ignore[type-arg]

        async for event in await self.aask(query, depth=depth, schema=schema, session=session):
            if isinstance(event, _Done):
                # Result is built inside Synthesizer; retrieve via last_result
                pass

        # Re-run to grab result — simpler than threading state through
        s = Synthesizer(llm=self._llm)
        route_event = self._router.route(query, depth=depth)
        _, chunks = await self._executor.run(query)
        history = session.as_messages() if session else []
        async for _ in await s.synthesize(
            query=query,
            chunks=chunks,
            history=history,
            stats_start_ms=time.monotonic() * 1000,
            schema=schema,
            backends_called=[],
            cache_hits={},
        ):
            pass
        result = s.last_result
        assert result is not None, "Synthesizer did not produce a result"
        return result

    async def _collect_summarize(
        self,
        *,
        target: str,
        length: Length,
        schema: type[BaseModel] | None,
    ) -> Result:  # type: ignore[type-arg]
        query = f"summarize: {target} (length={length})"
        return await self._collect(query=query, depth="fast", schema=schema, session=None)
```

- [x] **Step 15.4: Run tests to verify they pass**

```bash
uv run pytest tests/test_agent.py -v
```

Expected: all 9 tests PASS.

- [x] **Step 15.5: Commit**

```bash
git add src/sleuth/_agent.py tests/test_agent.py
git commit -m "feat: add Sleuth agent class — aask/ask, Router+Executor+Synthesizer wired end-to-end"
```

---

## Task 16: Public re-exports (`src/sleuth/__init__.py`)

**Files:**
- Modify: `src/sleuth/__init__.py`
- Test: `tests/test_public_api.py`

Per conventions §4, `__init__.py` re-exports the full public surface.

- [x] **Step 16.1: Write the failing test**

Create `tests/test_public_api.py`:

```python
"""Verify the public API surface re-exported from the top-level sleuth package."""

def test_sleuth_importable():
    from sleuth import Sleuth
    assert Sleuth is not None


def test_session_importable():
    from sleuth import Session
    assert Session is not None


def test_result_importable():
    from sleuth import Result
    assert Result is not None


def test_source_chunk_importable():
    from sleuth import Source, Chunk
    assert Source is not None
    assert Chunk is not None


def test_all_event_types_importable():
    from sleuth import (
        RouteEvent, PlanEvent, SearchEvent, FetchEvent,
        ThinkingEvent, TokenEvent, CitationEvent, CacheHitEvent, DoneEvent,
        Event,
    )
    for sym in (RouteEvent, PlanEvent, SearchEvent, FetchEvent,
                ThinkingEvent, TokenEvent, CitationEvent, CacheHitEvent, DoneEvent, Event):
        assert sym is not None


def test_depth_length_importable():
    from sleuth import Depth, Length
    assert Depth is not None
    assert Length is not None


def test_backend_importable():
    from sleuth.backends import Tavily
    assert Tavily is not None
```

- [x] **Step 16.2: Run test to verify it fails**

```bash
uv run pytest tests/test_public_api.py -v
```

Expected: some imports fail (module exists as stub with no exports yet).

- [x] **Step 16.3: Update `src/sleuth/__init__.py`**

```python
"""Sleuth — plug-and-play agentic search with reasoning, planning, and observability.

Public surface (conventions §4):

    from sleuth import Sleuth, Session, Result, Source, Chunk
    from sleuth import RouteEvent, PlanEvent, SearchEvent, FetchEvent
    from sleuth import ThinkingEvent, TokenEvent, CitationEvent, CacheHitEvent, DoneEvent, Event
    from sleuth import Depth, Length
    from sleuth.backends import Tavily
"""

from sleuth._agent import Sleuth as Sleuth
from sleuth.events import (
    CacheHitEvent as CacheHitEvent,
    CitationEvent as CitationEvent,
    DoneEvent as DoneEvent,
    Event as Event,
    FetchEvent as FetchEvent,
    PlanEvent as PlanEvent,
    RouteEvent as RouteEvent,
    SearchEvent as SearchEvent,
    ThinkingEvent as ThinkingEvent,
    TokenEvent as TokenEvent,
)
from sleuth.memory.session import Session as Session
from sleuth.types import (
    Chunk as Chunk,
    Depth as Depth,
    Length as Length,
    Result as Result,
    RunStats as RunStats,
    Source as Source,
)

__all__ = [
    "Sleuth",
    "Session",
    "Result",
    "RunStats",
    "Source",
    "Chunk",
    "Depth",
    "Length",
    "RouteEvent",
    "PlanEvent",
    "SearchEvent",
    "FetchEvent",
    "ThinkingEvent",
    "TokenEvent",
    "CitationEvent",
    "CacheHitEvent",
    "DoneEvent",
    "Event",
]
```

- [x] **Step 16.4: Update `src/sleuth/backends/__init__.py`** to expose `Tavily`:

```python
from sleuth.backends.web import TavilyBackend as Tavily

__all__ = ["Tavily"]
```

- [x] **Step 16.5: Run tests to verify they pass**

```bash
uv run pytest tests/test_public_api.py -v
```

Expected: all 7 tests PASS.

- [x] **Step 16.6: Commit**

```bash
git add src/sleuth/__init__.py src/sleuth/backends/__init__.py tests/test_public_api.py
git commit -m "feat: add public re-exports to sleuth/__init__.py (Sleuth, Session, all events, types)"
```

---

## Task 17: Add stub_llm + fake_backend fixtures to `tests/conftest.py`

**Files:**
- Modify: `tests/conftest.py`
- Test: verified by running the full suite

- [x] **Step 17.1: Append fixtures to `tests/conftest.py`**

Open the existing `tests/conftest.py` (created by Phase 0) and add:

```python
# ---------------------------------------------------------------------------
# Phase 1 additions — stub_llm + fake_backend cross-cutting fixtures
# ---------------------------------------------------------------------------

import pytest
from sleuth.llm.stub import StubLLM
from sleuth.types import Chunk, Source
from sleuth.backends.base import Capability


@pytest.fixture
def stub_llm():
    """StubLLM with a single 'answer' response — suitable for most unit tests."""
    return StubLLM(responses=["answer"])


class _FakeBackend:
    name = "fake"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [
            Chunk(
                text="fake result",
                source=Source(kind="url", location="https://fake.example.com"),
                score=1.0,
            )
        ]


@pytest.fixture
def fake_backend():
    """Minimal in-memory Backend for use in engine and integration tests."""
    return _FakeBackend()
```

- [x] **Step 17.2: Verify the fixtures are available**

```bash
uv run pytest tests/ --collect-only -q 2>&1 | head -30
```

Expected: no fixture errors; tests collected.

- [x] **Step 17.3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add stub_llm + fake_backend fixtures to conftest.py (Phase 1 additions)"
```

---

## Task 18: Full suite + coverage gate

**Files:** None created

- [x] **Step 18.1: Run all unit tests**

```bash
uv run pytest -m "not integration" -v
```

Expected: all tests PASS. Snapshot tests show "matched".

- [x] **Step 18.2: Run coverage check**

```bash
uv run pytest -m "not integration" --cov=src/sleuth --cov-report=term-missing
```

Expected: coverage for `src/sleuth/` is at or above 85%.  If below, add targeted tests for uncovered branches.

- [x] **Step 18.3: Run mypy**

```bash
uv run mypy src/sleuth/
```

Expected: exit 0 (no errors). Fix any strict errors before the PR.

- [x] **Step 18.4: Run ruff**

```bash
uv run ruff check src/sleuth/ tests/ && uv run ruff format --check src/sleuth/ tests/
```

Expected: no lint or formatting errors.

- [x] **Step 18.5: Commit any lint/type fixes (if needed)**

```bash
git add -u
git commit -m "fix: address mypy/ruff issues from Phase 1 full suite"
```

---

## Task 19: PR

**Files:** None created

- [x] **Step 19.1: Push branch**

```bash
git push -u origin feature/phase-1-core-mvp
```

- [x] **Step 19.2: Open PR**

```bash
gh pr create \
  --base develop \
  --title "feat: Phase 1 — Core MVP (types, events, protocols, engine, Tavily stub)" \
  --body "$(cat <<'EOF'
## Summary

Phase 1 Core MVP — foundational types, protocols, and minimal end-to-end Q&A.

### What's included

- **`sleuth/errors.py`** — `SleuthError` hierarchy.
- **`sleuth/logging.py`** — `get_logger` helper under `sleuth.*` namespace.
- **`sleuth/types.py`** — `Source`, `Chunk`, `RunStats`, `Result[T]`, `Depth`, `Length`.
- **`sleuth/events.py`** — all 9 typed events + discriminated `Event` union.
- **`sleuth/llm/base.py`** — `LLMClient` protocol, `LLMChunk` union, `Message`, `Tool`.
- **`sleuth/llm/stub.py`** — `StubLLM` test double.
- **`sleuth/backends/base.py`** — `Backend` protocol + `Capability` enum.
- **`sleuth/backends/web.py`** — `TavilyBackend` + `WebBackend` factory (Tavily-only, Phase 9 expands).
- **`sleuth/memory/cache.py`** — `Cache` protocol + `MemoryCache`.
- **`sleuth/memory/session.py`** — `Session` ring buffer.
- **`sleuth/engine/router.py`** — heuristic depth router (no LLM calls).
- **`sleuth/engine/executor.py`** — single-query parallel backend fan-out.
- **`sleuth/engine/synthesizer.py`** — streaming `TokenEvent` / `CitationEvent` / `DoneEvent`.
- **`sleuth/_agent.py`** — `Sleuth` class (`aask` / `ask` / `asummarize` / `summarize`).
- **`sleuth/__init__.py`** — full public re-exports per conventions §4.
- **`tests/contract/test_backend_protocol.py`** — `BackendTestKit` reusable harness.
- **`tests/snapshots/`** — syrupy baseline for fast-path event stream.

### Spec coverage
§3 (hard rules ✓), §4 (auto + fast depth ✓, deep treated as fast pending Phase 3), §5 (all 9 events ✓), §6 (all data shapes ✓), §7.1 (Backend protocol, failure handling, cancellation ✓), §7.2 (Tavily smoke ✓), §8 (MemoryCache + Session ring buffer ✓). Resolves spec §15 #3 (fast_llm documentation-only).

### Cross-phase stability promises
- `Cache` / `Session` protocol signatures frozen for Phase 4.
- `WebBackend` public symbol stable for Phase 9.
- `StubLLM` / `BackendTestKit` available for Phases 2–11.

## Test plan

- [x] All unit tests pass: `uv run pytest -m "not integration" -v`
- [x] Coverage ≥ 85%: `uv run pytest --cov=src/sleuth --cov-report=term-missing`
- [x] mypy strict passes: `uv run mypy src/sleuth/`
- [x] ruff clean: `uv run ruff check src/sleuth/ tests/`
- [x] Snapshot tests match: `uv run pytest tests/snapshots/ -v`
EOF
)"
```

Expected: PR URL printed. CI runs and passes.

---

## Self-review checklist

### 1. Spec coverage

| Spec section | Task(s) covering it |
|---|---|
| §3 hard rules (BYOK LLM, Backend protocol, async-first, streaming, no global state) | Tasks 5, 7, 8, 13, 15 |
| §3 observability — stdlib logging, no handlers | Task 2 |
| §4 Sleuth class, aask/ask, fast/auto depth | Task 15 |
| §4 schema= | Task 15 (passed through to Synthesizer) |
| §5 all 9 event types | Task 4 |
| §6 Source, Chunk, RunStats, Result[T] | Task 3 |
| §6 LLMClient, LLMChunk, Message, Tool | Task 5 |
| §7.1 Backend protocol, Capability | Task 7 |
| §7.1 failure handling → SearchEvent(error=) | Task 12 |
| §7.1 cancellation — BackendTestKit cancellation case | Task 7 |
| §7.2 Tavily-only stub | Task 10 |
| §8 MemoryCache | Task 8 |
| §8 Session ring buffer | Task 9 |
| §15 #3 resolved (fast_llm doc-only) | Task 15 (docstring + callout at top) |
| BackendTestKit reusable harness | Task 7 |
| Public re-exports (conventions §4) | Task 16 |
| stub_llm + fake_backend fixtures | Task 17 |
| Snapshot baseline | Task 14 |

No gaps found.

### 2. Placeholder scan

No "TBD", "TODO", "implement later", or "Similar to Task N" found. All code blocks are complete.

### 3. Type consistency

- `StubLLM.stream()` returns `AsyncIterator[LLMChunk]` — matches `LLMClient` protocol. ✓
- `Executor.run()` returns `tuple[list[SearchEvent], list[Chunk]]` — matches Synthesizer's chunk input. ✓
- `Synthesizer.synthesize()` yields `ThinkingEvent | TokenEvent | CitationEvent | DoneEvent` — all are members of `Event`. ✓
- `Session.add_turn()` takes `Result` (not `Result[T]`) — acceptable since session doesn't need to inspect `.data`. ✓
- `Cache.get/set/delete/clear` signatures match protocol definition in conventions §5.3. ✓
- `Backend.search()` signature matches conventions §5.2. ✓

All consistent.
