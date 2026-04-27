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

import pytest

from sleuth.backends.base import Capability
from sleuth.events import SearchEvent
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
    # error event emitted for slow backend (second SearchEvent for it is the error one)
    error_events = [e for e in events if isinstance(e, SearchEvent) and e.error is not None]
    assert len(error_events) == 1
    assert error_events[0].backend == "web-slow"
    # "timed out" or "timeout" — either form is acceptable
    assert error_events[0].error is not None
    err_lower = error_events[0].error.lower()
    assert "timeout" in err_lower or "timed out" in err_lower


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
