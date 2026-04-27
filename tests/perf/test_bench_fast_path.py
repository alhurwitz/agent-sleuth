"""
Phase 1 fast-path benchmarks.

Scenario: depth="fast", StubLLM, two fake backends (web + docs), no cache.

Benchmarks:
  test_bench_fast_path_first_token  — measures time-to-first-token (first TokenEvent).
  test_bench_fast_path_e2e          — measures full run latency (DoneEvent).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from sleuth import Sleuth
from sleuth.events import TokenEvent
from sleuth.memory.cache import MemoryCache
from tests.perf.conftest import run_stats_from_events

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect_events(agent: Sleuth, query: str) -> tuple[float, list[object]]:
    """Run aask and collect (time_to_first_token_ms, events)."""
    events: list[object] = []
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

    def run():
        # Fresh agent per benchmark iteration — MemoryCache so no file I/O.
        agent = Sleuth(
            llm=stub_llm_fast,
            backends=[fake_web_backend, fake_docs_backend],
            cache=MemoryCache(),
        )
        ftms, _events = asyncio.run(_collect_events(agent, "How does auth work?"))
        return ftms

    result = benchmark(run)
    # Hard gate: median first_token_ms must be < 1500 ms (checked in perf.yml too).
    assert result < 1500, f"first_token_ms={result:.1f} exceeded 1500 ms gate"


@pytest.mark.perf
def test_bench_fast_path_e2e(
    benchmark,
    stub_llm_fast,
    fake_web_backend,
    fake_docs_backend,
):
    """End-to-end latency for a fast-path run. Tracked as p50/p95 regression metric."""

    def run():
        agent = Sleuth(
            llm=stub_llm_fast,
            backends=[fake_web_backend, fake_docs_backend],
            cache=MemoryCache(),
        )
        _, events = asyncio.run(_collect_events(agent, "How does billing work?"))
        stats = run_stats_from_events(events)
        return stats.latency_ms if stats else 0

    benchmark(run)
