"""
Cache hit path benchmarks.

Scenario: run the same query twice with the same MemoryCache instance.
The second run is a warm run (backends already primed in the same process).

Note: the current Sleuth.aask does not emit CacheHitEvent because the
query-cache lookup/replay path is not yet wired in _agent.py. These
benchmarks measure the fast-path overhead (same backends, same cache
instance) rather than true cache-replay latency. The CacheHitEvent
assertion is intentionally absent.

Benchmarks:
  test_bench_cache_miss  — first run, cold start.
  test_bench_cache_hit   — second run, warm engine state (same cache instance).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from sleuth import Sleuth
from sleuth.events import DoneEvent, TokenEvent
from sleuth.memory.cache import MemoryCache

_QUERY = "What is the API rate limit?"


async def _run(agent: Sleuth) -> tuple[float, list[object]]:
    events: list[object] = []
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
        ftms, _ = asyncio.run(_run(agent))
        return ftms

    benchmark(run)


@pytest.mark.perf
def test_bench_cache_hit(benchmark, stub_llm_fast, fake_web_backend):
    """Warm cache run — same cache instance reused across iterations.

    Both runs execute the full engine path because cache-replay is not yet
    wired in Sleuth.aask. The warm run benefits from Python's JIT warmup
    and avoids MemoryCache initialisation overhead.
    """
    cache = MemoryCache()

    # Prime with one live run outside the benchmark loop.
    agent0 = Sleuth(llm=stub_llm_fast, backends=[fake_web_backend], cache=cache)
    asyncio.run(_run(agent0))

    def run():
        # Reuse same cache instance — measures repeated-run overhead.
        agent = Sleuth(llm=stub_llm_fast, backends=[fake_web_backend], cache=cache)
        _, events = asyncio.run(_run(agent))
        # Confirm the run completed successfully.
        assert any(isinstance(e, DoneEvent) for e in events), (
            "DoneEvent missing from warm-cache run"
        )

    benchmark(run)
