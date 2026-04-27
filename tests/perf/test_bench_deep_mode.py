"""
Phase 3 deep-mode perf benchmarks.

Scenario: depth="deep", max_iterations=2, StubLLM, fake backends.
Measures the overhead of the planner + speculative prefetch path.

StubLLM responses are not valid plan JSON, so the Planner falls back to
returning a single sub-query (same query text) — this is expected and means
the benchmark measures engine overhead rather than LLM latency.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sleuth import Sleuth
from sleuth.memory.cache import MemoryCache
from tests.perf.conftest import run_stats_from_events


async def _run_deep(agent: Sleuth, query: str) -> tuple[int, list[object]]:
    events: list[object] = []
    async for event in agent.aask(query, depth="deep", max_iterations=2):
        events.append(event)
    stats = run_stats_from_events(events)
    return (stats.latency_ms if stats else 0), events


@pytest.mark.perf
def test_bench_deep_mode_e2e(
    benchmark: Any,
    stub_llm_fast: Any,
    fake_web_backend: Any,
    fake_docs_backend: Any,
) -> None:
    """Deep-mode end-to-end latency with StubLLM (measures engine overhead only)."""

    def run() -> int:
        agent = Sleuth(
            llm=stub_llm_fast,
            backends=[fake_web_backend, fake_docs_backend],
            cache=MemoryCache(),
        )
        latency_ms, _ = asyncio.run(
            _run_deep(agent, "Explain our auth flow and how billing relates to seat count")
        )
        return latency_ms

    benchmark(run)


@pytest.mark.perf
def test_bench_deep_mode_speculative_prefetch(
    benchmark: Any,
    stub_llm_fast: Any,
    fake_web_backend: Any,
    fake_docs_backend: Any,
) -> None:
    """Speculative prefetch: backend search starts before planner finishes streaming.

    With StubLLM emitting instantly, this primarily measures that the prefetch
    code path does not add overhead compared to a sequential path.
    """

    def run() -> int:
        agent = Sleuth(
            llm=stub_llm_fast,
            backends=[fake_web_backend, fake_docs_backend],
            cache=MemoryCache(),
        )
        latency_ms, _events = asyncio.run(
            _run_deep(agent, "What are the deployment steps and rollback procedure?")
        )
        return latency_ms

    benchmark(run)
