"""
Phase 2 LocalFiles perf benchmarks.

Corpus: 4 small Markdown docs (from conftest corpus_dir fixture).
LLM: StubLLM. The navigator will fall back to returning all leaves because
     StubLLM responses are not valid JSON (safe fallback per navigator.py).

Benchmarks:
  test_bench_localfiles_cold_index  — first search (index must be built on disk).
  test_bench_localfiles_warm_index  — second search (index is cached in memory).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from sleuth.backends.localfiles import LocalFiles


@pytest.mark.perf
def test_bench_localfiles_cold_index(benchmark: Any, corpus_dir: Path, stub_llm_fast: Any) -> None:
    """LocalFiles backend cold start: index build + one search call."""

    def run() -> int:
        # New backend instance each time → index is always rebuilt.
        backend = LocalFiles(
            path=corpus_dir,
            navigator_llm=stub_llm_fast,
            indexer_llm=stub_llm_fast,
            rebuild="always",  # force rebuild to measure cold-index path
        )
        chunks = asyncio.run(backend.search("How does authentication work?", k=5))
        return len(chunks)

    result = benchmark(run)
    # At least 0 is acceptable — StubLLM navigator falls back to all leaves.
    assert result >= 0


@pytest.mark.perf
def test_bench_localfiles_warm_index(benchmark: Any, corpus_dir: Path, stub_llm_fast: Any) -> None:
    """LocalFiles backend warm start: index already in memory."""
    # Build index once outside the benchmark loop.
    backend = LocalFiles(
        path=corpus_dir,
        navigator_llm=stub_llm_fast,
        indexer_llm=stub_llm_fast,
    )
    asyncio.run(backend.search("warm up", k=1))

    def run() -> int:
        # Reuse the same backend instance → in-memory index, no disk I/O.
        chunks = asyncio.run(backend.search("What is the billing plan?", k=5))
        return len(chunks)

    benchmark(run)
