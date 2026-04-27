"""
Performance test fixtures for agent-sleuth.

Fixture inventory:
  corpus_dir        — a tmp directory with 4 synthetic Markdown docs (small, fast to read).
  stub_llm_fast     — StubLLM that yields a response with no artificial latency.
  stub_llm_100ms    — StubLLM that injects 100 ms delay before first token (first_token_ms tests).
  fake_web_backend  — Backend(capability=WEB) returning 3 Chunks instantly.
  fake_docs_backend — Backend(capability=DOCS) returning 3 Chunks instantly.
  baseline          — dict loaded from tests/perf/baselines/develop.json (or {} if absent).
  run_stats_from_events — helper: extracts RunStats from a list of events.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from sleuth.backends.base import Capability
from sleuth.events import DoneEvent
from sleuth.llm.base import LLMChunk, Stop, TextDelta
from sleuth.llm.stub import StubLLM
from sleuth.types import Chunk, RunStats, Source

BASELINES_PATH = Path(__file__).parent / "baselines" / "develop.json"

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

_DOCS = [
    (
        "auth.md",
        "# Authentication\n\nOur auth uses JWT tokens with a 15-minute expiry.\n\n"
        "## Refresh tokens\n\nRefresh tokens last 30 days and are stored server-side.\n",
    ),
    (
        "billing.md",
        "# Billing\n\nWe charge per seat per month. Pro plan: $25/seat.\n\n"
        "## Invoices\n\nInvoices are generated on the 1st of each month.\n",
    ),
    (
        "deploy.md",
        "# Deployment\n\nWe deploy to AWS ECS via GitHub Actions on every merge to main.\n\n"
        "## Rollback\n\nRollback is one button in the ECS console.\n",
    ),
    (
        "api.md",
        "# API Reference\n\nBase URL: https://api.example.com/v2\n\n"
        "## Rate limits\n\n1000 requests/minute per API key.\n",
    ),
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

    async def _delayed_stream(messages: list[object]) -> AsyncIterator[LLMChunk]:
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
                source=Source(
                    kind="url",
                    location=f"https://fake.example.com/{self.name}/{i}",
                ),
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
def baseline() -> dict[str, object]:
    """Load committed baseline JSON. Returns empty dict if file does not exist."""
    if BASELINES_PATH.exists():
        result: dict[str, object] = json.loads(BASELINES_PATH.read_text())
        return result
    return {}


def run_stats_from_events(events: list[object]) -> RunStats | None:
    """Extract RunStats from a DoneEvent in *events*. Returns None if not found."""
    for event in events:
        if isinstance(event, DoneEvent):
            return event.stats
    return None
