"""
Cross-cutting pytest fixtures for agent-sleuth.

Fixture inventory (Phase 0):
  - tmp_corpus:   a tmp_path subdirectory pre-created as an empty corpus root
  - respx_mock:   respx mock transport active for the test (async-safe)

Phase 1 adds:
  - stub_llm:     a StubLLM instance with a default "answer" response
  - fake_backend: a minimal in-memory Backend for engine/integration tests
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import respx as respx_module

from sleuth.backends.base import Capability
from sleuth.llm.stub import StubLLM
from sleuth.types import Chunk, Source


@pytest.fixture
def tmp_corpus(tmp_path: Path) -> Path:
    """Return a fresh empty directory suitable for use as a LocalFiles corpus root.

    Each test gets an isolated directory under pytest's tmp_path mechanism.
    The directory is created and ready to write files into.

    Usage::

        def test_something(tmp_corpus):
            (tmp_corpus / "doc.md").write_text("# Hello")
            backend = LocalFiles(path=tmp_corpus)
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return corpus


@pytest.fixture
def respx_mock() -> Generator[respx_module.MockRouter, None, None]:
    """Activate respx mock transport for the duration of the test.

    All httpx requests made during the test are intercepted. Unmatched
    requests raise ``respx.errors.AllMockedError`` so tests cannot make
    accidental real HTTP calls.

    Usage::

        async def test_fetch(respx_mock):
            respx_mock.get("https://api.example.com/search").respond(
                200, json={"results": []}
            )
            # ... code under test that uses httpx ...

    Note: ``respx_mock`` is synchronous-fixture-friendly even in async tests
    because respx patches the transport layer, not the event loop.
    """
    with respx_module.mock(assert_all_mocked=True, assert_all_called=False) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Phase 1 additions — stub_llm + fake_backend cross-cutting fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_llm() -> StubLLM:
    """StubLLM with a single 'answer' response — suitable for most unit tests."""
    return StubLLM(["answer"])


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
def fake_backend() -> _FakeBackend:
    """Minimal in-memory Backend for use in engine and integration tests."""
    return _FakeBackend()
