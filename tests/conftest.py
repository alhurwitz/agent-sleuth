"""
Cross-cutting pytest fixtures for agent-sleuth.

Fixture inventory (Phase 0):
  - tmp_corpus:   a tmp_path subdirectory pre-created as an empty corpus root
  - respx_mock:   respx mock transport active for the test (async-safe)

Phase 1 adds:
  - stub_llm:     a StubLLM instance with a default "hello" response
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import respx as respx_module


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
