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
import contextlib
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

    async def test_search_returns_chunk_list(self, backend: Backend) -> None:
        result = await backend.search("test query")
        assert_chunk_list(result)

    async def test_search_respects_k(self, backend: Backend) -> None:
        result = await backend.search("test query", k=1)
        assert len(result) <= 1

    async def test_backend_has_name(self, backend: Backend) -> None:
        assert isinstance(backend.name, str)
        assert len(backend.name) > 0

    async def test_backend_has_capabilities(self, backend: Backend) -> None:
        assert isinstance(backend.capabilities, frozenset)
        for cap in backend.capabilities:
            assert isinstance(cap, Capability)

    async def test_cancellation_safety(self, backend: Backend) -> None:
        """Backend search must honour asyncio cancellation without hanging.

        A fast (non-blocking) backend may complete before cancellation propagates —
        that is also acceptable.  The key assertion is that we get here without hanging.
        """
        task = asyncio.create_task(backend.search("query"))
        # Give it one scheduler turn to start, then cancel.
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        # The important assertion: we got here without hanging.


# ---------------------------------------------------------------------------
# Self-test — run BackendTestKit against FakeBackend
# ---------------------------------------------------------------------------


class TestFakeBackend(BackendTestKit):
    @pytest.fixture
    def backend(self) -> FakeBackend:
        return FakeBackend()

    async def test_error_propagates(self) -> None:
        b = FakeBackend(raise_on_search=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            await b.search("q")
