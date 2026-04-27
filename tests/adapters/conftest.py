"""Shared fixtures for all framework adapter tests.

Uses StubLLM + FakeBackend so tests never require real LLM or network.
"""

from __future__ import annotations

import pytest

from sleuth import Sleuth
from sleuth.backends.base import Capability
from sleuth.llm.stub import StubLLM
from sleuth.types import Chunk, Source


class FakeBackend:
    """Minimal Backend that returns a fixed chunk — no real search."""

    name = "fake"
    capabilities = frozenset({Capability.DOCS})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [
            Chunk(
                text=f"Fake result for: {query}",
                source=Source(kind="file", location="fake.md", title="Fake"),
                score=1.0,
            )
        ]


@pytest.fixture()
def stub_llm() -> StubLLM:
    """A StubLLM that emits a short answer then stops."""
    return StubLLM(responses=["The answer is 42."])


@pytest.fixture()
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture()
def sleuth_agent(stub_llm: StubLLM, fake_backend: FakeBackend) -> Sleuth:
    """A Sleuth instance wired to stub LLM + fake backend. No I/O."""
    return Sleuth(llm=stub_llm, backends=[fake_backend], cache=None)
