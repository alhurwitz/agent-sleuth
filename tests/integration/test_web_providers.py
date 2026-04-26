"""Integration tests for web backends against real APIs.

These tests are env-gated and run only in nightly CI (pytest -m integration).
Each test skips automatically if the required env var is absent.
"""

from __future__ import annotations

import os

import pytest

from sleuth.backends.web import (
    BraveBackend,
    ExaBackend,
    SerpAPIBackend,
    TavilyBackend,
)
from sleuth.types import Chunk

pytestmark = pytest.mark.integration


@pytest.fixture
def tavily_key() -> str:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        pytest.skip("TAVILY_API_KEY not set")
    return key


@pytest.fixture
def exa_key() -> str:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        pytest.skip("EXA_API_KEY not set")
    return key


@pytest.fixture
def brave_key() -> str:
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        pytest.skip("BRAVE_API_KEY not set")
    return key


@pytest.fixture
def serpapi_key() -> str:
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        pytest.skip("SERPAPI_API_KEY not set")
    return key


async def test_tavily_real_search(tavily_key: str) -> None:
    backend = TavilyBackend(api_key=tavily_key)  # pragma: allowlist secret
    chunks = await backend.search("Python asyncio tutorial", k=3)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source.location.startswith("http") for c in chunks)


async def test_exa_real_search(exa_key: str) -> None:
    backend = ExaBackend(api_key=exa_key)  # pragma: allowlist secret
    chunks = await backend.search("machine learning papers 2025", k=3)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


async def test_brave_real_search(brave_key: str) -> None:
    backend = BraveBackend(api_key=brave_key)  # pragma: allowlist secret
    chunks = await backend.search("open source AI tools", k=3)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


async def test_serpapi_real_search(serpapi_key: str) -> None:
    backend = SerpAPIBackend(api_key=serpapi_key)  # pragma: allowlist secret
    chunks = await backend.search("Python web frameworks comparison", k=3)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


async def test_tavily_fetch_mode_real(tavily_key: str) -> None:
    backend = TavilyBackend(  # pragma: allowlist secret
        api_key=tavily_key,
        fetch=True,
        fetch_top_n=1,
    )
    chunks = await backend.search("httpx Python library", k=2)
    # Should have at least one chunk (from API or fetch)
    assert len(chunks) >= 1
