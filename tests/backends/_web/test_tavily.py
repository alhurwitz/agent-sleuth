"""Tests for TavilyBackend."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from sleuth.backends._web.tavily import TavilyBackend
from sleuth.types import Chunk, Source


def _tavily_response(n: int = 2) -> dict[str, Any]:
    return {
        "results": [
            {
                "url": f"https://example.com/{i}",
                "title": f"Result {i}",
                "content": f"Content for result {i}. This is a sample snippet.",
                "score": 0.9 - i * 0.1,
            }
            for i in range(n)
        ]
    }


@pytest.mark.unit
async def test_tavily_backend_name_and_capabilities():
    backend = TavilyBackend(api_key="test-key")  # pragma: allowlist secret
    assert backend.name == "tavily"
    from sleuth.backends.base import Capability

    assert Capability.WEB in backend.capabilities
    assert Capability.FRESH in backend.capabilities


@pytest.mark.unit
async def test_tavily_search_returns_chunks():
    with respx.mock:
        respx.post("https://api.tavily.com/search").respond(200, json=_tavily_response(3))
        backend = TavilyBackend(api_key="test-key")  # pragma: allowlist secret
        chunks = await backend.search("python async", k=3)

    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source.kind == "url" for c in chunks)
    assert chunks[0].source.location == "https://example.com/0"
    assert chunks[0].score == pytest.approx(0.9)


@pytest.mark.unit
async def test_tavily_search_respects_k():
    """k parameter limits results returned."""
    with respx.mock:
        respx.post("https://api.tavily.com/search").respond(200, json=_tavily_response(5))
        backend = TavilyBackend(api_key="test-key")  # pragma: allowlist secret
        chunks = await backend.search("query", k=2)

    assert len(chunks) == 2


@pytest.mark.unit
async def test_tavily_raises_backend_error_on_5xx():
    from sleuth.errors import BackendError

    with respx.mock:
        # Fail all retries
        respx.post("https://api.tavily.com/search").respond(500)
        backend = TavilyBackend(api_key="test-key", _backoff_base=0.01)  # pragma: allowlist secret
        with pytest.raises(BackendError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_tavily_raises_immediately_on_401():
    """401 Unauthorized is not retried — surfaces as BackendError."""
    from sleuth.errors import BackendError

    with respx.mock:
        respx.post("https://api.tavily.com/search").respond(401)
        backend = TavilyBackend(api_key="bad-key", _backoff_base=0.01)  # pragma: allowlist secret
        with pytest.raises(BackendError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_tavily_fetch_mode_returns_extra_chunks():
    """fetch=True enriches results with page content."""
    from sleuth.backends._web._base import FetchPipeline

    fake_chunks = [
        Chunk(
            text="Full article text.",
            source=Source(kind="url", location="https://example.com/0"),
            score=None,
        )
    ]

    with respx.mock:
        respx.post("https://api.tavily.com/search").respond(200, json=_tavily_response(1))
        with patch.object(
            FetchPipeline,
            "fetch_and_chunk",
            new=AsyncMock(return_value=fake_chunks),
        ):
            backend = TavilyBackend(
                api_key="test-key",  # pragma: allowlist secret
                fetch=True,
                fetch_top_n=1,
            )
            chunks = await backend.search("python async", k=1)

    # Should include at least the fake fetched chunk
    locations = {c.source.location for c in chunks}
    assert "https://example.com/0" in locations


# ---------------------------------------------------------------------------
# BackendTestKit contract compliance
# ---------------------------------------------------------------------------


from tests.contract.test_backend_protocol import BackendTestKit  # noqa: E402


class TestTavilyBackendContract(BackendTestKit):
    @pytest.fixture
    def backend(self, respx_mock: Any) -> TavilyBackend:
        respx_mock.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=_tavily_response(3))
        )
        return TavilyBackend(api_key="test-key")  # pragma: allowlist secret
