"""Tests for SerpAPIBackend."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from sleuth.backends._web.serpapi import SerpAPIBackend
from sleuth.types import Chunk, Source

_SERPAPI_URL = "https://serpapi.com/search"


def _serpapi_response(n: int = 2) -> dict[str, Any]:
    return {
        "organic_results": [
            {
                "link": f"https://serpapi-example.com/{i}",
                "title": f"SerpAPI Result {i}",
                "snippet": f"Organic result snippet for query {i}.",
                "position": i + 1,
            }
            for i in range(n)
        ]
    }


@pytest.mark.unit
async def test_serpapi_backend_name_and_capabilities():
    backend = SerpAPIBackend(api_key="serpapi-key")  # pragma: allowlist secret
    assert backend.name == "serpapi"
    from sleuth.backends.base import Capability

    assert Capability.WEB in backend.capabilities


@pytest.mark.unit
async def test_serpapi_search_returns_chunks():
    with respx.mock:
        respx.get(_SERPAPI_URL).respond(200, json=_serpapi_response(3))
        backend = SerpAPIBackend(api_key="serpapi-key")  # pragma: allowlist secret
        chunks = await backend.search("best Python libraries", k=3)

    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source.kind == "url"
    assert chunks[0].source.location == "https://serpapi-example.com/0"


@pytest.mark.unit
async def test_serpapi_search_respects_k():
    with respx.mock:
        respx.get(_SERPAPI_URL).respond(200, json=_serpapi_response(5))
        backend = SerpAPIBackend(api_key="serpapi-key")  # pragma: allowlist secret
        chunks = await backend.search("query", k=2)

    assert len(chunks) == 2


@pytest.mark.unit
async def test_serpapi_raises_on_5xx():
    from sleuth.errors import BackendError

    with respx.mock:
        respx.get(_SERPAPI_URL).respond(500)
        backend = SerpAPIBackend(
            api_key="serpapi-key",  # pragma: allowlist secret
            _backoff_base=0.01,
        )
        with pytest.raises(BackendError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_serpapi_raises_immediately_on_403():
    """403 Forbidden is not retried — surfaces as BackendError."""
    from sleuth.errors import BackendError

    with respx.mock:
        respx.get(_SERPAPI_URL).respond(403)
        backend = SerpAPIBackend(api_key="bad-key", _backoff_base=0.01)  # pragma: allowlist secret
        with pytest.raises(BackendError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_serpapi_empty_organic_results():
    """Backend returns empty list when no organic results."""
    with respx.mock:
        respx.get(_SERPAPI_URL).respond(200, json={"organic_results": []})
        backend = SerpAPIBackend(api_key="serpapi-key")  # pragma: allowlist secret
        chunks = await backend.search("ultra niche query", k=5)

    assert chunks == []


@pytest.mark.unit
async def test_serpapi_fetch_mode():
    from sleuth.backends._web._base import FetchPipeline

    fake_chunk = Chunk(
        text="Full SerpAPI page content.",
        source=Source(kind="url", location="https://serpapi-example.com/0"),
        score=None,
    )

    with respx.mock:
        respx.get(_SERPAPI_URL).respond(200, json=_serpapi_response(1))
        with patch.object(
            FetchPipeline,
            "fetch_and_chunk",
            new=AsyncMock(return_value=[fake_chunk]),
        ):
            backend = SerpAPIBackend(
                api_key="serpapi-key",  # pragma: allowlist secret
                fetch=True,
                fetch_top_n=1,
            )
            chunks = await backend.search("query", k=1)

    assert any(c.text == "Full SerpAPI page content." for c in chunks)


# ---------------------------------------------------------------------------
# BackendTestKit contract compliance
# ---------------------------------------------------------------------------

from tests.contract.test_backend_protocol import BackendTestKit  # noqa: E402


class TestSerpAPIBackendContract(BackendTestKit):
    @pytest.fixture
    def backend(self, respx_mock: Any) -> SerpAPIBackend:
        respx_mock.get(_SERPAPI_URL).mock(
            return_value=httpx.Response(200, json=_serpapi_response(3))
        )
        return SerpAPIBackend(api_key="serpapi-key")  # pragma: allowlist secret
