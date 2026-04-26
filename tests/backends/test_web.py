"""Tests for the Tavily WebBackend smoke implementation.

Phase 9 will expand this file with Exa, Brave, SerpAPI tests.
"""

import httpx
import pytest

from sleuth.backends.base import Capability
from sleuth.backends.web import TavilyBackend, WebBackend
from sleuth.types import Chunk

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

FAKE_RESPONSE = {
    "results": [
        {
            "title": "Example Article",
            "url": "https://example.com/article",
            "content": "This is the snippet text.",
            "score": 0.95,
        },
        {
            "title": "Another Result",
            "url": "https://example.com/other",
            "content": "More content here.",
            "score": 0.80,
        },
    ]
}


@pytest.fixture
def tavily(respx_mock):
    respx_mock.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json=FAKE_RESPONSE))
    return TavilyBackend(api_key="test-key")  # pragma: allowlist secret


@pytest.fixture
def tavily_no_mock():
    """TavilyBackend without a respx_mock — for attribute-only tests."""
    return TavilyBackend(api_key="test-key")  # pragma: allowlist secret


async def test_search_returns_chunks(tavily):
    chunks = await tavily.search("what is Python?")
    assert len(chunks) == 2
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source.kind == "url"


async def test_search_maps_score(tavily):
    chunks = await tavily.search("python")
    assert chunks[0].score == pytest.approx(0.95)
    assert chunks[1].score == pytest.approx(0.80)


async def test_search_maps_title(tavily):
    chunks = await tavily.search("python")
    assert chunks[0].source.title == "Example Article"


async def test_search_maps_url(tavily):
    chunks = await tavily.search("python")
    assert chunks[0].source.location == "https://example.com/article"


async def test_search_respects_k(respx_mock):
    respx_mock.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json=FAKE_RESPONSE))
    backend = TavilyBackend(api_key="test-key")  # pragma: allowlist secret
    chunks = await backend.search("python", k=1)
    assert len(chunks) <= 1


def test_backend_capabilities(tavily_no_mock):
    assert Capability.WEB in tavily_no_mock.capabilities
    assert Capability.FRESH in tavily_no_mock.capabilities


def test_backend_name(tavily_no_mock):
    assert tavily_no_mock.name == "tavily"


async def test_http_error_raises_backend_error(respx_mock):
    respx_mock.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    from sleuth.errors import BackendError

    backend = TavilyBackend(api_key="bad-key")  # pragma: allowlist secret
    with pytest.raises(BackendError):
        await backend.search("q")


async def test_web_backend_factory_returns_tavily():
    """WebBackend(provider='tavily') is the stable public symbol for Phase 9."""
    b = WebBackend(provider="tavily", api_key="key")  # pragma: allowlist secret
    assert isinstance(b, TavilyBackend)


# ---------------------------------------------------------------------------
# BackendTestKit compliance
# ---------------------------------------------------------------------------

from tests.contract.test_backend_protocol import BackendTestKit  # noqa: E402


class TestTavilyBackendContract(BackendTestKit):
    @pytest.fixture
    def backend(self, respx_mock):
        respx_mock.post(TAVILY_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=FAKE_RESPONSE)
        )
        return TavilyBackend(api_key="test-key")  # pragma: allowlist secret
