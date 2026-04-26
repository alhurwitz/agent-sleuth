"""Tests for BraveBackend."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from sleuth.backends._web.brave import BraveBackend
from sleuth.types import Chunk, Source

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def _brave_response(n: int = 2) -> dict[str, Any]:
    return {
        "web": {
            "results": [
                {
                    "url": f"https://brave-example.com/{i}",
                    "title": f"Brave Result {i}",
                    "description": f"Description snippet for result {i}.",
                }
                for i in range(n)
            ]
        }
    }


@pytest.mark.unit
async def test_brave_backend_name_and_capabilities():
    backend = BraveBackend(api_key="brave-key")  # pragma: allowlist secret
    assert backend.name == "brave"
    from sleuth.backends.base import Capability

    assert Capability.WEB in backend.capabilities
    assert Capability.FRESH in backend.capabilities


@pytest.mark.unit
async def test_brave_search_returns_chunks():
    with respx.mock:
        respx.get(_BRAVE_URL).respond(200, json=_brave_response(3))
        backend = BraveBackend(api_key="brave-key")  # pragma: allowlist secret
        chunks = await backend.search("open source AI", k=3)

    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source.kind == "url"
    assert chunks[0].source.location == "https://brave-example.com/0"
    assert chunks[0].source.title == "Brave Result 0"


@pytest.mark.unit
async def test_brave_search_respects_k():
    with respx.mock:
        respx.get(_BRAVE_URL).respond(200, json=_brave_response(5))
        backend = BraveBackend(api_key="brave-key")  # pragma: allowlist secret
        chunks = await backend.search("query", k=2)

    assert len(chunks) == 2


@pytest.mark.unit
async def test_brave_raises_on_5xx():
    from sleuth.errors import BackendError

    with respx.mock:
        respx.get(_BRAVE_URL).respond(503)
        backend = BraveBackend(api_key="brave-key", _backoff_base=0.01)  # pragma: allowlist secret
        with pytest.raises(BackendError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_brave_raises_immediately_on_401():
    """401 Unauthorized is not retried — surfaces as BackendError."""
    from sleuth.errors import BackendError

    with respx.mock:
        respx.get(_BRAVE_URL).respond(401)
        backend = BraveBackend(api_key="bad-key", _backoff_base=0.01)  # pragma: allowlist secret
        with pytest.raises(BackendError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_brave_fetch_mode():
    from sleuth.backends._web._base import FetchPipeline

    fake_chunk = Chunk(
        text="Full page text.",
        source=Source(kind="url", location="https://brave-example.com/0"),
        score=None,
    )

    with respx.mock:
        respx.get(_BRAVE_URL).respond(200, json=_brave_response(1))
        with patch.object(
            FetchPipeline,
            "fetch_and_chunk",
            new=AsyncMock(return_value=[fake_chunk]),
        ):
            backend = BraveBackend(
                api_key="brave-key",  # pragma: allowlist secret
                fetch=True,
                fetch_top_n=1,
            )
            chunks = await backend.search("query", k=1)

    assert any(c.text == "Full page text." for c in chunks)


# ---------------------------------------------------------------------------
# BackendTestKit contract compliance
# ---------------------------------------------------------------------------

from tests.contract.test_backend_protocol import BackendTestKit  # noqa: E402


class TestBraveBackendContract(BackendTestKit):
    @pytest.fixture
    def backend(self, respx_mock: Any) -> BraveBackend:
        respx_mock.get(_BRAVE_URL).mock(return_value=httpx.Response(200, json=_brave_response(3)))
        return BraveBackend(api_key="brave-key")  # pragma: allowlist secret
