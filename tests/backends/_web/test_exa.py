"""Tests for ExaBackend."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sleuth.backends._web.exa import ExaBackend
from sleuth.types import Chunk, Source


def _exa_response(n: int = 2) -> dict[str, Any]:
    return {
        "results": [
            {
                "url": f"https://exa-example.com/{i}",
                "title": f"Exa Result {i}",
                "text": f"Snippet text for result {i}. Enough content to be a chunk.",
                "score": 0.95 - i * 0.05,
            }
            for i in range(n)
        ]
    }


@pytest.mark.unit
async def test_exa_backend_name_and_capabilities():
    backend = ExaBackend(api_key="exa-key")  # pragma: allowlist secret
    assert backend.name == "exa"
    from sleuth.backends.base import Capability

    assert Capability.WEB in backend.capabilities
    assert Capability.FRESH in backend.capabilities


@pytest.mark.unit
async def test_exa_search_returns_chunks():
    mock_exa = MagicMock()
    mock_results = MagicMock()
    mock_results.results = [
        MagicMock(
            url=f"https://exa-example.com/{i}",
            title=f"Exa Result {i}",
            text=f"Snippet text for result {i}.",
            score=0.95 - i * 0.05,
        )
        for i in range(3)
    ]
    mock_exa.search_and_contents = AsyncMock(return_value=mock_results)

    with patch("sleuth.backends._web.exa._make_exa_client", return_value=mock_exa):
        backend = ExaBackend(api_key="exa-key")  # pragma: allowlist secret
        chunks = await backend.search("machine learning", k=3)

    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source.kind == "url"
    assert chunks[0].source.location == "https://exa-example.com/0"


@pytest.mark.unit
async def test_exa_search_respects_k():
    mock_exa = MagicMock()
    mock_results = MagicMock()
    mock_results.results = [
        MagicMock(
            url=f"https://exa-example.com/{i}",
            title=f"Title {i}",
            text=f"Text {i}",
            score=0.9,
        )
        for i in range(5)
    ]
    mock_exa.search_and_contents = AsyncMock(return_value=mock_results)

    with patch("sleuth.backends._web.exa._make_exa_client", return_value=mock_exa):
        backend = ExaBackend(api_key="exa-key")  # pragma: allowlist secret
        chunks = await backend.search("query", k=2)

    assert len(chunks) == 2


@pytest.mark.unit
async def test_exa_fetch_mode():
    """fetch=True calls FetchPipeline in addition to Exa API."""
    from sleuth.backends._web._base import FetchPipeline

    mock_exa = MagicMock()
    mock_results = MagicMock()
    mock_results.results = [
        MagicMock(
            url="https://exa-example.com/0",
            title="Title 0",
            text="Text 0",
            score=0.9,
        )
    ]
    mock_exa.search_and_contents = AsyncMock(return_value=mock_results)

    fake_chunk = Chunk(
        text="Fetched content.",
        source=Source(kind="url", location="https://exa-example.com/0"),
        score=None,
    )

    with (
        patch("sleuth.backends._web.exa._make_exa_client", return_value=mock_exa),
        patch.object(
            FetchPipeline,
            "fetch_and_chunk",
            new=AsyncMock(return_value=[fake_chunk]),
        ),
    ):
        backend = ExaBackend(
            api_key="exa-key",  # pragma: allowlist secret
            fetch=True,
            fetch_top_n=1,
        )
        chunks = await backend.search("query", k=1)

    assert any(c.text == "Fetched content." for c in chunks)


@pytest.mark.unit
async def test_exa_import_error_message():
    """Helpful error when exa-py is not installed."""
    import sys

    with (
        patch.dict(sys.modules, {"exa_py": None}),
        pytest.raises(ImportError, match="exa"),
    ):
        ExaBackend(api_key="key")  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# BackendTestKit contract compliance
# ---------------------------------------------------------------------------

from tests.contract.test_backend_protocol import BackendTestKit  # noqa: E402


class TestExaBackendContract(BackendTestKit):
    @pytest.fixture
    def backend(self) -> ExaBackend:
        mock_exa = MagicMock()
        mock_results = MagicMock()
        mock_results.results = [
            MagicMock(
                url=f"https://exa.com/{i}",
                title=f"T{i}",
                text=f"Text {i}",
                score=0.9,
            )
            for i in range(3)
        ]
        mock_exa.search_and_contents = AsyncMock(return_value=mock_results)

        with patch("sleuth.backends._web.exa._make_exa_client", return_value=mock_exa):
            b = ExaBackend(api_key="exa-key")  # pragma: allowlist secret
        # Swap out the client so the mock is used at search time too
        b._client = mock_exa
        return b
