"""Web search backends.

Phase 1 ships a Tavily-only smoke implementation plus the ``WebBackend``
factory (currently returns ``TavilyBackend`` for all providers).  Phase 9
expands with Exa, Brave, and SerpAPI adapters and a richer factory.

The public symbol ``WebBackend`` is stable — downstream code should use it
rather than importing ``TavilyBackend`` directly.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from sleuth.backends.base import Capability
from sleuth.errors import BackendError
from sleuth.types import Chunk, Source

logger = logging.getLogger("sleuth.backends.web")

# Default timeout per spec §7.1
_DEFAULT_TIMEOUT_S = 8.0


class TavilyBackend:
    """Tavily search API backend.

    Implements the ``Backend`` Protocol structurally.  Uses ``httpx`` for
    async HTTP; respx is used in tests to mock requests.

    Args:
        api_key: Tavily API key (``TAVILY_API_KEY`` env var if not passed).
        timeout_s: Per-request timeout in seconds.  Default: 8s (spec §7.1).
    """

    name = "tavily"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB, Capability.FRESH})

    _SEARCH_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self._api_key = api_key
        self._timeout_s = timeout_s

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        """Search Tavily and return up to ``k`` chunks.

        Raises:
            BackendError: On any HTTP or API-level error.
        """
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": k,
        }
        logger.debug("Tavily search: query=%r k=%d", query, k)
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(self._SEARCH_URL, json=payload)
        except httpx.TimeoutException as exc:
            raise BackendError(f"Tavily request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"Tavily HTTP error: {exc}") from exc

        if resp.status_code != 200:
            raise BackendError(f"Tavily returned HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        chunks: list[Chunk] = []
        for item in data.get("results", [])[:k]:
            source = Source(
                kind="url",
                location=item.get("url", ""),
                title=item.get("title"),
            )
            chunks.append(
                Chunk(
                    text=item.get("content", ""),
                    source=source,
                    score=item.get("score"),
                )
            )
        return chunks


def WebBackend(
    provider: Literal["tavily"] = "tavily",
    *,
    api_key: str,
    **kwargs: Any,
) -> TavilyBackend:
    """Factory for web search backends.

    Phase 9 expands this to support ``provider="exa"``, ``"brave"``, ``"serpapi"``.
    The symbol is stable; callers should always use ``WebBackend(...)`` rather
    than importing ``TavilyBackend`` directly.

    Args:
        provider: Which provider to use.  Currently only ``"tavily"``.
        api_key: API key for the chosen provider.

    Returns:
        A backend instance implementing the ``Backend`` Protocol.
    """
    if provider == "tavily":
        return TavilyBackend(api_key=api_key, **kwargs)
    raise ValueError(f"Unknown web provider: {provider!r}.  Phase 9 adds exa/brave/serpapi.")
