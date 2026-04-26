"""Brave Search API backend."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from sleuth.backends._web._base import FetchPipeline, TokenBucket, with_backoff
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source

logger = logging.getLogger("sleuth.backends.web.brave")

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveBackend:
    """Backend adapter for the Brave Search API.

    Brave Search is a pure REST API — no SDK required.

    Args:
        api_key: Brave Search API subscription token.
        fetch: When True, parallel-fetches top ``fetch_top_n`` URLs.
        fetch_top_n: Number of top results to fetch when ``fetch=True``.
        rate_limit: Max requests per second.
        max_retries: Retries on 429/5xx.
        _backoff_base: Base delay for exponential backoff (testing hook).
    """

    name: str = "brave"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB, Capability.FRESH})

    def __init__(
        self,
        api_key: str,
        *,
        fetch: bool = False,
        fetch_top_n: int = 3,
        rate_limit: float = 5.0,
        max_retries: int = 3,
        _backoff_base: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self._fetch = fetch
        self._fetch_top_n = fetch_top_n
        self._max_retries = max_retries
        self._backoff_base = _backoff_base
        self._bucket = TokenBucket(rate=rate_limit, capacity=rate_limit)
        self._pipeline: FetchPipeline | None = FetchPipeline() if fetch else None

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        await self._bucket.acquire()

        async def _call() -> dict[str, Any]:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    _BRAVE_SEARCH_URL,
                    params={"q": query, "count": k},
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self._api_key,
                    },
                )
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]

        data = await with_backoff(
            _call, max_retries=self._max_retries, base_delay=self._backoff_base
        )

        results: list[dict[str, Any]] = data.get("web", {}).get("results", [])[:k]
        chunks = [
            Chunk(
                text=r.get("description", ""),
                source=Source(
                    kind="url",
                    location=r["url"],
                    title=r.get("title"),
                ),
                score=None,  # Brave API does not surface relevance scores
            )
            for r in results
        ]

        if self._fetch and self._pipeline and chunks:
            urls = [c.source.location for c in chunks[: self._fetch_top_n]]
            fetched = await self._pipeline.fetch_and_chunk(urls=urls, query=query)
            chunks = chunks + fetched

        logger.debug("brave: %d chunks for %r", len(chunks), query)
        return chunks
