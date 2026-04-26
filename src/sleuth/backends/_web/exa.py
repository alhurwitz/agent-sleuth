"""Exa (formerly Metaphor) web search backend."""

from __future__ import annotations

import logging
from typing import Any

from sleuth.backends._web._base import FetchPipeline, TokenBucket
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source

logger = logging.getLogger("sleuth.backends.web.exa")


def _make_exa_client(api_key: str) -> Any:
    try:
        from exa_py import Exa

        return Exa(api_key=api_key)
    except ImportError as exc:
        raise ImportError(
            "ExaBackend requires the 'exa' extra: pip install agent-sleuth[exa]"
        ) from exc


class ExaBackend:
    """Backend adapter for the Exa (Metaphor) neural search API.

    Args:
        api_key: Exa API key.
        fetch: When True, parallel-fetches top ``fetch_top_n`` URLs and
               appends page-content chunks.
        fetch_top_n: Number of top results to fetch when ``fetch=True``.
        rate_limit: Max requests per second.
        max_retries: Retries on transient errors.
        _backoff_base: Base delay for backoff (exposed for testing).
    """

    name: str = "exa"
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
        self._client = _make_exa_client(api_key)
        self._fetch = fetch
        self._fetch_top_n = fetch_top_n
        self._max_retries = max_retries
        self._backoff_base = _backoff_base
        self._bucket = TokenBucket(rate=rate_limit, capacity=rate_limit)
        self._pipeline: FetchPipeline | None = FetchPipeline() if fetch else None

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        await self._bucket.acquire()

        # exa-py's async search method
        result = await self._client.search_and_contents(
            query,
            num_results=k,
            text=True,
        )

        items = (result.results or [])[:k]
        chunks = [
            Chunk(
                text=item.text or "",
                source=Source(
                    kind="url",
                    location=item.url,
                    title=item.title,
                ),
                score=item.score if hasattr(item, "score") else None,
            )
            for item in items
        ]

        if self._fetch and self._pipeline and chunks:
            urls = [c.source.location for c in chunks[: self._fetch_top_n]]
            fetched = await self._pipeline.fetch_and_chunk(urls=urls, query=query)
            chunks = chunks + fetched

        logger.debug("exa: %d chunks for %r", len(chunks), query)
        return chunks
