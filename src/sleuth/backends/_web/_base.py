"""Shared utilities for web backends: rate limiting and HTTP backoff."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

from sleuth.errors import BackendError

logger = logging.getLogger("sleuth.backends.web")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (one instance per host)
# ---------------------------------------------------------------------------


class TokenBucket:
    """Simple async token-bucket rate limiter.

    Args:
        rate: Tokens added per second.
        capacity: Maximum token reservoir size (controls burst size).
    """

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # How long until we have enough tokens?
                wait = (tokens - self._tokens) / self._rate
                await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Exponential backoff wrapper
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_BACKOFF_S = 60.0


async def with_backoff(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Call ``coro_fn()`` with exponential backoff on retryable HTTP errors.

    Rules:
    - 4xx (except 429): raise immediately — no retry.
    - 429: honour ``Retry-After`` header if present; else exponential backoff.
    - 5xx: exponential backoff with cap of ``_MAX_BACKOFF_S``.
    - Exceeding ``max_retries``: raise ``BackendError``.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status not in _RETRYABLE_STATUS:
                raise  # 4xx non-429 — propagate immediately

            if attempt >= max_retries:
                last_exc = exc
                break

            if status == 429:
                retry_after_raw = exc.response.headers.get("Retry-After")
                if retry_after_raw is not None:
                    try:
                        delay = float(retry_after_raw)
                    except ValueError:
                        delay = base_delay * (2**attempt)
                else:
                    delay = base_delay * (2**attempt)
            else:
                delay = min(base_delay * (2**attempt), _MAX_BACKOFF_S)

            logger.warning(
                "HTTP %s from %s; retrying in %.1fs (attempt %d/%d)",
                status,
                exc.request.url,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)

    raise BackendError(
        f"Backend call failed after {max_retries} max retries: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Optional fetch pipeline (fetch=True mode)
# ---------------------------------------------------------------------------


def _lazy_import_trafilatura() -> Any:
    try:
        import trafilatura

        return trafilatura
    except ImportError as exc:
        raise ImportError(
            "fetch=True requires the 'web-fetch' extra: pip install agent-sleuth[web-fetch]"
        ) from exc


def _lazy_import_tiktoken() -> Any:
    try:
        import tiktoken

        return tiktoken
    except ImportError as exc:
        raise ImportError(
            "fetch=True requires the 'web-fetch' extra: pip install agent-sleuth[web-fetch]"
        ) from exc


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_tokens`` tokens."""
    tiktoken = _lazy_import_tiktoken()
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    chunks: list[str] = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i : i + max_tokens]
        chunks.append(enc.decode(chunk_ids))
    return chunks


class FetchPipeline:
    """Parallel-fetch top-N URLs and chunk their content.

    Args:
        max_tokens_per_chunk: Maximum tokens per returned ``Chunk``.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        max_tokens_per_chunk: int = 512,
        timeout: float = 10.0,
    ) -> None:
        self._max_tokens = max_tokens_per_chunk
        self._timeout = timeout

    async def fetch_and_chunk(
        self,
        urls: list[str],
        query: str,
    ) -> list[Any]:  # returns list[Chunk] — imported lazily to avoid circular dep
        from sleuth.types import Chunk, Source

        trafilatura = _lazy_import_trafilatura()

        async def _fetch_one(url: str) -> list[Chunk]:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(
                        url,
                        follow_redirects=True,
                        headers={"User-Agent": "agent-sleuth/0.1 (fetch=True)"},
                    )
                    resp.raise_for_status()
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                logger.debug("FetchPipeline: skipping %s — %s", url, exc)
                return []

            extracted: str | None = trafilatura.extract(resp.text, include_links=False)
            if not extracted:
                return []

            texts = _split_by_tokens(extracted, self._max_tokens)
            source = Source(kind="url", location=url)
            return [
                Chunk(text=t, source=source, score=None, metadata={"query": query})
                for t in texts
                if t.strip()
            ]

        results = await asyncio.gather(*(_fetch_one(u) for u in urls))
        return [chunk for page in results for chunk in page]
