"""Unit tests for shared rate-limit and backoff utilities."""

from __future__ import annotations

import time
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_token_bucket_allows_initial_burst():
    """First N tokens within rate are available immediately."""
    from sleuth.backends._web._base import TokenBucket

    bucket = TokenBucket(rate=10.0, capacity=10.0)
    start = time.monotonic()
    for _ in range(5):
        await bucket.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"Expected fast burst, got {elapsed:.3f}s"


@pytest.mark.unit
async def test_token_bucket_throttles_when_empty():
    """Acquiring more than capacity forces a delay."""
    from sleuth.backends._web._base import TokenBucket

    bucket = TokenBucket(rate=5.0, capacity=2.0)
    # Drain initial tokens quickly
    await bucket.acquire()
    await bucket.acquire()
    start = time.monotonic()
    # Third acquire must wait ~0.2s for one token at rate=5/s
    await bucket.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.15, f"Expected throttle delay, got {elapsed:.3f}s"


@pytest.mark.unit
async def test_token_bucket_per_host_isolation():
    """Two different hosts get independent buckets."""
    from sleuth.backends._web._base import TokenBucket

    b1 = TokenBucket(rate=100.0, capacity=100.0)
    b2 = TokenBucket(rate=1.0, capacity=1.0)
    # b1 should not block; b2 should after first token
    await b1.acquire()
    await b2.acquire()
    start = time.monotonic()
    await b1.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.1, "b1 should not be throttled"


# ---------------------------------------------------------------------------
# with_backoff
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_with_backoff_succeeds_on_first_try():
    """Non-raising coroutine returns immediately."""
    from sleuth.backends._web._base import with_backoff

    calls = []

    async def ok() -> str:
        calls.append(1)
        return "ok"

    result = await with_backoff(ok, max_retries=3, base_delay=0.01)
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.unit
async def test_with_backoff_retries_on_5xx():
    """5xx errors trigger retries up to max_retries."""
    import httpx

    from sleuth.backends._web._base import with_backoff

    attempt = 0

    async def flaky() -> str:
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("GET", "http://x.com"),
                response=httpx.Response(500),
            )
        return "recovered"

    result = await with_backoff(flaky, max_retries=3, base_delay=0.01)
    assert result == "recovered"
    assert attempt == 3


@pytest.mark.unit
async def test_with_backoff_no_retry_on_4xx():
    """4xx errors are not retried — raised immediately."""
    import httpx

    from sleuth.backends._web._base import with_backoff

    attempts = 0

    async def bad_request() -> str:
        nonlocal attempts
        attempts += 1
        raise httpx.HTTPStatusError(
            "Bad Request",
            request=httpx.Request("GET", "http://x.com"),
            response=httpx.Response(400),
        )

    with pytest.raises(httpx.HTTPStatusError):
        await with_backoff(bad_request, max_retries=3, base_delay=0.01)

    assert attempts == 1


@pytest.mark.unit
async def test_with_backoff_429_uses_retry_after_header():
    """429 with Retry-After header waits the header value (capped)."""
    import httpx

    from sleuth.backends._web._base import with_backoff

    attempt = 0

    async def rate_limited() -> str:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            response = httpx.Response(
                429,
                headers={"Retry-After": "0"},  # 0s for test speed
            )
            raise httpx.HTTPStatusError(
                "Too Many Requests",
                request=httpx.Request("GET", "http://x.com"),
                response=response,
            )
        return "ok"

    result = await with_backoff(rate_limited, max_retries=3, base_delay=0.01)
    assert result == "ok"
    assert attempt == 2


@pytest.mark.unit
async def test_with_backoff_raises_after_max_retries():
    """Permanent 5xx raises BackendError after exhausting retries."""
    import httpx

    from sleuth.backends._web._base import with_backoff
    from sleuth.errors import BackendError

    async def always_500() -> str:
        raise httpx.HTTPStatusError(
            "Server Error",
            request=httpx.Request("GET", "http://x.com"),
            response=httpx.Response(500),
        )

    with pytest.raises(BackendError, match="max retries"):
        await with_backoff(always_500, max_retries=3, base_delay=0.01)


# ---------------------------------------------------------------------------
# FetchPipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_fetch_pipeline_returns_chunks(respx_mock: Any) -> None:
    """FetchPipeline fetches URLs in parallel and returns Chunk objects."""
    import respx

    from sleuth.backends._web._base import FetchPipeline
    from sleuth.types import Chunk

    html = (
        "<html><body><article>"
        "<p>This is the first paragraph with enough content to be chunked.</p>"
        "<p>Second paragraph follows with additional context for the test.</p>"
        "</article></body></html>"
    )

    with respx.mock:
        respx.get("https://example.com/page").respond(200, text=html)
        pipeline = FetchPipeline(max_tokens_per_chunk=512)
        chunks = await pipeline.fetch_and_chunk(
            urls=["https://example.com/page"],
            query="test query",
        )

    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source.kind == "url" for c in chunks)
    assert all(c.source.location == "https://example.com/page" for c in chunks)


@pytest.mark.unit
async def test_fetch_pipeline_skips_failed_urls(respx_mock: Any) -> None:
    """FetchPipeline silently skips URLs that return HTTP errors."""
    import respx

    from sleuth.backends._web._base import FetchPipeline

    with respx.mock:
        respx.get("https://example.com/ok").respond(200, text="<p>Good content here for test.</p>")
        respx.get("https://example.com/fail").respond(500)

        pipeline = FetchPipeline(max_tokens_per_chunk=512)
        chunks = await pipeline.fetch_and_chunk(
            urls=["https://example.com/ok", "https://example.com/fail"],
            query="test",
        )

    # Only chunks from the successful URL
    assert all(c.source.location == "https://example.com/ok" for c in chunks)


@pytest.mark.unit
async def test_fetch_pipeline_respects_max_tokens(respx_mock: Any) -> None:
    """Chunks never exceed max_tokens_per_chunk tokens."""
    import respx
    import tiktoken

    from sleuth.backends._web._base import FetchPipeline

    # Build a large enough article so it must be split
    long_text = " ".join(["word"] * 2000)
    html = f"<html><body><article><p>{long_text}</p></article></body></html>"

    with respx.mock:
        respx.get("https://example.com/long").respond(200, text=html)
        pipeline = FetchPipeline(max_tokens_per_chunk=200)
        chunks = await pipeline.fetch_and_chunk(
            urls=["https://example.com/long"],
            query="test",
        )

    enc = tiktoken.get_encoding("cl100k_base")
    for c in chunks:
        token_count = len(enc.encode(c.text))
        assert token_count <= 220, f"Chunk too large: {token_count} tokens"
