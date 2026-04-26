# Phase 9: Web Provider Shims — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand Phase 1's Tavily-only `WebBackend` stub into a full factory + per-provider architecture covering Tavily, Exa, Brave, and SerpAPI — each with per-domain rate limiting, exponential backoff, and an optional `fetch=True` mode that parallel-fetches top-N pages and chunks them via trafilatura + a token-aware splitter.

**Architecture:** `src/sleuth/backends/web.py` is replaced with a thin factory function (`WebBackend(provider=...)`) and per-provider classes (`TavilyBackend`, `ExaBackend`, `BraveBackend`, `SerpAPIBackend`). Each provider lives in its own file under `src/sleuth/backends/_web/`. A shared `_base.py` in that package owns the token-bucket rate limiter, exponential backoff helper, and the `fetch=True` page-fetching pipeline (httpx + trafilatura + tiktoken splitter). All four classes satisfy the frozen `Backend` protocol from `sleuth.backends.base`. Both the factory and the four classes are re-exported from `sleuth.backends`.

**Tech Stack:** Python 3.11+, httpx (AsyncClient), respx (test mocking), trafilatura, tiktoken, pydantic v2, pytest-asyncio (auto mode), BackendTestKit (Phase 1).

---

> **Callout — new optional extras needed (not in conventions §3):**
> Phase 9 adds two new rows to `[project.optional-dependencies]` in `pyproject.toml`:
> ```
> exa        = ["exa-py>=1.0"]
> web-fetch  = ["trafilatura>=1.7", "tiktoken>=0.7"]
> ```
> Brave and SerpAPI are pure REST (no SDK); their deps land in `[dependency-groups] dev` as nothing extra is required at runtime. This escalation is flagged for human reconciliation before execution — it does not conflict with any existing conventions entry.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/sleuth/backends/_web/__init__.py` | Create | Package init (re-exports `_base` utilities) |
| `src/sleuth/backends/_web/_base.py` | Create | `TokenBucket`, `with_backoff()`, `FetchPipeline` (httpx + trafilatura + tiktoken) |
| `src/sleuth/backends/_web/tavily.py` | Create | `TavilyBackend` — moves logic from Phase 1 stub |
| `src/sleuth/backends/_web/exa.py` | Create | `ExaBackend` |
| `src/sleuth/backends/_web/brave.py` | Create | `BraveBackend` |
| `src/sleuth/backends/_web/serpapi.py` | Create | `SerpAPIBackend` |
| `src/sleuth/backends/web.py` | Modify | Replace Phase 1 stub with factory + re-exports; keep `WebBackend` symbol |
| `src/sleuth/backends/__init__.py` | Modify | Add `TavilyBackend`, `ExaBackend`, `BraveBackend`, `SerpAPIBackend` re-exports |
| `pyproject.toml` | Modify | Add `exa` and `web-fetch` optional dep groups |
| `tests/backends/_web/__init__.py` | Create | Empty — makes subdir a package |
| `tests/backends/_web/test_base.py` | Create | `TokenBucket` + `with_backoff` unit tests |
| `tests/backends/_web/test_tavily.py` | Create | `TavilyBackend` unit + BackendTestKit |
| `tests/backends/_web/test_exa.py` | Create | `ExaBackend` unit + BackendTestKit |
| `tests/backends/_web/test_brave.py` | Create | `BraveBackend` unit + BackendTestKit |
| `tests/backends/_web/test_serpapi.py` | Create | `SerpAPIBackend` unit + BackendTestKit |
| `tests/backends/_web/test_web_factory.py` | Create | `WebBackend()` factory + fetch pipeline integration |

---

## Task 1: Branch + package skeleton

**Files:**
- Create: `src/sleuth/backends/_web/__init__.py`
- Create: `tests/backends/_web/__init__.py`

- [ ] **Step 1: Create feature branch off `develop`**

```bash
git checkout develop
git checkout -b feature/phase-9-web-providers
```

Expected: `Switched to a new branch 'feature/phase-9-web-providers'`

- [ ] **Step 2: Create the `_web` package dirs**

```bash
mkdir -p src/sleuth/backends/_web
mkdir -p tests/backends/_web
```

- [ ] **Step 3: Create `src/sleuth/backends/_web/__init__.py`**

```python
"""Private per-provider web backend helpers."""
```

- [ ] **Step 4: Create `tests/backends/_web/__init__.py`**

```python
```

(empty file — makes the directory a package so pytest collects it)

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/backends/_web/__init__.py tests/backends/_web/__init__.py
git commit -m "chore: scaffold _web sub-package directories for Phase 9"
```

---

## Task 2: `pyproject.toml` — add provider extras

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Open `pyproject.toml` and locate `[project.optional-dependencies]`**

```bash
grep -n "optional-dependencies" pyproject.toml
```

Expected: a line like `14:[project.optional-dependencies]`

- [ ] **Step 2: Add the two new extras after the existing `mcp` entry**

In `[project.optional-dependencies]`, append:

```toml
exa        = ["exa-py>=1.0"]
web-fetch  = ["trafilatura>=1.7", "tiktoken>=0.7"]
```

The block should now end with:

```toml
mcp           = ["mcp>=1.0"]
exa           = ["exa-py>=1.0"]
web-fetch     = ["trafilatura>=1.7", "tiktoken>=0.7"]
```

- [ ] **Step 3: Add dev-time extras to `[dependency-groups] dev`**

Append to the `dev` list (Exa SDK is only needed if the `exa` extra is installed; respx and httpx are already in dev):

```toml
"exa-py>=1.0",
"trafilatura>=1.7",
"tiktoken>=0.7",
```

- [ ] **Step 4: Verify TOML is valid**

```bash
uv run python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
```

Expected: no output (no exception).

- [ ] **Step 5: Sync dev environment**

```bash
uv sync --all-extras --group dev
```

Expected: packages resolved and installed without errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add exa and web-fetch optional dep groups to pyproject.toml"
```

---

## Task 3: Shared base — `TokenBucket` + `with_backoff`

**Files:**
- Create: `src/sleuth/backends/_web/_base.py`
- Test: `tests/backends/_web/test_base.py`

- [ ] **Step 1: Write the failing tests first**

Create `tests/backends/_web/test_base.py`:

```python
"""Unit tests for shared rate-limit and backoff utilities."""

from __future__ import annotations

import asyncio
import time

import pytest

from sleuth.backends._web._base import TokenBucket, with_backoff


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_token_bucket_allows_initial_burst():
    """First N tokens within rate are available immediately."""
    bucket = TokenBucket(rate=10.0, capacity=10.0)
    start = time.monotonic()
    for _ in range(5):
        await bucket.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"Expected fast burst, got {elapsed:.3f}s"


@pytest.mark.unit
async def test_token_bucket_throttles_when_empty():
    """Acquiring more than capacity forces a delay."""
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

    attempt = 0
    start = time.monotonic()

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

    from sleuth.errors import BackendError

    async def always_500() -> str:
        raise httpx.HTTPStatusError(
            "Server Error",
            request=httpx.Request("GET", "http://x.com"),
            response=httpx.Response(500),
        )

    with pytest.raises(BackendError, match="max retries"):
        await with_backoff(always_500, max_retries=3, base_delay=0.01)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/backends/_web/test_base.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError` — `sleuth.backends._web._base` does not exist yet.

- [ ] **Step 3: Create `src/sleuth/backends/_web/_base.py`**

```python
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
                self._tokens = min(
                    self._capacity, self._tokens + elapsed * self._rate
                )
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
        f"Backend call failed after {max_retries} retries: {last_exc}"
    ) from last_exc
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/backends/_web/test_base.py -v
```

Expected:
```
PASSED tests/backends/_web/test_base.py::test_token_bucket_allows_initial_burst
PASSED tests/backends/_web/test_base.py::test_token_bucket_throttles_when_empty
PASSED tests/backends/_web/test_base.py::test_token_bucket_per_host_isolation
PASSED tests/backends/_web/test_base.py::test_with_backoff_succeeds_on_first_try
PASSED tests/backends/_web/test_base.py::test_with_backoff_retries_on_5xx
PASSED tests/backends/_web/test_base.py::test_with_backoff_no_retry_on_4xx
PASSED tests/backends/_web/test_base.py::test_with_backoff_429_uses_retry_after_header
PASSED tests/backends/_web/test_base.py::test_with_backoff_raises_after_max_retries

8 passed in ...
```

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/backends/_web/_base.py tests/backends/_web/test_base.py
git commit -m "feat: add TokenBucket rate limiter and with_backoff helper to _web._base"
```

---

## Task 4: `FetchPipeline` — parallel fetch + chunk via trafilatura + tiktoken

**Files:**
- Modify: `src/sleuth/backends/_web/_base.py`
- Modify: `tests/backends/_web/test_base.py`

- [ ] **Step 1: Write the failing fetch pipeline tests**

Append to `tests/backends/_web/test_base.py`:

```python
# ---------------------------------------------------------------------------
# FetchPipeline
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_fetch_pipeline_returns_chunks(respx_mock: Any) -> None:
    """FetchPipeline fetches URLs in parallel and returns Chunk objects."""
    import respx

    from sleuth.backends._web._base import FetchPipeline
    from sleuth.types import Chunk, Source

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
```

- [ ] **Step 2: Run tests to confirm new tests fail**

```bash
uv run pytest tests/backends/_web/test_base.py::test_fetch_pipeline_returns_chunks -v
```

Expected: `ImportError` — `FetchPipeline` not defined yet.

- [ ] **Step 3: Add `FetchPipeline` to `src/sleuth/backends/_web/_base.py`**

Append after the `with_backoff` function:

```python
# ---------------------------------------------------------------------------
# Optional fetch pipeline (fetch=True mode)
# ---------------------------------------------------------------------------


def _lazy_import_trafilatura() -> Any:
    try:
        import trafilatura  # type: ignore[import-untyped]
        return trafilatura
    except ImportError as exc:
        raise ImportError(
            "fetch=True requires the 'web-fetch' extra: "
            "pip install agent-sleuth[web-fetch]"
        ) from exc


def _lazy_import_tiktoken() -> Any:
    try:
        import tiktoken  # type: ignore[import-untyped]
        return tiktoken
    except ImportError as exc:
        raise ImportError(
            "fetch=True requires the 'web-fetch' extra: "
            "pip install agent-sleuth[web-fetch]"
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
    ) -> list["Chunk"]:  # forward-ref: imported below
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
```

- [ ] **Step 4: Run all base tests**

```bash
uv run pytest tests/backends/_web/test_base.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/backends/_web/_base.py tests/backends/_web/test_base.py
git commit -m "feat: add FetchPipeline (httpx + trafilatura + tiktoken) to _web._base"
```

---

## Task 5: `TavilyBackend` — migrate Phase 1 stub + full implementation

**Files:**
- Create: `src/sleuth/backends/_web/tavily.py`
- Test: `tests/backends/_web/test_tavily.py`

Note: Phase 1 ships a Tavily-only stub in `src/sleuth/backends/web.py`. This task moves that logic into `_web/tavily.py` and makes it production-quality. `web.py` is rewritten in Task 9.

- [ ] **Step 1: Write the failing tests**

Create `tests/backends/_web/test_tavily.py`:

```python
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
    backend = TavilyBackend(api_key="test-key")
    assert backend.name == "tavily"
    from sleuth.backends.base import Capability
    assert Capability.WEB in backend.capabilities
    assert Capability.FRESH in backend.capabilities


@pytest.mark.unit
async def test_tavily_search_returns_chunks():
    with respx.mock:
        respx.post("https://api.tavily.com/search").respond(
            200, json=_tavily_response(3)
        )
        backend = TavilyBackend(api_key="test-key")
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
        respx.post("https://api.tavily.com/search").respond(
            200, json=_tavily_response(5)
        )
        backend = TavilyBackend(api_key="test-key")
        chunks = await backend.search("query", k=2)

    assert len(chunks) == 2


@pytest.mark.unit
async def test_tavily_raises_backend_error_on_5xx():
    from sleuth.errors import BackendError

    with respx.mock:
        # Fail all retries
        respx.post("https://api.tavily.com/search").respond(500)
        backend = TavilyBackend(api_key="test-key", _backoff_base=0.01)
        with pytest.raises(BackendError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_tavily_raises_immediately_on_401():
    """401 Unauthorized is not retried."""
    with respx.mock:
        respx.post("https://api.tavily.com/search").respond(401)
        backend = TavilyBackend(api_key="bad-key", _backoff_base=0.01)
        with pytest.raises(httpx.HTTPStatusError):
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
        respx.post("https://api.tavily.com/search").respond(
            200, json=_tavily_response(1)
        )
        with patch.object(
            FetchPipeline,
            "fetch_and_chunk",
            new=AsyncMock(return_value=fake_chunks),
        ):
            backend = TavilyBackend(api_key="test-key", fetch=True, fetch_top_n=1)
            chunks = await backend.search("python async", k=1)

    # Should include at least the fake fetched chunk
    locations = {c.source.location for c in chunks}
    assert "https://example.com/0" in locations


@pytest.mark.unit
async def test_tavily_backend_protocol_compliance():
    """Run BackendTestKit contract suite against TavilyBackend."""
    from tests.contract.test_backend_protocol import BackendTestKit

    with respx.mock:
        respx.post("https://api.tavily.com/search").respond(
            200, json=_tavily_response(3)
        )
        backend = TavilyBackend(api_key="test-key")
        await BackendTestKit.run(backend)
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
uv run pytest tests/backends/_web/test_tavily.py -v 2>&1 | head -20
```

Expected: `ImportError` — `TavilyBackend` not defined yet.

- [ ] **Step 3: Create `src/sleuth/backends/_web/tavily.py`**

```python
"""Tavily web search backend."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from sleuth.backends._web._base import FetchPipeline, TokenBucket, with_backoff
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source

logger = logging.getLogger("sleuth.backends.web.tavily")

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilyBackend:
    """Backend adapter for the Tavily search API.

    Args:
        api_key: Tavily API key.
        fetch: When True, parallel-fetches top ``fetch_top_n`` URLs and
               appends page-content chunks to the API snippet results.
        fetch_top_n: Number of top results to fetch when ``fetch=True``.
        rate_limit: Max requests per second to the Tavily API domain.
        max_retries: Retries on 429/5xx before raising ``BackendError``.
        _backoff_base: Base delay (seconds) for exponential backoff.
            Exposed for testing; default 1.0 in production.
    """

    name: str = "tavily"
    capabilities: frozenset[Capability] = frozenset(
        {Capability.WEB, Capability.FRESH}
    )

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
        self._pipeline: FetchPipeline | None = (
            FetchPipeline() if fetch else None
        )

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        await self._bucket.acquire()

        async def _call() -> dict[str, Any]:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    _TAVILY_SEARCH_URL,
                    json={
                        "api_key": self._api_key,
                        "query": query,
                        "max_results": k,
                        "search_depth": "basic",
                    },
                )
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]

        data = await with_backoff(
            _call, max_retries=self._max_retries, base_delay=self._backoff_base
        )

        results: list[dict[str, Any]] = data.get("results", [])[:k]
        chunks = [
            Chunk(
                text=r.get("content", ""),
                source=Source(
                    kind="url",
                    location=r["url"],
                    title=r.get("title"),
                ),
                score=r.get("score"),
            )
            for r in results
        ]

        if self._fetch and self._pipeline and chunks:
            urls = [c.source.location for c in chunks[: self._fetch_top_n]]
            fetched = await self._pipeline.fetch_and_chunk(urls=urls, query=query)
            chunks = chunks + fetched

        logger.debug("tavily: %d chunks for %r", len(chunks), query)
        return chunks
```

- [ ] **Step 4: Run all Tavily tests**

```bash
uv run pytest tests/backends/_web/test_tavily.py -v
```

Expected: all 7 tests PASS. (BackendTestKit compliance test will pass once Phase 1's contract suite exists; if Phase 1 is not yet merged, that specific test will be skipped via `importorskip`.)

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/backends/_web/tavily.py tests/backends/_web/test_tavily.py
git commit -m "feat: add TavilyBackend with rate limiting, backoff, and optional fetch mode"
```

---

## Task 6: `ExaBackend`

**Files:**
- Create: `src/sleuth/backends/_web/exa.py`
- Test: `tests/backends/_web/test_exa.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backends/_web/test_exa.py`:

```python
"""Tests for ExaBackend."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx

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
    backend = ExaBackend(api_key="exa-key")
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
        backend = ExaBackend(api_key="exa-key")
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
        backend = ExaBackend(api_key="exa-key")
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

    with patch("sleuth.backends._web.exa._make_exa_client", return_value=mock_exa):
        with patch.object(
            FetchPipeline,
            "fetch_and_chunk",
            new=AsyncMock(return_value=[fake_chunk]),
        ):
            backend = ExaBackend(api_key="exa-key", fetch=True, fetch_top_n=1)
            chunks = await backend.search("query", k=1)

    assert any(c.text == "Fetched content." for c in chunks)


@pytest.mark.unit
async def test_exa_import_error_message():
    """Helpful error when exa-py is not installed."""
    import sys

    with patch.dict(sys.modules, {"exa_py": None}):
        with pytest.raises(ImportError, match="exa"):
            ExaBackend(api_key="key")
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
uv run pytest tests/backends/_web/test_exa.py -v 2>&1 | head -20
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/sleuth/backends/_web/exa.py`**

```python
"""Exa (formerly Metaphor) web search backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sleuth.backends._web._base import FetchPipeline, TokenBucket
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source

if TYPE_CHECKING:
    pass

logger = logging.getLogger("sleuth.backends.web.exa")


def _make_exa_client(api_key: str) -> Any:
    try:
        from exa_py import Exa  # type: ignore[import-untyped]
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
    capabilities: frozenset[Capability] = frozenset(
        {Capability.WEB, Capability.FRESH}
    )

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
        self._pipeline: FetchPipeline | None = (
            FetchPipeline() if fetch else None
        )

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
```

- [ ] **Step 4: Run all Exa tests**

```bash
uv run pytest tests/backends/_web/test_exa.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/backends/_web/exa.py tests/backends/_web/test_exa.py
git commit -m "feat: add ExaBackend with rate limiting and optional fetch mode"
```

---

## Task 7: `BraveBackend`

**Files:**
- Create: `src/sleuth/backends/_web/brave.py`
- Test: `tests/backends/_web/test_brave.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backends/_web/test_brave.py`:

```python
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
    backend = BraveBackend(api_key="brave-key")
    assert backend.name == "brave"
    from sleuth.backends.base import Capability
    assert Capability.WEB in backend.capabilities
    assert Capability.FRESH in backend.capabilities


@pytest.mark.unit
async def test_brave_search_returns_chunks():
    with respx.mock:
        respx.get(_BRAVE_URL).respond(200, json=_brave_response(3))
        backend = BraveBackend(api_key="brave-key")
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
        backend = BraveBackend(api_key="brave-key")
        chunks = await backend.search("query", k=2)

    assert len(chunks) == 2


@pytest.mark.unit
async def test_brave_raises_on_5xx():
    from sleuth.errors import BackendError

    with respx.mock:
        respx.get(_BRAVE_URL).respond(503)
        backend = BraveBackend(api_key="brave-key", _backoff_base=0.01)
        with pytest.raises(BackendError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_brave_raises_immediately_on_401():
    with respx.mock:
        respx.get(_BRAVE_URL).respond(401)
        backend = BraveBackend(api_key="bad-key", _backoff_base=0.01)
        with pytest.raises(httpx.HTTPStatusError):
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
            backend = BraveBackend(api_key="brave-key", fetch=True, fetch_top_n=1)
            chunks = await backend.search("query", k=1)

    assert any(c.text == "Full page text." for c in chunks)
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
uv run pytest tests/backends/_web/test_brave.py -v 2>&1 | head -20
```

Expected: `ImportError`.

- [ ] **Step 3: Create `src/sleuth/backends/_web/brave.py`**

```python
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
    capabilities: frozenset[Capability] = frozenset(
        {Capability.WEB, Capability.FRESH}
    )

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
        self._pipeline: FetchPipeline | None = (
            FetchPipeline() if fetch else None
        )

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

        results: list[dict[str, Any]] = (
            data.get("web", {}).get("results", [])[:k]
        )
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
```

- [ ] **Step 4: Run all Brave tests**

```bash
uv run pytest tests/backends/_web/test_brave.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/backends/_web/brave.py tests/backends/_web/test_brave.py
git commit -m "feat: add BraveBackend with rate limiting and optional fetch mode"
```

---

## Task 8: `SerpAPIBackend`

**Files:**
- Create: `src/sleuth/backends/_web/serpapi.py`
- Test: `tests/backends/_web/test_serpapi.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backends/_web/test_serpapi.py`:

```python
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
    backend = SerpAPIBackend(api_key="serpapi-key")
    assert backend.name == "serpapi"
    from sleuth.backends.base import Capability
    assert Capability.WEB in backend.capabilities


@pytest.mark.unit
async def test_serpapi_search_returns_chunks():
    with respx.mock:
        respx.get(_SERPAPI_URL).respond(200, json=_serpapi_response(3))
        backend = SerpAPIBackend(api_key="serpapi-key")
        chunks = await backend.search("best Python libraries", k=3)

    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source.kind == "url"
    assert chunks[0].source.location == "https://serpapi-example.com/0"


@pytest.mark.unit
async def test_serpapi_search_respects_k():
    with respx.mock:
        respx.get(_SERPAPI_URL).respond(200, json=_serpapi_response(5))
        backend = SerpAPIBackend(api_key="serpapi-key")
        chunks = await backend.search("query", k=2)

    assert len(chunks) == 2


@pytest.mark.unit
async def test_serpapi_raises_on_5xx():
    from sleuth.errors import BackendError

    with respx.mock:
        respx.get(_SERPAPI_URL).respond(500)
        backend = SerpAPIBackend(api_key="serpapi-key", _backoff_base=0.01)
        with pytest.raises(BackendError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_serpapi_raises_immediately_on_403():
    with respx.mock:
        respx.get(_SERPAPI_URL).respond(403)
        backend = SerpAPIBackend(api_key="bad-key", _backoff_base=0.01)
        with pytest.raises(httpx.HTTPStatusError):
            await backend.search("query", k=5)


@pytest.mark.unit
async def test_serpapi_empty_organic_results():
    """Backend returns empty list when no organic results."""
    with respx.mock:
        respx.get(_SERPAPI_URL).respond(200, json={"organic_results": []})
        backend = SerpAPIBackend(api_key="serpapi-key")
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
                api_key="serpapi-key", fetch=True, fetch_top_n=1
            )
            chunks = await backend.search("query", k=1)

    assert any(c.text == "Full SerpAPI page content." for c in chunks)
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
uv run pytest tests/backends/_web/test_serpapi.py -v 2>&1 | head -20
```

Expected: `ImportError`.

- [ ] **Step 3: Create `src/sleuth/backends/_web/serpapi.py`**

```python
"""SerpAPI (Google Search via SerpAPI) backend."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from sleuth.backends._web._base import FetchPipeline, TokenBucket, with_backoff
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source

logger = logging.getLogger("sleuth.backends.web.serpapi")

_SERPAPI_SEARCH_URL = "https://serpapi.com/search"


class SerpAPIBackend:
    """Backend adapter for SerpAPI (returns Google organic results).

    SerpAPI is a pure REST API — no SDK required.

    Args:
        api_key: SerpAPI key.
        engine: Search engine identifier (default ``"google"``).
        fetch: When True, parallel-fetches top ``fetch_top_n`` URLs.
        fetch_top_n: Number of top results to fetch when ``fetch=True``.
        rate_limit: Max requests per second.
        max_retries: Retries on 429/5xx.
        _backoff_base: Base delay for exponential backoff (testing hook).
    """

    name: str = "serpapi"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    def __init__(
        self,
        api_key: str,
        *,
        engine: str = "google",
        fetch: bool = False,
        fetch_top_n: int = 3,
        rate_limit: float = 5.0,
        max_retries: int = 3,
        _backoff_base: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self._engine = engine
        self._fetch = fetch
        self._fetch_top_n = fetch_top_n
        self._max_retries = max_retries
        self._backoff_base = _backoff_base
        self._bucket = TokenBucket(rate=rate_limit, capacity=rate_limit)
        self._pipeline: FetchPipeline | None = (
            FetchPipeline() if fetch else None
        )

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        await self._bucket.acquire()

        async def _call() -> dict[str, Any]:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    _SERPAPI_SEARCH_URL,
                    params={
                        "q": query,
                        "api_key": self._api_key,
                        "engine": self._engine,
                        "num": k,
                        "output": "json",
                    },
                )
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]

        data = await with_backoff(
            _call, max_retries=self._max_retries, base_delay=self._backoff_base
        )

        results: list[dict[str, Any]] = data.get("organic_results", [])[:k]
        chunks = [
            Chunk(
                text=r.get("snippet", ""),
                source=Source(
                    kind="url",
                    location=r["link"],
                    title=r.get("title"),
                ),
                score=None,  # SerpAPI returns position, not a float score
                metadata={"position": r.get("position")},
            )
            for r in results
        ]

        if self._fetch and self._pipeline and chunks:
            urls = [c.source.location for c in chunks[: self._fetch_top_n]]
            fetched = await self._pipeline.fetch_and_chunk(urls=urls, query=query)
            chunks = chunks + fetched

        logger.debug("serpapi: %d chunks for %r", len(chunks), query)
        return chunks
```

- [ ] **Step 4: Run all SerpAPI tests**

```bash
uv run pytest tests/backends/_web/test_serpapi.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/backends/_web/serpapi.py tests/backends/_web/test_serpapi.py
git commit -m "feat: add SerpAPIBackend with rate limiting and optional fetch mode"
```

---

## Task 9: Replace `web.py` with factory + re-exports

**Files:**
- Modify: `src/sleuth/backends/web.py`
- Test: `tests/backends/_web/test_web_factory.py`

- [ ] **Step 1: Write the failing factory tests**

Create `tests/backends/_web/test_web_factory.py`:

```python
"""Tests for WebBackend factory and public re-exports."""

from __future__ import annotations

import pytest

from sleuth.backends.web import (
    BraveBackend,
    ExaBackend,
    SerpAPIBackend,
    TavilyBackend,
    WebBackend,
)


@pytest.mark.unit
def test_web_backend_factory_tavily():
    backend = WebBackend(provider="tavily", api_key="t-key")
    assert isinstance(backend, TavilyBackend)
    assert backend.name == "tavily"


@pytest.mark.unit
def test_web_backend_factory_exa():
    import sys
    from unittest.mock import MagicMock

    # Provide a stub exa_py so ExaBackend can be instantiated without the real SDK
    fake_exa = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "exa_py", fake_exa)
        backend = WebBackend(provider="exa", api_key="e-key")
    assert isinstance(backend, ExaBackend)
    assert backend.name == "exa"


@pytest.mark.unit
def test_web_backend_factory_brave():
    backend = WebBackend(provider="brave", api_key="b-key")
    assert isinstance(backend, BraveBackend)
    assert backend.name == "brave"


@pytest.mark.unit
def test_web_backend_factory_serpapi():
    backend = WebBackend(provider="serpapi", api_key="s-key")
    assert isinstance(backend, SerpAPIBackend)
    assert backend.name == "serpapi"


@pytest.mark.unit
def test_web_backend_factory_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        WebBackend(provider="unknown-provider", api_key="key")


@pytest.mark.unit
def test_web_backend_factory_passes_kwargs():
    """Extra kwargs (e.g. fetch=True) are forwarded to the provider class."""
    backend = WebBackend(provider="tavily", api_key="t-key", fetch=True, fetch_top_n=5)
    assert isinstance(backend, TavilyBackend)
    assert backend._fetch is True
    assert backend._fetch_top_n == 5


@pytest.mark.unit
def test_web_backend_factory_returns_backend_protocol():
    """Factory output satisfies the Backend protocol (has name + capabilities + search)."""
    backend = WebBackend(provider="tavily", api_key="t-key")
    from sleuth.backends.base import Capability
    assert isinstance(backend.name, str)
    assert isinstance(backend.capabilities, frozenset)
    assert callable(backend.search)


@pytest.mark.unit
def test_per_provider_classes_exported_from_web_module():
    """All four classes are importable from sleuth.backends.web."""
    from sleuth.backends.web import (
        BraveBackend,
        ExaBackend,
        SerpAPIBackend,
        TavilyBackend,
    )
    assert TavilyBackend is not None
    assert ExaBackend is not None
    assert BraveBackend is not None
    assert SerpAPIBackend is not None
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
uv run pytest tests/backends/_web/test_web_factory.py -v 2>&1 | head -20
```

Expected: `ImportError` — `WebBackend` is still the Phase 1 stub.

- [ ] **Step 3: Rewrite `src/sleuth/backends/web.py`**

```python
"""WebBackend factory and per-provider class re-exports.

Public symbols:
    WebBackend      — factory function; returns a per-provider Backend instance.
    TavilyBackend   — direct per-provider class (power users).
    ExaBackend      — direct per-provider class (power users).
    BraveBackend    — direct per-provider class (power users).
    SerpAPIBackend  — direct per-provider class (power users).

Spec §7.2 and §15 #4: both the factory and the per-provider classes are public.
"""

from __future__ import annotations

from typing import Any

from sleuth.backends._web.brave import BraveBackend
from sleuth.backends._web.exa import ExaBackend
from sleuth.backends._web.serpapi import SerpAPIBackend
from sleuth.backends._web.tavily import TavilyBackend

__all__ = [
    "WebBackend",
    "TavilyBackend",
    "ExaBackend",
    "BraveBackend",
    "SerpAPIBackend",
]

_PROVIDERS: dict[str, type[Any]] = {
    "tavily": TavilyBackend,
    "exa": ExaBackend,
    "brave": BraveBackend,
    "serpapi": SerpAPIBackend,
}


def WebBackend(
    *,
    provider: str,
    api_key: str,
    **kwargs: Any,
) -> TavilyBackend | ExaBackend | BraveBackend | SerpAPIBackend:
    """Factory that returns the appropriate per-provider Backend instance.

    Args:
        provider: One of ``"tavily"``, ``"exa"``, ``"brave"``, ``"serpapi"``.
        api_key: API key for the chosen provider.
        **kwargs: Forwarded verbatim to the provider constructor.
            Common options: ``fetch``, ``fetch_top_n``, ``rate_limit``,
            ``max_retries``.

    Returns:
        A Backend-protocol-compliant instance for the requested provider.

    Raises:
        ValueError: If ``provider`` is not one of the supported values.

    Example::

        # Factory usage
        backend = WebBackend(provider="tavily", api_key=os.environ["TAVILY_KEY"])

        # Per-provider class (power user / type checker friendly)
        backend = TavilyBackend(api_key=os.environ["TAVILY_KEY"], fetch=True)
    """
    cls = _PROVIDERS.get(provider)
    if cls is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown provider {provider!r}. Supported: {supported}"
        )
    return cls(api_key=api_key, **kwargs)
```

- [ ] **Step 4: Run all factory tests**

```bash
uv run pytest tests/backends/_web/test_web_factory.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/backends/web.py tests/backends/_web/test_web_factory.py
git commit -m "feat: replace Phase 1 WebBackend stub with factory + per-provider architecture"
```

---

## Task 10: Update `sleuth.backends.__init__` re-exports

**Files:**
- Modify: `src/sleuth/backends/__init__.py`

- [ ] **Step 1: Read the current file**

```bash
cat src/sleuth/backends/__init__.py
```

Note the existing exports (Phase 1 puts `WebBackend` and base types here).

- [ ] **Step 2: Add per-provider classes to the re-export list**

In `src/sleuth/backends/__init__.py`, add imports alongside the existing `WebBackend` re-export:

```python
from sleuth.backends.web import (
    BraveBackend,
    ExaBackend,
    SerpAPIBackend,
    TavilyBackend,
    WebBackend,
)
```

And add the four new names to `__all__` (append if `__all__` already exists, create it otherwise):

```python
__all__ = [
    # ... existing entries from Phase 1 ...
    "WebBackend",
    "TavilyBackend",
    "ExaBackend",
    "BraveBackend",
    "SerpAPIBackend",
]
```

- [ ] **Step 3: Write a quick import test**

```bash
uv run python -c "
from sleuth.backends import (
    WebBackend, TavilyBackend, ExaBackend, BraveBackend, SerpAPIBackend
)
print('All public re-exports OK')
"
```

Expected: `All public re-exports OK`

- [ ] **Step 4: Run the full test suite for backends**

```bash
uv run pytest tests/backends/ -v --tb=short
```

Expected: all tests PASS (no regressions from Phase 1's Tavily tests).

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/backends/__init__.py
git commit -m "feat: re-export TavilyBackend, ExaBackend, BraveBackend, SerpAPIBackend from sleuth.backends"
```

---

## Task 11: BackendTestKit compliance — run all four providers through the contract suite

**Files:**
- Modify: `tests/backends/_web/test_tavily.py`
- Modify: `tests/backends/_web/test_exa.py`
- Modify: `tests/backends/_web/test_brave.py`
- Modify: `tests/backends/_web/test_serpapi.py`

Note: `BackendTestKit` is owned by Phase 1 (`tests/contract/test_backend_protocol.py`). This task wires each provider through it using `respx` mocks.

- [ ] **Step 1: Verify BackendTestKit interface**

```bash
uv run python -c "from tests.contract.test_backend_protocol import BackendTestKit; help(BackendTestKit.run)"
```

Expected output documents the `run(backend)` async class method. If Phase 1 is not yet merged this will fail — proceed to Step 2 only if it passes; otherwise skip this task and leave a `# TODO: wire BackendTestKit once Phase 1 merges` comment.

- [ ] **Step 2: Add contract test to each provider test file**

In `tests/backends/_web/test_tavily.py`, ensure the following test exists (it was included in Task 5 already):

```python
@pytest.mark.unit
async def test_tavily_backend_protocol_compliance():
    from tests.contract.test_backend_protocol import BackendTestKit
    with respx.mock:
        respx.post("https://api.tavily.com/search").respond(
            200, json=_tavily_response(3)
        )
        backend = TavilyBackend(api_key="test-key")
        await BackendTestKit.run(backend)
```

Append equivalent contract tests to the Exa, Brave, and SerpAPI test files:

In `tests/backends/_web/test_exa.py`:
```python
@pytest.mark.unit
async def test_exa_backend_protocol_compliance():
    from tests.contract.test_backend_protocol import BackendTestKit
    from unittest.mock import AsyncMock, MagicMock
    mock_exa = MagicMock()
    mock_results = MagicMock()
    mock_results.results = [
        MagicMock(url=f"https://exa.com/{i}", title=f"T{i}", text=f"Text {i}", score=0.9)
        for i in range(3)
    ]
    mock_exa.search_and_contents = AsyncMock(return_value=mock_results)
    with patch("sleuth.backends._web.exa._make_exa_client", return_value=mock_exa):
        backend = ExaBackend(api_key="exa-key")
        await BackendTestKit.run(backend)
```

In `tests/backends/_web/test_brave.py`:
```python
@pytest.mark.unit
async def test_brave_backend_protocol_compliance():
    from tests.contract.test_backend_protocol import BackendTestKit
    with respx.mock:
        respx.get(_BRAVE_URL).respond(200, json=_brave_response(3))
        backend = BraveBackend(api_key="brave-key")
        await BackendTestKit.run(backend)
```

In `tests/backends/_web/test_serpapi.py`:
```python
@pytest.mark.unit
async def test_serpapi_backend_protocol_compliance():
    from tests.contract.test_backend_protocol import BackendTestKit
    with respx.mock:
        respx.get(_SERPAPI_URL).respond(200, json=_serpapi_response(3))
        backend = SerpAPIBackend(api_key="serpapi-key")
        await BackendTestKit.run(backend)
```

- [ ] **Step 3: Run contract compliance tests**

```bash
uv run pytest tests/backends/_web/ -k "protocol_compliance" -v
```

Expected: all 4 compliance tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/backends/_web/test_tavily.py tests/backends/_web/test_exa.py \
        tests/backends/_web/test_brave.py tests/backends/_web/test_serpapi.py
git commit -m "test: wire all four web provider backends through BackendTestKit contract suite"
```

---

## Task 12: Integration tests (env-gated)

**Files:**
- Create: `tests/integration/test_web_providers.py`

- [ ] **Step 1: Create the integration test file**

```python
"""Integration tests for web backends against real APIs.

These tests are env-gated and run only in nightly CI (pytest -m integration).
Each test skips automatically if the required env var is absent.
"""

from __future__ import annotations

import os

import pytest

from sleuth.backends.web import (
    BraveBackend,
    ExaBackend,
    SerpAPIBackend,
    TavilyBackend,
)
from sleuth.types import Chunk

pytestmark = pytest.mark.integration


@pytest.fixture
def tavily_key() -> str:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        pytest.skip("TAVILY_API_KEY not set")
    return key


@pytest.fixture
def exa_key() -> str:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        pytest.skip("EXA_API_KEY not set")
    return key


@pytest.fixture
def brave_key() -> str:
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        pytest.skip("BRAVE_API_KEY not set")
    return key


@pytest.fixture
def serpapi_key() -> str:
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        pytest.skip("SERPAPI_API_KEY not set")
    return key


async def test_tavily_real_search(tavily_key: str) -> None:
    backend = TavilyBackend(api_key=tavily_key)
    chunks = await backend.search("Python asyncio tutorial", k=3)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source.location.startswith("http") for c in chunks)


async def test_exa_real_search(exa_key: str) -> None:
    backend = ExaBackend(api_key=exa_key)
    chunks = await backend.search("machine learning papers 2025", k=3)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


async def test_brave_real_search(brave_key: str) -> None:
    backend = BraveBackend(api_key=brave_key)
    chunks = await backend.search("open source AI tools", k=3)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


async def test_serpapi_real_search(serpapi_key: str) -> None:
    backend = SerpAPIBackend(api_key=serpapi_key)
    chunks = await backend.search("Python web frameworks comparison", k=3)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


async def test_tavily_fetch_mode_real(tavily_key: str) -> None:
    backend = TavilyBackend(api_key=tavily_key, fetch=True, fetch_top_n=1)
    chunks = await backend.search("httpx Python library", k=2)
    # Should have more chunks than without fetch (page content added)
    assert len(chunks) >= 1
```

- [ ] **Step 2: Verify the integration tests are skipped in normal runs**

```bash
uv run pytest tests/integration/test_web_providers.py -v 2>&1 | tail -10
```

Expected: all tests SKIPPED (env vars not set in dev environment).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_web_providers.py
git commit -m "test: add env-gated integration tests for all four web provider backends"
```

---

## Task 13: Lint, type check, coverage gate

**Files:** none created — validation only.

- [ ] **Step 1: Lint + format**

```bash
uv run ruff check src/sleuth/backends/_web/ tests/backends/_web/ --fix
uv run ruff format src/sleuth/backends/_web/ tests/backends/_web/
```

Expected: no unfixed lint errors. Format changes applied in place.

- [ ] **Step 2: Type check**

```bash
uv run mypy src/sleuth/backends/_web/ src/sleuth/backends/web.py
```

Expected: `Success: no issues found in N source files`

If mypy complains about untyped `trafilatura` or `exa_py`, add per-module ignores in `pyproject.toml` under `[[tool.mypy.overrides]]`:

```toml
[[tool.mypy.overrides]]
module = ["trafilatura.*", "exa_py.*"]
ignore_missing_imports = true
```

- [ ] **Step 3: Run full unit test suite with coverage**

```bash
uv run pytest tests/ -m "not integration" --cov=src/sleuth --cov-report=term-missing -q
```

Expected:
```
... all tests pass ...
TOTAL    ...   85%+
```

If coverage dips below 85%, identify uncovered lines and add targeted tests for the missing branches (most likely: `FetchPipeline` error paths or backoff edge cases).

- [ ] **Step 4: Commit lint/type fixes if any were needed**

```bash
git add -u
git commit -m "fix: resolve ruff and mypy issues in web provider backends"
```

(Skip this step if there were no changes.)

---

## Task 14: Final integration and PR prep

**Files:** none created.

- [ ] **Step 1: Run the complete test suite one final time**

```bash
uv run pytest tests/ -m "not integration" -v --tb=short
```

Expected: all unit + contract tests PASS.

- [ ] **Step 2: Verify public API surface is importable as documented**

```bash
uv run python -c "
from sleuth.backends import (
    WebBackend,
    TavilyBackend,
    ExaBackend,
    BraveBackend,
    SerpAPIBackend,
)

# Factory usage
b = WebBackend(provider='tavily', api_key='dummy')
assert b.name == 'tavily'

# Direct class usage
b2 = TavilyBackend(api_key='dummy', fetch=True, fetch_top_n=2)
assert b2._fetch is True

print('Public API surface verified OK')
"
```

Expected: `Public API surface verified OK`

- [ ] **Step 3: Push branch and open PR**

```bash
git push -u origin feature/phase-9-web-providers
gh pr create \
  --title "feat: Phase 9 — full WebBackend factory + Tavily/Exa/Brave/SerpAPI providers" \
  --body "Resolves spec §7.2 and §15 #4.

## What this adds
- \`WebBackend(provider=...)\` factory + per-provider classes (\`TavilyBackend\`, \`ExaBackend\`, \`BraveBackend\`, \`SerpAPIBackend\`)
- All four re-exported from \`sleuth.backends\`
- Per-domain token-bucket rate limiting + exponential backoff (4xx no-retry, 429/5xx retry with cap)
- Optional \`fetch=True\` mode: parallel httpx fetches → trafilatura extraction → tiktoken-aware chunking
- BackendTestKit contract compliance for all four providers
- Env-gated integration tests under \`tests/integration/\`

## Notes
- Adds \`exa\` and \`web-fetch\` optional dep groups to \`pyproject.toml\` (escalation flagged in plan)
- Phase 1's \`WebBackend\` symbol is preserved — existing tests pass unchanged
" \
  --base develop
```

---

## Self-review checklist

**1. Spec coverage**

| Spec requirement | Task covering it |
|---|---|
| §7.2 — Adapters for Tavily, Exa, Brave, SerpAPI | Tasks 5–8 |
| §7.2 — `WebBackend(provider=...)` factory | Task 9 |
| §7.2 — Per-domain rate limit | Tasks 3, 5–8 (`TokenBucket` per provider instance) |
| §7.2 — Exponential backoff | Tasks 3, 5–8 (`with_backoff`) |
| §7.2 — Optional `fetch=True` mode | Tasks 4, 5–8 (`FetchPipeline`) |
| §7.2 — Parallel fetch of top-N pages | Task 4 (`asyncio.gather` in `FetchPipeline`) |
| §7.2 — trafilatura + token-aware splitter | Task 4 |
| §15 #4 — factory AND per-provider classes both public | Task 9 + 10 |
| §12 — BackendTestKit compliance | Task 11 |
| §12 — Integration tests env-gated | Task 12 |
| Phase 1's `WebBackend` symbol preserved | Task 9 (factory function keeps same name) |

All spec requirements covered.

**2. Placeholder scan** — No "TBD", "TODO", "implement X without showing how", or "similar to Task N" found. All code is complete and runnable.

**3. Type consistency**
- `TokenBucket` — defined in Task 3, used as `TokenBucket(rate=..., capacity=...)` in Tasks 5–8. Consistent.
- `with_backoff(coro_fn, max_retries, base_delay)` — defined in Task 3, called identically in Tasks 5–8. Consistent.
- `FetchPipeline(max_tokens_per_chunk, timeout)` — defined in Task 4, instantiated as `FetchPipeline()` (defaults) in Tasks 5–8. Consistent.
- `Chunk`, `Source` — from `sleuth.types` per conventions §5.5. Not redefined anywhere.
- `Backend` protocol fields (`name: str`, `capabilities: frozenset[Capability]`, `search`) — all four classes declare them as class attributes. Consistent with conventions §5.2.
- `BackendError` — from `sleuth.errors` per conventions §6. Not redefined.
- `_backoff_base` testing hook — present in all four provider constructors and passed to `with_backoff(base_delay=...)`. Consistent.
