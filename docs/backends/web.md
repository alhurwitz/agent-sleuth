# Web providers

`WebBackend` is a factory that returns one of four per-provider backend instances. All providers share the same rate-limiting, backoff, and optional page-fetch pipeline.

---

## Factory usage

```python
from sleuth.backends.web import WebBackend

backend = WebBackend(
    provider="tavily",          # "tavily" | "exa" | "brave" | "serpapi"
    api_key="...",
    # common optional kwargs forwarded to the provider class:
    fetch=False,                # parallel-fetch top pages (requires web-fetch extra)
    fetch_top_n=3,
    rate_limit=5.0,             # requests per second
    max_retries=3,
)
```

`WebBackend` is a factory function, not a class. It instantiates and returns the matching per-provider class. Passing an unknown `provider` raises `ValueError`.

---

## Per-provider classes

For type-checker-friendly usage, import the class directly:

```python
from sleuth.backends.web import TavilyBackend, ExaBackend, BraveBackend, SerpAPIBackend
```

### `TavilyBackend`

```python
TavilyBackend(
    api_key: str,
    *,
    fetch: bool = False,
    fetch_top_n: int = 3,
    rate_limit: float = 5.0,    # requests/second
    max_retries: int = 3,
)
```

`capabilities = frozenset({Capability.WEB, Capability.FRESH})`

### `ExaBackend`

```python
from sleuth.backends.web import ExaBackend
```

Same constructor shape as `TavilyBackend`. `capabilities = frozenset({Capability.WEB})`.

Install for Exa: `pip install 'agent-sleuth[exa]'`

### `BraveBackend`

Same constructor shape. `capabilities = frozenset({Capability.WEB, Capability.FRESH})`.

### `SerpAPIBackend`

Same constructor shape. `capabilities = frozenset({Capability.WEB, Capability.FRESH})`.

---

## Rate limiting

Each provider instance maintains a **token-bucket rate limiter** scoped to that instance. The `rate_limit` parameter controls how many requests per second are allowed (default `5.0`). Burst capacity equals `rate_limit`.

```python
# Allow 2 requests/second with burst of 2
backend = WebBackend(provider="tavily", api_key="...", rate_limit=2.0)
```

Calls block asynchronously until a token is available — they never fail due to rate limiting, only slow down.

---

## HTTP backoff

On retryable errors:

| Status | Behaviour |
| --- | --- |
| 4xx (non-429) | Raise immediately — no retry |
| 429 | Honour `Retry-After` header if present; else exponential backoff |
| 5xx (500, 502, 503, 504) | Exponential backoff, capped at 60 s |
| Exceeded `max_retries` | Raise `BackendError` |

Failures propagate as `SearchEvent(error=...)` — the run continues with other backends' results.

---

## `fetch=True` mode

When `fetch=True`, the backend parallel-fetches the top `fetch_top_n` result URLs after the search API call. Page text is extracted with `trafilatura` and chunked into `max_tokens_per_chunk`-token windows using `tiktoken`.

**Requires:** `pip install 'agent-sleuth[web-fetch]'`

```python
backend = WebBackend(
    provider="tavily",
    api_key="...",
    fetch=True,
    fetch_top_n=3,    # fetch the top 3 URLs
)
```

Fetched chunks are appended to the API snippet chunks. Each fetch emits a `FetchEvent` into the event stream.

!!! note "Web-fetch extra"
    `fetch=True` requires `trafilatura` and `tiktoken`. Without the `web-fetch` extra, instantiating with `fetch=True` succeeds but the first `search()` call raises `ImportError` with a clear message.

---

## Example

```python
import os
from sleuth import Sleuth
from sleuth.backends.web import WebBackend
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[
        WebBackend(
            provider="tavily",
            api_key=os.environ["TAVILY_KEY"],
            fetch=True,       # fetch full page content for top 3 results
            fetch_top_n=3,
            rate_limit=5.0,
            max_retries=3,
        )
    ],
)

result = agent.ask("Latest release of Python 3.13")
print(result.text)
for src in result.citations:
    print(" -", src.location)
```
