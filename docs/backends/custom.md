# Custom backends

Write a custom backend in under 30 lines. Any object that satisfies the `Backend` protocol is a first-class backend.

---

## The protocol contract

```python
from sleuth.backends.base import Backend, Capability
from sleuth.types import Chunk

class Backend(Protocol):
    name: str                          # identifier in SearchEvent.backend
    capabilities: frozenset[Capability]

    async def search(self, query: str, k: int = 10) -> list[Chunk]: ...
```

No inheritance required. No registration. Just satisfy the structural protocol and pass the object to `Sleuth(backends=[...])`.

---

## Minimal example

A backend that searches a preloaded in-memory dictionary:

```python
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source

class DictBackend:
    name = "dict"
    capabilities = frozenset({Capability.DOCS})

    def __init__(self, corpus: dict[str, str]) -> None:
        self._corpus = corpus   # title → text

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        query_lower = query.lower()
        results = []
        for title, text in self._corpus.items():
            if query_lower in text.lower() or query_lower in title.lower():
                results.append(
                    Chunk(
                        text=text,
                        source=Source(kind="file", location=f"dict://{title}", title=title),
                        score=1.0,
                    )
                )
        return results[:k]
```

Use it like any built-in backend:

```python
from sleuth import Sleuth
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[
        DictBackend({"Auth overview": "The auth module uses JWT tokens ...",
                     "Rate limits": "The API is rate-limited to 100 req/min ..."})
    ],
)
result = agent.ask("How does auth work?")
```

---

## Advertising `Capability` flags

The `capabilities` frozenset tells the Planner which sub-queries should be routed to your backend in deep mode. Pick the flags that describe what your backend returns:

| Flag | When to use |
| --- | --- |
| `Capability.WEB` | Results from the public internet |
| `Capability.DOCS` | Local or private document corpora |
| `Capability.CODE` | Source code |
| `Capability.FRESH` | Results that reflect "now" (news, prices, status) |
| `Capability.PRIVATE` | Auth-gated systems (Notion, Linear, Slack, …) |

Multiple flags are fine:

```python
capabilities = frozenset({Capability.DOCS, Capability.PRIVATE})
```

---

## Validating with `BackendTestKit`

The `BackendTestKit` in `tests/contract/test_backend_protocol.py` runs five contract tests against any backend:

```python
# In your test file
import pytest
from tests.contract.test_backend_protocol import BackendTestKit

class TestDictBackend(BackendTestKit):
    @pytest.fixture
    def backend(self):
        return DictBackend({"Test entry": "some text about testing"})
```

The five tests verify:

1. `search()` returns a `list[Chunk]`
2. `search(k=1)` returns at most 1 result
3. `name` is a non-empty string
4. `capabilities` is a `frozenset[Capability]`
5. The backend handles `asyncio.CancelledError` without hanging

Run with:

```bash
uv run pytest tests/test_my_backend.py
```

---

## Error handling

Raise `BackendError` or `BackendTimeoutError` on failures. The engine catches these, emits `SearchEvent(error=...)`, and continues the run without this backend's results.

```python
from sleuth.errors import BackendError

async def search(self, query: str, k: int = 10) -> list[Chunk]:
    try:
        return await self._call_api(query, k)
    except SomeNetworkError as exc:
        raise BackendError(f"API call failed: {exc}") from exc
```

Never let unexpected exceptions escape silently — the engine's generic handler will catch them too, but with less informative error messages.

---

## Cancellation safety

Always propagate `asyncio.CancelledError`. If your backend uses `asyncio.wait_for` internally, it already does the right thing. If you're calling a synchronous library in a thread pool:

```python
import asyncio

async def search(self, query: str, k: int = 10) -> list[Chunk]:
    # asyncio.to_thread propagates CancelledError correctly
    return await asyncio.to_thread(self._sync_search, query, k)
```

Do not wrap `await` calls in `except CancelledError: pass` — let cancellation propagate so the engine can clean up.

---

## Per-backend timeout override

The engine applies capability-based default timeouts (8 s for web, 4 s for local). To override for a specific backend, set a `timeout_s` attribute (duck-typed; not in the frozen `Backend` protocol):

```python
class SlowBackend:
    name = "slow"
    capabilities = frozenset({Capability.PRIVATE})
    timeout_s = 30.0   # override the 4 s default

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        ...
```
