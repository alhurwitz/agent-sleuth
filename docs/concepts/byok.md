# BYOK & protocols

Sleuth never imports a model SDK as a hard dependency. You supply any object that satisfies small structural protocols — no base classes, no registration.

---

## The `LLMClient` protocol

Three attributes and one async method are the entire contract:

```python
from collections.abc import AsyncIterator
from typing import Protocol
from pydantic import BaseModel

class LLMClient(Protocol):
    name: str                        # e.g. "anthropic:claude-sonnet-4-6"
    supports_reasoning: bool         # True → engine emits ThinkingEvents
    supports_structured_output: bool # True → schema= passed natively; else JSON-parse fallback

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        ...
```

`LLMChunk` is a discriminated union of four dataclasses:

```python
@dataclass
class TextDelta:
    text: str

@dataclass
class ReasoningDelta:
    text: str         # extended-thinking / o-series reasoning token

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class Stop:
    reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "error"]

LLMChunk = TextDelta | ReasoningDelta | ToolCall | Stop
```

The `stream()` method must yield at least one `LLMChunk` and end with a `Stop` chunk.

---

## Built-in LLM shims

### `sleuth.llm.anthropic.Anthropic`

Install: `pip install 'agent-sleuth[anthropic]'`

The SDK is imported lazily — the `ImportError` is raised at instantiation, not at import time.

```python
from sleuth.llm.anthropic import Anthropic

llm = Anthropic(
    model="claude-sonnet-4-6",
    api_key=None,              # defaults to ANTHROPIC_API_KEY env var
    thinking=False,            # enable extended thinking (claude-opus-4-7, claude-sonnet-4-6)
    thinking_budget_tokens=5000,
    max_tokens=4096,
    base_url=None,
    timeout=120.0,
)
```

`supports_reasoning` is `True` for `claude-opus-4-7` (always) or for any model in `_THINKING_CAPABLE_MODELS` when `thinking=True`. When `True`, the engine emits `ThinkingEvent`s for `ReasoningDelta` chunks.

### `sleuth.llm.openai.OpenAI`

Install: `pip install 'agent-sleuth[openai]'`

```python
from sleuth.llm.openai import OpenAI

llm = OpenAI(
    model="gpt-4o",
    api_key=None,               # defaults to OPENAI_API_KEY env var
    base_url=None,              # override for Azure OpenAI / proxies
    timeout=120.0,
    max_completion_tokens=4096,
)
```

`supports_reasoning` is `True` for o-series models (name matches `^o\d`, e.g. `o1`, `o3`, `o4-mini`).

### `sleuth.llm.stub.StubLLM`

Zero-dependency test double. Available with the core install — no extra needed.

```python
from sleuth.llm.stub import StubLLM

llm = StubLLM(responses=["The answer is 42.", "Another response."])
# Each call to stream() advances through the list round-robin.
```

---

## The `Backend` protocol

One async method is the universal contract:

```python
from sleuth.backends.base import Backend, Capability
from sleuth.types import Chunk

class Backend(Protocol):
    name: str
    capabilities: frozenset[Capability]

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        ...
```

`Capability` is a `StrEnum`:

```python
class Capability(StrEnum):
    WEB     = "web"      # general web search
    DOCS    = "docs"     # local document corpora
    CODE    = "code"     # source code
    FRESH   = "fresh"    # results reflecting "now" (news, status pages)
    PRIVATE = "private"  # auth-gated systems
```

The planner uses `capabilities` to route sub-queries to appropriate backends. For example, a sub-query tagged `["fresh"]` will only be sent to backends advertising `Capability.FRESH`.

---

## Writing a custom LLM client

Any object with the three attributes and one async method works:

```python
from collections.abc import AsyncIterator
from sleuth.llm.base import LLMChunk, Message, TextDelta, Stop, Tool
from pydantic import BaseModel

class MyLLM:
    name = "my-llm"
    supports_reasoning = False
    supports_structured_output = False

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        # Call your model API here
        response_text = await call_my_api(messages)
        yield TextDelta(text=response_text)
        yield Stop(reason="end_turn")
```

---

## Writing and validating a custom backend

A minimal in-memory backend:

```python
from sleuth.backends.base import Backend, Capability
from sleuth.types import Chunk, Source

class DictBackend:
    """Search a pre-loaded dict of title → text snippets."""

    name = "dict"
    capabilities = frozenset({Capability.DOCS})

    def __init__(self, corpus: dict[str, str]) -> None:
        self._corpus = corpus

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

**Validate it with `BackendTestKit`** (from `tests/contract/test_backend_protocol.py`):

```python
import pytest
from tests.contract.test_backend_protocol import BackendTestKit

class TestDictBackend(BackendTestKit):
    @pytest.fixture
    def backend(self):
        return DictBackend({"Intro": "Sleuth is a search library.", "FAQ": "How does cache work?"})
```

`BackendTestKit` runs five contract tests automatically: returns a `list[Chunk]`, respects `k`, has a non-empty `name`, has a `frozenset[Capability]` for `capabilities`, and handles `asyncio.CancelledError` without hanging.

**Error handling:** raise `BackendError` or `BackendTimeoutError` (from `sleuth.errors`) on unrecoverable failures. The engine catches these, emits `SearchEvent(error=...)`, and continues without this backend's results.

```python
from sleuth.errors import BackendError

async def search(self, query: str, k: int = 10) -> list[Chunk]:
    try:
        return await self._call_api(query, k)
    except SomeNetworkError as exc:
        raise BackendError(f"API call failed: {exc}") from exc
```

**Cancellation:** propagate `asyncio.CancelledError` — never suppress it. If your backend uses `asyncio.wait_for` internally, it already does the right thing.

See [Custom backends](../backends/custom.md) for the full tutorial.
