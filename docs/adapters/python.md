# Python SDK

The native Python SDK gives you the full Sleuth surface: streaming events, sync/async twins, structured output, sessions, and eager index warm-up.

---

## `Sleuth` constructor

```python
from sleuth._agent import Sleuth

agent = Sleuth(
    llm: LLMClient,
    backends: list[Backend],
    *,
    fast_llm: LLMClient | None = None,
    cache: Cache | Literal["default"] | None = "default",
    semantic_cache: SemanticCache | bool = False,
    session: Session | None = None,
)
```

**`llm`** — any object satisfying the `LLMClient` protocol. Used for synthesis (the final answer-generation step).

**`backends`** — one or more `Backend` instances. All are searched in parallel on every query.

**`fast_llm`** — optional faster/cheaper LLM for routing and planning. Falls back to `llm` when not supplied. Recommended: a haiku-tier or mini-tier model.

**`cache`** — controls the query cache:
- `"default"` (default): uses `SqliteCache()` at `~/.sleuth/cache/sleuth_{namespace}.sqlite`.
- A `Cache` instance (e.g. `MemoryCache()`, `SqliteCache(base_path="...")`): uses that instance.
- `None`: disables caching entirely.

**`semantic_cache`** — controls the semantic similarity cache:
- `False` (default): disabled.
- `True`: enables `SemanticCache` with `FastembedEmbedder()` (requires `agent-sleuth[semantic]`), `threshold=0.92`, `window_s=600`.
- A `SemanticCache` instance: uses that configuration.

**`session`** — optional `Session` for multi-turn conversations. Applied to every call unless overridden per-call.

---

## `aask` — async event generator

```python
async def aask(
    self,
    query: str,
    *,
    depth: Depth = "auto",
    max_iterations: int = 4,
    schema: type[BaseModel] | None = None,
    session: Session | None = None,
) -> AsyncIterator[Event]:
```

Yields typed `Event` objects. Iterate with `async for`:

```python
async for event in agent.aask("How does auth work?"):
    if event.type == "token":
        print(event.text, end="")
    elif event.type == "done":
        print(f"\n{event.stats.latency_ms} ms")
```

**`depth`** — `"auto"` (router decides heuristically), `"fast"` (single fan-out, no planner), `"deep"` (planner + reflect loop).

**`max_iterations`** — maximum planner reflect iterations in deep mode. Default `4`.

**`schema`** — a Pydantic `BaseModel` subclass. When set, the synthesizer requests structured output from the LLM and populates `result.data`. Schema results bypass the cache (v0.1.0 limitation).

**`session`** — per-call session override. Overrides the instance-level `session=` if both are set.

---

## `ask` — synchronous wrapper

```python
def ask(
    self,
    query: str,
    *,
    depth: Depth = "auto",
    max_iterations: int = 4,
    schema: type[BaseModel] | None = None,
    session: Session | None = None,
) -> Result:
```

Blocks until the run completes. Returns a `Result[T]`:

```python
result = agent.ask("What is the cache TTL for query results?")
print(result.text)
for src in result.citations:
    print(" -", src.location)
print(result.stats.latency_ms, "ms")
```

`Result` fields:

```python
class Result(BaseModel, Generic[T]):
    text: str
    citations: list[Source]
    data: T | None = None     # populated when schema= is set
    stats: RunStats
```

---

## `asummarize` — async summarize generator

```python
async def asummarize(
    self,
    target: str,
    *,
    length: Length = "standard",
    schema: type[BaseModel] | None = None,
) -> AsyncIterator[Event]:
```

Summarizes a URL, file path, or free-text topic. When `target` is an existing file path and a `LocalFiles` backend is configured, the summary is built from the pre-built index (no LLM call needed for indexing). Otherwise falls back to `aask(f"summarize: {target}")`.

**`length`** — `"brief"` (one sentence from index root), `"standard"` (root + level-1 children), `"thorough"` (full tree walk).

---

## `summarize` — synchronous wrapper

```python
def summarize(
    self,
    target: str,
    *,
    length: Length = "standard",
    schema: type[BaseModel] | None = None,
) -> Result:
```

---

## `warm_index`

```python
async def warm_index(self) -> None:
```

Eagerly indexes all backends that support it (currently `LocalFiles`). Call this at startup to avoid a slow first query:

```python
import asyncio
asyncio.run(agent.warm_index())
```

---

## `Result[T]` collection semantics

`ask()` and `summarize()` both return `Result[T]`. The generic `T` is `BaseModel` when `schema=` is passed, and unbound otherwise. Type checkers see:

```python
from pydantic import BaseModel

class Verdict(BaseModel):
    answer: str
    confidence: float

result: Result[Verdict] = agent.ask("...", schema=Verdict)
reveal_type(result.data)  # Verdict | None
```

---

## Per-call session override

Instance-level sessions are used as the default, but every call can override:

```python
session_a = Session()
session_b = Session()

# Uses session_a
agent.ask("first question", session=session_a)

# Uses session_b (ignores instance-level session even if set)
agent.ask("second question", session=session_b)
```

---

## Complete example

```python
import asyncio, os
from sleuth import Sleuth, Session
from sleuth.backends import WebBackend
from sleuth.backends.localfiles import LocalFiles
from sleuth.llm.anthropic import Anthropic
from sleuth.memory.cache import SqliteCache

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    fast_llm=Anthropic(model="claude-haiku-4-5"),
    backends=[
        WebBackend(provider="tavily", api_key=os.environ["TAVILY_KEY"]),
        LocalFiles(path="./docs"),
    ],
    cache=SqliteCache(),         # explicit cache (same as "default")
    semantic_cache=True,         # enable fuzzy query matching
)

async def main():
    # Pre-warm the LocalFiles index
    await agent.warm_index()

    # Multi-turn conversation
    session = Session(max_turns=10)
    r1 = agent.ask("Who owns the rate limiter?", session=session)
    r2 = agent.ask("What tests cover it?", session=session)
    print(r2.text)

asyncio.run(main())
```
