# Event stream

Every Sleuth run emits a single ordered stream of typed events. The stream is the primary observability surface — no callbacks, no external tracers required.

---

## The `Event` discriminated union

All nine event types form a single Pydantic discriminated union keyed on the `type` literal:

```python
from typing import Annotated
from pydantic import Field

Event = Annotated[
    RouteEvent
    | PlanEvent
    | SearchEvent
    | FetchEvent
    | ThinkingEvent
    | TokenEvent
    | CitationEvent
    | CacheHitEvent
    | DoneEvent,
    Field(discriminator="type"),
]
```

Switch on `event.type` rather than `isinstance` checks:

```python
async for event in agent.aask(query):
    match event.type:
        case "route":   ...
        case "search":  ...
        case "token":   print(event.text, end="")
        case "done":    print(f"\n{event.stats.latency_ms} ms")
```

---

## Event types

### `RouteEvent`

Emitted **once per run**, immediately, before any backend search.

```python
class RouteEvent(BaseModel):
    type: Literal["route"]
    depth: Depth           # "fast" | "deep"
    reason: str            # human-readable heuristic explanation
```

`depth` tells you which code path will run. `reason` is a short string like `"complexity keyword: 'compare'"` or `"simple factual question pattern"` — useful for debugging routing decisions.

---

### `PlanEvent`

Emitted **once per reflect iteration** in deep mode only. Not emitted in fast mode.

```python
class PlanStep(BaseModel):
    query: str
    backends: list[str] | None = None   # None = router decides
    done: bool = False                  # end-of-iteration sentinel

class PlanEvent(BaseModel):
    type: Literal["plan"]
    steps: list[PlanStep]
```

`steps` lists the sub-queries the Planner decomposed the original question into. The `done=True` sentinel step is excluded from `PlanEvent.steps`. Use `PlanEvent`s to inspect the planner's decomposition — see [Deep mode](../recipes/deep-mode.md).

---

### `SearchEvent`

Emitted **once per backend call** (success or failure). Two `SearchEvent`s are emitted when a backend fails: an initial one (no error) and a second one with `error` set.

```python
class SearchEvent(BaseModel):
    type: Literal["search"]
    backend: str            # backend.name
    query: str              # the sub-query sent to this backend
    error: str | None = None  # set when the backend timed out or raised
```

Check `event.error is not None` to detect backend failures. The run continues regardless — other backends' results are still synthesized.

---

### `FetchEvent`

Emitted when a URL is fetched in `fetch=True` mode (WebBackend only).

```python
class FetchEvent(BaseModel):
    type: Literal["fetch"]
    url: str
    status: int             # HTTP status code
    error: str | None = None
```

---

### `ThinkingEvent`

Emitted for **extended reasoning tokens** when `llm.supports_reasoning is True` (Claude extended thinking models with `thinking=True`, or OpenAI o-series models).

```python
class ThinkingEvent(BaseModel):
    type: Literal["thinking"]
    text: str
```

These tokens represent the LLM's internal reasoning chain. They appear before `TokenEvent`s in the stream. Most consumers display them in a collapsible section or suppress them entirely.

---

### `TokenEvent`

Emitted **for each text token** from the Synthesizer LLM.

```python
class TokenEvent(BaseModel):
    type: Literal["token"]
    text: str
```

Tokens are not word-aligned — a single `TokenEvent` might carry `"the "`, `"answer"`, or a punctuation mark. Concatenate `event.text` across all `TokenEvent`s to reconstruct the full response.

!!! note "Cache hit behaviour"
    On a cache hit, a single `TokenEvent` carries the full cached response text rather than replaying it token-by-token. The `Result.text` is identical; only streaming granularity differs.

---

### `CitationEvent`

Emitted **once per contributing chunk** after the LLM stream closes.

```python
class CitationEvent(BaseModel):
    type: Literal["citation"]
    index: int              # 0-based position in the citations list
    source: Source          # kind + location + title
```

`CitationEvent`s are emitted in the same order as `result.citations`. Each `Source` carries a `kind` (`"url"`, `"file"`, or `"code"`), a `location` (URL, absolute path, or `repo:path:L1-L2`), and an optional `title`.

---

### `CacheHitEvent`

Emitted **before replaying a cached run**, instead of `RouteEvent`.

```python
class CacheHitEvent(BaseModel):
    type: Literal["cache_hit"]
    kind: str   # "query" | "fetch" | "plan" | "semantic"
    key: str    # the cache key
```

When you see a `CacheHitEvent`, the subsequent `TokenEvent` and `CitationEvent`s are replayed from the cache — no backends were called. The `DoneEvent` that follows has `first_token_ms=None` (per spec §6) and an incremented `cache_hits` counter.

---

### `DoneEvent`

The **final event** in every run.

```python
class DoneEvent(BaseModel):
    type: Literal["done"]
    stats: RunStats
```

`RunStats` carries:

```python
class RunStats(BaseModel):
    latency_ms: int               # wall-clock ms from first event to DoneEvent
    first_token_ms: int | None    # ms to first TokenEvent; None on cache hit
    tokens_in: int                # approximate (message count, not token count)
    tokens_out: int               # number of TokenEvents emitted
    cache_hits: dict[str, int]    # namespace → hit count, e.g. {"query": 1}
    backends_called: list[str]    # names of backends that succeeded
```

---

## `aask()` vs `ask()`

`aask()` is an **async generator** — iterate it with `async for`:

```python
async for event in agent.aask("..."):
    handle(event)
```

`ask()` is the **synchronous wrapper**: it drives `asyncio.run` internally, collects the stream, and returns a `Result[T]`:

```python
result = agent.ask("...")   # blocks until done
print(result.text)
print(result.citations)
print(result.stats)
```

`Result[T]` is defined as:

```python
class Result(BaseModel, Generic[T]):
    text: str
    citations: list[Source]
    data: T | None = None     # populated when schema= is passed
    stats: RunStats
```

---

## Fast-path ordering

```
RouteEvent(depth="fast")
SearchEvent × N   (one per backend, emitted as they're dispatched)
ThinkingEvent × M (optional, only if LLM emits reasoning tokens)
TokenEvent × K
CitationEvent × J
DoneEvent
```

## Deep-mode ordering

```
RouteEvent(depth="deep")
PlanEvent(steps=[...])       # iteration 1
SearchEvent × N
[PlanEvent, SearchEvent × N] # iterations 2..max_iterations (if not done)
ThinkingEvent × M            (optional)
TokenEvent × K
CitationEvent × J
DoneEvent
```

## Cache-hit ordering

```
RouteEvent(depth="fast"|"deep")   # still emitted before cache lookup
CacheHitEvent(kind="query")
TokenEvent(text=<full cached text>)
CitationEvent × J
DoneEvent(stats.first_token_ms=None)
```
