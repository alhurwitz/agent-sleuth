# Caching & memory

Sleuth has three explicit, user-visible memory layers. All are opt-in or configurable — none have hidden global state.

---

## The `Cache` protocol

All cache implementations satisfy this four-method protocol:

```python
class Cache(Protocol):
    async def get(self, namespace: str, key: str) -> Any | None: ...
    async def set(self, namespace: str, key: str, value: Any, *, ttl_s: int | None = None) -> None: ...
    async def delete(self, namespace: str, key: str) -> None: ...
    async def clear(self, namespace: str | None = None) -> None: ...
```

The namespaces in use are `"query"`, `"fetch"`, `"plan"`, and `"semantic"`.

---

## `MemoryCache`

In-memory, TTL-aware. Thread-safe for single-threaded async use.

```python
from sleuth.memory.cache import MemoryCache

cache = MemoryCache()
agent = Sleuth(llm=..., backends=[...], cache=cache)
```

Useful for tests and short-lived processes. Data does not survive process restart.

---

## `SqliteCache` (default)

Persistent, per-namespace SQLite files. The default when `cache="default"` is passed to `Sleuth` (which is the default).

```python
from sleuth.memory.cache import SqliteCache

# Default location: ~/.sleuth/cache/sleuth_{namespace}.sqlite
cache = SqliteCache()

# Custom location:
cache = SqliteCache(base_path="/var/cache/myapp/sleuth")
```

Each namespace gets its own file, e.g. `~/.sleuth/cache/sleuth_query.sqlite`. The default TTLs are:

| Namespace | TTL |
| --- | --- |
| `query` | 10 minutes |
| `fetch` | 24 hours |
| `plan` | 1 hour |

Override by passing `ttl_s=` to `cache.set(...)` directly, or pass `cache=None` to disable caching entirely.

**Disable caching:**
```python
agent = Sleuth(llm=..., backends=[...], cache=None)
```

---

## Cache-hit replay

When a query hits the cache, Sleuth replays the cached result through the same event stream — consumers need no special cache-aware code path.

The replay sequence:

```
RouteEvent(depth="fast"|"deep")
CacheHitEvent(type="cache_hit", kind="query", key="<sha256>")
TokenEvent(text=<full cached text>)   # single chunk, not token-by-token
CitationEvent(index=0, source=...)
CitationEvent(index=1, source=...)    # ... one per citation
DoneEvent(stats=RunStats(first_token_ms=None, cache_hits={"query": 1}))
```

`first_token_ms` is `None` on cache hits per spec §6. The `cache_hits` dict in `RunStats` tracks hit counts by namespace.

**Cache key derivation:** keyed on `sha256(query + "\x00" + depth + "\x00" + sorted_backend_names)`. Changing the backend set or depth produces a cache miss.

---

## `SemanticCache`

An opt-in similarity layer that catches near-duplicate queries even when exact text differs.

**Requires:** `pip install 'agent-sleuth[semantic]'`

```python
from sleuth.memory.semantic import SemanticCache, FastembedEmbedder
from sleuth.memory.cache import SqliteCache

sc = SemanticCache(
    cache=SqliteCache(),
    embedder=FastembedEmbedder(),   # BGE-small-en-v1.5, 384-dim
    threshold=0.92,                 # cosine similarity threshold
    window_s=600,                   # entries older than 10 min are ignored
)

agent = Sleuth(llm=..., backends=[...], semantic_cache=sc)
```

Or use the shorthand to get defaults:

```python
agent = Sleuth(llm=..., backends=[...], semantic_cache=True)
# Equivalent to SemanticCache(cache=SqliteCache(), embedder=FastembedEmbedder(), threshold=0.92, window_s=600)
```

On lookup, the `SemanticCache` embeds the query, compares against recent entries via cosine similarity, and returns a hit if the best score exceeds `threshold` within `window_s` seconds. A `CacheHitEvent(kind="semantic")` is emitted on a semantic hit.

**When to enable it:** queries that are rephrased versions of the same question — e.g. "how does auth work?" vs "explain the auth flow". Not useful when queries are intentionally varied or time-sensitive.

### Pluggable `Embedder`

```python
from sleuth.memory.semantic import Embedder

class MyEmbedder:
    name = "my-embedder"
    dim = 768

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Return one unit-norm float list per input text
        ...

sc = SemanticCache(cache=SqliteCache(), embedder=MyEmbedder())
```

The built-in `StubEmbedder` (zero-dependency, deterministic) is suitable for testing.

---

## `Session` — multi-turn ring buffer

`Session` stores recent conversation turns and provides them as LLM message history.

```python
from sleuth.memory.session import Session

session = Session(max_turns=20)   # default max_turns=20

r1 = agent.ask("Who maintains the auth middleware?", session=session)
r2 = agent.ask("What did they change recently?", session=session)
# r2's LLM call includes r1's Q+A as context
```

**Per-call override:** pass `session=` to individual `aask`/`ask` calls to override the instance-level session.

### Persistence

```python
# Synchronous save (atomic write on supported platforms)
session.save("./my_session.json")

# Load (classmethod)
session = Session.load("./my_session.json")

# Flush any pending background write
await session.flush()
```

`Session.load` raises `FileNotFoundError` if the path does not exist.

`as_messages()` returns the buffer as alternating `user` / `assistant` `Message` objects — the format expected by `LLMClient.stream()`.

---

## Index cache (`LocalFiles`)

`LocalFiles` maintains its own per-corpus index at `<corpus>/.sleuth/index/<version_hash>.json`. This is distinct from the query cache and is not governed by `SqliteCache` TTLs. It invalidates when the corpus version changes (controlled by the `rebuild` parameter: `"mtime"`, `"hash"`, or `"always"`).

See [Local files](../backends/local-files.md) for details.
