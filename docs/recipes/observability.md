# Observability

Sleuth exposes three observability surfaces. You need no external tracer, no callbacks, no extra configuration to get started.

---

## Surface 1 — the event stream (primary)

The typed event stream is Sleuth's primary observability surface. Every run emits `RouteEvent`, `SearchEvent`, `TokenEvent`, `CitationEvent`, and `DoneEvent` at minimum. Deep mode adds `PlanEvent`. Extended-thinking LLMs add `ThinkingEvent`. Cache hits add `CacheHitEvent`.

See [Event stream](../concepts/events.md) for the full reference.

**15-line event tail with timing:**

```python
import asyncio, time
from sleuth import Sleuth
from sleuth.backends import WebBackend
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[WebBackend(provider="tavily", api_key="...")],
)

async def tail(query: str) -> None:
    t0 = time.monotonic()
    async for event in agent.aask(query):
        elapsed = int((time.monotonic() - t0) * 1000)
        match event.type:
            case "route":   print(f"+{elapsed:4d}ms  route  depth={event.depth}")
            case "search":  print(f"+{elapsed:4d}ms  search [{event.backend}] {event.query[:60]}")
            case "token":   pass   # suppress token spam
            case "citation":print(f"+{elapsed:4d}ms  cite   [{event.index}] {event.source.location[:60]}")
            case "done":
                s = event.stats
                print(f"+{elapsed:4d}ms  done   latency={s.latency_ms}ms first_token={s.first_token_ms}ms "
                      f"tokens_out={s.tokens_out} backends={s.backends_called}")

asyncio.run(tail("How does rate limiting work in our API?"))
```

---

## Surface 2 — stdlib `logging` under `sleuth.*`

All internal modules log under the `sleuth` namespace hierarchy using Python's standard `logging`. No handlers are attached by default — configure your own:

```python
import logging

# Enable INFO-level logs from all Sleuth modules
logging.basicConfig(level="WARNING")
logging.getLogger("sleuth").setLevel(logging.INFO)
```

Log levels:

| Level | What you see |
| --- | --- |
| `DEBUG` | Per-chunk details, cache key hashes, tree navigation steps |
| `INFO` | Index build progress, backend selection decisions |
| `WARNING` | Backend timeouts, parse errors in `LocalFiles`, retry attempts |
| `ERROR` | Unexpected backend exceptions (with traceback) |

Enable per-module granularity:

```python
# Show only web-backend warnings
logging.getLogger("sleuth.backends.web").setLevel(logging.WARNING)

# Show all engine debug output
logging.getLogger("sleuth.engine").setLevel(logging.DEBUG)
```

Named loggers in use:

| Logger name | Module |
| --- | --- |
| `sleuth.engine.router` | `engine/router.py` |
| `sleuth.engine.executor` | `engine/executor.py` |
| `sleuth.engine.synthesizer` | `engine/synthesizer.py` |
| `sleuth.backends.localfiles` | `backends/localfiles.py` |
| `sleuth.backends.web` | `backends/_web/_base.py` |
| `sleuth.backends.web.tavily` | `backends/_web/tavily.py` |
| `sleuth.backends.vectorstore` | `backends/vectorstore.py` |

---

## Surface 3 — `RunStats` in `DoneEvent`

Every run ends with a `DoneEvent` carrying `RunStats`:

```python
class RunStats(BaseModel):
    latency_ms: int               # wall-clock ms
    first_token_ms: int | None    # ms to first token (None on cache hit)
    tokens_in: int                # approximate (message count)
    tokens_out: int               # number of TokenEvents
    cache_hits: dict[str, int]    # {"query": 1} on a cache hit
    backends_called: list[str]    # names of backends that succeeded
```

Use `RunStats` to build your own metrics:

```python
async def measure(agent, query: str) -> RunStats:
    async for event in agent.aask(query):
        if event.type == "done":
            return event.stats
    raise RuntimeError("No DoneEvent received")
```

---

## OpenTelemetry (v0.2+)

An OpenTelemetry adapter that maps Sleuth events to spans and metrics is planned for v0.2. The event stream is designed to carry all the information needed (start times, backend names, token counts, cache hit rates) — the OTel adapter will be a thin wrapper.

Until then, the event stream and `logging` provide full observability without any external dependency.
