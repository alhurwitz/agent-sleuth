# Structured output

Pass `schema=YourModel` to get a typed Pydantic instance in `result.data` alongside the normal synthesized text.

---

## How it works

When `schema=` is set:

1. The Synthesizer passes the schema to `LLMClient.stream(messages, schema=YourModel)`.
2. If `llm.supports_structured_output is True` (both built-in shims set this to `True`), the shim uses the provider's native mechanism:
    - **Anthropic:** injects a `structured_output` tool and forces the model to call it.
    - **OpenAI:** uses `response_format={"type": "json_schema", ...}`.
3. If `supports_structured_output is False`, the engine falls back to requesting JSON in the system prompt and parsing `result.text` manually.
4. The parsed model instance is stored in `result.data`.

---

## End-to-end example

```python
from pydantic import BaseModel
from sleuth import Sleuth
from sleuth.backends import WebBackend
from sleuth.llm.anthropic import Anthropic

# 1. Define your schema
class Verdict(BaseModel):
    answer: str
    confidence: float   # 0.0 – 1.0
    sources: list[str]
    caveats: list[str] = []

# 2. Construct Sleuth
agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[WebBackend(provider="tavily", api_key="...")],
)

# 3. Ask with schema=
result = agent.ask("Is the deploy script idempotent?", schema=Verdict)

# 4. Access both text and structured data
print(result.text)               # synthesized prose
print(result.data.answer)        # typed field
print(result.data.confidence)    # float
print(result.data.sources)       # list[str]

# 5. Citations are still populated
for src in result.citations:
    print(" -", src.location)
```

---

## Streaming with schema

`aask()` streams `TokenEvent`s for the prose response. After the stream closes, the structured data is accessible by collecting the `DoneEvent`:

```python
import asyncio
from pydantic import BaseModel
from sleuth import Sleuth
from sleuth.backends import WebBackend
from sleuth.llm.anthropic import Anthropic

class Summary(BaseModel):
    headline: str
    key_points: list[str]

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[WebBackend(provider="tavily", api_key="...")],
)

async def main():
    result_data = None
    async for event in agent.aask("Summarize Python 3.13 changes", schema=Summary):
        if event.type == "token":
            print(event.text, end="", flush=True)
        elif event.type == "done":
            # Use ask() instead if you need result.data synchronously
            pass

    # For the typed result, use ask() which returns Result[Summary]
    result = agent.ask("Summarize Python 3.13 changes", schema=Summary)
    for point in result.data.key_points:
        print("-", point)

asyncio.run(main())
```

!!! tip "Use `ask()` for structured results"
    `ask()` returns `Result[T]` with `result.data` directly populated. Use `aask()` when you need to stream tokens AND access `result.data` — but note that `aask()` doesn't expose `result.data` directly; you'd need to re-call `ask()` or use the `Synthesizer.last_result` pattern internally.

---

## Cache bypass

!!! warning "v0.1.0 limitation"
    Schema-typed results are not currently round-trippable through JSON serialization, so cache writes are skipped when `schema=` is set. Every call with `schema=` hits the backends and the LLM. This is a known limitation planned for v0.2.

To avoid redundant calls while iterating, consider caching the result yourself:

```python
import json
from pathlib import Path
from pydantic import BaseModel

class Verdict(BaseModel):
    answer: str
    confidence: float

CACHE_FILE = Path("./verdict_cache.json")

if CACHE_FILE.exists():
    verdict = Verdict.model_validate_json(CACHE_FILE.read_text())
else:
    result = agent.ask("Is the deploy script idempotent?", schema=Verdict)
    verdict = result.data
    CACHE_FILE.write_text(verdict.model_dump_json())

print(verdict.answer)
```
