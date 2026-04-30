# Deep mode

`depth="deep"` engages the Planner and reflect loop, turning multi-part queries into parallel sub-searches with iterative refinement.

---

## When to use deep mode

The Router auto-selects `"deep"` when it detects complexity keywords (`compare`, `tradeoffs`, `explain`, `versus`, `all the`, `across`, …) or when the query exceeds 10 words without a simple-factual pattern. You can force it or suppress it:

```python
# Force deep mode
result = agent.ask("Compare OAuth and OIDC", depth="deep")

# Force fast mode (single fan-out, no planner)
result = agent.ask("Compare OAuth and OIDC", depth="fast")

# Let the router decide (default)
result = agent.ask("Compare OAuth and OIDC")  # → "deep" (complexity keyword detected)
```

---

## How the reflect loop works

```
Iteration 1:
  Planner LLM → [sub-query 1, sub-query 2, sub-query 3]
  Executor fans out → search results from all backends × 3 queries
  Results summarized → appended to planner context

Iteration 2 (if not done and < max_iterations):
  Planner LLM receives original query + prior results
  → [sub-query 4, done: true]   ← LLM signals "I have enough"
  Executor fans out → more results

Synthesizer → final answer from all accumulated chunks
```

**Speculative prefetch:** the executor starts searching on the planner's first emitted sub-query while the planner is still streaming the rest. This hides planner latency behind search latency.

---

## Full deep-mode event ordering

```
RouteEvent(type="route", depth="deep", reason="complexity keyword: 'compare'")
PlanEvent(type="plan", steps=[PlanStep(query="OAuth overview"), PlanStep(query="OIDC overview")])
SearchEvent(type="search", backend="tavily", query="OAuth overview")
SearchEvent(type="search", backend="tavily", query="OIDC overview")
SearchEvent(type="search", backend="localfiles", query="OAuth overview")
... (parallel searches across all backends)
# optional second iteration:
PlanEvent(type="plan", steps=[PlanStep(query="OAuth vs OIDC token lifetimes")])
SearchEvent(...)
...
TokenEvent(text="OAuth is ...")   ×N
CitationEvent(index=0, source=...)   ×K
DoneEvent(stats=RunStats(...))
```

---

## Controlling the loop

```python
result = agent.ask(
    "Compare OAuth and OIDC: tradeoffs, use cases, and token lifetimes",
    depth="deep",
    max_iterations=4,  # default 4; lower for speed, higher for thoroughness
)
```

The loop stops when:
1. The Planner LLM includes `"done": true` in its output — it has decided enough information was gathered.
2. `max_iterations` is reached.

---

## Inspecting the planner's decomposition

Collect `PlanEvent`s from `aask()` to see what sub-queries the planner chose:

```python
import asyncio
from sleuth import Sleuth
from sleuth.backends import WebBackend
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    fast_llm=Anthropic(model="claude-haiku-4-5"),  # planner uses fast_llm
    backends=[WebBackend(provider="tavily", api_key="...")],
)

async def main():
    plan_events = []
    tokens = []

    async for event in agent.aask(
        "Compare REST and GraphQL: performance, caching, and developer experience",
        depth="deep",
        max_iterations=3,
    ):
        if event.type == "plan":
            plan_events.append(event)
            print(f"\n[Iteration {len(plan_events)}] Plan:")
            for step in event.steps:
                print(f"  - {step.query}")
                if step.backends:
                    print(f"    backends: {step.backends}")
        elif event.type == "search":
            err = f" [ERROR: {event.error}]" if event.error else ""
            print(f"  Searching [{event.backend}]: {event.query}{err}")
        elif event.type == "token":
            tokens.append(event.text)
        elif event.type == "done":
            print(f"\nDone in {event.stats.latency_ms} ms")
            print(f"Backends called: {event.stats.backends_called}")

    print("\nAnswer:")
    print("".join(tokens))

asyncio.run(main())
```

---

## Deep mode with the fast_llm

The `fast_llm` is used for planning (routing + sub-query decomposition). The main `llm` is used only for synthesis. Assign a faster/cheaper model to `fast_llm` to reduce planning cost:

```python
agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),       # synthesis
    fast_llm=Anthropic(model="claude-haiku-4-5"),   # planning
    backends=[...],
)
```

If `fast_llm` is not set, the main `llm` is used for both.

---

## When deep mode is wasteful

Deep mode makes multiple LLM calls (one per planner iteration plus synthesis). For simple, single-answer questions, `depth="fast"` is faster and cheaper. Trust the router for most cases — it only chooses `"deep"` when it has strong signal.
