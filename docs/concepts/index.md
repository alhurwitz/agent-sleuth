# Concepts

Four ideas underpin everything in Sleuth. Understand them once and the rest of the API follows naturally.

---

## Architecture

Sleuth routes every query through four sequential stages: **Router → Planner → Executor → Synthesizer**. The Router classifies depth heuristically (no LLM call). The Planner decomposes complex queries into sub-queries (deep mode only). The Executor fans sub-queries out to all backends in parallel with per-backend timeouts. The Synthesizer streams tokens and citations from the LLM.

A single typed **event stream** is the primary output surface — not callbacks, not separate return values.

[Architecture in depth →](architecture.md)

---

## Event stream

Every run — live or cached — emits the same nine event types: `RouteEvent`, `PlanEvent`, `SearchEvent`, `FetchEvent`, `ThinkingEvent`, `TokenEvent`, `CitationEvent`, `CacheHitEvent`, `DoneEvent`. They form a discriminated Pydantic union you can `match` on the `type` literal field.

`aask()` is an async generator that yields these events. `ask()` consumes the stream and returns a `Result[T]`.

[Event stream reference →](events.md)

---

## BYOK

Sleuth never imports a model SDK as a hard dependency. You provide any object that satisfies the three-attribute, one-method `LLMClient` protocol. Thin optional shims for Anthropic and OpenAI live behind extras and lazy-import their SDK only when you instantiate them.

The same protocol pattern applies to backends: implement one async `search(query, k) -> list[Chunk]` method and you have a first-class backend.

[BYOK & protocols →](byok.md)

---

## Caching & memory

Sleuth has three explicit, user-visible memory layers. `SqliteCache` (on by default) deduplicates identical queries at `~/.sleuth/cache/`. `SemanticCache` (opt-in) catches near-duplicate queries via cosine similarity. `Session` stores conversation turns in a ring buffer with optional JSON persistence.

Cache hits replay through the same event stream — prefixed with `CacheHitEvent` — so consumers never need a special cache-aware code path.

[Caching & memory →](caching.md)
