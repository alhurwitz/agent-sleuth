# Recipes

Practical, copy-paste guides for common Sleuth patterns.

---

## [Sessions & multi-turn](sessions.md)

Build a conversational interface with `Session`. The ring buffer stores recent turns and prepends them as conversation history to every LLM call. This page covers construction, per-call overrides, and JSON persistence (`save` / `load` / `flush`) so sessions survive process restarts.

---

## [Structured output](structured-output.md)

Pass `schema=YourPydanticModel` to get a typed `result.data` alongside the normal text stream. The Anthropic and OpenAI shims use their native structured-output mechanisms; the fallback is JSON-parse of the text response. Includes an end-to-end example with a `Verdict` model and a note on the v0.1.0 cache bypass for schema results.

---

## [Deep mode](deep-mode.md)

Force `depth="deep"` to engage the Planner and reflect loop. The Planner decomposes multi-part queries into sub-queries, searches them in parallel with speculative prefetch, and iterates up to `max_iterations` times. This page explains the full deep-mode event ordering and shows how to inspect the planner's decomposition by collecting `PlanEvent`s.

---

## [Observability](observability.md)

Three surfaces for understanding what Sleuth is doing: the primary event stream, stdlib `logging` under the `sleuth` namespace, and `RunStats` in `DoneEvent`. Includes a 15-line "tail every event with timing" snippet and a note on the v0.2+ OpenTelemetry roadmap.
