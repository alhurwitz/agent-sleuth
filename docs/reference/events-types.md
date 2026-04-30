# Events & types

Public data shapes used throughout Sleuth. All are Pydantic v2 models (serializable, validated).

See [Event stream](../concepts/events.md) for ordering diagrams and consumer patterns.

---

## Event types

::: sleuth.events.RouteEvent
    options:
      show_root_heading: true

::: sleuth.events.PlanStep
    options:
      show_root_heading: true

::: sleuth.events.PlanEvent
    options:
      show_root_heading: true

::: sleuth.events.SearchEvent
    options:
      show_root_heading: true

::: sleuth.events.FetchEvent
    options:
      show_root_heading: true

::: sleuth.events.ThinkingEvent
    options:
      show_root_heading: true

::: sleuth.events.TokenEvent
    options:
      show_root_heading: true

::: sleuth.events.CitationEvent
    options:
      show_root_heading: true

::: sleuth.events.CacheHitEvent
    options:
      show_root_heading: true

::: sleuth.events.DoneEvent
    options:
      show_root_heading: true

---

## `Event` discriminated union

```python
Event = Annotated[
    RouteEvent | PlanEvent | SearchEvent | FetchEvent | ThinkingEvent
    | TokenEvent | CitationEvent | CacheHitEvent | DoneEvent,
    Field(discriminator="type"),
]
```

Switch on `event.type` to handle each variant.

---

## Core types

::: sleuth.types.Source
    options:
      show_root_heading: true

::: sleuth.types.Chunk
    options:
      show_root_heading: true

::: sleuth.types.RunStats
    options:
      show_root_heading: true

::: sleuth.types.Result
    options:
      show_root_heading: true

---

## Literals

```python
Depth = Literal["auto", "fast", "deep"]
Length = Literal["brief", "standard", "thorough"]
```
