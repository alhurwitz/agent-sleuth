# `sleuth` — the `Sleuth` class

The `Sleuth` class is the top-level entry point. It wires Router → Executor → Synthesizer into a single `aask` async generator and provides an `ask` sync wrapper.

See [Python SDK](../adapters/python.md) for usage examples and parameter explanations.

---

::: sleuth._agent.Sleuth
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - aask
        - ask
        - asummarize
        - summarize
        - warm_index
