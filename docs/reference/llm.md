# LLM clients reference

The `LLMClient` protocol, streaming chunk types, and the three built-in shims.

See [BYOK & protocols](../concepts/byok.md) for the design rationale and custom-client examples.

---

## Protocol & streaming types

::: sleuth.llm.base.LLMClient
    options:
      show_root_heading: true

::: sleuth.llm.base.Message
    options:
      show_root_heading: true

::: sleuth.llm.base.Tool
    options:
      show_root_heading: true

### `LLMChunk` union

```python
LLMChunk = TextDelta | ReasoningDelta | ToolCall | Stop
```

::: sleuth.llm.base.TextDelta
    options:
      show_root_heading: true

::: sleuth.llm.base.ReasoningDelta
    options:
      show_root_heading: true

::: sleuth.llm.base.ToolCall
    options:
      show_root_heading: true

::: sleuth.llm.base.Stop
    options:
      show_root_heading: true

---

## `Anthropic` shim

Install: `pip install 'agent-sleuth[anthropic]'`

::: sleuth.llm.anthropic.Anthropic
    options:
      show_root_heading: true
      members:
        - __init__
        - name
        - supports_reasoning
        - stream

---

## `OpenAI` shim

Install: `pip install 'agent-sleuth[openai]'`

::: sleuth.llm.openai.OpenAI
    options:
      show_root_heading: true
      members:
        - __init__
        - name
        - supports_reasoning
        - stream

---

## `StubLLM`

Deterministic test double. No extra required.

::: sleuth.llm.stub.StubLLM
    options:
      show_root_heading: true
      members:
        - __init__
        - stream
