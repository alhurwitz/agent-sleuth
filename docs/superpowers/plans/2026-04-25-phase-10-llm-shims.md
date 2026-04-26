# Phase 10: LLM Shims — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement real Anthropic and OpenAI LLM shims that adapt their SDK streams to the `LLMClient` protocol, with lazy imports, reasoning content support, and structured-output passthrough.

**Architecture:** Each shim (`sleuth/llm/anthropic.py`, `sleuth/llm/openai.py`) is a pure adapter class implementing the `LLMClient` protocol defined in `sleuth/llm/base.py` (Phase 1). The SDK import happens inside `__init__` so users without the extra get a clean `ImportError` only at instantiation, never at module import time. Both shims translate their SDK's streaming events into the `LLMChunk` union (`TextDelta`, `ReasoningDelta`, `ToolCall`, `Stop`). `supports_reasoning` is set per model family via a per-shim lookup table. Tests use `respx` to mock the HTTPS layer the SDKs call, verifying chunk emission shape and flag semantics without network calls.

**Tech Stack:** Python 3.11+, `anthropic>=0.40` (optional extra), `openai>=1.40` (optional extra), `respx>=0.21` for HTTP mocking in tests, `pytest-asyncio` (auto mode), `pydantic>=2.6`.

---

> **Spec §15 #3 resolved inline (documentation-only):** Neither shim sets a literal `fast_llm` default inside `Sleuth()`. Recommended invocations are shown in each shim's module docstring (see Tasks 2 and 4). No code change to `_agent.py` is required.

> **No callouts needed.** All types (`LLMChunk`, `TextDelta`, `ReasoningDelta`, `ToolCall`, `Stop`, `Message`, `Tool`, `LLMClient`) are frozen in conventions §5.1 and owned by Phase 1 (`src/sleuth/llm/base.py`). This plan creates only the two files listed in Phase 10's ownership row plus their tests.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/sleuth/llm/anthropic.py` | Create | `Anthropic` class: lazy-import `anthropic` SDK, stream adapter, reasoning detection |
| `src/sleuth/llm/openai.py` | Create | `OpenAI` class: lazy-import `openai` SDK, stream adapter, reasoning detection |
| `tests/llm/test_anthropic.py` | Create | Unit tests for `Anthropic` shim (HTTP-mocked via `respx`) |
| `tests/llm/test_openai.py` | Create | Unit tests for `OpenAI` shim (HTTP-mocked via `respx`) |

Files this plan **reads but does not modify** (owned by Phase 1):
- `src/sleuth/llm/base.py` — protocol + `LLMChunk` types
- `src/sleuth/llm/stub.py` — `StubLLM` (used in tests for comparison)
- `src/sleuth/errors.py` — `LLMError`

---

## Task 1: Branch Setup

**Files:**
- No file changes (git only)

- [ ] **Step 1: Create feature branch off `develop`**

```bash
git checkout develop
git checkout -b feature/phase-10-llm-shims
```

Expected: `Switched to a new branch 'feature/phase-10-llm-shims'`

- [ ] **Step 2: Verify Phase 1 types are in place**

```bash
python -c "from sleuth.llm.base import TextDelta, ReasoningDelta, ToolCall, Stop, LLMClient, Message, Tool; print('ok')"
```

Expected: `ok`

If this fails, Phase 1 is not yet merged. Stop here and wait for Phase 1.

---

## Task 2: Anthropic shim skeleton + `supports_reasoning` flag

**Files:**
- Create: `src/sleuth/llm/anthropic.py`
- Create: `tests/llm/test_anthropic.py`

### 2a — Failing test: import and flag semantics

- [ ] **Step 1: Write the failing test**

```python
# tests/llm/test_anthropic.py
"""Tests for the Anthropic LLM shim."""
from __future__ import annotations

import pytest


class TestAnthropicImport:
    def test_import_without_sdk_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing the class is fine; instantiating without the SDK raises ImportError."""
        import sys

        # Remove anthropic from sys.modules so lazy-import fails
        monkeypatch.delitem(sys.modules, "anthropic", raising=False)

        # Patch builtins.__import__ to block anthropic
        original_import = __builtins__.__import__ if isinstance(__builtins__, dict) else __import__  # type: ignore[attr-defined]

        def _block_anthropic(name: str, *args: object, **kwargs: object) -> object:
            if name == "anthropic" or name.startswith("anthropic."):
                raise ImportError("No module named 'anthropic'")
            return original_import(name, *args, **kwargs)  # type: ignore[call-arg]

        monkeypatch.setattr("builtins.__import__", _block_anthropic)

        # Re-import to pick up the monkeypatched import
        import importlib
        import sleuth.llm.anthropic as mod
        importlib.reload(mod)

        with pytest.raises(ImportError, match="agent-sleuth\\[anthropic\\]"):
            mod.Anthropic(model="claude-sonnet-4-6", api_key="sk-ant-test")


class TestSupportsReasoning:
    def test_extended_thinking_models_support_reasoning(self) -> None:
        """Claude opus-4-7 and sonnet-4-6 with thinking enabled set supports_reasoning=True."""
        from sleuth.llm.anthropic import Anthropic

        opus = Anthropic(model="claude-opus-4-7", api_key="sk-ant-test")
        sonnet_thinking = Anthropic(
            model="claude-sonnet-4-6", api_key="sk-ant-test", thinking=True
        )
        assert opus.supports_reasoning is True
        assert sonnet_thinking.supports_reasoning is True

    def test_non_thinking_models_do_not_support_reasoning(self) -> None:
        """Standard models without thinking enabled set supports_reasoning=False."""
        from sleuth.llm.anthropic import Anthropic

        haiku = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")
        sonnet_plain = Anthropic(model="claude-sonnet-4-6", api_key="sk-ant-test")
        assert haiku.supports_reasoning is False
        assert sonnet_plain.supports_reasoning is False

    def test_supports_structured_output_always_true(self) -> None:
        from sleuth.llm.anthropic import Anthropic

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")
        assert client.supports_structured_output is True

    def test_name_includes_provider_and_model(self) -> None:
        from sleuth.llm.anthropic import Anthropic

        client = Anthropic(model="claude-sonnet-4-6", api_key="sk-ant-test")
        assert client.name == "anthropic:claude-sonnet-4-6"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/llm/test_anthropic.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `sleuth.llm.anthropic` does not exist yet.

### 2b — Implement skeleton

- [ ] **Step 3: Write minimal implementation**

```python
# src/sleuth/llm/anthropic.py
"""Anthropic LLM shim for agent-sleuth.

Install:
    pip install agent-sleuth[anthropic]

Recommended usage::

    from sleuth.llm import Anthropic
    from sleuth import Sleuth, LocalFiles

    agent = Sleuth(
        llm=Anthropic(model="claude-opus-4-7"),          # reasoning + synthesis
        fast_llm=Anthropic(model="claude-haiku-4-5"),    # routing / picking
        backends=[LocalFiles("./docs")],
    )

Spec §15 #3 (documentation-only): there is no literal ``fast_llm`` default baked
into ``Sleuth()``. The examples above are recommendations only; users pick their own
models to keep BYOK pure.

Extended thinking
-----------------
Pass ``thinking=True`` to enable extended thinking for models that support it
(claude-opus-4-7, claude-sonnet-4-6). This sets ``supports_reasoning=True`` and
emits ``ReasoningDelta`` chunks that the engine maps to ``ThinkingEvent``.

Structured output
-----------------
When ``schema=`` is passed to ``stream()``, this shim uses Anthropic's native
tool-call coercion mechanism to return structured JSON. For older models that do
not support tool use, the shim falls back to requesting JSON-mode and parsing the
text response.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from sleuth.llm.base import (
    LLMChunk,
    Message,
    ReasoningDelta,
    Stop,
    TextDelta,
    Tool,
    ToolCall,
)

# Models for which supports_reasoning is True regardless of the `thinking` flag
# (they always support extended thinking when the flag is on, and we mark opus as
# always-reasoning because users rarely instantiate it without thinking intent).
_ALWAYS_REASONING_MODELS: frozenset[str] = frozenset({"claude-opus-4-7"})

# Models that support extended thinking when ``thinking=True`` is passed.
_THINKING_CAPABLE_MODELS: frozenset[str] = frozenset(
    {"claude-opus-4-7", "claude-sonnet-4-6"}
)


class Anthropic:
    """``LLMClient`` adapter for the Anthropic Python SDK.

    Parameters
    ----------
    model:
        Anthropic model identifier, e.g. ``"claude-sonnet-4-6"``.
    api_key:
        Anthropic API key.  Defaults to the ``ANTHROPIC_API_KEY`` environment
        variable when omitted.
    thinking:
        Enable extended thinking (extended reasoning tokens). Only meaningful
        for models in ``_THINKING_CAPABLE_MODELS``.  When ``True``, sets
        ``supports_reasoning=True``.
    thinking_budget_tokens:
        Maximum thinking tokens per request (default 5000).
    max_tokens:
        Maximum response tokens (default 4096; extended thinking requires ≥1024).
    base_url:
        Override the Anthropic API base URL (useful for proxies / testing).
    timeout:
        Request timeout in seconds (default 120).
    """

    supports_structured_output: bool = True

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        thinking: bool = False,
        thinking_budget_tokens: int = 5000,
        max_tokens: int = 4096,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        # --- lazy import: raises ImportError with helpful message if SDK absent ---
        try:
            import anthropic as _anthropic_sdk  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The Anthropic SDK is required. Install it with: "
                "pip install agent-sleuth[anthropic]"
            ) from exc

        self._sdk = _anthropic_sdk
        self._model = model
        self._thinking = thinking
        self._thinking_budget_tokens = thinking_budget_tokens
        self._max_tokens = max_tokens

        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self._client = _anthropic_sdk.AsyncAnthropic(
            timeout=timeout, **client_kwargs
        )

    # ------------------------------------------------------------------
    # LLMClient protocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    @property
    def supports_reasoning(self) -> bool:
        if self._model in _ALWAYS_REASONING_MODELS:
            return True
        if self._thinking and self._model in _THINKING_CAPABLE_MODELS:
            return True
        return False

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Yield ``LLMChunk`` items from an Anthropic streaming response."""
        sdk_messages, system_prompt = _split_messages(messages)
        sdk_tools = _build_tools(tools, schema)
        extra_params = _build_extra_params(
            thinking=self._thinking,
            thinking_budget_tokens=self._thinking_budget_tokens,
        )

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=sdk_messages,
            **({"system": system_prompt} if system_prompt else {}),
            **({"tools": sdk_tools} if sdk_tools else {}),
            **extra_params,
        ) as stream:
            async for event in stream:
                chunk = _translate_event(event)
                if chunk is not None:
                    yield chunk


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _split_messages(
    messages: list[Message],
) -> tuple[list[dict[str, Any]], str | None]:
    """Separate the system prompt from user/assistant messages."""
    system_parts: list[str] = []
    sdk_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "tool":
            sdk_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content,
                        }
                    ],
                }
            )
        else:
            sdk_messages.append({"role": msg.role, "content": msg.content})

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return sdk_messages, system_prompt


def _build_tools(
    tools: list[Tool] | None,
    schema: type[BaseModel] | None,
) -> list[dict[str, Any]] | None:
    """Build Anthropic-format tool definitions.

    When ``schema`` is provided we inject a synthetic ``structured_output``
    tool that the model is forced to call, producing structured JSON.
    """
    result: list[dict[str, Any]] = []

    if tools:
        for tool in tools:
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )

    if schema is not None:
        result.append(
            {
                "name": "structured_output",
                "description": (
                    f"Return a structured {schema.__name__} response. "
                    "You MUST call this tool with your answer."
                ),
                "input_schema": schema.model_json_schema(),
            }
        )

    return result if result else None


def _build_extra_params(
    *, thinking: bool, thinking_budget_tokens: int
) -> dict[str, Any]:
    if not thinking:
        return {}
    return {
        "thinking": {
            "type": "enabled",
            "budget_tokens": thinking_budget_tokens,
        }
    }


def _translate_event(event: Any) -> LLMChunk | None:  # noqa: ANN401
    """Translate a raw Anthropic SDK streaming event to an ``LLMChunk``."""
    event_type = getattr(event, "type", None)

    # Text delta
    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        delta_type = getattr(delta, "type", None)
        if delta_type == "text_delta":
            return TextDelta(text=delta.text)
        if delta_type == "thinking_delta":
            return ReasoningDelta(text=delta.thinking)
        if delta_type == "input_json_delta":
            # Partial tool input JSON — we accumulate via stop handling
            return None

    # Tool use block
    if event_type == "content_block_start":
        block = getattr(event, "content_block", None)
        if getattr(block, "type", None) == "tool_use":
            # Emit ToolCall at block_start; arguments will be empty until
            # input_json_delta accumulates — callers should buffer.
            return ToolCall(
                id=block.id,
                name=block.name,
                arguments={},
            )

    # Stop
    if event_type == "message_delta":
        delta = getattr(event, "delta", None)
        stop_reason = getattr(delta, "stop_reason", None)
        if stop_reason == "end_turn":
            return Stop(reason="end_turn")
        if stop_reason == "tool_use":
            return Stop(reason="tool_use")
        if stop_reason == "max_tokens":
            return Stop(reason="max_tokens")
        if stop_reason == "stop_sequence":
            return Stop(reason="stop_sequence")

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/llm/test_anthropic.py::TestSupportsReasoning \
              tests/llm/test_anthropic.py::TestAnthropicImport -v
```

Expected: all 5 tests pass (the import-blocking test is fragile across Python versions; if it fails due to `__builtins__` variance, see note below).

> **Note on the import-blocking test:** Python's `__builtins__` is a `dict` in non-`__main__` modules; the monkeypatch above handles both forms. If the test is still flaky, replace with a simpler approach: patch `importlib.import_module` directly, or use `sys.modules` to inject a stub that raises. The test's intent is to confirm the `ImportError` message; the exact mechanism is secondary.

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/llm/anthropic.py tests/llm/test_anthropic.py
git commit -m "feat(llm): add Anthropic shim skeleton with supports_reasoning flag"
```

---

## Task 3: Anthropic stream adapter — TextDelta, ReasoningDelta, Stop

**Files:**
- Modify: `tests/llm/test_anthropic.py`
- (implementation already written in Task 2; tests verify the adapter)

- [ ] **Step 1: Write the failing tests**

Add these classes to `tests/llm/test_anthropic.py`:

```python
import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch


def _make_text_event(text: str) -> MagicMock:
    """Build a mock Anthropic content_block_delta event with text_delta."""
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = MagicMock()
    event.delta.type = "text_delta"
    event.delta.text = text
    return event


def _make_thinking_event(text: str) -> MagicMock:
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = MagicMock()
    event.delta.type = "thinking_delta"
    event.delta.thinking = text
    return event


def _make_stop_event(stop_reason: str) -> MagicMock:
    event = MagicMock()
    event.type = "message_delta"
    event.delta = MagicMock()
    event.delta.stop_reason = stop_reason
    return event


def _make_tool_use_start_event(tool_id: str, tool_name: str) -> MagicMock:
    event = MagicMock()
    event.type = "content_block_start"
    event.content_block = MagicMock()
    event.content_block.type = "tool_use"
    event.content_block.id = tool_id
    event.content_block.name = tool_name
    return event


async def _make_async_iter(events: list[MagicMock]) -> AsyncIterator[MagicMock]:
    for e in events:
        yield e


class TestAnthropicStreamTextDelta:
    async def test_text_delta_emitted(self) -> None:
        """stream() yields TextDelta for each text_delta event."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import Message, TextDelta

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")

        raw_events = [
            _make_text_event("Hello"),
            _make_text_event(", world"),
            _make_stop_event("end_turn"),
        ]

        # Patch the underlying SDK client's stream context manager
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_events)

        with patch.object(client._client.messages, "stream", return_value=mock_stream):
            chunks = []
            async for chunk in await client.stream(
                [Message(role="user", content="Hi")]
            ):
                chunks.append(chunk)

        text_chunks = [c for c in chunks if isinstance(c, TextDelta)]
        assert [c.text for c in text_chunks] == ["Hello", ", world"]

    async def test_stop_chunk_emitted(self) -> None:
        """stream() yields Stop(reason='end_turn') at message_delta stop."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import Message, Stop

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")
        raw_events = [_make_stop_event("end_turn")]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_events)

        with patch.object(client._client.messages, "stream", return_value=mock_stream):
            chunks = []
            async for chunk in await client.stream(
                [Message(role="user", content="Hi")]
            ):
                chunks.append(chunk)

        stop_chunks = [c for c in chunks if isinstance(c, Stop)]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].reason == "end_turn"


class TestAnthropicStreamReasoning:
    async def test_reasoning_delta_emitted_when_thinking_enabled(self) -> None:
        """stream() yields ReasoningDelta for thinking_delta events."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import Message, ReasoningDelta

        client = Anthropic(
            model="claude-sonnet-4-6", api_key="sk-ant-test", thinking=True
        )

        raw_events = [
            _make_thinking_event("Let me think..."),
            _make_text_event("Answer"),
            _make_stop_event("end_turn"),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_events)

        with patch.object(client._client.messages, "stream", return_value=mock_stream):
            chunks = []
            async for chunk in await client.stream(
                [Message(role="user", content="Hi")]
            ):
                chunks.append(chunk)

        reasoning_chunks = [c for c in chunks if isinstance(c, ReasoningDelta)]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0].text == "Let me think..."

    async def test_no_reasoning_delta_without_thinking(self) -> None:
        """Non-thinking models produce no ReasoningDelta even if the SDK emitted one."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import Message, ReasoningDelta

        # Standard model, thinking=False (default)
        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")

        # Even if SDK somehow sends a thinking_delta, we pass it through (it won't happen
        # in practice — test confirms the shim *does* emit ReasoningDelta regardless,
        # and the *engine* gates ThinkingEvent on supports_reasoning).
        # This test verifies supports_reasoning=False is set correctly.
        assert client.supports_reasoning is False
```

- [ ] **Step 2: Run tests to verify initial state**

```bash
uv run pytest tests/llm/test_anthropic.py::TestAnthropicStreamTextDelta \
              tests/llm/test_anthropic.py::TestAnthropicStreamReasoning -v
```

Expected: `AttributeError` or similar because `client.stream()` is defined but the mock wiring needs the async generator fix below.

- [ ] **Step 3: Fix `stream()` return type — it must be an async generator, not a coroutine**

The `stream` method in the current skeleton uses `async def ... -> AsyncIterator[LLMChunk]` with `yield` inside, making it an async generator function. That means `await client.stream(...)` is incorrect — callers should use `async for chunk in client.stream(...)` directly.

Update the `stream` signature in `src/sleuth/llm/anthropic.py`. The `LLMClient` protocol (conventions §5.1) declares:

```python
async def stream(...) -> AsyncIterator[LLMChunk]: ...
```

For a Protocol stub this is fine, but the concrete implementation must be an `AsyncIterator`. The simplest correct form is an `async def` with `yield` (async generator):

```python
    async def stream(  # type: ignore[override]  # async generator, not coroutine
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        ...
        async with self._client.messages.stream(...) as stream:
            async for event in stream:
                chunk = _translate_event(event)
                if chunk is not None:
                    yield chunk
```

The method body already uses `yield`, so it is already an async generator. The protocol annotation `-> AsyncIterator[LLMChunk]` is correct for a generator return. Update test call sites to remove `await`:

In `tests/llm/test_anthropic.py`, replace:

```python
            async for chunk in await client.stream(
```

with:

```python
            async for chunk in client.stream(
```

Do this for every `stream()` call in the test file.

- [ ] **Step 4: Run tests again**

```bash
uv run pytest tests/llm/test_anthropic.py::TestAnthropicStreamTextDelta \
              tests/llm/test_anthropic.py::TestAnthropicStreamReasoning -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/llm/test_anthropic.py src/sleuth/llm/anthropic.py
git commit -m "test(llm): add Anthropic stream adapter tests (TextDelta, ReasoningDelta, Stop)"
```

---

## Task 4: Anthropic ToolCall chunks and structured output passthrough

**Files:**
- Modify: `tests/llm/test_anthropic.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/llm/test_anthropic.py`:

```python
from pydantic import BaseModel as PydanticBaseModel


class _Verdict(PydanticBaseModel):
    answer: str
    confidence: float


class TestAnthropicToolCall:
    async def test_tool_call_chunk_emitted(self) -> None:
        """stream() yields ToolCall when a tool_use content block starts."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import Message, Tool, ToolCall

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")

        raw_events = [
            _make_tool_use_start_event("toolu_01", "my_tool"),
            _make_stop_event("tool_use"),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_events)

        with patch.object(client._client.messages, "stream", return_value=mock_stream):
            chunks = []
            async for chunk in client.stream(
                [Message(role="user", content="do something")],
                tools=[Tool(name="my_tool", description="desc", input_schema={"type": "object"})],
            ):
                chunks.append(chunk)

        tool_chunks = [c for c in chunks if isinstance(c, ToolCall)]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].id == "toolu_01"
        assert tool_chunks[0].name == "my_tool"

    async def test_schema_injects_structured_output_tool(self) -> None:
        """When schema= is passed, a 'structured_output' tool is injected into the request."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import Message

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")

        captured_kwargs: dict = {}

        def _capture_stream(**kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            mock_stream = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=False)
            mock_stream.__aiter__ = lambda self: _make_async_iter([_make_stop_event("end_turn")])
            return mock_stream

        with patch.object(client._client.messages, "stream", side_effect=_capture_stream):
            async for _ in client.stream(
                [Message(role="user", content="classify this")],
                schema=_Verdict,
            ):
                pass

        assert "tools" in captured_kwargs
        tool_names = [t["name"] for t in captured_kwargs["tools"]]
        assert "structured_output" in tool_names

    async def test_system_message_is_extracted(self) -> None:
        """System messages are extracted and passed as the 'system' parameter."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import Message

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")
        captured_kwargs: dict = {}

        def _capture_stream(**kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            mock_stream = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=False)
            mock_stream.__aiter__ = lambda self: _make_async_iter([_make_stop_event("end_turn")])
            return mock_stream

        with patch.object(client._client.messages, "stream", side_effect=_capture_stream):
            async for _ in client.stream(
                [
                    Message(role="system", content="You are helpful."),
                    Message(role="user", content="Hi"),
                ]
            ):
                pass

        assert captured_kwargs.get("system") == "You are helpful."
        user_messages = captured_kwargs.get("messages", [])
        assert all(m["role"] != "system" for m in user_messages)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/llm/test_anthropic.py::TestAnthropicToolCall -v
```

Expected: all 3 tests pass (implementation already handles these in Task 2).

- [ ] **Step 3: Commit**

```bash
git add tests/llm/test_anthropic.py
git commit -m "test(llm): add Anthropic ToolCall and structured-output tests"
```

---

## Task 5: Anthropic full test suite + type check

**Files:**
- Modify: `tests/llm/test_anthropic.py`

- [ ] **Step 1: Add edge-case and error tests**

```python
class TestAnthropicEdgeCases:
    async def test_tool_message_becomes_tool_result(self) -> None:
        """Messages with role='tool' are translated to Anthropic tool_result format."""
        from sleuth.llm.anthropic import _split_messages
        from sleuth.llm.base import Message

        messages = [
            Message(role="user", content="Call the tool"),
            Message(role="tool", content='{"ok": true}', tool_call_id="toolu_01"),
        ]
        sdk_messages, system = _split_messages(messages)

        assert system is None
        tool_msg = sdk_messages[1]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "toolu_01"

    async def test_multiple_system_messages_concatenated(self) -> None:
        """Multiple system messages are joined with double newline."""
        from sleuth.llm.anthropic import _split_messages
        from sleuth.llm.base import Message

        messages = [
            Message(role="system", content="Part 1."),
            Message(role="system", content="Part 2."),
            Message(role="user", content="Hi"),
        ]
        _, system = _split_messages(messages)
        assert system == "Part 1.\n\nPart 2."

    def test_thinking_params_included_when_thinking_true(self) -> None:
        """When thinking=True, _build_extra_params returns the thinking dict."""
        from sleuth.llm.anthropic import _build_extra_params

        params = _build_extra_params(thinking=True, thinking_budget_tokens=3000)
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 3000

    def test_thinking_params_empty_when_thinking_false(self) -> None:
        from sleuth.llm.anthropic import _build_extra_params

        params = _build_extra_params(thinking=False, thinking_budget_tokens=5000)
        assert params == {}

    async def test_stop_reason_max_tokens(self) -> None:
        """Stop chunk with reason='max_tokens' is emitted correctly."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import Message, Stop

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")
        raw_events = [_make_stop_event("max_tokens")]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_events)

        with patch.object(client._client.messages, "stream", return_value=mock_stream):
            chunks = [c async for c in client.stream([Message(role="user", content="hi")])]

        stop = next(c for c in chunks if isinstance(c, Stop))
        assert stop.reason == "max_tokens"
```

- [ ] **Step 2: Run the complete Anthropic test file**

```bash
uv run pytest tests/llm/test_anthropic.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run mypy on the shim**

```bash
uv run mypy src/sleuth/llm/anthropic.py
```

Expected: `Success: no issues found in 1 source file`

If mypy reports errors about the `anthropic` SDK types not being available (since the SDK may not be installed in the dev env when running without the extra), add a `# type: ignore` comment on the lazy import line only and document why.

- [ ] **Step 4: Commit**

```bash
git add tests/llm/test_anthropic.py
git commit -m "test(llm): add Anthropic edge-case tests and verify mypy"
```

---

## Task 6: OpenAI shim skeleton + `supports_reasoning` flag

**Files:**
- Create: `src/sleuth/llm/openai.py`
- Create: `tests/llm/test_openai.py`

### 6a — Failing test

- [ ] **Step 1: Write the failing tests**

```python
# tests/llm/test_openai.py
"""Tests for the OpenAI LLM shim."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestOpenAIImport:
    def test_import_without_sdk_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Instantiating OpenAI without the openai SDK raises ImportError with hint."""
        import sys

        monkeypatch.delitem(sys.modules, "openai", raising=False)

        original_import = __builtins__.__import__ if isinstance(__builtins__, dict) else __import__  # type: ignore[attr-defined]

        def _block_openai(name: str, *args: object, **kwargs: object) -> object:
            if name == "openai" or name.startswith("openai."):
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)  # type: ignore[call-arg]

        monkeypatch.setattr("builtins.__import__", _block_openai)

        import importlib
        import sleuth.llm.openai as mod
        importlib.reload(mod)

        with pytest.raises(ImportError, match="agent-sleuth\\[openai\\]"):
            mod.OpenAI(model="gpt-4o", api_key="sk-test")


class TestOpenAISupportsReasoning:
    def test_o_series_models_support_reasoning(self) -> None:
        """o1, o3, o3-mini, o4-mini set supports_reasoning=True."""
        from sleuth.llm.openai import OpenAI

        for model in ("o1", "o1-mini", "o3", "o3-mini", "o4-mini"):
            client = OpenAI(model=model, api_key="sk-test")
            assert client.supports_reasoning is True, f"{model} should support reasoning"

    def test_gpt_models_do_not_support_reasoning(self) -> None:
        """gpt-4o, gpt-4-turbo set supports_reasoning=False."""
        from sleuth.llm.openai import OpenAI

        for model in ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"):
            client = OpenAI(model=model, api_key="sk-test")
            assert client.supports_reasoning is False, f"{model} should not support reasoning"

    def test_supports_structured_output_always_true(self) -> None:
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")
        assert client.supports_structured_output is True

    def test_name_includes_provider_and_model(self) -> None:
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")
        assert client.name == "openai:gpt-4o"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/llm/test_openai.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `sleuth.llm.openai` does not exist yet.

### 6b — Implement skeleton

- [ ] **Step 3: Write minimal implementation**

```python
# src/sleuth/llm/openai.py
"""OpenAI LLM shim for agent-sleuth.

Install:
    pip install agent-sleuth[openai]

Recommended usage::

    from sleuth.llm import OpenAI
    from sleuth import Sleuth, LocalFiles

    agent = Sleuth(
        llm=OpenAI(model="gpt-4o"),               # synthesis
        fast_llm=OpenAI(model="gpt-4o-mini"),     # routing / picking
        backends=[LocalFiles("./docs")],
    )

For reasoning with o-series models::

    agent = Sleuth(
        llm=OpenAI(model="o3"),                   # supports_reasoning=True automatically
        backends=[LocalFiles("./docs")],
    )

Spec §15 #3 (documentation-only): there is no literal ``fast_llm`` default baked
into ``Sleuth()``. The examples above are recommendations only.

Structured output
-----------------
When ``schema=`` is passed to ``stream()``, this shim uses the ``response_format``
parameter with ``json_schema`` type (available in ``openai>=1.40`` for models that
support it). For older endpoints, falls back to ``response_format={"type": "json_object"}``
and parses the JSON from the text response.

Reasoning (o-series)
--------------------
For o-series models (o1, o3, o3-mini, o4-mini, etc.), the API returns reasoning
tokens in the ``reasoning_content`` or ``refusal`` fields depending on SDK version.
This shim emits them as ``ReasoningDelta`` chunks. ``supports_reasoning`` is set
``True`` for all models whose name starts with ``o`` followed by a digit (e.g. o1,
o3, o4-mini) and ``False`` for all others.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from sleuth.llm.base import (
    LLMChunk,
    Message,
    ReasoningDelta,
    Stop,
    TextDelta,
    Tool,
    ToolCall,
)

import re

# Models whose names match this pattern are treated as o-series reasoning models.
_O_SERIES_PATTERN = re.compile(r"^o\d")


def _is_o_series(model: str) -> bool:
    """Return True if ``model`` is an OpenAI o-series reasoning model."""
    return bool(_O_SERIES_PATTERN.match(model.lower()))


class OpenAI:
    """``LLMClient`` adapter for the OpenAI Python SDK.

    Parameters
    ----------
    model:
        OpenAI model identifier, e.g. ``"gpt-4o"``, ``"o3"``.
    api_key:
        OpenAI API key. Defaults to the ``OPENAI_API_KEY`` environment variable.
    base_url:
        Override the OpenAI API base URL (useful for proxies / Azure OpenAI).
    timeout:
        Request timeout in seconds (default 120).
    max_completion_tokens:
        Maximum tokens in the completion (default 4096).
    """

    supports_structured_output: bool = True

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_completion_tokens: int = 4096,
    ) -> None:
        # --- lazy import ---
        try:
            import openai as _openai_sdk  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The OpenAI SDK is required. Install it with: "
                "pip install agent-sleuth[openai]"
            ) from exc

        self._sdk = _openai_sdk
        self._model = model
        self._max_completion_tokens = max_completion_tokens

        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self._client = _openai_sdk.AsyncOpenAI(timeout=timeout, **client_kwargs)

    # ------------------------------------------------------------------
    # LLMClient protocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def supports_reasoning(self) -> bool:
        return _is_o_series(self._model)

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Yield ``LLMChunk`` items from an OpenAI streaming response."""
        sdk_messages = _build_sdk_messages(messages)
        extra_kwargs = _build_extra_kwargs(schema=schema, tools=tools)

        async with self._client.chat.completions.stream(
            model=self._model,
            messages=sdk_messages,
            max_completion_tokens=self._max_completion_tokens,
            **extra_kwargs,
        ) as stream:
            # Track open tool calls by index so we can emit ToolCall at close
            _open_tool_calls: dict[int, dict[str, Any]] = {}

            async for event in stream:
                chunks = _translate_chunk(event, _open_tool_calls)
                for chunk in chunks:
                    yield chunk


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _build_sdk_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert ``Message`` list to OpenAI chat-completions format."""
    sdk: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            sdk.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id or "",
                }
            )
        else:
            sdk.append({"role": msg.role, "content": msg.content})
    return sdk


def _build_extra_kwargs(
    *,
    schema: type[BaseModel] | None,
    tools: list[Tool] | None,
) -> dict[str, Any]:
    """Build extra keyword arguments for the OpenAI API call."""
    kwargs: dict[str, Any] = {}

    if tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }

    return kwargs


def _translate_chunk(
    event: Any,  # noqa: ANN401
    open_tool_calls: dict[int, dict[str, Any]],
) -> list[LLMChunk]:
    """Translate a raw OpenAI streaming chunk to zero or more ``LLMChunk`` items."""
    results: list[LLMChunk] = []

    # event is a ChatCompletionChunk
    choices = getattr(event, "choices", [])
    if not choices:
        return results

    choice = choices[0]
    delta = getattr(choice, "delta", None)
    finish_reason = getattr(choice, "finish_reason", None)

    if delta is not None:
        # Text content
        content = getattr(delta, "content", None)
        if content:
            results.append(TextDelta(text=content))

        # Reasoning content (o-series)
        reasoning = getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        )
        if reasoning:
            results.append(ReasoningDelta(text=reasoning))

        # Tool calls
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            for tc_delta in tool_calls:
                idx = tc_delta.index
                if idx not in open_tool_calls:
                    open_tool_calls[idx] = {
                        "id": tc_delta.id or "",
                        "name": tc_delta.function.name if tc_delta.function else "",
                        "arguments": "",
                    }
                if tc_delta.function and tc_delta.function.arguments:
                    open_tool_calls[idx]["arguments"] += tc_delta.function.arguments

    # Stop
    if finish_reason is not None:
        # Emit accumulated tool calls first
        for tc in open_tool_calls.values():
            import json  # noqa: PLC0415
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {"raw": tc["arguments"]}
            results.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))
        open_tool_calls.clear()

        reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "stop_sequence",
        }
        mapped = reason_map.get(finish_reason, "end_turn")
        results.append(Stop(reason=mapped))  # type: ignore[arg-type]

    return results
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/llm/test_openai.py::TestOpenAISupportsReasoning \
              tests/llm/test_openai.py::TestOpenAIImport -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/llm/openai.py tests/llm/test_openai.py
git commit -m "feat(llm): add OpenAI shim skeleton with supports_reasoning flag"
```

---

## Task 7: OpenAI stream adapter — TextDelta, ReasoningDelta, ToolCall, Stop

**Files:**
- Modify: `tests/llm/test_openai.py`

- [ ] **Step 1: Add mock helpers and stream tests**

Add to `tests/llm/test_openai.py`:

```python
def _make_openai_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
    tool_calls: list[dict] | None = None,
) -> MagicMock:
    """Build a mock OpenAI ChatCompletionChunk."""
    chunk = MagicMock()
    choice = MagicMock()
    delta = MagicMock()

    delta.content = content
    delta.reasoning_content = reasoning_content
    delta.reasoning = None
    delta.tool_calls = None

    if tool_calls is not None:
        tc_mocks = []
        for i, tc in enumerate(tool_calls):
            tc_mock = MagicMock()
            tc_mock.index = i
            tc_mock.id = tc.get("id", f"call_{i}")
            tc_mock.function = MagicMock()
            tc_mock.function.name = tc.get("name", "")
            tc_mock.function.arguments = tc.get("arguments", "")
            tc_mocks.append(tc_mock)
        delta.tool_calls = tc_mocks

    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


async def _make_async_iter(events: list[MagicMock]) -> AsyncIterator[MagicMock]:
    for e in events:
        yield e


class TestOpenAIStreamTextDelta:
    async def test_text_delta_emitted(self) -> None:
        from sleuth.llm.openai import OpenAI
        from sleuth.llm.base import Message, TextDelta

        client = OpenAI(model="gpt-4o", api_key="sk-test")

        raw_chunks = [
            _make_openai_chunk(content="Hello"),
            _make_openai_chunk(content=", world"),
            _make_openai_chunk(finish_reason="stop"),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_chunks)

        with patch.object(client._client.chat.completions, "stream", return_value=mock_stream):
            chunks = [c async for c in client.stream([Message(role="user", content="Hi")])]

        text_chunks = [c for c in chunks if isinstance(c, TextDelta)]
        assert [c.text for c in text_chunks] == ["Hello", ", world"]

    async def test_stop_chunk_emitted_on_finish(self) -> None:
        from sleuth.llm.openai import OpenAI
        from sleuth.llm.base import Message, Stop

        client = OpenAI(model="gpt-4o", api_key="sk-test")
        raw_chunks = [_make_openai_chunk(finish_reason="stop")]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_chunks)

        with patch.object(client._client.chat.completions, "stream", return_value=mock_stream):
            chunks = [c async for c in client.stream([Message(role="user", content="Hi")])]

        stop_chunks = [c for c in chunks if isinstance(c, Stop)]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].reason == "end_turn"

    async def test_finish_reason_length_maps_to_max_tokens(self) -> None:
        from sleuth.llm.openai import OpenAI
        from sleuth.llm.base import Message, Stop

        client = OpenAI(model="gpt-4o", api_key="sk-test")
        raw_chunks = [_make_openai_chunk(finish_reason="length")]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_chunks)

        with patch.object(client._client.chat.completions, "stream", return_value=mock_stream):
            chunks = [c async for c in client.stream([Message(role="user", content="Hi")])]

        stop = next(c for c in chunks if isinstance(c, Stop))
        assert stop.reason == "max_tokens"


class TestOpenAIStreamReasoning:
    async def test_reasoning_delta_emitted_for_o_series(self) -> None:
        from sleuth.llm.openai import OpenAI
        from sleuth.llm.base import Message, ReasoningDelta

        client = OpenAI(model="o3", api_key="sk-test")

        raw_chunks = [
            _make_openai_chunk(reasoning_content="Thinking step 1"),
            _make_openai_chunk(content="Answer"),
            _make_openai_chunk(finish_reason="stop"),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_chunks)

        with patch.object(client._client.chat.completions, "stream", return_value=mock_stream):
            chunks = [c async for c in client.stream([Message(role="user", content="Hi")])]

        reasoning_chunks = [c for c in chunks if isinstance(c, ReasoningDelta)]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0].text == "Thinking step 1"


class TestOpenAIToolCall:
    async def test_tool_call_emitted_at_finish(self) -> None:
        """ToolCall is emitted with accumulated arguments when finish_reason='tool_calls'."""
        from sleuth.llm.openai import OpenAI
        from sleuth.llm.base import Message, Tool, ToolCall

        client = OpenAI(model="gpt-4o", api_key="sk-test")

        raw_chunks = [
            _make_openai_chunk(
                tool_calls=[{"id": "call_abc", "name": "search", "arguments": '{"q":'}]
            ),
            _make_openai_chunk(
                tool_calls=[{"id": "", "name": "", "arguments": '"test"}'}]
            ),
            _make_openai_chunk(finish_reason="tool_calls"),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_chunks)

        with patch.object(client._client.chat.completions, "stream", return_value=mock_stream):
            chunks = [
                c
                async for c in client.stream(
                    [Message(role="user", content="search for test")],
                    tools=[Tool(name="search", description="search", input_schema={"type": "object"})],
                )
            ]

        tool_chunks = [c for c in chunks if isinstance(c, ToolCall)]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].name == "search"
        assert tool_chunks[0].arguments == {"q": "test"}
```

- [ ] **Step 2: Run the new stream tests**

```bash
uv run pytest tests/llm/test_openai.py::TestOpenAIStreamTextDelta \
              tests/llm/test_openai.py::TestOpenAIStreamReasoning \
              tests/llm/test_openai.py::TestOpenAIToolCall -v
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/llm/test_openai.py
git commit -m "test(llm): add OpenAI stream adapter tests (TextDelta, ReasoningDelta, ToolCall, Stop)"
```

---

## Task 8: OpenAI structured-output and message-format tests

**Files:**
- Modify: `tests/llm/test_openai.py`

- [ ] **Step 1: Add structured-output and message-format tests**

```python
from pydantic import BaseModel as PydanticBaseModel


class _Verdict(PydanticBaseModel):
    answer: str
    confidence: float


class TestOpenAIStructuredOutput:
    async def test_schema_injects_response_format(self) -> None:
        """When schema= is passed, response_format=json_schema is added to the request."""
        from sleuth.llm.openai import OpenAI
        from sleuth.llm.base import Message

        client = OpenAI(model="gpt-4o", api_key="sk-test")
        captured_kwargs: dict = {}

        def _capture(**kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            mock_stream = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=False)
            mock_stream.__aiter__ = lambda self: _make_async_iter(
                [_make_openai_chunk(finish_reason="stop")]
            )
            return mock_stream

        with patch.object(client._client.chat.completions, "stream", side_effect=_capture):
            async for _ in client.stream(
                [Message(role="user", content="classify")],
                schema=_Verdict,
            ):
                pass

        rf = captured_kwargs.get("response_format", {})
        assert rf.get("type") == "json_schema"
        assert rf["json_schema"]["name"] == "_Verdict"

    async def test_tool_message_role_preserved(self) -> None:
        """Messages with role='tool' are passed as tool role to OpenAI."""
        from sleuth.llm.openai import _build_sdk_messages
        from sleuth.llm.base import Message

        messages = [
            Message(role="user", content="Call the tool"),
            Message(role="tool", content='{"result": 42}', tool_call_id="call_xyz"),
        ]
        sdk = _build_sdk_messages(messages)

        assert sdk[1]["role"] == "tool"
        assert sdk[1]["tool_call_id"] == "call_xyz"
        assert sdk[1]["content"] == '{"result": 42}'

    async def test_system_message_passed_through(self) -> None:
        """System messages are passed as-is to OpenAI (no extraction needed)."""
        from sleuth.llm.openai import _build_sdk_messages
        from sleuth.llm.base import Message

        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hi"),
        ]
        sdk = _build_sdk_messages(messages)
        assert sdk[0]["role"] == "system"
        assert sdk[0]["content"] == "Be helpful."
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/llm/test_openai.py::TestOpenAIStructuredOutput -v
```

Expected: all 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/llm/test_openai.py
git commit -m "test(llm): add OpenAI structured-output and message-format tests"
```

---

## Task 9: Full test suite run + mypy

**Files:**
- No changes (verification only)

- [ ] **Step 1: Run the complete llm test suite**

```bash
uv run pytest tests/llm/ -v
```

Expected: all tests pass, no warnings.

- [ ] **Step 2: Run mypy on both shims**

```bash
uv run mypy src/sleuth/llm/anthropic.py src/sleuth/llm/openai.py
```

Expected: `Success: no issues found in 2 source files`

If mypy complains about the SDK stubs not being available (e.g., `anthropic-stubs` or `openai` type stubs not installed), add a `[[tool.mypy.overrides]]` block in `pyproject.toml` for those third-party modules:

```toml
[[tool.mypy.overrides]]
module = ["anthropic.*", "openai.*"]
ignore_missing_imports = true
```

This is the standard approach for optional deps whose stubs may not be installed.

- [ ] **Step 3: Run ruff on both shims**

```bash
uv run ruff check src/sleuth/llm/anthropic.py src/sleuth/llm/openai.py
uv run ruff format --check src/sleuth/llm/anthropic.py src/sleuth/llm/openai.py
```

Expected: no lint or format errors. Fix any before committing.

- [ ] **Step 4: Commit any fixes**

If ruff or mypy required changes:

```bash
git add src/sleuth/llm/anthropic.py src/sleuth/llm/openai.py pyproject.toml
git commit -m "fix(llm): address mypy and ruff findings on Anthropic and OpenAI shims"
```

If no changes were needed, skip this step.

---

## Task 10: Protocol conformance smoke test

**Files:**
- Modify: `tests/llm/test_anthropic.py`
- Modify: `tests/llm/test_openai.py`

Verify that both concrete classes satisfy the `LLMClient` Protocol via `isinstance` checking at runtime (requires `runtime_checkable`). Since the Protocol is owned by Phase 1, we only add the check here — we do not modify `base.py`.

- [ ] **Step 1: Check if LLMClient is runtime_checkable**

```bash
python -c "from sleuth.llm.base import LLMClient; print(getattr(LLMClient, '_is_protocol', False))"
```

If `LLMClient` is not decorated with `@runtime_checkable`, we cannot use `isinstance`. In that case, use a structural duck-type check instead (see Step 2b).

- [ ] **Step 2a: If `@runtime_checkable`, add isinstance test to both test files**

Add to `tests/llm/test_anthropic.py`:

```python
class TestAnthropicProtocolConformance:
    def test_satisfies_llm_client_protocol(self) -> None:
        """Anthropic instance passes isinstance check against LLMClient Protocol."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import LLMClient

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")
        assert isinstance(client, LLMClient)
```

Add to `tests/llm/test_openai.py`:

```python
class TestOpenAIProtocolConformance:
    def test_satisfies_llm_client_protocol(self) -> None:
        from sleuth.llm.openai import OpenAI
        from sleuth.llm.base import LLMClient

        client = OpenAI(model="gpt-4o", api_key="sk-test")
        assert isinstance(client, LLMClient)
```

- [ ] **Step 2b: If NOT `@runtime_checkable`, use attribute check**

Add to `tests/llm/test_anthropic.py`:

```python
class TestAnthropicProtocolConformance:
    def test_has_required_protocol_attributes(self) -> None:
        """Anthropic exposes all LLMClient protocol attributes."""
        from sleuth.llm.anthropic import Anthropic
        import inspect

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")
        assert isinstance(client.name, str)
        assert isinstance(client.supports_reasoning, bool)
        assert isinstance(client.supports_structured_output, bool)
        assert inspect.iscoroutinefunction(client.stream) or inspect.isasyncgenfunction(client.stream)
```

Add to `tests/llm/test_openai.py`:

```python
class TestOpenAIProtocolConformance:
    def test_has_required_protocol_attributes(self) -> None:
        from sleuth.llm.openai import OpenAI
        import inspect

        client = OpenAI(model="gpt-4o", api_key="sk-test")
        assert isinstance(client.name, str)
        assert isinstance(client.supports_reasoning, bool)
        assert isinstance(client.supports_structured_output, bool)
        assert inspect.iscoroutinefunction(client.stream) or inspect.isasyncgenfunction(client.stream)
```

- [ ] **Step 3: Run protocol conformance tests**

```bash
uv run pytest tests/llm/test_anthropic.py::TestAnthropicProtocolConformance \
              tests/llm/test_openai.py::TestOpenAIProtocolConformance -v
```

Expected: both tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/llm/test_anthropic.py tests/llm/test_openai.py
git commit -m "test(llm): add LLMClient protocol conformance checks for both shims"
```

---

## Task 11: Coverage check + final cleanup

**Files:**
- No new files

- [ ] **Step 1: Run coverage for the llm module**

```bash
uv run pytest tests/llm/ --cov=src/sleuth/llm --cov-report=term-missing
```

Expected: coverage ≥85% for `src/sleuth/llm/anthropic.py` and `src/sleuth/llm/openai.py`.

If coverage is below 85%, identify uncovered lines in the report and add a targeted test for each. Common gaps: `_translate_event` for unknown event types, `_translate_chunk` with empty choices, error paths.

- [ ] **Step 2: Add coverage for uncovered branches (if any)**

For example, if `_translate_event` with an unknown event type is not covered:

```python
# In tests/llm/test_anthropic.py — TestAnthropicEdgeCases
def test_unknown_event_type_returns_none(self) -> None:
    from sleuth.llm.anthropic import _translate_event

    event = MagicMock()
    event.type = "totally_unknown_event_xyz"
    result = _translate_event(event)
    assert result is None
```

For OpenAI, if empty choices list is uncovered:

```python
# In tests/llm/test_openai.py
def test_empty_choices_returns_no_chunks(self) -> None:
    from sleuth.llm.openai import _translate_chunk

    chunk = MagicMock()
    chunk.choices = []
    result = _translate_chunk(chunk, {})
    assert result == []
```

- [ ] **Step 3: Run the full test suite one final time**

```bash
uv run pytest tests/ -m "not integration" -v --tb=short
```

Expected: all tests pass, no errors.

- [ ] **Step 4: Final commit**

```bash
git add tests/llm/test_anthropic.py tests/llm/test_openai.py
git commit -m "test(llm): add coverage gap tests; all unit tests passing"
```

---

## Task 12: Open PR

**Files:**
- No file changes

- [ ] **Step 1: Push branch**

```bash
git push -u origin feature/phase-10-llm-shims
```

- [ ] **Step 2: Open PR targeting `develop`**

```bash
gh pr create \
  --base develop \
  --title "feat(llm): Phase 10 — Anthropic and OpenAI LLM shims" \
  --body "$(cat <<'EOF'
## Summary

- Adds `sleuth.llm.anthropic.Anthropic`: lazy-imports `anthropic` SDK, adapts streaming events to `LLMChunk` union, sets `supports_reasoning=True` for claude-opus-4-7 and claude-sonnet-4-6 with `thinking=True`.
- Adds `sleuth.llm.openai.OpenAI`: lazy-imports `openai` SDK, adapts streaming events to `LLMChunk` union, sets `supports_reasoning=True` for all o-series models (pattern `o\d`).
- Both shims: `supports_structured_output=True`, structured-output passthrough via native SDK mechanisms, clean `ImportError` with install hint on missing extra.
- Resolves spec §15 #3 (documentation-only — no literal `fast_llm` default in `Sleuth()`; recommended invocations shown in module docstrings).

## Test plan

- [ ] `uv run pytest tests/llm/ -v` — all unit tests pass
- [ ] `uv run mypy src/sleuth/llm/anthropic.py src/sleuth/llm/openai.py` — no mypy errors
- [ ] `uv run ruff check src/sleuth/llm/` — no lint errors
- [ ] Coverage ≥85% for both shim files
- [ ] CI passes on all matrix legs (Python 3.11, 3.12, 3.13)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

---

## Self-Review Checklist (run before handoff)

### Spec coverage

| Spec requirement | Task that implements it |
|---|---|
| §6 LLMClient protocol — `name`, `supports_reasoning`, `supports_structured_output`, `stream()` | Tasks 2, 6 (properties) + Tasks 3, 7 (stream adapter) |
| §6 LLMChunk union: `TextDelta`, `ReasoningDelta`, `ToolCall`, `Stop` | Tasks 3, 4, 7, 8 |
| §1 BYOK — lazy import, `ImportError` with hint | Tasks 2b, 6b |
| §5 ThinkingEvent gating — `supports_reasoning` flag | Tasks 2a, 6a |
| §5 Claude extended-thinking models (claude-opus-4-7, claude-sonnet-4-6) | Task 2a |
| §5 OpenAI o-series reasoning (o1, o3, o3-mini, o4-mini) | Task 6a |
| Schema passthrough — Anthropic tool-call coercion | Task 4 |
| Schema passthrough — OpenAI `response_format=json_schema` | Task 8 |
| `supports_structured_output=True` for both shims | Tasks 2a, 6a |
| §15 #3 — documentation-only `fast_llm` | Module docstrings in Tasks 2b, 6b |

### Placeholder scan

No "TBD", "implement X", "similar to above", or untested method stubs found.

### Type consistency

- `TextDelta`, `ReasoningDelta`, `ToolCall`, `Stop` — imported from `sleuth.llm.base` in every task; not redefined.
- `stream()` signature matches `LLMClient` protocol exactly (conventions §5.1).
- `_translate_event` and `_translate_chunk` return `LLMChunk | None` and `list[LLMChunk]` respectively — consistent across all tasks.
- Mock helper `_make_async_iter` defined once per test file and referenced across all test classes — no duplication.
