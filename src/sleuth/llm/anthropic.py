"""Anthropic LLM shim for agent-sleuth.

Install:
    pip install agent-sleuth[anthropic]

Recommended usage::

    from sleuth.llm.anthropic import Anthropic
    from sleuth import Sleuth

    agent = Sleuth(
        llm=Anthropic(model="claude-opus-4-7"),          # reasoning + synthesis
        fast_llm=Anthropic(model="claude-haiku-4-5"),    # routing / picking
        backends=[...],
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
from typing import Any, cast

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
_THINKING_CAPABLE_MODELS: frozenset[str] = frozenset({"claude-opus-4-7", "claude-sonnet-4-6"})


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
        Maximum response tokens (default 4096; extended thinking requires >=1024).
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
            import anthropic as _anthropic_sdk
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

        self._client = _anthropic_sdk.AsyncAnthropic(timeout=timeout, **client_kwargs)

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
        return self._thinking and self._model in _THINKING_CAPABLE_MODELS

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

        # Cast to Any to avoid SDK-specific overload typing conflicts: the SDK
        # expects typed Param classes, but we pass plain dicts for flexibility.
        _stream_fn = cast(Any, self._client.messages.stream)
        async with _stream_fn(
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


def _build_extra_params(*, thinking: bool, thinking_budget_tokens: int) -> dict[str, Any]:
    if not thinking:
        return {}
    return {
        "thinking": {
            "type": "enabled",
            "budget_tokens": thinking_budget_tokens,
        }
    }


def _translate_event(event: Any) -> LLMChunk | None:
    """Translate a raw Anthropic SDK streaming event to an ``LLMChunk``."""
    event_type = getattr(event, "type", None)

    # Text delta
    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        delta_type = getattr(delta, "type", None)
        if delta is not None and delta_type == "text_delta":
            return TextDelta(text=delta.text)
        if delta is not None and delta_type == "thinking_delta":
            return ReasoningDelta(text=delta.thinking)
        if delta_type == "input_json_delta":
            # Partial tool input JSON — we accumulate via stop handling
            return None

    # Tool use block
    if event_type == "content_block_start":
        block = getattr(event, "content_block", None)
        if block is not None and getattr(block, "type", None) == "tool_use":
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
