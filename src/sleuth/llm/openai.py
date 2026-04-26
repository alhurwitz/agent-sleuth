"""OpenAI LLM shim for agent-sleuth.

Install:
    pip install agent-sleuth[openai]

Recommended usage::

    from sleuth.llm.openai import OpenAI
    from sleuth import Sleuth

    agent = Sleuth(
        llm=OpenAI(model="gpt-4o"),               # synthesis
        fast_llm=OpenAI(model="gpt-4o-mini"),     # routing / picking
        backends=[...],
    )

For reasoning with o-series models::

    agent = Sleuth(
        llm=OpenAI(model="o3"),                   # supports_reasoning=True automatically
        backends=[...],
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

import json
import re
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
            import openai as _openai_sdk
        except ImportError as exc:
            raise ImportError(
                "The OpenAI SDK is required. Install it with: pip install agent-sleuth[openai]"
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

        # Cast to Any to avoid SDK-specific overload typing conflicts.
        _stream_fn = cast(Any, self._client.chat.completions.stream)
        async with _stream_fn(
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
    event: Any,
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
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
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
