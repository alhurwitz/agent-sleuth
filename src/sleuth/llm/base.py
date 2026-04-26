"""LLM protocol and supporting types.

The package never imports a model SDK as a hard dependency.  Users pass any
object that satisfies the ``LLMClient`` Protocol.  Optional shims in
``sleuth.llm.{anthropic,openai}`` adapt those SDKs to this shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# LLMChunk — discriminated union for streaming chunks
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    """A text token from the model's response."""

    text: str


@dataclass
class ReasoningDelta:
    """A reasoning/thinking token (Claude extended thinking, OpenAI o-series)."""

    text: str


@dataclass
class ToolCall:
    """A tool-use request emitted by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Stop:
    """Stream terminator — carries the stop reason."""

    reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "error"]


LLMChunk = TextDelta | ReasoningDelta | ToolCall | Stop


# ---------------------------------------------------------------------------
# Message + Tool — Pydantic models (public, serializable)
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


class Tool(BaseModel):
    """A tool definition passed to the LLM for function-calling."""

    name: str
    description: str
    input_schema: dict[str, Any]


# ---------------------------------------------------------------------------
# LLMClient Protocol
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Structural protocol for all LLM clients used by Sleuth.

    Attributes:
        name: Human-readable identifier, e.g. ``"anthropic:claude-sonnet-4-6"``.
        supports_reasoning: When True the engine emits ThinkingEvents.
        supports_structured_output: When True schema= is passed through natively;
            otherwise the engine falls back to JSON-parse of the text response.
    """

    name: str
    supports_reasoning: bool
    supports_structured_output: bool

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]:  # pragma: no cover
        """Stream LLM output as ``LLMChunk`` items.

        Must yield at least one chunk and end with a ``Stop`` chunk.
        """
        raise NotImplementedError
        yield  # make this a generator to satisfy the return type
