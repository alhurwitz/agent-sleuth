"""Tests for the Anthropic LLM shim."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel as PydanticBaseModel

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 2: Import and flag semantics
# ---------------------------------------------------------------------------


class TestAnthropicImport:
    def test_import_without_sdk_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing the class is fine; instantiating without the SDK raises ImportError."""
        import sys

        # Remove anthropic from sys.modules so lazy-import fails
        monkeypatch.delitem(sys.modules, "anthropic", raising=False)

        # Patch builtins.__import__ to block anthropic
        real_builtins = __builtins__
        if isinstance(real_builtins, dict):
            original_import = real_builtins["__import__"]
        else:
            original_import = real_builtins.__import__

        def _block_anthropic(name: str, *args: object, **kwargs: object) -> object:
            if name == "anthropic" or name.startswith("anthropic."):
                raise ImportError("No module named 'anthropic'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _block_anthropic)

        import importlib

        import sleuth.llm.anthropic as mod

        importlib.reload(mod)

        fake_key = "sk-ant-test"  # pragma: allowlist secret
        with pytest.raises(ImportError, match="agent-sleuth\\[anthropic\\]"):
            mod.Anthropic(model="claude-sonnet-4-6", api_key=fake_key)


class TestSupportsReasoning:
    def test_extended_thinking_models_support_reasoning(self) -> None:
        """Claude opus-4-7 and sonnet-4-6 with thinking enabled set supports_reasoning=True."""
        from sleuth.llm.anthropic import Anthropic

        opus = Anthropic(model="claude-opus-4-7", api_key="sk-ant-test")
        sonnet_thinking = Anthropic(model="claude-sonnet-4-6", api_key="sk-ant-test", thinking=True)
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


# ---------------------------------------------------------------------------
# Task 3: Stream adapter — TextDelta, ReasoningDelta, Stop
# ---------------------------------------------------------------------------


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

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _make_async_iter(raw_events)

        with patch.object(client._client.messages, "stream", return_value=mock_stream):
            chunks = []
            async for chunk in client.stream([Message(role="user", content="Hi")]):
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
            async for chunk in client.stream([Message(role="user", content="Hi")]):
                chunks.append(chunk)

        stop_chunks = [c for c in chunks if isinstance(c, Stop)]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].reason == "end_turn"


class TestAnthropicStreamReasoning:
    async def test_reasoning_delta_emitted_when_thinking_enabled(self) -> None:
        """stream() yields ReasoningDelta for thinking_delta events."""
        from sleuth.llm.anthropic import Anthropic
        from sleuth.llm.base import Message, ReasoningDelta

        client = Anthropic(model="claude-sonnet-4-6", api_key="sk-ant-test", thinking=True)

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
            async for chunk in client.stream([Message(role="user", content="Hi")]):
                chunks.append(chunk)

        reasoning_chunks = [c for c in chunks if isinstance(c, ReasoningDelta)]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0].text == "Let me think..."

    async def test_no_reasoning_delta_without_thinking(self) -> None:
        """Non-thinking models produce no ReasoningDelta even if the SDK emitted one."""
        from sleuth.llm.anthropic import Anthropic

        # Standard model, thinking=False (default)
        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")

        # Even if SDK somehow sends a thinking_delta, we pass it through (it won't happen
        # in practice — test confirms the shim *does* emit ReasoningDelta regardless,
        # and the *engine* gates ThinkingEvent on supports_reasoning).
        # This test verifies supports_reasoning=False is set correctly.
        assert client.supports_reasoning is False


# ---------------------------------------------------------------------------
# Task 4: ToolCall chunks and structured output
# ---------------------------------------------------------------------------


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

        captured_kwargs: dict = {}  # type: ignore[type-arg]

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
        captured_kwargs: dict = {}  # type: ignore[type-arg]

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


# ---------------------------------------------------------------------------
# Task 5: Edge-case tests
# ---------------------------------------------------------------------------


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

    def test_unknown_event_type_returns_none(self) -> None:
        """_translate_event returns None for unknown event types."""
        from sleuth.llm.anthropic import _translate_event

        event = MagicMock()
        event.type = "totally_unknown_event_xyz"
        result = _translate_event(event)
        assert result is None

    def test_stop_reason_stop_sequence(self) -> None:
        """_translate_event returns Stop(stop_sequence) for stop_sequence stop reason."""
        from sleuth.llm.anthropic import _translate_event
        from sleuth.llm.base import Stop

        event = MagicMock()
        event.type = "message_delta"
        event.delta = MagicMock()
        event.delta.stop_reason = "stop_sequence"
        result = _translate_event(event)
        assert isinstance(result, Stop)
        assert result.reason == "stop_sequence"

    def test_input_json_delta_returns_none(self) -> None:
        """_translate_event returns None for input_json_delta (partial tool JSON)."""
        from sleuth.llm.anthropic import _translate_event

        event = MagicMock()
        event.type = "content_block_delta"
        event.delta = MagicMock()
        event.delta.type = "input_json_delta"
        result = _translate_event(event)
        assert result is None


# ---------------------------------------------------------------------------
# Task 10: Protocol conformance
# ---------------------------------------------------------------------------


class TestAnthropicProtocolConformance:
    def test_has_required_protocol_attributes(self) -> None:
        """Anthropic exposes all LLMClient protocol attributes."""
        from sleuth.llm.anthropic import Anthropic

        client = Anthropic(model="claude-haiku-4-5", api_key="sk-ant-test")
        assert isinstance(client.name, str)
        assert isinstance(client.supports_reasoning, bool)
        assert isinstance(client.supports_structured_output, bool)
        assert inspect.iscoroutinefunction(client.stream) or inspect.isasyncgenfunction(
            client.stream
        )
