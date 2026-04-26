"""Tests for the OpenAI LLM shim."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel as PydanticBaseModel

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_openai_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
    tool_calls: list[dict] | None = None,  # type: ignore[type-arg]
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


# ---------------------------------------------------------------------------
# Task 6: Import and flag semantics
# ---------------------------------------------------------------------------


class TestOpenAIImport:
    def test_import_without_sdk_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Instantiating OpenAI without the openai SDK raises ImportError with hint."""
        import sys

        monkeypatch.delitem(sys.modules, "openai", raising=False)

        real_builtins = __builtins__
        if isinstance(real_builtins, dict):
            original_import = real_builtins["__import__"]
        else:
            original_import = real_builtins.__import__

        def _block_openai(name: str, *args: object, **kwargs: object) -> object:
            if name == "openai" or name.startswith("openai."):
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _block_openai)

        import importlib

        import sleuth.llm.openai as mod

        importlib.reload(mod)

        fake_key = "sk-test"  # pragma: allowlist secret
        with pytest.raises(ImportError, match="agent-sleuth\\[openai\\]"):
            mod.OpenAI(model="gpt-4o", api_key=fake_key)


class TestOpenAISupportsReasoning:
    def test_o_series_models_support_reasoning(self) -> None:
        """o1, o3, o3-mini, o4-mini set supports_reasoning=True."""
        from sleuth.llm.openai import OpenAI

        for model in ("o1", "o1-mini", "o3", "o3-mini", "o4-mini"):
            client = OpenAI(model=model, api_key="sk-test")  # pragma: allowlist secret
            assert client.supports_reasoning is True, f"{model} should support reasoning"

    def test_gpt_models_do_not_support_reasoning(self) -> None:
        """gpt-4o, gpt-4-turbo set supports_reasoning=False."""
        from sleuth.llm.openai import OpenAI

        for model in ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"):
            client = OpenAI(model=model, api_key="sk-test")  # pragma: allowlist secret
            assert client.supports_reasoning is False, f"{model} should not support reasoning"

    def test_supports_structured_output_always_true(self) -> None:
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")  # pragma: allowlist secret
        assert client.supports_structured_output is True

    def test_name_includes_provider_and_model(self) -> None:
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")  # pragma: allowlist secret
        assert client.name == "openai:gpt-4o"


# ---------------------------------------------------------------------------
# Task 7: Stream adapter — TextDelta, ReasoningDelta, ToolCall, Stop
# ---------------------------------------------------------------------------


class TestOpenAIStreamTextDelta:
    async def test_text_delta_emitted(self) -> None:
        from sleuth.llm.base import Message, TextDelta
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")  # pragma: allowlist secret

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
        from sleuth.llm.base import Message, Stop
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")  # pragma: allowlist secret
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
        from sleuth.llm.base import Message, Stop
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")  # pragma: allowlist secret
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
        from sleuth.llm.base import Message, ReasoningDelta
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="o3", api_key="sk-test")  # pragma: allowlist secret

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
        from sleuth.llm.base import Message, Tool, ToolCall
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")  # pragma: allowlist secret

        raw_chunks = [
            _make_openai_chunk(
                tool_calls=[{"id": "call_abc", "name": "search", "arguments": '{"q":'}]
            ),
            _make_openai_chunk(tool_calls=[{"id": "", "name": "", "arguments": '"test"}'}]),
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
                    tools=[
                        Tool(name="search", description="search", input_schema={"type": "object"})
                    ],
                )
            ]

        tool_chunks = [c for c in chunks if isinstance(c, ToolCall)]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].name == "search"
        assert tool_chunks[0].arguments == {"q": "test"}


# ---------------------------------------------------------------------------
# Task 8: Structured output and message format
# ---------------------------------------------------------------------------


class _Verdict(PydanticBaseModel):
    answer: str
    confidence: float


class TestOpenAIStructuredOutput:
    async def test_schema_injects_response_format(self) -> None:
        """When schema= is passed, response_format=json_schema is added to the request."""
        from sleuth.llm.base import Message
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")  # pragma: allowlist secret
        captured_kwargs: dict = {}  # type: ignore[type-arg]

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
        from sleuth.llm.base import Message
        from sleuth.llm.openai import _build_sdk_messages

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
        from sleuth.llm.base import Message
        from sleuth.llm.openai import _build_sdk_messages

        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hi"),
        ]
        sdk = _build_sdk_messages(messages)
        assert sdk[0]["role"] == "system"
        assert sdk[0]["content"] == "Be helpful."


# ---------------------------------------------------------------------------
# Task 10: Protocol conformance
# ---------------------------------------------------------------------------


class TestOpenAIProtocolConformance:
    def test_has_required_protocol_attributes(self) -> None:
        from sleuth.llm.openai import OpenAI

        client = OpenAI(model="gpt-4o", api_key="sk-test")  # pragma: allowlist secret
        assert isinstance(client.name, str)
        assert isinstance(client.supports_reasoning, bool)
        assert isinstance(client.supports_structured_output, bool)
        assert inspect.iscoroutinefunction(client.stream) or inspect.isasyncgenfunction(
            client.stream
        )


# ---------------------------------------------------------------------------
# Task 11: Coverage gap tests
# ---------------------------------------------------------------------------


class TestOpenAICoverageGaps:
    def test_empty_choices_returns_no_chunks(self) -> None:
        """_translate_chunk returns [] for events with no choices."""
        from sleuth.llm.openai import _translate_chunk

        chunk = MagicMock()
        chunk.choices = []
        result = _translate_chunk(chunk, {})
        assert result == []

    def test_invalid_json_in_tool_arguments(self) -> None:
        """_translate_chunk handles JSON decode errors gracefully."""
        from sleuth.llm.base import ToolCall
        from sleuth.llm.openai import _translate_chunk

        open_tool_calls = {0: {"id": "call_1", "name": "mytool", "arguments": "not-json"}}

        chunk = MagicMock()
        choice = MagicMock()
        choice.delta = None
        choice.finish_reason = "stop"
        chunk.choices = [choice]

        results = _translate_chunk(chunk, open_tool_calls)
        tool_calls = [r for r in results if isinstance(r, ToolCall)]
        assert len(tool_calls) == 1
        assert tool_calls[0].arguments == {"raw": "not-json"}

    def test_unknown_finish_reason_maps_to_end_turn(self) -> None:
        """Unknown finish reasons map to 'end_turn'."""
        from sleuth.llm.base import Stop
        from sleuth.llm.openai import _translate_chunk

        chunk = MagicMock()
        choice = MagicMock()
        choice.delta = None
        choice.finish_reason = "some_unknown_reason"
        chunk.choices = [choice]

        results = _translate_chunk(chunk, {})
        stop_chunks = [r for r in results if isinstance(r, Stop)]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].reason == "end_turn"
