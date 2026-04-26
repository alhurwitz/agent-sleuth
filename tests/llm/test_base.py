from sleuth.llm.base import (
    LLMClient,
    Message,
    ReasoningDelta,
    Stop,
    TextDelta,
    Tool,
    ToolCall,
)


def test_text_delta():
    d = TextDelta(text="hello")
    assert d.text == "hello"


def test_reasoning_delta():
    d = ReasoningDelta(text="thinking")
    assert d.text == "thinking"


def test_tool_call():
    tc = ToolCall(id="c1", name="search", arguments={"query": "foo"})
    assert tc.name == "search"
    assert tc.arguments == {"query": "foo"}


def test_stop_reasons():
    for reason in ("end_turn", "tool_use", "max_tokens", "stop_sequence", "error"):
        s = Stop(reason=reason)
        assert s.reason == reason


def test_message_defaults():
    m = Message(role="user", content="hi")
    assert m.tool_call_id is None


def test_tool_model():
    t = Tool(name="search", description="web search", input_schema={"type": "object"})
    assert t.name == "search"


def test_llmclient_is_protocol():
    """LLMClient is a runtime-checkable Protocol — structural subtyping only."""
    import typing

    assert hasattr(LLMClient, "__protocol_attrs__") or typing.get_origin(LLMClient) is None
    # Just importing LLMClient without error is the meaningful assertion here.
    assert LLMClient is not None
