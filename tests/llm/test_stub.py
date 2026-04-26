import pytest

from sleuth.llm.base import LLMChunk, Message, Stop, TextDelta
from sleuth.llm.stub import StubLLM


async def collect(stub: StubLLM, msgs: list[Message]) -> list[LLMChunk]:
    return [c async for c in stub.stream(msgs)]


@pytest.fixture
def msgs() -> list[Message]:
    return [Message(role="user", content="hi")]


async def test_string_response_emits_text_delta_then_stop(msgs):
    stub = StubLLM(["hello"])
    chunks = await collect(stub, msgs)
    assert chunks == [TextDelta("hello"), Stop("end_turn")]


async def test_calls_cycle_through_responses(msgs):
    stub = StubLLM(["first", "second"])
    c1 = await collect(stub, msgs)
    c2 = await collect(stub, msgs)
    c3 = await collect(stub, msgs)  # cycles back to "first"
    assert c1[0] == TextDelta("first")
    assert c2[0] == TextDelta("second")
    assert c3[0] == TextDelta("first")


async def test_llmchunk_response_passes_through(msgs):
    stub = StubLLM([Stop("max_tokens")])
    chunks = await collect(stub, msgs)
    assert chunks == [Stop("max_tokens")]


async def test_list_response_yields_each_chunk_in_order(msgs):
    stub = StubLLM([[TextDelta("a"), TextDelta("b"), Stop("end_turn")]])
    chunks = await collect(stub, msgs)
    assert chunks == [TextDelta("a"), TextDelta("b"), Stop("end_turn")]


async def test_callable_owns_full_response(msgs):
    async def responder(messages):
        yield TextDelta("dynamic")
        yield Stop("end_turn")

    stub = StubLLM(responder)
    chunks = await collect(stub, msgs)
    assert chunks == [TextDelta("dynamic"), Stop("end_turn")]


async def test_attributes():
    stub = StubLLM(["ok"])
    assert stub.name == "stub"
    assert stub.supports_reasoning is False
    assert stub.supports_structured_output is True
