"""Tests for SleuthQueryEngine — Sleuth as a LlamaIndex QueryEngine."""

import pytest

pytest.importorskip("llama_index", reason="llamaindex extra not installed")

from llama_index.core.query_engine import BaseQueryEngine

from sleuth import Sleuth
from sleuth.llamaindex import SleuthQueryEngine


@pytest.mark.adapter
def test_query_engine_is_base_query_engine(sleuth_agent: Sleuth) -> None:
    engine = SleuthQueryEngine(agent=sleuth_agent)
    assert isinstance(engine, BaseQueryEngine)


@pytest.mark.adapter
async def test_query_engine_aquery_returns_response(sleuth_agent: Sleuth) -> None:
    engine = SleuthQueryEngine(agent=sleuth_agent)
    response = await engine.aquery("What is 42?")
    # LlamaIndex Response objects have a .response str attribute
    assert hasattr(response, "response")
    assert isinstance(response.response, str)


@pytest.mark.adapter
def test_query_engine_query_returns_response(sleuth_agent: Sleuth) -> None:
    engine = SleuthQueryEngine(agent=sleuth_agent)
    response = engine.query("What is 42?")
    assert hasattr(response, "response")
    assert isinstance(response.response, str)
