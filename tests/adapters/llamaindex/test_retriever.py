"""Tests for SleuthRetriever — Sleuth as a LlamaIndex BaseRetriever."""

import pytest

pytest.importorskip("llama_index", reason="llamaindex extra not installed")

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from sleuth import Sleuth
from sleuth.llamaindex import SleuthRetriever


@pytest.mark.adapter
def test_retriever_is_base_retriever(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    assert isinstance(retriever, BaseRetriever)


@pytest.mark.adapter
async def test_retriever_aretrieve_returns_nodes(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    nodes = await retriever.aretrieve(QueryBundle(query_str="What is Sleuth?"))
    assert isinstance(nodes, list)
    for node in nodes:
        assert isinstance(node, NodeWithScore)
