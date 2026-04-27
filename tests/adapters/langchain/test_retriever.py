"""Tests for SleuthRetriever — the LangChain retriever surface."""

import pytest

pytest.importorskip("langchain_core", reason="langchain extra not installed")

from langchain_core.retrievers import BaseRetriever

from sleuth import Sleuth
from sleuth.langchain import SleuthRetriever


@pytest.mark.adapter
def test_sleuth_retriever_is_base_retriever(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    assert isinstance(retriever, BaseRetriever)


@pytest.mark.adapter
async def test_sleuth_retriever_get_relevant_documents(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    # In newer langchain, invoke() replaces deprecated get_relevant_documents()
    docs = await retriever.ainvoke("What is Sleuth?")
    assert isinstance(docs, list)
    # Each doc should have page_content populated
    for doc in docs:
        assert hasattr(doc, "page_content")
        assert isinstance(doc.page_content, str)


@pytest.mark.adapter
async def test_sleuth_retriever_aget_relevant_documents(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    docs = await retriever.ainvoke("What is Sleuth?")
    assert isinstance(docs, list)
    assert len(docs) >= 1
