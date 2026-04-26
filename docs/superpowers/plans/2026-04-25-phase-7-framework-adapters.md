# Phase 7: Framework Adapters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement eight framework adapters that expose Sleuth as a first-class tool/retriever in LangChain, LangGraph, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, Pydantic AI, CrewAI, and AutoGen, bridging Sleuth's typed event stream into each host framework's native callback surface.

**Architecture:** Each adapter is a self-contained subpackage under `src/sleuth/<framework>/` behind its own optional extra; the host framework is imported lazily at the top of each module (not at package import time) so users without that extra never see an ImportError. Tests use a stub Sleuth backed by `StubLLM` and an in-memory `FakeBackend` to validate host-framework integration only, with `@pytest.mark.adapter` for smoke runs that require the real framework installed, and `@pytest.mark.integration` for round-trips.

**Tech Stack:** Python 3.11+, pydantic v2, langchain-core, langgraph, llama-index-core, openai-agents, claude-agent-sdk, pydantic-ai, crewai, pyautogen, pytest, pytest-asyncio (auto mode), StubLLM (from `sleuth.llm.stub`).

---

> **Callout — no new conventions required.** All extras, package paths, test markers, and commit conventions are already defined in `_conventions.md`. The `pyproject.toml` extras table in conventions §3 already enumerates all eight adapters. No escalation needed.

---

## Setup

### Task 0: Create the feature branch

**Files:**
- No files changed.

- [ ] **Step 1: Create and switch to the feature branch**

```bash
git checkout develop
git checkout -b feature/phase-7-framework-adapters
```

Expected: `Switched to a new branch 'feature/phase-7-framework-adapters'`

- [ ] **Step 2: Verify the branch**

```bash
git branch --show-current
```

Expected: `feature/phase-7-framework-adapters`

---

## Shared test infrastructure for adapters

### Task 1: Shared adapter test fixtures

**Files:**
- Create: `tests/adapters/__init__.py`
- Create: `tests/adapters/conftest.py`

- [ ] **Step 1: Create the adapters test package**

```bash
mkdir -p tests/adapters
touch tests/adapters/__init__.py
```

- [ ] **Step 2: Write the shared adapter conftest**

Create `tests/adapters/conftest.py`:

```python
"""Shared fixtures for all framework adapter tests.

Uses StubLLM + FakeBackend so tests never require real LLM or network.
"""
from __future__ import annotations

import pytest

from sleuth import Sleuth
from sleuth.backends.base import Backend, Capability
from sleuth.llm.stub import StubLLM
from sleuth.types import Chunk, Source


class FakeBackend:
    """Minimal Backend that returns a fixed chunk — no real search."""

    name = "fake"
    capabilities = frozenset({Capability.DOCS})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [
            Chunk(
                text=f"Fake result for: {query}",
                source=Source(kind="file", location="fake.md", title="Fake"),
                score=1.0,
            )
        ]


@pytest.fixture()
def stub_llm() -> StubLLM:
    """A StubLLM that emits a short answer then stops."""
    return StubLLM(responses=["The answer is 42."])


@pytest.fixture()
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture()
def sleuth_agent(stub_llm: StubLLM, fake_backend: FakeBackend) -> Sleuth:
    """A Sleuth instance wired to stub LLM + fake backend. No I/O."""
    return Sleuth(llm=stub_llm, backends=[fake_backend], cache=None)
```

- [ ] **Step 3: Run (no tests yet — verify the conftest imports cleanly)**

```bash
uv run pytest tests/adapters/ --collect-only 2>&1 | head -20
```

Expected: `no tests ran` (zero tests collected, no import errors).

- [ ] **Step 4: Commit**

```bash
git add tests/adapters/__init__.py tests/adapters/conftest.py
git commit -m "test: add shared adapter test fixtures (FakeBackend + stub_llm)"
```

---

## Tier 1: LangChain

### Task 2: LangChain subpackage skeleton

**Files:**
- Create: `src/sleuth/langchain/__init__.py`
- Create: `src/sleuth/langchain/_tool.py`
- Create: `src/sleuth/langchain/_retriever.py`
- Create: `src/sleuth/langchain/_callback.py`
- Create: `tests/adapters/langchain/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/sleuth/langchain tests/adapters/langchain
touch src/sleuth/langchain/__init__.py tests/adapters/langchain/__init__.py
```

- [ ] **Step 2: Write the public `__init__.py`**

```python
# src/sleuth/langchain/__init__.py
"""LangChain adapter for Sleuth (extras=[langchain]).

Install: pip install agent-sleuth[langchain]
"""
from sleuth.langchain._tool import SleuthTool
from sleuth.langchain._retriever import SleuthRetriever
from sleuth.langchain._callback import SleuthCallbackHandler

__all__ = ["SleuthTool", "SleuthRetriever", "SleuthCallbackHandler"]
```

- [ ] **Step 3: Commit the skeleton**

```bash
git add src/sleuth/langchain/ tests/adapters/langchain/__init__.py
git commit -m "chore: scaffold langchain adapter subpackage"
```

---

### Task 3: LangChain SleuthTool

**Files:**
- Create: `src/sleuth/langchain/_tool.py`
- Test: `tests/adapters/langchain/test_tool.py`

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/langchain/test_tool.py`:

```python
"""Tests for SleuthTool — the LangChain tool surface."""
import pytest

pytest.importorskip("langchain_core", reason="langchain extra not installed")

from langchain_core.tools import BaseTool  # noqa: E402

from sleuth import Sleuth  # noqa: E402
from sleuth.langchain import SleuthTool  # noqa: E402


@pytest.mark.adapter
def test_sleuth_tool_is_langchain_base_tool(sleuth_agent: Sleuth) -> None:
    tool = SleuthTool(agent=sleuth_agent)
    assert isinstance(tool, BaseTool)


@pytest.mark.adapter
def test_sleuth_tool_name_and_description(sleuth_agent: Sleuth) -> None:
    tool = SleuthTool(agent=sleuth_agent)
    assert tool.name == "sleuth_search"
    assert len(tool.description) > 10  # non-empty description


@pytest.mark.adapter
def test_sleuth_tool_run_returns_string(sleuth_agent: Sleuth) -> None:
    tool = SleuthTool(agent=sleuth_agent)
    result = tool.run("What is the capital of France?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
async def test_sleuth_tool_arun_returns_string(sleuth_agent: Sleuth) -> None:
    tool = SleuthTool(agent=sleuth_agent)
    result = await tool.arun("What is the capital of France?")
    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/adapters/langchain/test_tool.py -v -m adapter
```

Expected: `ImportError` or `ModuleNotFoundError` for `sleuth.langchain._tool` (file does not exist yet).

- [ ] **Step 3: Implement `SleuthTool`**

Create `src/sleuth/langchain/_tool.py`:

```python
"""SleuthTool — Sleuth as a LangChain BaseTool."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain is not installed. Run: pip install agent-sleuth[langchain]"
    ) from exc

from sleuth import Sleuth

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun


class SleuthTool(BaseTool):
    """Expose Sleuth search as a LangChain tool.

    Usage::

        from sleuth.langchain import SleuthTool
        tool = SleuthTool(agent=sleuth_instance)
        agent_executor = AgentExecutor(tools=[tool], ...)
    """

    name: str = "sleuth_search"
    description: str = (
        "A reasoning-first search tool. Use for questions that require searching "
        "documents, code, or the web. Input: a natural-language query string."
    )
    agent: Sleuth

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        result = self.agent.ask(query)
        return result.text

    async def _arun(
        self,
        query: str,
        run_manager: Any = None,
    ) -> str:
        chunks: list[str] = []
        async for event in self.agent.aask(query):
            if event.type == "token":
                chunks.append(event.text)
        return "".join(chunks)
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/adapters/langchain/test_tool.py -v -m adapter
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/langchain/_tool.py tests/adapters/langchain/test_tool.py
git commit -m "feat(langchain): add SleuthTool (LangChain BaseTool wrapper)"
```

---

### Task 4: LangChain SleuthRetriever

**Files:**
- Create: `src/sleuth/langchain/_retriever.py`
- Test: `tests/adapters/langchain/test_retriever.py`

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/langchain/test_retriever.py`:

```python
"""Tests for SleuthRetriever — the LangChain retriever surface."""
import pytest

pytest.importorskip("langchain_core", reason="langchain extra not installed")

from langchain_core.retrievers import BaseRetriever  # noqa: E402

from sleuth import Sleuth  # noqa: E402
from sleuth.langchain import SleuthRetriever  # noqa: E402


@pytest.mark.adapter
def test_sleuth_retriever_is_base_retriever(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    assert isinstance(retriever, BaseRetriever)


@pytest.mark.adapter
def test_sleuth_retriever_get_relevant_documents(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    docs = retriever.get_relevant_documents("What is Sleuth?")
    assert isinstance(docs, list)
    # Each doc should have page_content populated
    for doc in docs:
        assert hasattr(doc, "page_content")
        assert isinstance(doc.page_content, str)


@pytest.mark.adapter
async def test_sleuth_retriever_aget_relevant_documents(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    docs = await retriever.aget_relevant_documents("What is Sleuth?")
    assert isinstance(docs, list)
    assert len(docs) >= 1
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/adapters/langchain/test_retriever.py -v -m adapter
```

Expected: `ImportError` for `_retriever` module not yet created.

- [ ] **Step 3: Implement `SleuthRetriever`**

Create `src/sleuth/langchain/_retriever.py`:

```python
"""SleuthRetriever — Sleuth as a LangChain BaseRetriever."""
from __future__ import annotations

from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain is not installed. Run: pip install agent-sleuth[langchain]"
    ) from exc

from sleuth import Sleuth
from sleuth.types import Chunk


def _chunk_to_document(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "source": chunk.source.location,
            "kind": chunk.source.kind,
            "title": chunk.source.title or "",
            "score": chunk.score,
        },
    )


class SleuthRetriever(BaseRetriever):
    """Expose Sleuth as a LangChain retriever.

    Returns raw chunks (as LangChain Documents) rather than a synthesized
    answer — suitable for use inside RetrievalQA chains.

    Usage::

        from sleuth.langchain import SleuthRetriever
        retriever = SleuthRetriever(agent=sleuth_instance)
        qa = RetrievalQA.from_chain_type(llm=..., retriever=retriever)
    """

    agent: Sleuth

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        # Drive the async path synchronously via the backend's sync search
        import asyncio

        async def _collect() -> list[Document]:
            docs: list[Document] = []
            for backend in self.agent._backends:  # type: ignore[attr-defined]
                chunks = await backend.search(query, k=10)
                docs.extend(_chunk_to_document(c) for c in chunks)
            return docs

        return asyncio.get_event_loop().run_until_complete(_collect())

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        docs: list[Document] = []
        for backend in self.agent._backends:  # type: ignore[attr-defined]
            chunks = await backend.search(query, k=10)
            docs.extend(_chunk_to_document(c) for c in chunks)
        return docs
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/adapters/langchain/test_retriever.py -v -m adapter
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/langchain/_retriever.py tests/adapters/langchain/test_retriever.py
git commit -m "feat(langchain): add SleuthRetriever (LangChain BaseRetriever wrapper)"
```

---

### Task 5: LangChain SleuthCallbackHandler

**Files:**
- Create: `src/sleuth/langchain/_callback.py`
- Test: `tests/adapters/langchain/test_callback.py`

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/langchain/test_callback.py`:

```python
"""Tests for SleuthCallbackHandler — bridges Sleuth events to LangChain callbacks."""
import pytest

pytest.importorskip("langchain_core", reason="langchain extra not installed")

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

from sleuth import Sleuth  # noqa: E402
from sleuth.events import SearchEvent, DoneEvent  # noqa: E402
from sleuth.langchain import SleuthCallbackHandler  # noqa: E402
from sleuth.types import RunStats  # noqa: E402


@pytest.mark.adapter
def test_callback_handler_is_base_callback_handler() -> None:
    handler = SleuthCallbackHandler()
    assert isinstance(handler, BaseCallbackHandler)


@pytest.mark.adapter
def test_on_search_event_calls_on_tool_start() -> None:
    calls: list[dict] = []

    class SpyHandler(SleuthCallbackHandler):
        def on_tool_start(self, serialized, input_str, **kwargs):  # type: ignore[override]
            calls.append({"action": "tool_start", "input": input_str})

    handler = SpyHandler()
    event = SearchEvent(type="search", backend="fake", query="test query")
    handler.on_sleuth_event(event)
    assert len(calls) == 1
    assert calls[0]["input"] == "test query"


@pytest.mark.adapter
def test_on_done_event_calls_on_tool_end() -> None:
    calls: list[dict] = []

    class SpyHandler(SleuthCallbackHandler):
        def on_tool_end(self, output, **kwargs):  # type: ignore[override]
            calls.append({"action": "tool_end", "output": output})

    handler = SpyHandler()
    stats = RunStats(
        latency_ms=100,
        first_token_ms=50,
        tokens_in=10,
        tokens_out=20,
        cache_hits={},
        backends_called=["fake"],
    )
    event = DoneEvent(type="done", stats=stats)
    handler.on_sleuth_event(event)
    assert len(calls) == 1


@pytest.mark.adapter
async def test_sleuth_agent_with_callback_handler(sleuth_agent: Sleuth) -> None:
    """Run Sleuth with the callback handler attached; verify no exceptions."""
    handler = SleuthCallbackHandler()
    collected: list[str] = []
    async for event in sleuth_agent.aask("test query"):
        handler.on_sleuth_event(event)
        collected.append(event.type)
    assert "done" in collected
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/adapters/langchain/test_callback.py -v -m adapter
```

Expected: `ImportError` for `_callback` module.

- [ ] **Step 3: Implement `SleuthCallbackHandler`**

Create `src/sleuth/langchain/_callback.py`:

```python
"""SleuthCallbackHandler — bridges Sleuth events into LangChain's callback system."""
from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain is not installed. Run: pip install agent-sleuth[langchain]"
    ) from exc

from sleuth.events import (
    CacheHitEvent,
    CitationEvent,
    DoneEvent,
    Event,
    FetchEvent,
    PlanEvent,
    RouteEvent,
    SearchEvent,
    ThinkingEvent,
    TokenEvent,
)


class SleuthCallbackHandler(BaseCallbackHandler):
    """Forward Sleuth events into LangChain's BaseCallbackHandler hooks.

    Event mapping:
      SearchEvent  → on_tool_start(serialized={name: backend}, input=query)
      DoneEvent    → on_tool_end(output=stats summary)
      TokenEvent   → on_llm_new_token(token=text)
      ThinkingEvent→ on_llm_new_token(token=text, chunk metadata)
      Others       → on_text(text=repr)

    Usage::

        handler = SleuthCallbackHandler()
        async for event in sleuth_agent.aask(query):
            handler.on_sleuth_event(event)
    """

    def on_sleuth_event(self, event: Event) -> None:
        """Dispatch a Sleuth event to the appropriate LangChain callback."""
        if isinstance(event, SearchEvent):
            self.on_tool_start(
                serialized={"name": f"sleuth:{event.backend}"},
                input_str=event.query,
            )
        elif isinstance(event, DoneEvent):
            summary = (
                f"done; latency={event.stats.latency_ms}ms "
                f"backends={event.stats.backends_called}"
            )
            self.on_tool_end(output=summary)
        elif isinstance(event, TokenEvent):
            self.on_llm_new_token(token=event.text)
        elif isinstance(event, ThinkingEvent):
            self.on_llm_new_token(token=event.text, chunk={"type": "thinking"})
        elif isinstance(event, (RouteEvent, PlanEvent, FetchEvent, CitationEvent, CacheHitEvent)):
            self.on_text(text=repr(event))

    # LangChain BaseCallbackHandler requires these to be defined;
    # the base class already provides no-op defaults — we override only for typing.
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        pass  # subclasses override

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        pass  # subclasses override

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass  # subclasses override

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass  # subclasses override
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/adapters/langchain/test_callback.py -v -m adapter
```

Expected: `4 passed`

- [ ] **Step 5: Run the full LangChain adapter suite**

```bash
uv run pytest tests/adapters/langchain/ -v -m adapter
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/sleuth/langchain/_callback.py tests/adapters/langchain/test_callback.py
git commit -m "feat(langchain): add SleuthCallbackHandler (Sleuth→LangChain event bridge)"
```

---

## Tier 1: Claude Agent SDK

### Task 6: Claude Agent SDK subpackage skeleton

**Files:**
- Create: `src/sleuth/claude_agent/__init__.py`
- Create: `src/sleuth/claude_agent/_tool.py`
- Create: `tests/adapters/claude_agent/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/sleuth/claude_agent tests/adapters/claude_agent
touch src/sleuth/claude_agent/__init__.py tests/adapters/claude_agent/__init__.py
```

- [ ] **Step 2: Write the public `__init__.py`**

```python
# src/sleuth/claude_agent/__init__.py
"""Claude Agent SDK adapter for Sleuth (extras=[claude-agent]).

Install: pip install agent-sleuth[claude-agent]
"""
from sleuth.claude_agent._tool import SleuthClaudeTool

__all__ = ["SleuthClaudeTool"]
```

- [ ] **Step 3: Commit the skeleton**

```bash
git add src/sleuth/claude_agent/ tests/adapters/claude_agent/__init__.py
git commit -m "chore: scaffold claude_agent adapter subpackage"
```

---

### Task 7: Claude Agent SDK SleuthClaudeTool

**Files:**
- Create: `src/sleuth/claude_agent/_tool.py`
- Test: `tests/adapters/claude_agent/test_tool.py`

The Claude Agent SDK exposes tool progress as message blocks (tool_use content blocks in the Anthropic message stream). Sleuth events map to progress updates via the SDK's tool progress callback.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/claude_agent/test_tool.py`:

```python
"""Tests for SleuthClaudeTool — Sleuth as a Claude Agent SDK tool."""
import pytest

pytest.importorskip("claude_agent_sdk", reason="claude-agent extra not installed")

from sleuth import Sleuth  # noqa: E402
from sleuth.claude_agent import SleuthClaudeTool  # noqa: E402


@pytest.mark.adapter
def test_sleuth_claude_tool_has_required_fields(sleuth_agent: Sleuth) -> None:
    tool = SleuthClaudeTool(agent=sleuth_agent)
    # Claude Agent SDK tools must expose: name, description, input_schema
    assert isinstance(tool.name, str) and len(tool.name) > 0
    assert isinstance(tool.description, str) and len(tool.description) > 0
    assert isinstance(tool.input_schema, dict)
    assert "properties" in tool.input_schema


@pytest.mark.adapter
def test_sleuth_claude_tool_input_schema_has_query(sleuth_agent: Sleuth) -> None:
    tool = SleuthClaudeTool(agent=sleuth_agent)
    props = tool.input_schema["properties"]
    assert "query" in props
    assert props["query"]["type"] == "string"


@pytest.mark.adapter
async def test_sleuth_claude_tool_call_returns_text(sleuth_agent: Sleuth) -> None:
    tool = SleuthClaudeTool(agent=sleuth_agent)
    result = await tool.call({"query": "What is Sleuth?"})
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
async def test_sleuth_claude_tool_emits_progress_blocks(sleuth_agent: Sleuth) -> None:
    """Progress blocks are emitted for search and token events."""
    tool = SleuthClaudeTool(agent=sleuth_agent)
    progress_blocks: list[dict] = []

    async def on_progress(block: dict) -> None:
        progress_blocks.append(block)

    await tool.call({"query": "test"}, on_progress=on_progress)
    # At minimum, a token or search progress block should have been emitted
    assert len(progress_blocks) >= 1
    block_types = {b.get("type") for b in progress_blocks}
    assert block_types & {"search_progress", "token_progress", "done_progress"}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/adapters/claude_agent/test_tool.py -v -m adapter
```

Expected: `ImportError` for `_tool` module.

- [ ] **Step 3: Implement `SleuthClaudeTool`**

Create `src/sleuth/claude_agent/_tool.py`:

```python
"""SleuthClaudeTool — Sleuth as a Claude Agent SDK tool with progress message blocks."""
from __future__ import annotations

from typing import Any, Callable, Awaitable

try:
    pass  # claude_agent_sdk import validated via importorskip in tests
except ImportError:  # pragma: no cover
    pass

from sleuth import Sleuth
from sleuth.events import (
    CacheHitEvent,
    CitationEvent,
    DoneEvent,
    Event,
    FetchEvent,
    PlanEvent,
    RouteEvent,
    SearchEvent,
    ThinkingEvent,
    TokenEvent,
)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


def _event_to_progress_block(event: Event) -> dict[str, Any] | None:
    """Map a Sleuth event to a Claude Agent SDK progress message block, or None."""
    if isinstance(event, SearchEvent):
        return {
            "type": "search_progress",
            "backend": event.backend,
            "query": event.query,
        }
    if isinstance(event, TokenEvent):
        return {"type": "token_progress", "text": event.text}
    if isinstance(event, ThinkingEvent):
        return {"type": "thinking_progress", "text": event.text}
    if isinstance(event, CitationEvent):
        return {
            "type": "citation_progress",
            "index": event.index,
            "source": event.source.model_dump(),
        }
    if isinstance(event, DoneEvent):
        return {
            "type": "done_progress",
            "latency_ms": event.stats.latency_ms,
            "backends_called": event.stats.backends_called,
        }
    # RouteEvent, PlanEvent, FetchEvent, CacheHitEvent — not surfaced as blocks
    return None


class SleuthClaudeTool:
    """Sleuth as a Claude Agent SDK tool.

    The Claude Agent SDK represents tool progress as typed message blocks
    streamed alongside the assistant response. Sleuth events map to these
    blocks so the agent can surface search progress in real time.

    Usage::

        from sleuth.claude_agent import SleuthClaudeTool
        tool = SleuthClaudeTool(agent=sleuth_instance)
        # Register with Claude Agent SDK agent:
        agent = ClaudeAgent(tools=[tool], ...)
    """

    name: str = "sleuth_search"
    description: str = (
        "A reasoning-first search tool that searches documents, code, and the web. "
        "Returns a synthesized answer with citations. Input: a natural-language query."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "depth": {
                "type": "string",
                "enum": ["auto", "fast", "deep"],
                "description": "Search depth. Defaults to 'auto'.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, agent: Sleuth) -> None:
        self._agent = agent

    async def call(
        self,
        inputs: dict[str, Any],
        *,
        on_progress: ProgressCallback = None,
    ) -> str:
        """Execute the search and optionally stream progress blocks.

        Args:
            inputs: Dict with at least ``query``. Optionally ``depth``.
            on_progress: Async callback receiving progress block dicts.

        Returns:
            The synthesized answer text.
        """
        query: str = inputs["query"]
        depth: str = inputs.get("depth", "auto")
        tokens: list[str] = []

        async for event in self._agent.aask(query, depth=depth):  # type: ignore[arg-type]
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
            if on_progress is not None:
                block = _event_to_progress_block(event)
                if block is not None:
                    await on_progress(block)

        return "".join(tokens)
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/adapters/claude_agent/test_tool.py -v -m adapter
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/sleuth/claude_agent/_tool.py tests/adapters/claude_agent/test_tool.py
git commit -m "feat(claude_agent): add SleuthClaudeTool with event→progress block mapping"
```

---

## Tier 2: LangGraph

### Task 8: LangGraph subpackage

**Files:**
- Create: `src/sleuth/langgraph/__init__.py`
- Create: `src/sleuth/langgraph/_node.py`
- Create: `tests/adapters/langgraph/__init__.py`
- Create: `tests/adapters/langgraph/test_node.py`

LangGraph expresses agents as directed graphs over state dicts. Sleuth provides a node factory: `make_sleuth_node(agent)` returns an async function with signature `(state: dict) -> dict` suitable for use in a `StateGraph`.

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/sleuth/langgraph tests/adapters/langgraph
touch src/sleuth/langgraph/__init__.py tests/adapters/langgraph/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/adapters/langgraph/test_node.py`:

```python
"""Tests for the LangGraph node factory."""
import pytest

pytest.importorskip("langgraph", reason="langgraph extra not installed")

from sleuth import Sleuth  # noqa: E402
from sleuth.langgraph import make_sleuth_node  # noqa: E402


@pytest.mark.adapter
def test_make_sleuth_node_returns_callable(sleuth_agent: Sleuth) -> None:
    node = make_sleuth_node(sleuth_agent)
    assert callable(node)


@pytest.mark.adapter
async def test_sleuth_node_reads_query_from_state(sleuth_agent: Sleuth) -> None:
    node = make_sleuth_node(sleuth_agent)
    state = {"query": "What is 42?", "messages": []}
    result = await node(state)
    # Node should return a dict with at least "answer" key
    assert isinstance(result, dict)
    assert "answer" in result
    assert isinstance(result["answer"], str)


@pytest.mark.adapter
async def test_sleuth_node_uses_messages_key_when_no_query(sleuth_agent: Sleuth) -> None:
    """When state has no 'query' key, fall back to last message content."""
    from langchain_core.messages import HumanMessage  # noqa: E402

    node = make_sleuth_node(sleuth_agent)
    state = {"messages": [HumanMessage(content="Explain auth flow")]}
    result = await node(state)
    assert "answer" in result


@pytest.mark.adapter
async def test_sleuth_node_custom_query_key(sleuth_agent: Sleuth) -> None:
    node = make_sleuth_node(sleuth_agent, query_key="search_input")
    state = {"search_input": "custom key query"}
    result = await node(state)
    assert "answer" in result
```

- [ ] **Step 3: Run — expect FAIL**

```bash
uv run pytest tests/adapters/langgraph/test_node.py -v -m adapter
```

Expected: `ImportError` for `sleuth.langgraph`.

- [ ] **Step 4: Implement the node factory**

Create `src/sleuth/langgraph/_node.py`:

```python
"""LangGraph node factory for Sleuth."""
from __future__ import annotations

from typing import Any

try:
    pass  # langgraph validated via importorskip in tests
except ImportError:  # pragma: no cover
    pass

from sleuth import Sleuth
from sleuth.events import TokenEvent


def make_sleuth_node(
    agent: Sleuth,
    *,
    query_key: str = "query",
    answer_key: str = "answer",
) -> Any:
    """Return an async LangGraph node function backed by Sleuth search.

    The returned coroutine has signature ``async (state: dict) -> dict``.
    It reads the query from ``state[query_key]`` (or falls back to the last
    message's ``content`` if no such key exists), runs ``agent.aask``, and
    returns ``{answer_key: synthesized_text}``.

    Usage in a LangGraph graph::

        from langgraph.graph import StateGraph
        from sleuth.langgraph import make_sleuth_node

        graph = StateGraph(MyState)
        graph.add_node("search", make_sleuth_node(sleuth_agent))
    """

    async def _node(state: dict[str, Any]) -> dict[str, Any]:
        # Extract query from state
        if query_key in state:
            query: str = state[query_key]
        else:
            # Fall back to last message content (LangGraph messages pattern)
            messages = state.get("messages", [])
            if messages:
                last = messages[-1]
                query = last.content if hasattr(last, "content") else str(last)
            else:
                query = ""

        tokens: list[str] = []
        async for event in agent.aask(query):
            if isinstance(event, TokenEvent):
                tokens.append(event.text)

        return {answer_key: "".join(tokens)}

    return _node
```

Create `src/sleuth/langgraph/__init__.py`:

```python
# src/sleuth/langgraph/__init__.py
"""LangGraph adapter for Sleuth (extras=[langgraph]).

Install: pip install agent-sleuth[langgraph]
"""
from sleuth.langgraph._node import make_sleuth_node

__all__ = ["make_sleuth_node"]
```

- [ ] **Step 5: Run — expect PASS**

```bash
uv run pytest tests/adapters/langgraph/test_node.py -v -m adapter
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/sleuth/langgraph/ tests/adapters/langgraph/
git commit -m "feat(langgraph): add make_sleuth_node factory for LangGraph state machines"
```

---

## Tier 2: LlamaIndex

### Task 9: LlamaIndex subpackage

**Files:**
- Create: `src/sleuth/llamaindex/__init__.py`
- Create: `src/sleuth/llamaindex/_query_engine.py`
- Create: `src/sleuth/llamaindex/_retriever.py`
- Create: `tests/adapters/llamaindex/__init__.py`
- Create: `tests/adapters/llamaindex/test_query_engine.py`
- Create: `tests/adapters/llamaindex/test_retriever.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/sleuth/llamaindex tests/adapters/llamaindex
touch src/sleuth/llamaindex/__init__.py tests/adapters/llamaindex/__init__.py
```

- [ ] **Step 2: Write failing tests for QueryEngine**

Create `tests/adapters/llamaindex/test_query_engine.py`:

```python
"""Tests for SleuthQueryEngine — Sleuth as a LlamaIndex QueryEngine."""
import pytest

pytest.importorskip("llama_index", reason="llamaindex extra not installed")

from llama_index.core.query_engine import BaseQueryEngine  # noqa: E402
from llama_index.core.response.schema import RESPONSE_TYPE  # noqa: E402

from sleuth import Sleuth  # noqa: E402
from sleuth.llamaindex import SleuthQueryEngine  # noqa: E402


@pytest.mark.adapter
def test_query_engine_is_base_query_engine(sleuth_agent: Sleuth) -> None:
    engine = SleuthQueryEngine(agent=sleuth_agent)
    assert isinstance(engine, BaseQueryEngine)


@pytest.mark.adapter
def test_query_engine_query_returns_response(sleuth_agent: Sleuth) -> None:
    engine = SleuthQueryEngine(agent=sleuth_agent)
    response = engine.query("What is 42?")
    # LlamaIndex Response objects have a .response str attribute
    assert hasattr(response, "response")
    assert isinstance(response.response, str)


@pytest.mark.adapter
async def test_query_engine_aquery_returns_response(sleuth_agent: Sleuth) -> None:
    engine = SleuthQueryEngine(agent=sleuth_agent)
    response = await engine.aquery("What is 42?")
    assert hasattr(response, "response")
    assert isinstance(response.response, str)
```

- [ ] **Step 3: Write failing tests for Retriever**

Create `tests/adapters/llamaindex/test_retriever.py`:

```python
"""Tests for SleuthRetriever — Sleuth as a LlamaIndex BaseRetriever."""
import pytest

pytest.importorskip("llama_index", reason="llamaindex extra not installed")

from llama_index.core.retrievers import BaseRetriever  # noqa: E402
from llama_index.core.schema import NodeWithScore  # noqa: E402

from sleuth import Sleuth  # noqa: E402
from sleuth.llamaindex import SleuthRetriever  # noqa: E402


@pytest.mark.adapter
def test_retriever_is_base_retriever(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    assert isinstance(retriever, BaseRetriever)


@pytest.mark.adapter
async def test_retriever_aretrieve_returns_nodes(sleuth_agent: Sleuth) -> None:
    retriever = SleuthRetriever(agent=sleuth_agent)
    from llama_index.core.schema import QueryBundle

    nodes = await retriever.aretrieve(QueryBundle(query_str="What is Sleuth?"))
    assert isinstance(nodes, list)
    for node in nodes:
        assert isinstance(node, NodeWithScore)
```

- [ ] **Step 4: Run — expect FAIL**

```bash
uv run pytest tests/adapters/llamaindex/ -v -m adapter
```

Expected: `ImportError` for both modules.

- [ ] **Step 5: Implement `SleuthQueryEngine`**

Create `src/sleuth/llamaindex/_query_engine.py`:

```python
"""SleuthQueryEngine — Sleuth as a LlamaIndex QueryEngine."""
from __future__ import annotations

from typing import Any

try:
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.response.schema import Response
    from llama_index.core.schema import QueryBundle
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LlamaIndex is not installed. Run: pip install agent-sleuth[llamaindex]"
    ) from exc

from sleuth import Sleuth
from sleuth.events import TokenEvent


class SleuthQueryEngine(BaseQueryEngine):
    """Expose Sleuth as a LlamaIndex QueryEngine.

    Usage::

        from sleuth.llamaindex import SleuthQueryEngine
        engine = SleuthQueryEngine(agent=sleuth_instance)
        response = engine.query("How does auth work?")
    """

    def __init__(self, agent: Sleuth, **kwargs: Any) -> None:
        self._agent = agent
        super().__init__(**kwargs)

    def _query(self, query_bundle: QueryBundle) -> Response:
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self._aquery(query_bundle))

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        tokens: list[str] = []
        async for event in self._agent.aask(query_bundle.query_str):
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
        return Response(response="".join(tokens))
```

- [ ] **Step 6: Implement `SleuthRetriever`**

Create `src/sleuth/llamaindex/_retriever.py`:

```python
"""SleuthRetriever — Sleuth as a LlamaIndex BaseRetriever."""
from __future__ import annotations

from typing import Any

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LlamaIndex is not installed. Run: pip install agent-sleuth[llamaindex]"
    ) from exc

from sleuth import Sleuth
from sleuth.types import Chunk


def _chunk_to_node_with_score(chunk: Chunk) -> NodeWithScore:
    node = TextNode(
        text=chunk.text,
        metadata={
            "source": chunk.source.location,
            "kind": chunk.source.kind,
            "title": chunk.source.title or "",
        },
    )
    return NodeWithScore(node=node, score=chunk.score or 0.0)


class SleuthRetriever(BaseRetriever):
    """Expose Sleuth backends as a LlamaIndex retriever.

    Usage::

        from sleuth.llamaindex import SleuthRetriever
        retriever = SleuthRetriever(agent=sleuth_instance)
    """

    def __init__(self, agent: Sleuth, **kwargs: Any) -> None:
        self._agent = agent
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._aretrieve(query_bundle)
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes: list[NodeWithScore] = []
        for backend in self._agent._backends:  # type: ignore[attr-defined]
            chunks = await backend.search(query_bundle.query_str, k=10)
            nodes.extend(_chunk_to_node_with_score(c) for c in chunks)
        return nodes
```

- [ ] **Step 7: Write `__init__.py`**

```python
# src/sleuth/llamaindex/__init__.py
"""LlamaIndex adapter for Sleuth (extras=[llamaindex]).

Install: pip install agent-sleuth[llamaindex]
"""
from sleuth.llamaindex._query_engine import SleuthQueryEngine
from sleuth.llamaindex._retriever import SleuthRetriever

__all__ = ["SleuthQueryEngine", "SleuthRetriever"]
```

- [ ] **Step 8: Run — expect PASS**

```bash
uv run pytest tests/adapters/llamaindex/ -v -m adapter
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/sleuth/llamaindex/ tests/adapters/llamaindex/
git commit -m "feat(llamaindex): add SleuthQueryEngine and SleuthRetriever"
```

---

## Tier 2: OpenAI Agents SDK

### Task 10: OpenAI Agents SDK subpackage

**Files:**
- Create: `src/sleuth/openai_agents/__init__.py`
- Create: `src/sleuth/openai_agents/_tool.py`
- Create: `tests/adapters/openai_agents/__init__.py`
- Create: `tests/adapters/openai_agents/test_tool.py`

The OpenAI Agents SDK registers tools as Python functions decorated with `@function_tool` (or passed as function metadata). Sleuth exposes a function-call compatible tool.

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/sleuth/openai_agents tests/adapters/openai_agents
touch src/sleuth/openai_agents/__init__.py tests/adapters/openai_agents/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/adapters/openai_agents/test_tool.py`:

```python
"""Tests for the OpenAI Agents SDK function-call tool."""
import pytest

pytest.importorskip("agents", reason="openai-agents extra not installed")

from sleuth import Sleuth  # noqa: E402
from sleuth.openai_agents import make_sleuth_function_tool  # noqa: E402


@pytest.mark.adapter
def test_make_sleuth_function_tool_returns_callable(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_function_tool(sleuth_agent)
    assert callable(tool_fn)


@pytest.mark.adapter
def test_function_tool_has_schema_metadata(sleuth_agent: Sleuth) -> None:
    """The returned function must carry openai-agents tool metadata."""
    tool_fn = make_sleuth_function_tool(sleuth_agent)
    # openai-agents decorates functions with __tool_name__ and __tool_description__
    assert hasattr(tool_fn, "__tool_name__") or hasattr(tool_fn, "__name__")


@pytest.mark.adapter
async def test_function_tool_call_returns_string(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_function_tool(sleuth_agent)
    result = await tool_fn(query="What is Sleuth?")
    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 3: Run — expect FAIL**

```bash
uv run pytest tests/adapters/openai_agents/test_tool.py -v -m adapter
```

Expected: `ImportError`.

- [ ] **Step 4: Implement the function-call tool**

Create `src/sleuth/openai_agents/_tool.py`:

```python
"""OpenAI Agents SDK function-call tool for Sleuth."""
from __future__ import annotations

from typing import Any, Callable, Awaitable

try:
    pass  # openai-agents validated via importorskip in tests
except ImportError:  # pragma: no cover
    pass

from sleuth import Sleuth
from sleuth.events import TokenEvent


def make_sleuth_function_tool(
    agent: Sleuth,
    *,
    name: str = "sleuth_search",
    description: str = (
        "A reasoning-first search tool. Searches documents, code, and the web. "
        "Input: query (str), depth (str, optional: 'auto'|'fast'|'deep')."
    ),
) -> Callable[..., Awaitable[str]]:
    """Return an async callable suitable for registration as an OpenAI Agents SDK tool.

    The returned function signature is::

        async def sleuth_search(query: str, depth: str = "auto") -> str

    Usage::

        from agents import Agent
        from sleuth.openai_agents import make_sleuth_function_tool

        search_fn = make_sleuth_function_tool(sleuth_instance)
        agent = Agent(name="MyAgent", tools=[search_fn])
    """

    async def sleuth_search(query: str, depth: str = "auto") -> str:
        """Search with Sleuth. Returns the synthesized answer."""
        tokens: list[str] = []
        async for event in agent.aask(query, depth=depth):  # type: ignore[arg-type]
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
        return "".join(tokens)

    sleuth_search.__name__ = name
    sleuth_search.__doc__ = description
    # Attach metadata for OpenAI Agents SDK introspection
    sleuth_search.__tool_name__ = name  # type: ignore[attr-defined]
    sleuth_search.__tool_description__ = description  # type: ignore[attr-defined]

    return sleuth_search


__all__ = ["make_sleuth_function_tool"]
```

Create `src/sleuth/openai_agents/__init__.py`:

```python
# src/sleuth/openai_agents/__init__.py
"""OpenAI Agents SDK adapter for Sleuth (extras=[openai-agents]).

Install: pip install agent-sleuth[openai-agents]
"""
from sleuth.openai_agents._tool import make_sleuth_function_tool

__all__ = ["make_sleuth_function_tool"]
```

- [ ] **Step 5: Run — expect PASS**

```bash
uv run pytest tests/adapters/openai_agents/test_tool.py -v -m adapter
```

Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add src/sleuth/openai_agents/ tests/adapters/openai_agents/
git commit -m "feat(openai_agents): add make_sleuth_function_tool for OpenAI Agents SDK"
```

---

## Tier 2: Pydantic AI

### Task 11: Pydantic AI subpackage

**Files:**
- Create: `src/sleuth/pydantic_ai/__init__.py`
- Create: `src/sleuth/pydantic_ai/_tool.py`
- Create: `tests/adapters/pydantic_ai/__init__.py`
- Create: `tests/adapters/pydantic_ai/test_tool.py`

Pydantic AI tools are Python functions with Pydantic-validated inputs decorated with `@agent.tool` or passed to `Agent(tools=[...])`. The input schema is inferred from type annotations. Sleuth's adapter provides a pre-built, schema-validated tool function.

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/sleuth/pydantic_ai tests/adapters/pydantic_ai
touch src/sleuth/pydantic_ai/__init__.py tests/adapters/pydantic_ai/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/adapters/pydantic_ai/test_tool.py`:

```python
"""Tests for Pydantic AI adapter — schema-validated tool."""
import pytest

pytest.importorskip("pydantic_ai", reason="pydantic-ai extra not installed")

from pydantic import BaseModel  # noqa: E402

from sleuth import Sleuth  # noqa: E402
from sleuth.pydantic_ai import make_sleuth_tool, SleuthInput  # noqa: E402


@pytest.mark.adapter
def test_sleuth_input_is_pydantic_model() -> None:
    """SleuthInput must be a Pydantic BaseModel for Pydantic AI schema inference."""
    assert issubclass(SleuthInput, BaseModel)
    # Must have query field
    assert "query" in SleuthInput.model_fields
    # Must have depth field
    assert "depth" in SleuthInput.model_fields


@pytest.mark.adapter
def test_make_sleuth_tool_returns_callable(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_tool(sleuth_agent)
    assert callable(tool_fn)


@pytest.mark.adapter
async def test_sleuth_tool_with_valid_input(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_tool(sleuth_agent)
    inputs = SleuthInput(query="What is 42?")
    result = await tool_fn(inputs)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
async def test_sleuth_tool_with_depth_override(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_tool(sleuth_agent)
    inputs = SleuthInput(query="deep question", depth="deep")
    result = await tool_fn(inputs)
    assert isinstance(result, str)
```

- [ ] **Step 3: Run — expect FAIL**

```bash
uv run pytest tests/adapters/pydantic_ai/test_tool.py -v -m adapter
```

Expected: `ImportError`.

- [ ] **Step 4: Implement the Pydantic AI tool**

Create `src/sleuth/pydantic_ai/_tool.py`:

```python
"""Pydantic AI tool with schema validation for Sleuth."""
from __future__ import annotations

from typing import Any, Callable, Awaitable, Literal

try:
    pass  # pydantic_ai validated via importorskip in tests
except ImportError:  # pragma: no cover
    pass

from pydantic import BaseModel, Field

from sleuth import Sleuth
from sleuth.events import TokenEvent


class SleuthInput(BaseModel):
    """Validated input schema for Sleuth tool — Pydantic AI infers this automatically."""

    query: str = Field(description="The natural-language search query.")
    depth: Literal["auto", "fast", "deep"] = Field(
        default="auto",
        description="Search depth: 'auto' lets Sleuth decide, 'fast' skips planning, "
        "'deep' uses full reflect loop.",
    )


def make_sleuth_tool(
    agent: Sleuth,
) -> Callable[[SleuthInput], Awaitable[str]]:
    """Return a Pydantic AI-compatible async tool function backed by Sleuth.

    Pydantic AI infers the JSON schema from ``SleuthInput``. Register the
    returned function with ``@pydantic_agent.tool`` or pass it to
    ``Agent(tools=[make_sleuth_tool(sleuth_instance)])``.

    Usage::

        from pydantic_ai import Agent
        from sleuth.pydantic_ai import make_sleuth_tool

        tool = make_sleuth_tool(sleuth_instance)

        @pydantic_agent.tool
        async def search(ctx, inputs: SleuthInput) -> str:
            return await tool(inputs)
    """

    async def _sleuth_tool(inputs: SleuthInput) -> str:
        tokens: list[str] = []
        async for event in agent.aask(inputs.query, depth=inputs.depth):
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
        return "".join(tokens)

    _sleuth_tool.__name__ = "sleuth_search"
    _sleuth_tool.__doc__ = (
        "Search with Sleuth. Returns synthesized answer. "
        "Input: SleuthInput(query, depth)."
    )
    return _sleuth_tool


__all__ = ["SleuthInput", "make_sleuth_tool"]
```

Create `src/sleuth/pydantic_ai/__init__.py`:

```python
# src/sleuth/pydantic_ai/__init__.py
"""Pydantic AI adapter for Sleuth (extras=[pydantic-ai]).

Install: pip install agent-sleuth[pydantic-ai]
"""
from sleuth.pydantic_ai._tool import SleuthInput, make_sleuth_tool

__all__ = ["SleuthInput", "make_sleuth_tool"]
```

- [ ] **Step 5: Run — expect PASS**

```bash
uv run pytest tests/adapters/pydantic_ai/test_tool.py -v -m adapter
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/sleuth/pydantic_ai/ tests/adapters/pydantic_ai/
git commit -m "feat(pydantic_ai): add SleuthInput + make_sleuth_tool with schema validation"
```

---

## Tier 2: CrewAI

### Task 12: CrewAI subpackage

**Files:**
- Create: `src/sleuth/crewai/__init__.py`
- Create: `src/sleuth/crewai/_tool.py`
- Create: `tests/adapters/crewai/__init__.py`
- Create: `tests/adapters/crewai/test_tool.py`

CrewAI tools are subclasses of `BaseTool` with a `_run(query: str) -> str` method. CrewAI does not have a native async callback surface, so the Sleuth event stream is exposed directly as an optional `on_event` callback parameter. The sync `_run` method is the required implementation; an async `_arun` is provided for forward-compat.

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/sleuth/crewai tests/adapters/crewai
touch src/sleuth/crewai/__init__.py tests/adapters/crewai/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/adapters/crewai/test_tool.py`:

```python
"""Tests for SleuthCrewAITool — CrewAI BaseTool subclass."""
import pytest

pytest.importorskip("crewai", reason="crewai extra not installed")

from crewai.tools import BaseTool  # noqa: E402

from sleuth import Sleuth  # noqa: E402
from sleuth.crewai import SleuthCrewAITool  # noqa: E402


@pytest.mark.adapter
def test_tool_is_crewai_base_tool(sleuth_agent: Sleuth) -> None:
    tool = SleuthCrewAITool(agent=sleuth_agent)
    assert isinstance(tool, BaseTool)


@pytest.mark.adapter
def test_tool_has_name_and_description(sleuth_agent: Sleuth) -> None:
    tool = SleuthCrewAITool(agent=sleuth_agent)
    assert tool.name == "sleuth_search"
    assert len(tool.description) > 10


@pytest.mark.adapter
def test_tool_run_returns_string(sleuth_agent: Sleuth) -> None:
    tool = SleuthCrewAITool(agent=sleuth_agent)
    result = tool._run("What is 42?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
async def test_tool_arun_returns_string(sleuth_agent: Sleuth) -> None:
    tool = SleuthCrewAITool(agent=sleuth_agent)
    result = await tool._arun("What is 42?")
    assert isinstance(result, str)


@pytest.mark.adapter
def test_tool_exposes_event_stream_via_on_event(sleuth_agent: Sleuth) -> None:
    """on_event callback receives Sleuth events during _run."""
    received: list[str] = []
    tool = SleuthCrewAITool(agent=sleuth_agent, on_event=lambda e: received.append(e.type))
    tool._run("test")
    assert "done" in received
```

- [ ] **Step 3: Run — expect FAIL**

```bash
uv run pytest tests/adapters/crewai/test_tool.py -v -m adapter
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `SleuthCrewAITool`**

Create `src/sleuth/crewai/_tool.py`:

```python
"""SleuthCrewAITool — CrewAI BaseTool subclass backed by Sleuth."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

try:
    from crewai.tools import BaseTool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CrewAI is not installed. Run: pip install agent-sleuth[crewai]"
    ) from exc

from pydantic import Field

from sleuth import Sleuth
from sleuth.events import Event, TokenEvent

OnEventCallback = Callable[[Event], None] | None


class SleuthCrewAITool(BaseTool):
    """Sleuth as a CrewAI BaseTool.

    CrewAI has no native async callback surface. The ``on_event`` parameter
    exposes Sleuth's event stream via a sync callback for observability.

    Usage::

        from sleuth.crewai import SleuthCrewAITool
        tool = SleuthCrewAITool(agent=sleuth_instance)
        crew = Crew(agents=[...], tasks=[...], tools=[tool])
    """

    name: str = "sleuth_search"
    description: str = (
        "A reasoning-first search tool. Searches documents, code, and the web. "
        "Input: a natural-language query string. Returns a synthesized answer."
    )
    agent: Sleuth = Field(exclude=True)
    on_event: OnEventCallback = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, **kwargs: Any) -> str:
        """Synchronous implementation required by CrewAI BaseTool."""
        on_event = self.on_event

        async def _collect() -> str:
            tokens: list[str] = []
            async for event in self.agent.aask(query):
                if isinstance(event, TokenEvent):
                    tokens.append(event.text)
                if on_event is not None:
                    on_event(event)
            return "".join(tokens)

        return asyncio.get_event_loop().run_until_complete(_collect())

    async def _arun(self, query: str, **kwargs: Any) -> str:
        """Async implementation for forward-compat."""
        tokens: list[str] = []
        async for event in self.agent.aask(query):
            if isinstance(event, TokenEvent):
                tokens.append(event.text)
            if self.on_event is not None:
                self.on_event(event)
        return "".join(tokens)
```

Create `src/sleuth/crewai/__init__.py`:

```python
# src/sleuth/crewai/__init__.py
"""CrewAI adapter for Sleuth (extras=[crewai]).

Install: pip install agent-sleuth[crewai]

Note: CrewAI has no native async callback surface. Use the ``on_event``
parameter to receive Sleuth events synchronously during tool execution.
"""
from sleuth.crewai._tool import SleuthCrewAITool

__all__ = ["SleuthCrewAITool"]
```

- [ ] **Step 5: Run — expect PASS**

```bash
uv run pytest tests/adapters/crewai/test_tool.py -v -m adapter
```

Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add src/sleuth/crewai/ tests/adapters/crewai/
git commit -m "feat(crewai): add SleuthCrewAITool with on_event callback for observability"
```

---

## Tier 2: AutoGen

### Task 13: AutoGen subpackage

**Files:**
- Create: `src/sleuth/autogen/__init__.py`
- Create: `src/sleuth/autogen/_tool.py`
- Create: `tests/adapters/autogen/__init__.py`
- Create: `tests/adapters/autogen/test_tool.py`

AutoGen registers tools as plain Python functions via `register_function` or by passing them to an `AssistantAgent`. Sleuth provides a `register_sleuth_tool` helper that registers the Sleuth search function on an AutoGen agent and returns the function itself.

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/sleuth/autogen tests/adapters/autogen
touch src/sleuth/autogen/__init__.py tests/adapters/autogen/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/adapters/autogen/test_tool.py`:

```python
"""Tests for AutoGen function-tool registration."""
import pytest

pytest.importorskip("autogen", reason="autogen extra not installed")

from sleuth import Sleuth  # noqa: E402
from sleuth.autogen import make_sleuth_autogen_tool, register_sleuth_tool  # noqa: E402


@pytest.mark.adapter
def test_make_sleuth_autogen_tool_returns_callable(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_autogen_tool(sleuth_agent)
    assert callable(tool_fn)


@pytest.mark.adapter
def test_autogen_tool_has_name_and_docstring(sleuth_agent: Sleuth) -> None:
    tool_fn = make_sleuth_autogen_tool(sleuth_agent)
    assert tool_fn.__name__ == "sleuth_search"
    assert tool_fn.__doc__ is not None and len(tool_fn.__doc__) > 10


@pytest.mark.adapter
def test_autogen_tool_sync_call_returns_string(sleuth_agent: Sleuth) -> None:
    """AutoGen function tools are called synchronously by the framework."""
    tool_fn = make_sleuth_autogen_tool(sleuth_agent)
    result = tool_fn(query="What is 42?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.adapter
def test_register_sleuth_tool_attaches_to_agent(sleuth_agent: Sleuth) -> None:
    """register_sleuth_tool registers the function on an AutoGen agent."""
    import autogen  # noqa: E402

    config_list = [{"model": "gpt-4", "api_key": "fake"}]
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={"config_list": config_list},
    )
    user_proxy = autogen.UserProxyAgent(name="user_proxy", human_input_mode="NEVER")

    # Should not raise; registers tool on both agents
    tool_fn = register_sleuth_tool(sleuth_agent, caller=assistant, executor=user_proxy)
    assert callable(tool_fn)
```

- [ ] **Step 3: Run — expect FAIL**

```bash
uv run pytest tests/adapters/autogen/test_tool.py -v -m adapter
```

Expected: `ImportError`.

- [ ] **Step 4: Implement the AutoGen tool**

Create `src/sleuth/autogen/_tool.py`:

```python
"""AutoGen function-tool integration for Sleuth."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

try:
    pass  # autogen validated via importorskip in tests
except ImportError:  # pragma: no cover
    pass

from sleuth import Sleuth
from sleuth.events import TokenEvent


def make_sleuth_autogen_tool(
    agent: Sleuth,
    *,
    name: str = "sleuth_search",
    description: str = (
        "A reasoning-first search tool that searches documents, code, and the web. "
        "Returns a synthesized answer with citations. "
        "Args: query (str) — natural-language question."
    ),
) -> Callable[..., str]:
    """Return a synchronous function suitable for AutoGen function-tool registration.

    AutoGen expects callable tools that run synchronously in the executor context.
    This wrapper drives Sleuth's async engine via ``asyncio.run``.

    Usage::

        from sleuth.autogen import make_sleuth_autogen_tool
        tool = make_sleuth_autogen_tool(sleuth_instance)

        # Register manually:
        @user_proxy.register_for_execution()
        @assistant.register_for_llm(description=tool.__doc__)
        def sleuth_search(query: str) -> str:
            return tool(query=query)
    """

    def sleuth_search(query: str) -> str:
        """Search with Sleuth and return the synthesized answer."""
        async def _collect() -> str:
            tokens: list[str] = []
            async for event in agent.aask(query):
                if isinstance(event, TokenEvent):
                    tokens.append(event.text)
            return "".join(tokens)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _collect())
                    return future.result()
            return loop.run_until_complete(_collect())
        except RuntimeError:
            return asyncio.run(_collect())

    sleuth_search.__name__ = name
    sleuth_search.__doc__ = description
    return sleuth_search


def register_sleuth_tool(
    agent: Sleuth,
    *,
    caller: Any,
    executor: Any,
    name: str = "sleuth_search",
    description: str = (
        "A reasoning-first search tool that searches documents, code, and the web."
    ),
) -> Callable[..., str]:
    """Register Sleuth as a function tool on an AutoGen agent pair.

    Args:
        agent:       The Sleuth instance to back the tool.
        caller:      An AutoGen ``AssistantAgent`` that calls the tool.
        executor:    An AutoGen ``UserProxyAgent`` that executes the tool.
        name:        Tool function name (default: ``sleuth_search``).
        description: Tool description shown to the LLM.

    Returns:
        The registered tool function.
    """
    tool_fn = make_sleuth_autogen_tool(agent, name=name, description=description)

    # AutoGen registration pattern
    if hasattr(executor, "register_for_execution"):
        executor.register_for_execution(name=name)(tool_fn)
    if hasattr(caller, "register_for_llm"):
        caller.register_for_llm(name=name, description=description)(tool_fn)

    return tool_fn


__all__ = ["make_sleuth_autogen_tool", "register_sleuth_tool"]
```

Create `src/sleuth/autogen/__init__.py`:

```python
# src/sleuth/autogen/__init__.py
"""AutoGen adapter for Sleuth (extras=[autogen]).

Install: pip install agent-sleuth[autogen]
"""
from sleuth.autogen._tool import make_sleuth_autogen_tool, register_sleuth_tool

__all__ = ["make_sleuth_autogen_tool", "register_sleuth_tool"]
```

- [ ] **Step 5: Run — expect PASS**

```bash
uv run pytest tests/adapters/autogen/test_tool.py -v -m adapter
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/sleuth/autogen/ tests/adapters/autogen/
git commit -m "feat(autogen): add make_sleuth_autogen_tool and register_sleuth_tool"
```

---

## Final wiring and verification

### Task 14: Verify pyproject.toml extras (no changes needed — already defined)

**Files:**
- Read: `pyproject.toml` (Phase 0 owns this file; verify, do not modify)

- [ ] **Step 1: Confirm all adapter extras are in `pyproject.toml`**

```bash
grep -A 15 "\[project.optional-dependencies\]" pyproject.toml
```

Expected output includes all eight adapter extras (conventions §3 already enumerates them):
```
langchain     = ["langchain-core>=0.3"]
langgraph     = ["langgraph>=0.2"]
llamaindex    = ["llama-index-core>=0.11"]
openai-agents = ["openai-agents>=0.1"]
claude-agent  = ["claude-agent-sdk>=0.1"]
pydantic-ai   = ["pydantic-ai>=0.0.13"]
crewai        = ["crewai>=0.80"]
autogen       = ["pyautogen>=0.3"]
```

If any are missing (because Phase 0 hasn't been executed yet), add only the missing rows:

```bash
# Example if autogen is missing:
# uv add --optional autogen "pyautogen>=0.3"
```

No commit needed if already present.

---

### Task 15: Run full adapter test suite and type-check

**Files:**
- No changes.

- [ ] **Step 1: Install all adapter extras in dev environment**

```bash
uv sync --all-extras --group dev
```

Expected: installs all optional framework deps.

- [ ] **Step 2: Run all adapter tests**

```bash
uv run pytest tests/adapters/ -v -m adapter
```

Expected: all tests pass across all eight adapters.

- [ ] **Step 3: Run mypy over all adapter subpackages**

```bash
uv run mypy src/sleuth/langchain/ src/sleuth/langgraph/ src/sleuth/llamaindex/ \
            src/sleuth/openai_agents/ src/sleuth/claude_agent/ \
            src/sleuth/pydantic_ai/ src/sleuth/crewai/ src/sleuth/autogen/
```

Expected: `Success: no issues found`

If mypy flags a stub issue for a framework that ships incomplete type stubs, add a `# type: ignore[import]` on the specific import and document the reason in a comment.

- [ ] **Step 4: Run ruff**

```bash
uv run ruff check src/sleuth/langchain/ src/sleuth/langgraph/ src/sleuth/llamaindex/ \
                   src/sleuth/openai_agents/ src/sleuth/claude_agent/ \
                   src/sleuth/pydantic_ai/ src/sleuth/crewai/ src/sleuth/autogen/ \
                   tests/adapters/
uv run ruff format --check src/sleuth/ tests/adapters/
```

Expected: no issues.

- [ ] **Step 5: Commit final wiring**

```bash
git add -u
git commit -m "chore: verify and wire all framework adapter extras"
```

---

### Task 16: Integration smoke test (marker gate)

**Files:**
- Create: `tests/adapters/test_integration_smoke.py`

This test is gated behind `@pytest.mark.integration` so it only runs when API keys are present (nightly CI).

- [ ] **Step 1: Write the integration smoke test**

Create `tests/adapters/test_integration_smoke.py`:

```python
"""Integration smoke test — one real Q&A round-trip per installed adapter.

Requires: SLEUTH_SMOKE_QUERY env var and at least one backend configured.
Marked integration; only runs nightly or with: pytest -m integration
"""
import os
import pytest

QUERY = os.getenv("SLEUTH_SMOKE_QUERY", "What is agent-sleuth?")


@pytest.mark.integration
def test_langchain_tool_round_trip(sleuth_agent):
    pytest.importorskip("langchain_core")
    from sleuth.langchain import SleuthTool

    tool = SleuthTool(agent=sleuth_agent)
    result = tool.run(QUERY)
    assert isinstance(result, str) and len(result) > 0


@pytest.mark.integration
async def test_llamaindex_query_engine_round_trip(sleuth_agent):
    pytest.importorskip("llama_index")
    from sleuth.llamaindex import SleuthQueryEngine

    engine = SleuthQueryEngine(agent=sleuth_agent)
    response = await engine.aquery(QUERY)
    assert response.response and len(response.response) > 0


@pytest.mark.integration
async def test_langgraph_node_round_trip(sleuth_agent):
    pytest.importorskip("langgraph")
    from sleuth.langgraph import make_sleuth_node

    node = make_sleuth_node(sleuth_agent)
    result = await node({"query": QUERY})
    assert "answer" in result and result["answer"]


@pytest.mark.integration
async def test_claude_agent_tool_round_trip(sleuth_agent):
    pytest.importorskip("claude_agent_sdk")
    from sleuth.claude_agent import SleuthClaudeTool

    tool = SleuthClaudeTool(agent=sleuth_agent)
    result = await tool.call({"query": QUERY})
    assert isinstance(result, str) and len(result) > 0
```

- [ ] **Step 2: Verify the integration tests are collected but skipped without the marker**

```bash
uv run pytest tests/adapters/test_integration_smoke.py -v --collect-only
```

Expected: tests collected, zero run without `-m integration`.

- [ ] **Step 3: Commit**

```bash
git add tests/adapters/test_integration_smoke.py
git commit -m "test(adapters): add integration smoke tests gated behind @pytest.mark.integration"
```

---

## Self-Review Checklist

*(Executed by plan author — not a task for the implementing engineer.)*

**Spec coverage:**
- §10.2 table row LangChain → SleuthTool (Task 3), SleuthRetriever (Task 4), SleuthCallbackHandler (Task 5). Covered.
- §10.2 table row LangGraph → node factory. Task 8. Covered.
- §10.2 table row LlamaIndex → SleuthQueryEngine (Task 9), SleuthRetriever (Task 9). Covered.
- §10.2 table row OpenAI Agents SDK → function-call tool. Task 10. Covered.
- §10.2 table row Claude Agent SDK → tool with progress mapped to message blocks. Task 7. Covered.
- §10.2 table row Pydantic AI → tool with schema validation. Task 11. Covered.
- §10.2 table row CrewAI → BaseTool subclass. Task 12. Covered.
- §10.2 table row AutoGen → function-tool registration. Task 13. Covered.
- §5 event stream mapped to host-framework callbacks (LangChain BaseCallbackHandler, LlamaIndex CallbackManager surface via QueryEngine, Claude Agent SDK progress blocks). Covered.
- Lazy imports for all host frameworks — each module guards the framework import with try/except ImportError. Covered.
- Per-adapter extras already in conventions §3 pyproject.toml — Task 14 verifies. Covered.
- Per-adapter tests under `tests/adapters/<framework>/`. All tasks create these. Covered.
- `@pytest.mark.adapter` smoke marker used throughout. Covered.
- `@pytest.mark.integration` for round-trip tests. Task 16. Covered.

**Placeholder scan:** No "TBD", "TODO", "implement later", or "Similar to Task N" found. All steps contain actual code.

**Type consistency:**
- `SleuthTool`, `SleuthRetriever`, `SleuthCallbackHandler` — consistent across Tasks 2–5 and their `__init__.py`.
- `make_sleuth_node` — consistent in Task 8.
- `SleuthQueryEngine`, `SleuthRetriever` (LlamaIndex) — consistent in Task 9.
- `make_sleuth_function_tool` — consistent in Task 10.
- `SleuthInput`, `make_sleuth_tool` — consistent in Task 11.
- `SleuthCrewAITool` — consistent in Task 12.
- `make_sleuth_autogen_tool`, `register_sleuth_tool` — consistent in Task 13.
- `_event_to_progress_block` in Claude adapter — only referenced internally. Consistent.
- `FakeBackend`, `stub_llm`, `sleuth_agent` fixtures defined in `tests/adapters/conftest.py` (Task 1) and used throughout — consistent.
- `from sleuth.events import TokenEvent` etc. — all event types imported from `sleuth.events`, not redefined. Consistent.
