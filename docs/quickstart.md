# Quickstart

Get a cited, streaming answer from Sleuth in five minutes.

## Install

`agent-sleuth` has a zero-dependency core. Install only what your use case needs:

```bash
# Core (no LLM/framework deps)
pip install agent-sleuth

# LLM shims (BYOK — pick one or both)
pip install 'agent-sleuth[anthropic]'      # Anthropic SDK shim (claude-* models)
pip install 'agent-sleuth[openai]'         # OpenAI SDK shim (gpt-4o, o3, …)

# Framework adapters
pip install 'agent-sleuth[langchain]'      # SleuthTool, SleuthRetriever, SleuthCallbackHandler
pip install 'agent-sleuth[claude-agent]'   # SleuthClaudeTool with progress blocks
pip install 'agent-sleuth[langgraph]'      # make_sleuth_node graph-node factory
pip install 'agent-sleuth[llamaindex]'     # SleuthQueryEngine + SleuthRetriever
pip install 'agent-sleuth[openai-agents]'  # make_sleuth_function_tool
pip install 'agent-sleuth[pydantic-ai]'    # make_sleuth_tool + SleuthInput schema
pip install 'agent-sleuth[crewai]'         # SleuthCrewAITool with on_event callback
pip install 'agent-sleuth[autogen]'        # make_sleuth_autogen_tool, register_sleuth_tool

# Backend extras
pip install 'agent-sleuth[localfiles]'     # LocalFiles — PDFs via pymupdf
pip install 'agent-sleuth[code]'           # CodeSearch — tree-sitter parsing
pip install 'agent-sleuth[code-embed]'     # CodeSearch optional embedding re-rank
pip install 'agent-sleuth[exa]'            # Exa web provider
pip install 'agent-sleuth[web-fetch]'      # fetch=True mode (trafilatura, tiktoken)

# Vector-store vendors
pip install 'agent-sleuth[pinecone]'
pip install 'agent-sleuth[qdrant]'
pip install 'agent-sleuth[chroma]'
pip install 'agent-sleuth[weaviate]'

# Memory
pip install 'agent-sleuth[semantic]'       # SemanticCache with fastembed BGE-small

# MCP server
pip install 'agent-sleuth[mcp]'            # sleuth-mcp binary
```

---

## Scenario 1 — Web search (Tavily)

The minimal working example: one web backend, one LLM, one async search.

```python
import asyncio, os
from sleuth import Sleuth
from sleuth.backends import WebBackend
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[WebBackend(provider="tavily", api_key=os.environ["TAVILY_KEY"])],
)

async def main():
    async for event in agent.aask("Who maintains the Anthropic Agent SDK?"):
        if event.type == "token":
            print(event.text, end="", flush=True)
        elif event.type == "citation":
            print(f"\n[{event.index}] {event.source.location}")
        elif event.type == "done":
            print(f"\nLatency: {event.stats.latency_ms} ms")

asyncio.run(main())
```

**What you'll see:** tokens stream as the LLM generates them, followed by citation lines and a latency summary.

For the synchronous path (no `asyncio.run` required) use `agent.ask(...)` — it blocks and returns a [`Result`](concepts/events.md):

```python
result = agent.ask("Latest Python release date?")
print(result.text)
for src in result.citations:
    print(" -", src.location)
```

See [Web providers](backends/web.md) for all four providers and `fetch=True` mode.

---

## Scenario 2 — Search your local docs (no embeddings)

`LocalFiles` builds a hierarchical tree-of-contents over your corpus the first time it runs. Subsequent calls reuse the persisted index at `<corpus>/.sleuth/index/`.

```python
from sleuth import Sleuth
from sleuth.backends.localfiles import LocalFiles
from sleuth.llm.anthropic import Anthropic

llm = Anthropic(model="claude-sonnet-4-6")
agent = Sleuth(
    llm=llm,
    fast_llm=Anthropic(model="claude-haiku-4-5"),  # faster model for indexing/navigation
    backends=[LocalFiles(path="./docs")],
)

result = agent.ask("Where do we document refresh-token rotation?")
print(result.text)
```

!!! note "First-run cost"
    Indexing runs the LLM over each document to build branch summaries. For ~200 documents expect a few minutes and a small API cost. The index persists at `./docs/.sleuth/index/` and only rebuilds when files change.

Pre-warm the index before the first query with:

```python
import asyncio
asyncio.run(agent.warm_index())
```

See [Local files](backends/local-files.md) for all constructor options.

---

## Scenario 3 — Structured output

Pass `schema=` to get a typed Pydantic model in `result.data` alongside the streaming tokens.

```python
from pydantic import BaseModel
from sleuth import Sleuth
from sleuth.backends import WebBackend
from sleuth.llm.anthropic import Anthropic

class Verdict(BaseModel):
    answer: str
    confidence: float   # 0.0–1.0
    sources: list[str]

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[WebBackend(provider="tavily", api_key="...")],
)

result = agent.ask("Is the deploy script idempotent?", schema=Verdict)

# Both are available:
print(result.text)               # synthesized prose
print(result.data.confidence)    # typed field from Verdict
print(result.data.sources)
```

!!! warning "Schema results bypass the cache"
    Schema-typed results are not currently round-trippable through JSON, so cache writes are skipped when `schema=` is set. This is a known v0.1.0 limitation — see [Structured output](recipes/structured-output.md).

---

## Scenario 4 — Multi-turn with Session

Pass a `Session` to maintain context across calls. Sleuth stores turns in a ring buffer (default: last 20) and prepends them as conversation history to the LLM.

```python
from sleuth import Sleuth, Session
from sleuth.backends import WebBackend
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[WebBackend(provider="tavily", api_key="...")],
)

session = Session()

r1 = agent.ask("Who maintains the auth middleware?", session=session)
print(r1.text)

r2 = agent.ask("What changed in their last commit?", session=session)
print(r2.text)  # LLM has context from r1
```

Persist across restarts:

```python
session.save("./session.json")
# ...later:
session = Session.load("./session.json")
```

See [Sessions & multi-turn](recipes/sessions.md) for the full persistence API.

---

## Scenario 5 — Deep mode (multi-step reasoning)

`depth="deep"` engages the Planner and reflect loop. The router would have auto-selected deep mode for this query, but you can force it.

```python
import asyncio
from sleuth import Sleuth
from sleuth.backends import WebBackend
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[WebBackend(provider="tavily", api_key="...")],
)

async def main():
    async for event in agent.aask(
        "Compare OAuth and OIDC: tradeoffs, use cases, and token lifetimes",
        depth="deep",
        max_iterations=4,
    ):
        if event.type == "plan":
            print(f"Plan: {[s.query for s in event.steps]}")
        elif event.type == "search":
            print(f"Searching [{event.backend}]: {event.query}")
        elif event.type == "token":
            print(event.text, end="", flush=True)

asyncio.run(main())
```

Deep mode emits `PlanEvent` before each iteration and multiple `SearchEvent`s as sub-queries fan out in parallel. See [Deep mode](recipes/deep-mode.md).

---

## Scenario 6 — LangChain tool

Drop Sleuth into any LangChain agent or chain that accepts a `BaseTool`.

```python
from sleuth import Sleuth
from sleuth.backends import WebBackend
from sleuth.langchain import SleuthTool
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[WebBackend(provider="tavily", api_key="...")],
)

tool = SleuthTool(agent=agent)
# Pass tool to AgentExecutor, LCEL chains, etc.
```

See [Frameworks](adapters/frameworks.md) for all eight framework adapters.

---

## Scenario 7 — MCP server

Run `sleuth-mcp` and wire it to Claude Desktop or any other MCP client.

**1. Create the config file:**

```toml
# ~/.config/sleuth/mcp.toml
[llm]
name = "anthropic:claude-sonnet-4-6"

[[backends]]
type = "web"
provider = "tavily"
api_key_env = "TAVILY_API_KEY"

[[backends]]
type = "localfiles"
path = "/var/data/docs"
```

**2. Start the server:**

```bash
# stdio (for Claude Desktop / most MCP clients)
sleuth-mcp --transport stdio

# HTTP (for browser-based or remote clients)
sleuth-mcp --transport http --host 127.0.0.1 --port 4737
```

**3. Wire into Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "sleuth": {
      "command": "sleuth-mcp",
      "args": ["--transport", "stdio"]
    }
  }
}
```

See [MCP server](adapters/mcp.md) for the full TOML schema and transport reference.

---

## What's next

| Goal | Page |
| --- | --- |
| Understand the event stream | [Event stream](concepts/events.md) |
| Search source code | [Code search](backends/code-search.md) |
| Use your existing vector index | [Vector stores](backends/vector-store.md) |
| Write a custom backend | [Custom backends](backends/custom.md) |
| Observe what the engine is doing | [Observability](recipes/observability.md) |
| Full `Sleuth` constructor reference | [Python SDK](adapters/python.md) |
