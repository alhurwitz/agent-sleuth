# Framework adapters

Eight adapters let you drop Sleuth into an existing pipeline without re-architecting. All are behind optional extras and lazy-import their framework.

---

## LangChain

**Install:** `pip install 'agent-sleuth[langchain]'`

**Public surface:** `SleuthTool`, `SleuthRetriever`, `SleuthCallbackHandler`

### `SleuthTool` — as a `BaseTool`

```python
from sleuth.langchain import SleuthTool

tool = SleuthTool(agent=sleuth_instance)
# Pass to AgentExecutor, LCEL chains, etc.
agent_executor = AgentExecutor(tools=[tool], llm=..., agent=...)
```

`SleuthTool` subclasses `BaseTool`. It implements both `_run` (sync, via thread pool) and `_arun` (async). The tool description is: "A reasoning-first search tool. Use for questions that require searching documents, code, or the web."

### `SleuthRetriever` — inside `RetrievalQA`

```python
from sleuth.langchain import SleuthRetriever
from langchain.chains import RetrievalQA

retriever = SleuthRetriever(agent=sleuth_instance)
qa = RetrievalQA.from_chain_type(llm=..., retriever=retriever)
```

`SleuthRetriever` bypasses Sleuth's synthesizer and returns raw `Chunk`s as LangChain `Document` objects — suitable for chains that do their own synthesis.

### `SleuthCallbackHandler` — event mapping

```python
from sleuth.langchain import SleuthCallbackHandler

handler = SleuthCallbackHandler()
async for event in sleuth_instance.aask(query):
    handler.on_sleuth_event(event)
```

Event → LangChain callback mapping:

| Sleuth event | LangChain callback |
| --- | --- |
| `SearchEvent` | `on_tool_start(serialized={name: "sleuth:<backend>"}, input_str=query)` |
| `DoneEvent` | `on_tool_end(output="done; latency=…ms backends=[…]")` |
| `TokenEvent` | `on_llm_new_token(token=text)` |
| `ThinkingEvent` | `on_llm_new_token(token=text, chunk={"type": "thinking"})` |
| Others | `on_text(text=repr(event))` |

Subclass `SleuthCallbackHandler` and override only the methods you need.

---

## Claude Agent SDK

**Install:** `pip install 'agent-sleuth[claude-agent]'`

**Public surface:** `SleuthClaudeTool`

```python
from sleuth.claude_agent import SleuthClaudeTool

tool = SleuthClaudeTool(agent=sleuth_instance)
# Register with your Claude Agent SDK agent:
agent = ClaudeAgent(tools=[tool], ...)
```

The Claude Agent SDK represents tool progress as typed **message blocks** streamed alongside the assistant response. `SleuthClaudeTool.call()` accepts an `on_progress` async callback:

```python
async def my_progress(block: dict) -> None:
    print(block)  # {"type": "search_progress", "backend": "tavily", "query": "..."}

result_text = await tool.call(
    {"query": "How does OAuth work?", "depth": "auto"},
    on_progress=my_progress,
)
```

Event → progress block mapping:

| Sleuth event | Block type |
| --- | --- |
| `SearchEvent` | `{"type": "search_progress", "backend": ..., "query": ...}` |
| `TokenEvent` | `{"type": "token_progress", "text": ...}` |
| `ThinkingEvent` | `{"type": "thinking_progress", "text": ...}` |
| `CitationEvent` | `{"type": "citation_progress", "index": ..., "source": {...}}` |
| `DoneEvent` | `{"type": "done_progress", "latency_ms": ..., "backends_called": [...]}` |

`RouteEvent`, `PlanEvent`, `FetchEvent`, `CacheHitEvent` are not surfaced as progress blocks.

---

## LangGraph

**Install:** `pip install 'agent-sleuth[langgraph]'`

**Public surface:** `make_sleuth_node`

```python
from langgraph.graph import StateGraph
from sleuth.langgraph import make_sleuth_node

graph = StateGraph(MyState)
graph.add_node("search", make_sleuth_node(sleuth_instance))
```

The returned coroutine has signature `async (state: dict) -> dict`. It reads the query from `state["query"]` (or falls back to the last message's `content`), runs `agent.aask`, and returns `{"answer": synthesized_text}`.

```python
node = make_sleuth_node(
    sleuth_instance,
    query_key="query",    # key to read from state (default: "query")
    answer_key="answer",  # key to write to state (default: "answer")
)
```

---

## LlamaIndex

**Install:** `pip install 'agent-sleuth[llamaindex]'`

**Public surface:** `SleuthQueryEngine`, `SleuthRetriever`

### `SleuthQueryEngine`

```python
from sleuth.llamaindex import SleuthQueryEngine

engine = SleuthQueryEngine(agent=sleuth_instance)
response = engine.query("How does auth work?")
print(response.response)
```

Subclasses `BaseQueryEngine`. Implements `_query` (sync via thread) and `_aquery` (async). Returns LlamaIndex `Response` objects.

### `SleuthRetriever` (LlamaIndex)

```python
from sleuth.llamaindex import SleuthRetriever as LlamaRetriever

retriever = LlamaRetriever(agent=sleuth_instance)
nodes = retriever.retrieve("auth middleware")
```

---

## OpenAI Agents SDK

**Install:** `pip install 'agent-sleuth[openai-agents]'`

**Public surface:** `make_sleuth_function_tool`

```python
from agents import Agent
from sleuth.openai_agents import make_sleuth_function_tool

search_fn = make_sleuth_function_tool(sleuth_instance)
agent = Agent(name="MyAgent", tools=[search_fn])
```

The returned async function has signature `async (query: str, depth: str = "auto") -> str`. Its `__name__` and `__doc__` are set for OpenAI Agents SDK introspection.

---

## Pydantic AI

**Install:** `pip install 'agent-sleuth[pydantic-ai]'`

**Public surface:** `make_sleuth_tool`, `SleuthInput`

```python
from pydantic_ai import Agent as PydanticAgent
from sleuth.pydantic_ai import make_sleuth_tool, SleuthInput

tool = make_sleuth_tool(sleuth_instance)

@pydantic_agent.tool
async def search(ctx, inputs: SleuthInput) -> str:
    return await tool(inputs)
```

`SleuthInput` is a Pydantic model Pydantic AI uses to infer the JSON schema:

```python
class SleuthInput(BaseModel):
    query: str
    depth: Literal["auto", "fast", "deep"] = "auto"
```

---

## CrewAI

**Install:** `pip install 'agent-sleuth[crewai]'`

**Public surface:** `SleuthCrewAITool`

```python
from sleuth.crewai import SleuthCrewAITool
from crewai import Crew

tool = SleuthCrewAITool(agent=sleuth_instance)
crew = Crew(agents=[...], tasks=[...], tools=[tool])
```

CrewAI has no native async callback surface. Use the `on_event` parameter for observability:

```python
def watch(event):
    if event.type == "search":
        print(f"Searching [{event.backend}]: {event.query}")

tool = SleuthCrewAITool(agent=sleuth_instance, on_event=watch)
```

`on_event` is called synchronously for every event during `_run`.

---

## AutoGen

**Install:** `pip install 'agent-sleuth[autogen]'`

**Public surface:** `make_sleuth_autogen_tool`, `register_sleuth_tool`

```python
from autogen_agentchat.agents import AssistantAgent
from sleuth.autogen import make_sleuth_autogen_tool

tool = make_sleuth_autogen_tool(sleuth_instance)
assistant = AssistantAgent(name="assistant", model_client=..., tools=[tool])
```

For the legacy autogen v0.2/v0.3 API, use `register_sleuth_tool` to register on an agent pair:

```python
from sleuth.autogen import register_sleuth_tool

register_sleuth_tool(
    agent=sleuth_instance,
    caller=assistant_agent,
    executor=user_proxy_agent,
    name="sleuth_search",
)
```

`register_sleuth_tool` calls both `register_for_execution` on the executor and `register_for_llm` on the caller when those methods exist.

!!! note "AutoGen package name"
    The extra resolves to `autogen-agentchat`. The adapter targets the v0.4+ API (`model_client=` instead of `llm_config=`).
