# Adapters

Sleuth integrates with three first-class surfaces and eight popular framework ecosystems.

---

## Tier model

### Tier 1 — full support

| Adapter | Module | Surface |
| --- | --- | --- |
| [Python SDK](python.md) | `sleuth` | `Sleuth.aask` / `ask` / `asummarize` / `summarize` / `warm_index` |
| [LangChain](frameworks.md#langchain) | `sleuth.langchain` | `SleuthTool`, `SleuthRetriever`, `SleuthCallbackHandler` |
| [Claude Agent SDK](frameworks.md#claude-agent-sdk) | `sleuth.claude_agent` | `SleuthClaudeTool` (events → progress blocks) |
| [MCP server](mcp.md) | `sleuth-mcp` binary | `search`, `summarize` tools over stdio + HTTP |

### Tier 2 — best-effort

| Adapter | Module | Surface |
| --- | --- | --- |
| [LangGraph](frameworks.md#langgraph) | `sleuth.langgraph` | `make_sleuth_node` |
| [LlamaIndex](frameworks.md#llamaindex) | `sleuth.llamaindex` | `SleuthQueryEngine`, `SleuthRetriever` |
| [OpenAI Agents SDK](frameworks.md#openai-agents-sdk) | `sleuth.openai_agents` | `make_sleuth_function_tool` |
| [Pydantic AI](frameworks.md#pydantic-ai) | `sleuth.pydantic_ai` | `make_sleuth_tool`, `SleuthInput` |
| [CrewAI](frameworks.md#crewai) | `sleuth.crewai` | `SleuthCrewAITool` |
| [AutoGen](frameworks.md#autogen) | `sleuth.autogen` | `make_sleuth_autogen_tool`, `register_sleuth_tool` |

---

## Lazy imports and extras

Each adapter imports its framework only when the module is first used — not at `import sleuth`. The `ImportError` is raised at instantiation time with a clear message pointing at the required extra:

```bash
# Install only the adapter(s) you need
pip install 'agent-sleuth[langchain]'
pip install 'agent-sleuth[claude-agent]'
pip install 'agent-sleuth[langgraph]'
pip install 'agent-sleuth[llamaindex]'
pip install 'agent-sleuth[openai-agents]'
pip install 'agent-sleuth[pydantic-ai]'
pip install 'agent-sleuth[crewai]'
pip install 'agent-sleuth[autogen]'
```

The core `agent-sleuth` package has zero framework dependencies.

---

## Which adapter should I use?

- **Starting fresh or framework-agnostic:** use the [Python SDK](python.md) directly — it has the full event stream, Session, structured output, and warm-index.
- **LangChain agent or LCEL chain:** `SleuthTool` (as a `BaseTool`) or `SleuthRetriever` (inside `RetrievalQA`).
- **LangGraph state machine:** `make_sleuth_node` returns a node function that reads from state and writes back the answer.
- **Claude Desktop / any MCP client:** run `sleuth-mcp --transport stdio` and point the client at it — see [MCP server](mcp.md).
