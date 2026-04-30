# MCP server

`sleuth-mcp` exposes Sleuth's `search` and `summarize` capabilities as MCP tools over stdio or HTTP. Configure backends and LLM server-side in a TOML file — MCP clients need no Sleuth-specific code.

---

## Install

```bash
pip install 'agent-sleuth[mcp]'
```

---

## CLI usage

```bash
# stdio (for Claude Desktop and most MCP clients)
sleuth-mcp --transport stdio

# HTTP (for remote or browser-based clients)
sleuth-mcp --transport http --host 127.0.0.1 --port 4737

# Explicit config file
sleuth-mcp --config /path/to/mcp.toml --transport stdio
```

| Flag | Default | Description |
| --- | --- | --- |
| `--config` | XDG lookup (see below) | Path to the TOML config file |
| `--transport` | `stdio` | `stdio` or `http` |
| `--host` | `127.0.0.1` (or from config) | HTTP bind host |
| `--port` | `4737` (or from config) | HTTP bind port |

---

## Config file lookup order

1. `--config <path>` (explicit CLI flag)
2. `$XDG_CONFIG_HOME/sleuth/mcp.toml`
3. `~/.config/sleuth/mcp.toml`

---

## TOML config schema

```toml
# ~/.config/sleuth/mcp.toml

[llm]
name = "anthropic:claude-sonnet-4-6"
# name format: "anthropic:<model>" | "openai:<model>" | "stub"

[[backends]]
type = "web"
provider = "tavily"           # "tavily" | "exa" | "brave" | "serpapi"
api_key_env = "TAVILY_API_KEY"  # env var name containing the API key

[[backends]]
type = "localfiles"
path = "/var/data/docs"

[[backends]]
type = "codesearch"
path = "/var/data/src"

[server]
host = "127.0.0.1"    # default; overridable by --host
port = 4737           # default; overridable by --port
default_depth = "auto"  # "auto" | "fast" | "deep"
```

The `[server]` section is optional. The `[[backends]]` array must have at least one entry.

**LLM name formats:**

| Value | Effect |
| --- | --- |
| `"anthropic:<model>"` | Loads `sleuth.llm.anthropic.Anthropic` (requires `agent-sleuth[anthropic]`) |
| `"openai:<model>"` | Loads `sleuth.llm.openai.OpenAI` (requires `agent-sleuth[openai]`) |
| `"stub"` | Loads `StubLLM` — for testing only |

---

## Tool schemas

### `search`

```json
{
  "name": "search",
  "description": "Search across configured backends (web, local files, code) and return a cited answer. Streaming progress is emitted as MCP progress notifications.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "The question or search query." },
      "depth": { "type": "string", "enum": ["auto", "fast", "deep"], "description": "Search depth. Default: auto (router decides)." },
      "schema": { "type": "object", "description": "Optional JSON Schema for structured output in result.data." }
    },
    "required": ["query"]
  }
}
```

Returns a JSON object:
```json
{
  "type": "result",
  "text": "...",
  "citations": [{"kind": "url", "location": "https://...", "title": "..."}],
  "stats": {"latency_ms": 423, "tokens_in": 12, "tokens_out": 87}
}
```

### `summarize`

```json
{
  "name": "summarize",
  "description": "Summarize a URL, file path, or document corpus. Returns a cited summary at the requested length.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "target": { "type": "string", "description": "URL, file path, or free-text query to summarize." },
      "length": { "type": "string", "enum": ["brief", "standard", "thorough"], "description": "Summary length. Default: standard." }
    },
    "required": ["target"]
  }
}
```

---

## Event → progress notification mapping

Sleuth events are mapped to MCP progress notifications during streaming. `DoneEvent` is suppressed (the caller reads the final `Result` directly).

| Sleuth event | Progress message format |
| --- | --- |
| `RouteEvent` | `[route] depth=<depth>: <reason>` |
| `PlanEvent` | `[plan] <N> step(s) planned` |
| `SearchEvent` | `[search:<backend>] <query>` |
| `FetchEvent` | `[fetch] <url> -> <status>` |
| `ThinkingEvent` | `[thinking] <text>` |
| `TokenEvent` | `<text>` (raw token) |
| `CitationEvent` | `[citation <index>] <location>` |
| `CacheHitEvent` | `[cache_hit:<kind>] <key>` |

---

## Wiring to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent on your platform:

```json
{
  "mcpServers": {
    "sleuth": {
      "command": "sleuth-mcp",
      "args": ["--transport", "stdio"],
      "env": {
        "TAVILY_API_KEY": "tvly-..."
      }
    }
  }
}
```

Claude Desktop will launch `sleuth-mcp` on startup and expose the `search` and `summarize` tools to Claude.

!!! tip "Config file location"
    If your config file is not at the default XDG path, add `--config /path/to/mcp.toml` to `args`.

---

## HTTP transport

The HTTP transport uses FastMCP's streamable-HTTP app served by uvicorn:

```bash
sleuth-mcp --transport http --host 0.0.0.0 --port 4737
```

Connect any MCP-over-HTTP client to `http://<host>:<port>/`. The server handles multiple concurrent sessions.
