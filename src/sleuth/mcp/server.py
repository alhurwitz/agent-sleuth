"""MCP server core for Sleuth.

Exposes two MCP tools:
  - search(query, depth?, schema?) -> Result
  - summarize(target, length?) -> Result

Sleuth events are mapped to MCP progress notifications:
  RouteEvent, PlanEvent, SearchEvent, FetchEvent, ThinkingEvent, TokenEvent, CitationEvent,
  CacheHitEvent -> progress notification {"type": "progress", "message": <human-readable>}
  DoneEvent     -> suppressed (caller uses the synchronous Result return)

Transport notes:
  - stdio: uses mcp.server.fastmcp.FastMCP.run_stdio_async()
  - HTTP:  uses mcp.server.fastmcp.FastMCP.streamable_http_app() + uvicorn
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sleuth._agent import Sleuth
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
from sleuth.types import Depth, Length

# ---------------------------------------------------------------------------
# Tool schema builders
# ---------------------------------------------------------------------------


def build_search_tool() -> dict[str, Any]:
    """Return the MCP tool descriptor for `search`."""
    return {
        "name": "search",
        "description": (
            "Search across configured backends (web, local files, code) and return "
            "a cited answer. Streaming progress is emitted as MCP progress notifications."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or search query.",
                },
                "depth": {
                    "type": "string",
                    "enum": ["auto", "fast", "deep"],
                    "description": "Search depth. Default: auto (router decides).",
                },
                "schema": {
                    "type": "object",
                    "description": "Optional JSON Schema for structured output in result.data.",
                },
            },
            "required": ["query"],
        },
    }


def build_summarize_tool() -> dict[str, Any]:
    """Return the MCP tool descriptor for `summarize`."""
    return {
        "name": "summarize",
        "description": (
            "Summarize a URL, file path, or document corpus. "
            "Returns a cited summary at the requested length."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "URL, file path, or free-text query to summarize.",
                },
                "length": {
                    "type": "string",
                    "enum": ["brief", "standard", "thorough"],
                    "description": "Summary length. Default: standard.",
                },
            },
            "required": ["target"],
        },
    }


# ---------------------------------------------------------------------------
# Event → MCP progress notification mapping
# ---------------------------------------------------------------------------


async def events_to_notifications(
    events: list[Event] | AsyncIterator[Event],
) -> AsyncIterator[dict[str, Any]]:
    """Yield MCP progress notification dicts for each Sleuth event.

    DoneEvent is suppressed — it becomes the final Result, not a notification.
    All other event types produce a {"type": "progress", "message": ...} dict.
    """
    if isinstance(events, list):

        async def _iter() -> AsyncIterator[Event]:
            for e in events:
                yield e

        source: AsyncIterator[Event] = _iter()
    else:
        source = events

    async for event in source:
        match event:
            case DoneEvent():
                # Suppressed: DoneEvent marks end-of-stream; caller reads Result directly.
                pass
            case RouteEvent(depth=depth, reason=reason):
                yield {"type": "progress", "message": f"[route] depth={depth}: {reason}"}
            case PlanEvent(steps=steps):
                yield {"type": "progress", "message": f"[plan] {len(steps)} step(s) planned"}
            case SearchEvent(backend=backend, query=query):
                yield {"type": "progress", "message": f"[search:{backend}] {query}"}
            case FetchEvent(url=url, status=status):
                yield {"type": "progress", "message": f"[fetch] {url} -> {status}"}
            case ThinkingEvent(text=text):
                yield {"type": "progress", "message": f"[thinking] {text}"}
            case TokenEvent(text=text):
                yield {"type": "progress", "message": text}
            case CitationEvent(index=index, source=source_obj):
                yield {
                    "type": "progress",
                    "message": f"[citation {index}] {source_obj.location}",
                }
            case CacheHitEvent(kind=kind, key=key):
                yield {"type": "progress", "message": f"[cache_hit:{kind}] {key}"}
            case _:
                # Unknown future event types: emit a generic notification
                yield {"type": "progress", "message": f"[event] {event!r}"}


# ---------------------------------------------------------------------------
# SleuthMcpServer
# ---------------------------------------------------------------------------


class SleuthMcpServer:
    """Core MCP server logic — transport-agnostic.

    Wraps a Sleuth instance and handles tool dispatch. Transport adapters
    (stdio, HTTP) call list_tools() and call_tool() and layer MCP framing on top.
    """

    def __init__(self, sleuth: Sleuth) -> None:
        self._sleuth = sleuth
        self._tools: list[dict[str, Any]] = [
            build_search_tool(),
            build_summarize_tool(),
        ]

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool descriptors for all exposed tools."""
        return list(self._tools)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch a tool call and return a structured MCP result dict.

        Returns a dict with keys: type="result", text, citations, stats, [data].
        """
        if name == "search":
            return await self._call_search(arguments)
        elif name == "summarize":
            return await self._call_summarize(arguments)
        else:
            raise ValueError(f"Unknown tool: {name!r}")

    async def _call_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query: str = args["query"]
        depth: Depth = args.get("depth", "auto")
        result = self._sleuth.ask(query, depth=depth)
        return _result_to_mcp(result)

    async def _call_summarize(self, args: dict[str, Any]) -> dict[str, Any]:
        target: str = args["target"]
        length: Length = args.get("length", "standard")
        result = self._sleuth.summarize(target, length=length)
        return _result_to_mcp(result)


def _result_to_mcp(result: Any) -> dict[str, Any]:
    """Convert a sleuth.types.Result into an MCP result dict."""
    payload: dict[str, Any] = {
        "type": "result",
        "text": result.text,
        "citations": [
            {
                "kind": src.kind,
                "location": src.location,
                "title": src.title,
            }
            for src in result.citations
        ],
        "stats": {
            "latency_ms": result.stats.latency_ms,
            "tokens_in": result.stats.tokens_in,
            "tokens_out": result.stats.tokens_out,
        },
    }
    if result.data is not None:
        payload["data"] = (
            result.data.model_dump() if hasattr(result.data, "model_dump") else result.data
        )
    return payload


# ---------------------------------------------------------------------------
# stdio transport entry point
# ---------------------------------------------------------------------------


async def run_stdio(server: SleuthMcpServer) -> None:
    """Run the MCP server over stdio using the official mcp SDK (FastMCP).

    Requires `mcp>=1.0` (agent-sleuth[mcp] extra).
    """
    import json

    from mcp.server.fastmcp import FastMCP

    mcp_app = FastMCP("sleuth-mcp")

    search_desc = build_search_tool()["description"]
    summarize_desc = build_summarize_tool()["description"]

    async def _search(query: str, depth: str = "auto") -> str:
        result = await server.call_tool("search", {"query": query, "depth": depth})
        return json.dumps(result)

    async def _summarize(target: str, length: str = "standard") -> str:
        result = await server.call_tool("summarize", {"target": target, "length": length})
        return json.dumps(result)

    mcp_app.add_tool(_search, name="search", description=search_desc)
    mcp_app.add_tool(_summarize, name="summarize", description=summarize_desc)

    await mcp_app.run_stdio_async()


# ---------------------------------------------------------------------------
# HTTP transport entry point
# ---------------------------------------------------------------------------


def create_http_app(server: SleuthMcpServer, host: str = "127.0.0.1", port: int = 4737) -> Any:
    """Return a configured FastMCP instance for HTTP (streamable-HTTP) transport.

    Bind to host/port in __main__.py via uvicorn using .streamable_http_app().
    """
    import json

    from mcp.server.fastmcp import FastMCP

    mcp_app = FastMCP("sleuth-mcp", host=host, port=port)

    search_desc = build_search_tool()["description"]
    summarize_desc = build_summarize_tool()["description"]

    async def _search(query: str, depth: str = "auto") -> str:
        result = await server.call_tool("search", {"query": query, "depth": depth})
        return json.dumps(result)

    async def _summarize(target: str, length: str = "standard") -> str:
        result = await server.call_tool("summarize", {"target": target, "length": length})
        return json.dumps(result)

    mcp_app.add_tool(_search, name="search", description=search_desc)
    mcp_app.add_tool(_summarize, name="summarize", description=summarize_desc)

    return mcp_app
