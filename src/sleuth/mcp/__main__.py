"""CLI entrypoint for sleuth-mcp.

Usage:
    sleuth-mcp [--config PATH] [--transport stdio|http] [--host HOST] [--port PORT]

Config path lookup order:
  1. --config <path>
  2. $XDG_CONFIG_HOME/sleuth/mcp.toml
  3. ~/.config/sleuth/mcp.toml
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from sleuth.errors import ConfigError
from sleuth.mcp.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sleuth-mcp",
        description="Sleuth MCP server — exposes search and summarize tools via MCP protocol.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        type=Path,
        default=None,
        help=(
            "Path to TOML config file. "
            "Defaults to $XDG_CONFIG_HOME/sleuth/mcp.toml or ~/.config/sleuth/mcp.toml."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to use. Default: stdio.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="HTTP transport: bind host. Default: from config or 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP transport: bind port. Default: from config or 4737.",
    )
    return parser


def _build_sleuth(cfg: Any) -> Any:
    """Construct a Sleuth instance from a SleuthMcpConfig.

    Backend wiring:
      - type=web        -> sleuth.backends.web.WebBackend
      - type=localfiles -> sleuth.backends.localfiles.LocalFiles
      - type=codesearch -> sleuth.backends.codesearch.CodeSearch
    LLM wiring:
      - name="anthropic:<model>" -> sleuth.llm.anthropic.Anthropic (requires [anthropic] extra)
      - name="openai:<model>"    -> sleuth.llm.openai.OpenAI (requires [openai] extra)
      - name="stub"              -> sleuth.llm.stub.StubLLM (dev/test only)
    """
    from sleuth._agent import Sleuth

    llm = _build_llm(cfg.llm.name)
    backends = [_build_backend(b) for b in cfg.backends]
    return Sleuth(llm=llm, backends=backends)


def _build_llm(name: str) -> Any:
    if name == "stub":
        from sleuth.llm.stub import StubLLM

        return StubLLM(responses=["(stub response)"])
    elif name.startswith("anthropic:"):
        model = name.split(":", 1)[1]
        from sleuth.llm.anthropic import Anthropic

        return Anthropic(model=model)
    elif name.startswith("openai:"):
        model = name.split(":", 1)[1]
        from sleuth.llm.openai import OpenAI

        return OpenAI(model=model)
    else:
        raise ConfigError(
            f"Unrecognised LLM name {name!r}. "
            "Expected 'stub', 'anthropic:<model>', or 'openai:<model>'."
        )


def _build_backend(b: Any) -> Any:
    from sleuth.mcp.config import BackendConfig

    assert isinstance(b, BackendConfig)
    if b.type == "web":
        import os

        from sleuth.backends.web import WebBackend

        api_key: str = os.environ.get(b.api_key_env or "", "") if b.api_key_env else ""
        return WebBackend(provider=b.provider or "tavily", api_key=api_key)
    elif b.type == "localfiles":
        from sleuth.backends.localfiles import LocalFiles

        if not b.path:
            raise ConfigError("localfiles backend requires a 'path' key")
        return LocalFiles(path=b.path)
    elif b.type == "codesearch":
        from sleuth.backends.codesearch import CodeSearch

        if not b.path:
            raise ConfigError("codesearch backend requires a 'path' key")
        return CodeSearch(path=b.path)
    else:
        raise ConfigError(f"Unknown backend type: {b.type!r}")


async def _run_async(args: argparse.Namespace) -> None:
    from sleuth.mcp.server import SleuthMcpServer, create_http_app, run_stdio

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    sleuth_instance = _build_sleuth(cfg)
    server = SleuthMcpServer(sleuth=sleuth_instance)

    if args.transport == "stdio":
        await run_stdio(server)
    else:
        host = args.host or cfg.server.host
        port = args.port or cfg.server.port
        import uvicorn

        mcp_app = create_http_app(server, host=host, port=port)
        starlette_app = mcp_app.streamable_http_app()
        config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
        uv_server = uvicorn.Server(config)
        await uv_server.serve()


def main() -> None:
    """Entry point registered as sleuth-mcp in pyproject.toml [project.scripts]."""
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()
