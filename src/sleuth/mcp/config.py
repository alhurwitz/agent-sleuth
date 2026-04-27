"""TOML configuration loader for sleuth-mcp.

Config path lookup order:
  1. --config <path> CLI flag (passed explicitly to load_config)
  2. $XDG_CONFIG_HOME/sleuth/mcp.toml
  3. ~/.config/sleuth/mcp.toml
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sleuth.errors import ConfigError
from sleuth.types import Depth


@dataclass
class LlmConfig:
    name: str  # e.g. "anthropic:claude-sonnet-4-6" or "stub"


@dataclass
class BackendConfig:
    type: Literal["web", "localfiles", "codesearch"]
    # web fields
    provider: str | None = None
    api_key_env: str | None = None
    # localfiles / codesearch fields
    path: str | None = None


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 4737
    default_depth: Depth = "auto"


@dataclass
class SleuthMcpConfig:
    llm: LlmConfig
    backends: list[BackendConfig]
    server: ServerConfig = field(default_factory=ServerConfig)


def _default_config_path() -> Path:
    """Return the default config path, respecting XDG_CONFIG_HOME."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "sleuth" / "mcp.toml"


def load_config(path: Path | None = None) -> SleuthMcpConfig:
    """Load a TOML config file into SleuthMcpConfig.

    Args:
        path: Explicit path. If None, falls back to XDG / ~/.config lookup.

    Raises:
        ConfigError: If the file is missing, unparseable, or missing required sections.
    """
    resolved = path or _default_config_path()
    if not resolved.exists():
        raise ConfigError(f"Config file not found: {resolved}")

    try:
        with resolved.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Failed to parse config file {resolved}: {exc}") from exc

    if "llm" not in data:
        raise ConfigError("Config missing required [llm] section")
    if "backends" not in data or not data["backends"]:
        raise ConfigError("Config missing required [[backends]] section (must have at least one)")

    llm = LlmConfig(name=data["llm"]["name"])

    backends: list[BackendConfig] = []
    for raw in data["backends"]:
        backends.append(
            BackendConfig(
                type=raw["type"],
                provider=raw.get("provider"),
                api_key_env=raw.get("api_key_env"),
                path=raw.get("path"),
            )
        )

    server_raw = data.get("server", {})
    server = ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=server_raw.get("port", 4737),
        default_depth=server_raw.get("default_depth", "auto"),
    )

    return SleuthMcpConfig(llm=llm, backends=backends, server=server)
