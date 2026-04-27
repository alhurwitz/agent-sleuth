"""Tests for sleuth.mcp.config TOML loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from sleuth.errors import ConfigError
from sleuth.mcp.config import SleuthMcpConfig, load_config

_API_KEY_ENV = "TAVILY_API_KEY"  # pragma: allowlist secret

MINIMAL_TOML = textwrap.dedent(f"""\
    [llm]
    name = "stub"

    [[backends]]
    type = "web"
    provider = "tavily"
    api_key_env = "{_API_KEY_ENV}"
""")

FULL_TOML = textwrap.dedent(f"""\
    [llm]
    name = "anthropic:claude-sonnet-4-6"

    [[backends]]
    type = "web"
    provider = "tavily"
    api_key_env = "{_API_KEY_ENV}"

    [[backends]]
    type = "localfiles"
    path = "/var/data/docs"

    [server]
    host = "127.0.0.1"
    port = 9876
    default_depth = "fast"
""")


@pytest.mark.unit
def test_load_minimal(tmp_path: Path) -> None:
    p = tmp_path / "mcp.toml"
    p.write_text(MINIMAL_TOML)
    cfg = load_config(p)
    assert isinstance(cfg, SleuthMcpConfig)
    assert cfg.llm.name == "stub"
    assert len(cfg.backends) == 1
    assert cfg.backends[0].type == "web"


@pytest.mark.unit
def test_load_full(tmp_path: Path) -> None:
    p = tmp_path / "mcp.toml"
    p.write_text(FULL_TOML)
    cfg = load_config(p)
    assert cfg.llm.name == "anthropic:claude-sonnet-4-6"
    assert len(cfg.backends) == 2
    assert cfg.backends[1].type == "localfiles"
    assert cfg.backends[1].path == "/var/data/docs"
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 9876
    assert cfg.server.default_depth == "fast"


@pytest.mark.unit
def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nonexistent.toml")


@pytest.mark.unit
def test_load_invalid_toml(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[[[ broken")
    with pytest.raises(ConfigError, match="parse"):
        load_config(p)


@pytest.mark.unit
def test_missing_llm_section(tmp_path: Path) -> None:
    p = tmp_path / "nollm.toml"
    p.write_text("[[backends]]\ntype = 'web'\nprovider = 'tavily'\napi_key_env = 'X'\n")
    with pytest.raises(ConfigError, match="llm"):
        load_config(p)


@pytest.mark.unit
def test_missing_backends_section(tmp_path: Path) -> None:
    p = tmp_path / "nobacks.toml"
    p.write_text("[llm]\nname = 'stub'\n")
    with pytest.raises(ConfigError, match="backends"):
        load_config(p)


@pytest.mark.unit
def test_default_server_values(tmp_path: Path) -> None:
    p = tmp_path / "mcp.toml"
    p.write_text(MINIMAL_TOML)
    cfg = load_config(p)
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 4737
    assert cfg.server.default_depth == "auto"
