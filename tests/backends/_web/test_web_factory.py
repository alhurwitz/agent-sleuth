"""Tests for WebBackend factory and public re-exports."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from sleuth.backends.web import (
    BraveBackend,
    ExaBackend,
    SerpAPIBackend,
    TavilyBackend,
    WebBackend,
)


@pytest.mark.unit
def test_web_backend_factory_tavily():
    backend = WebBackend(provider="tavily", api_key="t-key")  # pragma: allowlist secret
    assert isinstance(backend, TavilyBackend)
    assert backend.name == "tavily"


@pytest.mark.unit
def test_web_backend_factory_exa():
    fake_exa = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "exa_py", fake_exa)
        backend = WebBackend(provider="exa", api_key="e-key")  # pragma: allowlist secret
    assert isinstance(backend, ExaBackend)
    assert backend.name == "exa"


@pytest.mark.unit
def test_web_backend_factory_brave():
    backend = WebBackend(provider="brave", api_key="b-key")  # pragma: allowlist secret
    assert isinstance(backend, BraveBackend)
    assert backend.name == "brave"


@pytest.mark.unit
def test_web_backend_factory_serpapi():
    backend = WebBackend(provider="serpapi", api_key="s-key")  # pragma: allowlist secret
    assert isinstance(backend, SerpAPIBackend)
    assert backend.name == "serpapi"


@pytest.mark.unit
def test_web_backend_factory_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        WebBackend(provider="unknown-provider", api_key="key")  # pragma: allowlist secret


@pytest.mark.unit
def test_web_backend_factory_passes_kwargs():
    """Extra kwargs (e.g. fetch=True) are forwarded to the provider class."""
    backend = WebBackend(
        provider="tavily",
        api_key="t-key",  # pragma: allowlist secret
        fetch=True,
        fetch_top_n=5,
    )
    assert isinstance(backend, TavilyBackend)
    assert backend._fetch is True
    assert backend._fetch_top_n == 5


@pytest.mark.unit
def test_web_backend_factory_returns_backend_protocol():
    """Factory output satisfies the Backend protocol (has name + capabilities + search)."""
    backend = WebBackend(provider="tavily", api_key="t-key")  # pragma: allowlist secret

    assert isinstance(backend.name, str)
    assert isinstance(backend.capabilities, frozenset)
    assert callable(backend.search)


@pytest.mark.unit
def test_per_provider_classes_exported_from_web_module():
    """All four classes are importable from sleuth.backends.web."""
    from sleuth.backends.web import (
        BraveBackend,
        ExaBackend,
        SerpAPIBackend,
        TavilyBackend,
    )

    assert TavilyBackend is not None
    assert ExaBackend is not None
    assert BraveBackend is not None
    assert SerpAPIBackend is not None
