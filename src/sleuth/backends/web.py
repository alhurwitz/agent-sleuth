"""WebBackend factory and per-provider class re-exports.

Public symbols:
    WebBackend      — factory function; returns a per-provider Backend instance.
    TavilyBackend   — direct per-provider class (power users).
    ExaBackend      — direct per-provider class (power users).
    BraveBackend    — direct per-provider class (power users).
    SerpAPIBackend  — direct per-provider class (power users).

Spec §7.2 and §15 #4: both the factory and the per-provider classes are public.
"""

from __future__ import annotations

from typing import Any

from sleuth.backends._web.brave import BraveBackend
from sleuth.backends._web.exa import ExaBackend
from sleuth.backends._web.serpapi import SerpAPIBackend
from sleuth.backends._web.tavily import TavilyBackend

__all__ = [
    "BraveBackend",
    "ExaBackend",
    "SerpAPIBackend",
    "TavilyBackend",
    "WebBackend",
]

_PROVIDERS: dict[str, type[Any]] = {
    "tavily": TavilyBackend,
    "exa": ExaBackend,
    "brave": BraveBackend,
    "serpapi": SerpAPIBackend,
}

# Type alias for the union of all provider types
WebBackendType = TavilyBackend | ExaBackend | BraveBackend | SerpAPIBackend


def WebBackend(
    *,
    provider: str,
    api_key: str,
    **kwargs: Any,
) -> WebBackendType:
    """Factory that returns the appropriate per-provider Backend instance.

    Args:
        provider: One of ``"tavily"``, ``"exa"``, ``"brave"``, ``"serpapi"``.
        api_key: API key for the chosen provider.
        **kwargs: Forwarded verbatim to the provider constructor.
            Common options: ``fetch``, ``fetch_top_n``, ``rate_limit``,
            ``max_retries``.

    Returns:
        A Backend-protocol-compliant instance for the requested provider.

    Raises:
        ValueError: If ``provider`` is not one of the supported values.

    Example::

        # Factory usage
        backend = WebBackend(provider="tavily", api_key=os.environ["TAVILY_KEY"])

        # Per-provider class (power user / type checker friendly)
        backend = TavilyBackend(api_key=os.environ["TAVILY_KEY"], fetch=True)
    """
    cls = _PROVIDERS.get(provider)
    if cls is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown provider {provider!r}. Supported: {supported}")
    return cls(api_key=api_key, **kwargs)  # type: ignore[no-any-return]
