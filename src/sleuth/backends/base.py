"""Backend protocol and Capability enum.

Every search backend — built-in or user-supplied — implements the ``Backend``
Protocol.  The engine never does isinstance checks; it relies on structural
typing (enforced by mypy ``--strict``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from sleuth.errors import BackendError, BackendTimeoutError
from sleuth.types import Chunk


class Capability(StrEnum):
    """Tag set that tells the planner which backends are eligible per sub-query."""

    WEB = "web"  # general web search
    DOCS = "docs"  # local document corpora
    CODE = "code"  # source code
    FRESH = "fresh"  # results that reflect "now" (news, prices, status pages)
    PRIVATE = "private"  # auth-gated systems (Notion, Linear, Slack, etc.)


class Backend(Protocol):
    """Structural protocol for all Sleuth backends.

    Attributes:
        name: Short identifier used in ``SearchEvent.backend`` and ``RunStats``.
        capabilities: Frozenset of ``Capability`` values — drives planner routing.
    """

    name: str
    capabilities: frozenset[Capability]

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        """Search this backend and return up to ``k`` ``Chunk`` objects.

        Raises:
            BackendError: On unrecoverable search failure.
            BackendTimeoutError: When the backend's own deadline is exceeded.
        """
        ...


__all__ = ["Backend", "BackendError", "BackendTimeoutError", "Capability"]
