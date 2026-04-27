"""Heuristic depth router — no LLM calls.

Routes each query to a ``Depth`` value by inspecting the query text.
When ``depth`` is already ``"fast"`` or ``"deep"`` the caller's value is passed
through unchanged.  Phase 3 extended the ``Router`` class with ``_is_deep()``
and added a module-level ``route()`` async-generator convenience wrapper.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from sleuth.events import RouteEvent
from sleuth.types import Depth

logger = logging.getLogger("sleuth.engine.router")

# ---------------------------------------------------------------------------
# Heuristic signals
# ---------------------------------------------------------------------------

# Queries shorter than this word count are almost always fast-path questions.
_FAST_WORD_LIMIT = 10

# Keywords that strongly suggest a complex, multi-step query needing planning.
_DEEP_KEYWORDS = re.compile(
    r"\b(compare|tradeoffs?|all the|every|across|between|vs\.?|versus|"
    r"how do|explain|design|rationale|breaking changes?|history of|"
    r"comprehensive|in depth|exhaustive|walk me through)\b",
    re.IGNORECASE,
)

# Simple question starters that almost always resolve in one search pass.
_FAST_STARTS = re.compile(
    r"^(what|who|when|where|define|does|is|are|how many|which)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Phase 3: module-level helpers
# ---------------------------------------------------------------------------


# Additional patterns used only by _is_deep (richer than _DEEP_KEYWORDS to catch
# multi-part queries, research requests, and change-diff patterns).
_DEEP_EXTRA = re.compile(
    r"\b(research\b|all\s+(?:the\s+)?ways\b|each\b.{0,40}\bhandle\b"
    r"|and\b.{0,30}\bwhat\b.{0,30}\bchanged\b"
    r"|differences?\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def _is_deep(query: str) -> bool:
    """Return True if the query heuristically requires deep (multi-step) planning.

    Purely regex/keyword — no LLM calls. Errs on the side of fast when ambiguous.
    Checks the shared ``_DEEP_KEYWORDS`` pattern plus ``_DEEP_EXTRA`` patterns
    for research / multi-part / change-diff constructions.
    """
    return bool(_DEEP_KEYWORDS.search(query)) or bool(_DEEP_EXTRA.search(query))


async def route(query: str, *, depth: Depth = "auto") -> AsyncIterator[RouteEvent]:
    """Module-level async-generator wrapper around ``Router().route()``.

    Yields a single ``RouteEvent``. The async-generator form lets callers use
    ``async for event in route(query)`` uniformly with other event-stream sources.

    Args:
        query: The user's search query.
        depth: ``"auto"`` to run heuristics; ``"fast"`` / ``"deep"`` pass through.
    """
    yield Router().route(query, depth=depth)


class Router:
    """Heuristic depth router.

    Determines whether a query should be answered with a single search fan-out
    (``"fast"``) or a full planning loop (``"deep"``).  No LLM calls are made
    here — this is intentionally cheap and synchronous.

    Phase 3 will extend this to use an LLM classifier for edge cases, but the
    ``route()`` API signature is frozen.
    """

    def route(self, query: str, *, depth: Depth = "auto") -> RouteEvent:
        """Classify a query and return a ``RouteEvent``.

        Args:
            query: The user's search query.
            depth: ``"auto"`` to run heuristics; ``"fast"`` / ``"deep"`` pass through.

        Returns:
            A ``RouteEvent`` with the resolved depth and a short reason string.
        """
        if depth in ("fast", "deep"):
            logger.debug("Router: passthrough depth=%s", depth)
            return RouteEvent(
                type="route",
                depth=depth,
                reason=f"caller-specified depth={depth}",
            )

        resolved, reason = self._classify(query)
        logger.debug("Router: auto → %s (%s)", resolved, reason)
        return RouteEvent(type="route", depth=resolved, reason=reason)

    def _classify(self, query: str) -> tuple[Depth, str]:
        words = query.split()

        # Short queries are almost always fast
        if len(words) <= _FAST_WORD_LIMIT and not _DEEP_KEYWORDS.search(query):
            if _FAST_STARTS.match(query.strip()):
                return "fast", "simple factual question pattern"
            if len(words) <= 5:
                return "fast", "very short query"

        # Explicit complexity signals → deep
        m = _DEEP_KEYWORDS.search(query)
        if m:
            return "deep", f"complexity keyword: {m.group()!r}"

        # Long queries default to deep
        if len(words) > _FAST_WORD_LIMIT:
            return "deep", f"long query ({len(words)} words)"

        return "fast", "no complexity signals detected"
