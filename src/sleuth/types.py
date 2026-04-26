"""Public data shapes used throughout Sleuth.

All types here are Pydantic v2 models (serializable, validated).
Internal-only hot-path structs live in engine/ as plain dataclasses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Depth / Length literals
# ---------------------------------------------------------------------------

Depth = Literal["auto", "fast", "deep"]
Length = Literal["brief", "standard", "thorough"]

# ---------------------------------------------------------------------------
# Source, Chunk
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """A citable location that contributed to an answer."""

    kind: Literal["url", "file", "code"]
    location: str  # URL, absolute file path, or "repo:path:line_range"
    title: str | None = None
    fetched_at: datetime | None = None


class Chunk(BaseModel):
    """A scored text fragment from a backend search result."""

    text: str
    source: Source
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RunStats
# ---------------------------------------------------------------------------


class RunStats(BaseModel):
    """Latency and token accounting emitted in DoneEvent."""

    latency_ms: int
    first_token_ms: int | None  # None on cache hit
    tokens_in: int
    tokens_out: int
    cache_hits: dict[str, int]  # namespace → hit count
    backends_called: list[str]


# ---------------------------------------------------------------------------
# Result[T]
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class Result(BaseModel, Generic[T]):
    """The return value of agent.ask(...).

    ``data`` is populated when ``schema=`` is passed; otherwise ``None``.
    Static typing is preserved via the ``Result[T]`` generic.
    """

    text: str
    citations: list[Source]
    data: T | None = None
    stats: RunStats
