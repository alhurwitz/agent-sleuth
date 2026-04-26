"""Typed event stream for Sleuth runs.

Every run emits an ordered stream of these events. The discriminated
``Event`` union makes it easy to switch on ``event.type`` without
isinstance checks.  Cached runs emit the same events as live runs,
prefixed with ``CacheHitEvent``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from sleuth.types import Depth, RunStats, Source


class PlanStep(BaseModel):
    """One step in a planner-generated search plan."""

    query: str
    backends: list[str] | None = None  # backend names to try; None = router decides
    done: bool = False  # end-of-iteration sentinel (deep mode)


class RouteEvent(BaseModel):
    """Emitted once per run when the router decides the depth."""

    type: Literal["route"]
    depth: Depth
    reason: str


class PlanEvent(BaseModel):
    """Emitted when the planner produces sub-queries (deep mode only)."""

    type: Literal["plan"]
    steps: list[PlanStep]


class SearchEvent(BaseModel):
    """Emitted when a backend search call starts (or errors)."""

    type: Literal["search"]
    backend: str
    query: str
    error: str | None = None  # set when the backend timed out or raised


class FetchEvent(BaseModel):
    """Emitted when a URL is fetched (WebBackend fetch=True mode)."""

    type: Literal["fetch"]
    url: str
    status: int
    error: str | None = None


class ThinkingEvent(BaseModel):
    """Emitted for reasoning tokens when the LLM supports extended thinking."""

    type: Literal["thinking"]
    text: str


class TokenEvent(BaseModel):
    """Emitted for each text token from the synthesizer LLM."""

    type: Literal["token"]
    text: str


class CitationEvent(BaseModel):
    """Emitted as sources resolve during synthesis."""

    type: Literal["citation"]
    index: int
    source: Source


class CacheHitEvent(BaseModel):
    """Emitted before replaying a cached run."""

    type: Literal["cache_hit"]
    kind: str  # "query" | "fetch" | "plan"
    key: str


class DoneEvent(BaseModel):
    """Final event in every run — carries accounting stats."""

    type: Literal["done"]
    stats: RunStats


Event = Annotated[
    RouteEvent
    | PlanEvent
    | SearchEvent
    | FetchEvent
    | ThinkingEvent
    | TokenEvent
    | CitationEvent
    | CacheHitEvent
    | DoneEvent,
    Field(discriminator="type"),
]
