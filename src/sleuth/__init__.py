"""Sleuth — plug-and-play agentic search with reasoning, planning, and observability.

Public surface (conventions §4):

    from sleuth import Sleuth, Session, Result, Source, Chunk
    from sleuth import RouteEvent, PlanEvent, SearchEvent, FetchEvent
    from sleuth import ThinkingEvent, TokenEvent, CitationEvent, CacheHitEvent, DoneEvent, Event
    from sleuth import Depth, Length
    from sleuth.backends import Tavily
"""

from sleuth._agent import Sleuth as Sleuth
from sleuth._version import __version__ as __version__
from sleuth.events import (
    CacheHitEvent as CacheHitEvent,
)
from sleuth.events import (
    CitationEvent as CitationEvent,
)
from sleuth.events import (
    DoneEvent as DoneEvent,
)
from sleuth.events import (
    Event as Event,
)
from sleuth.events import (
    FetchEvent as FetchEvent,
)
from sleuth.events import (
    PlanEvent as PlanEvent,
)
from sleuth.events import (
    RouteEvent as RouteEvent,
)
from sleuth.events import (
    SearchEvent as SearchEvent,
)
from sleuth.events import (
    ThinkingEvent as ThinkingEvent,
)
from sleuth.events import (
    TokenEvent as TokenEvent,
)
from sleuth.memory.session import Session as Session
from sleuth.types import (
    Chunk as Chunk,
)
from sleuth.types import (
    Depth as Depth,
)
from sleuth.types import (
    Length as Length,
)
from sleuth.types import (
    Result as Result,
)
from sleuth.types import (
    RunStats as RunStats,
)
from sleuth.types import (
    Source as Source,
)

__all__ = [
    "CacheHitEvent",
    "Chunk",
    "CitationEvent",
    "Depth",
    "DoneEvent",
    "Event",
    "FetchEvent",
    "Length",
    "PlanEvent",
    "Result",
    "RouteEvent",
    "RunStats",
    "SearchEvent",
    "Session",
    "Sleuth",
    "Source",
    "ThinkingEvent",
    "TokenEvent",
    "__version__",
]
