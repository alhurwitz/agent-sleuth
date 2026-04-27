"""SleuthCallbackHandler — bridges Sleuth events into LangChain's callback system."""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain is not installed. Run: pip install agent-sleuth[langchain]"
    ) from exc

from sleuth.events import (
    CacheHitEvent,
    CitationEvent,
    DoneEvent,
    Event,
    FetchEvent,
    PlanEvent,
    RouteEvent,
    SearchEvent,
    ThinkingEvent,
    TokenEvent,
)


class SleuthCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """Forward Sleuth events into LangChain's BaseCallbackHandler hooks.

    Event mapping:
      SearchEvent  → on_tool_start(serialized={name: backend}, input=query)
      DoneEvent    → on_tool_end(output=stats summary)
      TokenEvent   → on_llm_new_token(token=text)
      ThinkingEvent→ on_llm_new_token(token=text, chunk metadata)
      Others       → on_text(text=repr)

    Usage::

        handler = SleuthCallbackHandler()
        async for event in sleuth_agent.aask(query):
            handler.on_sleuth_event(event)
    """

    def on_sleuth_event(self, event: Event) -> None:
        """Dispatch a Sleuth event to the appropriate LangChain callback."""
        if isinstance(event, SearchEvent):
            self.on_tool_start(
                serialized={"name": f"sleuth:{event.backend}"},
                input_str=event.query,
            )
        elif isinstance(event, DoneEvent):
            summary = (
                f"done; latency={event.stats.latency_ms}ms backends={event.stats.backends_called}"
            )
            self.on_tool_end(output=summary)
        elif isinstance(event, TokenEvent):
            self.on_llm_new_token(token=event.text)
        elif isinstance(event, ThinkingEvent):
            self.on_llm_new_token(token=event.text, chunk={"type": "thinking"})
        elif isinstance(event, RouteEvent | PlanEvent | FetchEvent | CitationEvent | CacheHitEvent):
            self.on_text(text=repr(event))

    # LangChain BaseCallbackHandler requires these to be defined;
    # the base class already provides no-op defaults — we override only for typing.
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        pass  # subclasses override

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        pass  # subclasses override

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass  # subclasses override

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass  # subclasses override
