"""StubLLM — deterministic test double for all engine and backend tests.

Never import this in production code.  It is always importable (no extras
required) because all CI needs it without installing a real LLM SDK.

Contract (one response per call, cycling):
    - ``responses`` is a ``Sequence[str | LLMChunk | list[LLMChunk]]`` or a
      ``Callable[[list[Message]], AsyncIterator[LLMChunk]]``.
    - Each call to ``stream()`` advances an internal index modulo
      ``len(responses)`` (cycles).
    - ``str`` item → emits ``[TextDelta(s), Stop("end_turn")]``.
    - ``LLMChunk`` item → emits ``[chunk]`` (single chunk, no Stop appended).
    - ``list[LLMChunk]`` item → emits each chunk in order.
    - Callable → owns the full async-generator response per call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence

from pydantic import BaseModel

from sleuth.llm.base import LLMChunk, Message, Stop, TextDelta, Tool


class StubLLM:
    """Deterministic LLM double that replays scripted responses.

    Pass ``responses`` as:

    * ``Sequence[str | LLMChunk | list[LLMChunk]]`` — each item is one *response*
      (one call to ``stream()``).  Calls cycle through the list round-robin.

      - ``str`` → ``TextDelta(s)`` + ``Stop("end_turn")``.
      - ``LLMChunk`` → emitted verbatim (single chunk, no Stop appended).
      - ``list[LLMChunk]`` → each chunk emitted in order.

    * ``Callable[[list[Message]], AsyncIterator[LLMChunk]]`` — full control;
      the callable owns the stream and is called once per ``stream()`` call.
    """

    name = "stub"
    supports_reasoning = False
    supports_structured_output = True

    def __init__(
        self,
        responses: (
            Sequence[str | LLMChunk | list[LLMChunk]] | Callable[..., AsyncIterator[LLMChunk]]
        ),
    ) -> None:
        self._callable: Callable[..., AsyncIterator[LLMChunk]] | None = None
        self._responses: list[str | LLMChunk | list[LLMChunk]] | None = None
        self._index: int = 0

        if callable(responses) and not isinstance(responses, list | tuple):
            self._callable = responses
        else:
            self._responses = list(responses)

    async def stream(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Yield LLMChunk items for this call (async generator)."""
        if self._callable is not None:
            async for chunk in self._callable(messages):
                yield chunk
            return

        # List-based cycling
        assert self._responses is not None
        item = self._responses[self._index % len(self._responses)]
        self._index += 1

        if isinstance(item, str):
            yield TextDelta(item)
            yield Stop("end_turn")
        elif isinstance(item, list):
            for chunk in item:
                yield chunk
        else:
            # Single LLMChunk — yield verbatim
            yield item
