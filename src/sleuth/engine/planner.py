"""LLM Planner for deep-mode query decomposition.

The Planner is a thin async generator that calls the LLM once per reflect
iteration and yields PlanStep objects. The Executor wraps it for speculative
prefetch and the reflect loop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sleuth.events import PlanEvent, PlanStep
from sleuth.llm.base import LLMClient, Message, TextDelta

if TYPE_CHECKING:
    pass

logger = logging.getLogger("sleuth.engine.planner")


@dataclass
class _PlannerState:
    """Mutable state threaded through reflect iterations."""

    iteration: int = 0
    context_snippets: list[str] = field(default_factory=list)


class Planner:
    """Decomposes a query into sub-queries via LLM; supports reflect loop.

    Usage::

        planner = Planner(llm=llm_client)
        async for step in planner.plan(query, state):
            ...  # handle PlanStep; done=True signals end of this iteration
    """

    _SYSTEM_PROMPT = (
        "You are a search planning assistant. Given a user query and optional "
        "prior search results, decompose the query into focused sub-queries that "
        "together answer the original question. Output a JSON array of objects with "
        'keys "query" (string) and optionally "backends" (list of strings: "web", '
        '"docs", "code", "fresh", "private"). When all needed information has been '
        'gathered, include a final object with "done": true and "query": "". '
        "Output only the JSON array, no prose."
    )

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._last_explicitly_done: bool = False

    async def plan(
        self,
        query: str,
        state: _PlannerState,
        *,
        on_plan_event: Callable[[PlanEvent], Any] | None = None,
    ) -> AsyncIterator[PlanStep]:
        """Yield PlanSteps for one reflect iteration.

        Yields PlanSteps as soon as they are parsed. Calls on_plan_event with a
        PlanEvent after all steps are collected (excluding the done sentinel).

        Args:
            query: The original user query.
            state: Mutable reflect-loop state (carries prior search results).
            on_plan_event: Optional callback fired with a PlanEvent after all
                steps are collected from the LLM for this iteration.
        """
        messages: list[Message] = [
            Message(role="system", content=self._SYSTEM_PROMPT),
            Message(
                role="user",
                content=self._build_user_message(query, state),
            ),
        ]

        raw_text = ""
        async for chunk in self._llm.stream(messages):
            if isinstance(chunk, TextDelta):
                raw_text += chunk.text

        steps, self._last_explicitly_done = self._parse_steps(raw_text)

        # Emit PlanEvent with all real steps (exclude done sentinel)
        if on_plan_event is not None:
            real_steps = [s for s in steps if not s.done]
            event = PlanEvent(
                type="plan",
                steps=real_steps,
            )
            on_plan_event(event)

        for step in steps:
            yield step

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, query: str, state: _PlannerState) -> str:
        parts = [f"Original query: {query}"]
        if state.context_snippets:
            joined = "\n---\n".join(state.context_snippets[:5])  # cap context
            parts.append(f"Prior search results (summarised):\n{joined}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_steps(raw: str) -> tuple[list[PlanStep], bool]:
        """Parse LLM JSON output into PlanSteps; degrade gracefully on bad JSON.

        Always ensures the last step has ``done=True`` so the step generator has
        a clean terminator. Returns ``(steps, explicitly_done)`` where
        ``explicitly_done`` is ``True`` only when the LLM itself included a
        ``{"done": true}`` item — as opposed to the auto-appended sentinel.

        The ``reflect_loop`` uses ``explicitly_done`` to decide whether to
        continue iterating or stop.
        """
        raw = raw.strip()
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("expected JSON array")
        except (json.JSONDecodeError, ValueError):
            logger.warning("planner output was not valid JSON; treating as single query: %r", raw)
            return [PlanStep(query=raw or "search"), PlanStep(query="", done=True)], False

        steps: list[PlanStep] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            done = bool(item.get("done", False))
            q = str(item.get("query", ""))
            backends_raw = item.get("backends")
            backends = list(backends_raw) if isinstance(backends_raw, list) else None
            steps.append(PlanStep(query=q, backends=backends, done=done))

        if not steps:
            return [PlanStep(query="search"), PlanStep(query="", done=True)], False

        # Check if the LLM explicitly included a done sentinel
        explicitly_done = steps[-1].done

        if not steps[-1].done:
            # Auto-append a structural terminator
            steps.append(PlanStep(query="", done=True))

        return steps, explicitly_done
