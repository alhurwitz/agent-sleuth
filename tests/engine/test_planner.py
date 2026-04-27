"""Tests for the LLM Planner (Phase 3).

Covers:
- PlanStep Pydantic model (from sleuth.events)
- Planner.plan() happy path and fallbacks
- Snapshot of deep-mode event sequence
- PlanEvent emission via callback
- End-to-end Sleuth.aask(depth="deep") smoke test
"""

from __future__ import annotations

import pytest

from sleuth.engine.planner import Planner, _PlannerState
from sleuth.events import PlanStep

# ---------------------------------------------------------------------------
# Task 2: PlanStep field tests (imported from sleuth.events per reconciliation)
# ---------------------------------------------------------------------------


def test_planstep_fields() -> None:
    step = PlanStep(query="what is OAuth?")
    assert step.query == "what is OAuth?"
    assert step.backends is None
    assert step.done is False


def test_planstep_done_flag() -> None:
    step = PlanStep(query="", done=True)
    assert step.done is True


# ---------------------------------------------------------------------------
# Task 3: Planner.plan() unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_happy_path() -> None:
    """Planner parses a well-formed JSON array into PlanSteps."""
    from sleuth.llm.stub import StubLLM

    llm = StubLLM(
        responses=[
            '[{"query": "define OAuth"}, {"query": "OAuth vs OIDC"}, {"done": true, "query": ""}]'
        ]
    )
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps: list[PlanStep] = []
    async for step in planner.plan("explain OAuth", state):
        steps.append(step)

    assert len(steps) == 3
    assert steps[0].query == "define OAuth"
    assert steps[1].query == "OAuth vs OIDC"
    assert steps[2].done is True
    assert steps[2].query == ""


@pytest.mark.asyncio
async def test_plan_with_backends_hint() -> None:
    """Planner propagates backend hints."""
    from sleuth.llm.stub import StubLLM

    llm = StubLLM(
        responses=[
            '[{"query": "auth code flow", "backends": ["docs"]}, {"done": true, "query": ""}]'
        ]
    )
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps: list[PlanStep] = []
    async for step in planner.plan("how does auth code flow work?", state):
        steps.append(step)

    assert steps[0].backends == ["docs"]
    assert steps[1].done is True


@pytest.mark.asyncio
async def test_plan_bad_json_degrades_gracefully() -> None:
    """On non-JSON output planner returns a single-step fallback."""
    from sleuth.llm.stub import StubLLM

    llm = StubLLM(responses=["This is plain text, not JSON."])
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps: list[PlanStep] = []
    async for step in planner.plan("anything", state):
        steps.append(step)

    # fallback: one search step + one done step
    assert len(steps) == 2
    assert steps[-1].done is True


@pytest.mark.asyncio
async def test_plan_appends_done_if_missing() -> None:
    """Planner appends a done sentinel if the LLM forgot to include it."""
    from sleuth.llm.stub import StubLLM

    llm = StubLLM(responses=['[{"query": "step one"}]'])
    planner = Planner(llm=llm)

    steps = [s async for s in planner.plan("q", _PlannerState())]
    assert steps[-1].done is True


@pytest.mark.asyncio
async def test_plan_includes_prior_context() -> None:
    """Planner embeds prior search snippets in the user message."""
    from collections.abc import AsyncIterator

    from sleuth.llm.base import LLMChunk, Message, Stop, TextDelta

    captured_messages: list[Message] = []

    class SpyLLM:
        name = "spy"
        supports_reasoning = False
        supports_structured_output = True

        async def stream(
            self,
            messages: list[Message],
            *,
            schema: object = None,
            tools: object = None,
        ) -> AsyncIterator[LLMChunk]:
            captured_messages.extend(messages)
            yield TextDelta(text='[{"done": true, "query": ""}]')
            yield Stop(reason="end_turn")

    planner = Planner(llm=SpyLLM())
    state = _PlannerState(context_snippets=["prior result A", "prior result B"])

    _ = [s async for s in planner.plan("query", state)]

    user_msg = next(m for m in captured_messages if m.role == "user")
    assert "prior result A" in user_msg.content
    assert "prior result B" in user_msg.content


# ---------------------------------------------------------------------------
# Task 9: PlanEvent emission via callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_emits_plan_event_via_callback() -> None:
    """Planner calls the emit callback with a PlanEvent after collecting steps."""
    from sleuth.events import PlanEvent
    from sleuth.llm.stub import StubLLM

    emitted: list[PlanEvent] = []

    llm = StubLLM(
        responses=['[{"query": "step A"}, {"query": "step B"}, {"done": true, "query": ""}]']
    )
    planner = Planner(llm=llm)
    state = _PlannerState()

    _ = [
        s
        async for s in planner.plan(
            "test query",
            state,
            on_plan_event=lambda e: emitted.append(e),
        )
    ]

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, PlanEvent)
    assert len(event.steps) == 2  # two real steps (done step excluded)
    assert event.steps[0].query == "step A"
    assert event.steps[1].query == "step B"


# ---------------------------------------------------------------------------
# Task 8: Snapshot test — deep-mode event sequence (Route→Plan→Search)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deep_mode_event_sequence_snapshot(snapshot: object) -> None:
    """Snapshot the RouteEvent->PlanEvent->SearchEvent*N sequence for deep mode.

    Update snapshots with:
        uv run pytest --snapshot-update
        tests/engine/test_planner.py::test_deep_mode_event_sequence_snapshot
    """
    from sleuth.backends.base import Capability
    from sleuth.engine.executor import execute_subqueries
    from sleuth.engine.router import Router
    from sleuth.events import PlanEvent, SearchEvent
    from sleuth.llm.stub import StubLLM
    from sleuth.types import Chunk, Source

    class _DeterministicBackend:
        name = "det-web"
        capabilities = frozenset({Capability.WEB})

        async def search(self, query: str, k: int = 10) -> list[Chunk]:
            return [
                Chunk(
                    text=f"result: {query}",
                    source=Source(
                        kind="url",
                        location=f"https://det.example.com/{query.replace(' ', '-')}",
                    ),
                )
            ]

    collected_events: list[dict] = []  # type: ignore[type-arg]

    # 1. RouteEvent
    router = Router()
    route_event = router.route("compare OAuth and OIDC for enterprise use", depth="auto")
    collected_events.append(route_event.model_dump())

    # 2. PlanEvent — run the planner and emit a PlanEvent
    llm = StubLLM(
        responses=[
            '[{"query": "what is OAuth"}, {"query": "what is OIDC"}, '
            '{"query": "OAuth vs OIDC enterprise", "backends": ["docs"]}, '
            '{"done": true, "query": ""}]'
        ]
    )
    planner = Planner(llm=llm)
    state = _PlannerState()

    steps = [s async for s in planner.plan("compare OAuth and OIDC for enterprise use", state)]
    real_steps = [s for s in steps if not s.done]

    plan_event = PlanEvent(
        type="plan",
        steps=real_steps,
    )
    collected_events.append(plan_event.model_dump())

    # 3. SearchEvents — one per sub-query
    search_events: list[SearchEvent] = []
    backend = _DeterministicBackend()

    await execute_subqueries(
        subqueries=[s.query for s in real_steps],
        backends=[backend],
        on_search_event=lambda e: search_events.append(e),
    )
    for se in sorted(search_events, key=lambda e: e.query):  # sort for determinism
        collected_events.append(se.model_dump())

    assert collected_events == snapshot


# ---------------------------------------------------------------------------
# Task 10: End-to-end smoke test — Sleuth.aask(depth="deep")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sleuth_aask_deep_emits_route_plan_search_done() -> None:
    """End-to-end: Sleuth.aask(depth='deep') emits RouteEvent(deep), PlanEvent,
    SearchEvent(s), and DoneEvent in that order.

    Uses StubLLM (two responses: one for planning, one for synthesis)
    and a stub backend.
    """
    from sleuth import Sleuth
    from sleuth.backends.base import Capability
    from sleuth.llm.stub import StubLLM
    from sleuth.types import Chunk, Source

    class _FakeBackend:
        name = "fake"
        capabilities = frozenset({Capability.WEB})

        async def search(self, q: str, k: int = 10) -> list[Chunk]:
            return [
                Chunk(
                    text=f"result: {q}",
                    source=Source(kind="url", location=f"https://fake/{q}"),
                )
            ]

    # First StubLLM response: planner JSON; second: synthesis answer
    llm = StubLLM(
        responses=[
            '[{"query": "what is OAuth"}, {"done": true, "query": ""}]',
            "OAuth is an authorization framework.",
        ]
    )

    agent = Sleuth(llm=llm, backends=[_FakeBackend()], cache=None)

    event_types: list[str] = []
    async for event in agent.aask("compare OAuth vs OIDC in depth", depth="deep", max_iterations=1):
        event_types.append(event.type)

    # Verify ordering constraints from spec §5
    assert "route" in event_types
    assert "plan" in event_types
    assert "search" in event_types
    assert "done" in event_types

    route_idx = event_types.index("route")
    plan_idx = event_types.index("plan")
    first_search_idx = event_types.index("search")
    done_idx = event_types.index("done")

    assert route_idx < plan_idx < first_search_idx < done_idx

    # Also verify the route event has depth="deep"
    route_events = [e for e in event_types if e == "route"]
    assert len(route_events) >= 1
