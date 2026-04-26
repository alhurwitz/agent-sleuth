"""Snapshot tests for the full event stream.

These use syrupy to freeze the event sequence.  First run generates
the snapshot under tests/snapshots/__snapshots__/.  Subsequent runs
compare against it.

To update snapshots after an intentional change:
    uv run pytest tests/snapshots/ --snapshot-update
"""

from sleuth._agent import Sleuth
from sleuth.backends.base import Capability
from sleuth.llm.base import Stop, TextDelta
from sleuth.llm.stub import StubLLM
from sleuth.types import Chunk, Source


class FakeWebBackend:
    name = "fake_web"
    capabilities: frozenset[Capability] = frozenset({Capability.WEB})

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        return [
            Chunk(
                text="Python was created by Guido van Rossum.",
                source=Source(kind="url", location="https://python.org/history"),
                score=0.99,
            )
        ]


async def test_fast_path_event_stream_snapshot(snapshot):
    """Snapshot the complete event sequence for a fast-path Q&A."""
    stub = StubLLM([[TextDelta("Python was created by Guido van Rossum."), Stop("end_turn")]])
    agent = Sleuth(llm=stub, backends=[FakeWebBackend()])

    events = []
    async for event in agent.aask("who created Python?", depth="fast"):
        events.append(event.type)

    assert events == snapshot
