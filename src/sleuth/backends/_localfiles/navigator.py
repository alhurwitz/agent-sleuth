"""LLM-driven branch selection for the LocalFiles query path.

The navigator receives the compact TOC text (1-5 KB) and the user query,
then asks the navigator LLM to pick which top-level branch IDs to descend.
It recursively collects all leaf chunks under selected branches, up to k.

The LLM is asked for structured JSON: {"selected_ids": ["id1", "id2", ...]}.
On any parse failure the navigator falls back to returning all leaves (safe).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from sleuth.backends._localfiles.models import IndexNode, IndexTree
from sleuth.types import Chunk, Source

if TYPE_CHECKING:
    from sleuth.llm.base import LLMClient

logger = logging.getLogger("sleuth.backends.localfiles.navigator")

_NAVIGATE_PROMPT = """\
You are navigating a document index to answer a user query. Below is the \
table of contents for the document corpus. Each line shows [node_id] title — summary.

TABLE OF CONTENTS:
{toc}

USER QUERY: {query}

Select the node IDs whose subtrees are most likely to contain the answer. \
Return ONLY valid JSON in this exact format with no other text:
{{"selected_ids": ["id1", "id2"]}}

Select between 1 and {max_branches} node IDs. Prefer specificity over breadth."""


async def _ask_navigator(llm: LLMClient, toc: str, query: str, max_branches: int) -> list[str]:
    """Call navigator LLM; return list of selected node IDs. Returns [] on failure."""
    from sleuth.llm.base import Message, TextDelta

    prompt = _NAVIGATE_PROMPT.format(toc=toc, query=query, max_branches=max_branches)
    messages = [Message(role="user", content=prompt)]
    parts: list[str] = []
    async for chunk in llm.stream(messages):
        if isinstance(chunk, TextDelta):
            parts.append(chunk.text)
    raw = "".join(parts).strip()
    try:
        # Extract JSON even if LLM wraps it in markdown fences
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        data: dict[str, Any] = json.loads(raw)
        ids = data.get("selected_ids", [])
        return [str(i) for i in ids] if isinstance(ids, list) else []
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Navigator LLM returned invalid JSON; falling back to all leaves.")
        return []


def _collect_leaves(node: IndexNode, depth: int, max_depth: int) -> list[IndexNode]:
    """Recursively collect leaf nodes. Prunes at max_depth."""
    if node.is_leaf:
        return [node]
    if depth >= max_depth:
        # Return a synthetic representation: the node itself treated as a leaf
        # by copying its summary as text (for compacted large corpora).
        node.text = node.summary  # mutate in-place — acceptable for a temp result
        return [node]
    leaves: list[IndexNode] = []
    for child in node.children:
        leaves.extend(_collect_leaves(child, depth + 1, max_depth))
    return leaves


def _node_to_chunk(node: IndexNode) -> Chunk:
    return Chunk(
        text=node.text or node.summary or node.title,
        source=Source(
            kind="file",
            location=node.source_path,
            title=node.source_section,
        ),
        score=None,
        metadata={"node_id": node.id, "page_or_line": node.page_or_line},
    )


async def navigate(
    query: str,
    tree: IndexTree,
    navigator_llm: LLMClient,
    k: int = 10,
    max_branch_descent: int = 3,
) -> list[Chunk]:
    """Main entry point: select relevant branches and return up to k leaf Chunks."""
    toc = tree.toc_text()
    selected_ids = await _ask_navigator(navigator_llm, toc, query, max_branches=k)

    candidate_nodes: list[IndexNode] = []
    if selected_ids:
        for node_id in selected_ids:
            node = tree.find_node(node_id)
            if node is not None:
                candidate_nodes.append(node)

    # Fallback: all top-level nodes
    if not candidate_nodes:
        candidate_nodes = tree.nodes

    leaves: list[IndexNode] = []
    for node in candidate_nodes:
        leaves.extend(_collect_leaves(node, depth=0, max_depth=max_branch_descent))

    return [_node_to_chunk(leaf) for leaf in leaves[:k]]
