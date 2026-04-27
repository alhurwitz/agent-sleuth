"""Builds an IndexTree from a list of ParsedDoc objects.

Calls the indexer LLM once per non-leaf node to generate a summary.
The LLM call uses a simple prompt; the response is the entire TextDelta
stream joined into a string.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sleuth.backends._localfiles.models import IndexNode, IndexTree, NodeKind, ParsedDoc

if TYPE_CHECKING:
    from sleuth.llm.base import LLMClient


_SUMMARY_PROMPT = """\
You are indexing a document for search. Write a one-sentence summary of the \
following section heading and content snippet. Be concise; this summary \
will appear in a table-of-contents used by an AI to decide which sections \
to read.

Section heading: {heading}
Content snippet (first 400 chars): {snippet}

Respond with only the summary sentence."""


async def _llm_summary(llm: LLMClient, heading: str, text: str) -> str:
    """Collect a one-sentence summary from the indexer LLM."""
    from sleuth.llm.base import Message, TextDelta

    snippet = text[:400].replace("\n", " ")
    prompt = _SUMMARY_PROMPT.format(heading=heading, snippet=snippet)
    messages = [Message(role="user", content=prompt)]
    parts: list[str] = []
    async for chunk in llm.stream(messages):
        if isinstance(chunk, TextDelta):
            parts.append(chunk.text)
    return "".join(parts).strip()


def _build_node_tree(doc: ParsedDoc) -> IndexNode:
    """Convert a ParsedDoc flat section list into a tree of IndexNodes.

    Level-1 sections become direct children of the root.
    Level-2+ sections become children of the nearest ancestor with level - 1.
    The root node represents the entire file.
    """
    root_id = f"{doc.path}::root"
    root = IndexNode(
        id=root_id,
        title=doc.title,
        kind=NodeKind.ROOT,
        summary=None,
        text=None,
        source_path=doc.path,
        source_section=doc.title,
        page_or_line=1,
        children=[],
    )

    # Stack tracks (level, node) for the current insertion path
    stack: list[tuple[int, IndexNode]] = [(0, root)]

    for i, section in enumerate(doc.sections):
        level = section.level if section.level > 0 else 1
        node_id = f"{doc.path}::{i}"
        breadcrumb_parts = [n.title for _, n in stack[1:]] + [section.heading]
        source_section = " > ".join(p for p in breadcrumb_parts if p)

        kind = (
            NodeKind.SECTION
            if level == 1
            else (NodeKind.SUBSECTION if level == 2 else NodeKind.LEAF)
        )

        node = IndexNode(
            id=node_id,
            title=section.heading or doc.title,
            kind=kind,
            summary=None,
            text=section.text,
            source_path=doc.path,
            source_section=source_section,
            page_or_line=section.page_or_line,
        )

        # Pop stack until we find a parent at a lower level
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()

        stack[-1][1].children.append(node)
        stack.append((level, node))

    return root


async def _summarize_tree(node: IndexNode, llm: LLMClient) -> None:
    """DFS: assign LLM summaries to all non-leaf nodes, leaves get text as-is."""
    if node.is_leaf:
        return  # leaf already has .text
    # Summarize children first (bottom-up)
    await asyncio.gather(*[_summarize_tree(child, llm) for child in node.children])
    # Build context from children's headings + summaries
    child_context = "; ".join(
        f"{c.title}: {c.summary or c.text or ''}"[:120] for c in node.children
    )
    node.summary = await _llm_summary(llm, node.title, child_context)


async def build_tree(
    docs: list[ParsedDoc],
    *,
    corpus_path: str,
    indexer_llm: LLMClient,
    version: str = "unknown",
) -> IndexTree:
    """Build and summarize an IndexTree from a list of ParsedDocs."""
    root_nodes = [_build_node_tree(doc) for doc in docs]
    # Summarize each doc's root node in parallel
    await asyncio.gather(*[_summarize_tree(node, indexer_llm) for node in root_nodes])
    return IndexTree(corpus_path=corpus_path, version=version, nodes=root_nodes)
