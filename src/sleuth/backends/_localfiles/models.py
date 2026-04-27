"""Internal dataclasses for the LocalFiles index tree.

These are hot-path structs — plain dataclasses, not Pydantic models.
Public output types (Chunk, Source) come from sleuth.types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeKind(StrEnum):
    ROOT = "root"
    SECTION = "section"
    SUBSECTION = "subsection"
    LEAF = "leaf"


@dataclass
class IndexNode:
    """One node in the per-document (or per-corpus) tree-of-contents."""

    id: str  # "<rel_path>::<seq>" — stable within a tree version
    title: str  # heading text or file name for root nodes
    kind: NodeKind
    summary: str | None  # LLM-written summary; None until indexer runs
    text: str | None  # leaf chunk text; None for branch nodes
    source_path: str  # absolute file path
    source_section: str  # heading breadcrumb e.g. "# Intro > ## Overview"
    page_or_line: int  # PDF page (1-indexed) or file line (1-indexed)
    children: list[IndexNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def toc_line(self, indent: int = 0) -> str:
        """One-line representation for the navigator LLM's TOC view."""
        prefix = "  " * indent
        summary_fragment = f" — {self.summary}" if self.summary else ""
        return f"{prefix}[{self.id}] {self.title}{summary_fragment}"

    def all_leaves(self) -> list[IndexNode]:
        """DFS collection of all leaf descendants (or self if already a leaf)."""
        if self.is_leaf:
            return [self]
        result: list[IndexNode] = []
        for child in self.children:
            result.extend(child.all_leaves())
        return result


@dataclass
class IndexTree:
    """The full tree-of-contents for a corpus (possibly many files)."""

    corpus_path: str  # absolute path to the indexed directory
    version: str  # hash of (sorted file paths + mtimes)
    nodes: list[IndexNode]  # top-level nodes (one per indexed file, typically)

    def toc_text(self) -> str:
        """Compact TOC string fed to the navigator LLM. Stays under ~5 KB."""
        lines: list[str] = []
        for node in self.nodes:
            lines.append(node.toc_line(indent=0))
            for child in node.children:
                lines.append(child.toc_line(indent=1))
                for grandchild in child.children:
                    lines.append(grandchild.toc_line(indent=2))
                    # stop at depth 2 to keep TOC compact; navigator descends further
        return "\n".join(lines)

    def find_node(self, node_id: str) -> IndexNode | None:
        """BFS lookup by node id."""
        queue = list(self.nodes)
        while queue:
            node = queue.pop(0)
            if node.id == node_id:
                return node
            queue.extend(node.children)
        return None


@dataclass
class ParsedDoc:
    """Intermediate output of a format parser before tree-building."""

    path: str  # absolute file path
    title: str  # inferred doc title
    sections: list[ParsedSection]  # flat ordered list of sections


@dataclass
class ParsedSection:
    """One heading + its text content from a parsed document."""

    heading: str  # heading text (empty string for pre-heading content)
    level: int  # 1 = h1/# , 2 = h2/## , etc.; 0 = preamble
    text: str  # body text under this heading
    page_or_line: int  # source location
