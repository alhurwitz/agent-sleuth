"""tree-sitter helpers: parse source, expand a hit line to its enclosing node."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import StrEnum

try:
    from tree_sitter import Language, Parser
except ImportError:  # pragma: no cover
    Language = object  # type: ignore[misc,assignment]
    Parser = object  # type: ignore[misc,assignment]

logger = logging.getLogger("sleuth.backends.codesearch.treesitter")

# Node types that count as "enclosing context" — ordered from most-specific to
# least-specific.  The expander climbs until it finds one of these.
_ENCLOSING_TYPES: frozenset[str] = frozenset(
    {
        # Python
        "function_definition",
        "async_function_definition",
        "class_definition",
        # JavaScript / TypeScript
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "class_declaration",
        "class_expression",
        "lexical_declaration",  # const fn = () => {}
    }
)


class SupportedLanguage(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"


@dataclass(frozen=True, slots=True)
class ExpandedNode:
    text: str
    start_byte: int
    end_byte: int
    start_line: int  # 0-based (tree-sitter convention)
    end_line: int  # 0-based, inclusive
    node_type: str  # e.g. "function_definition"


@functools.lru_cache(maxsize=4)
def _get_language(lang: SupportedLanguage) -> Language:
    """Return a cached tree-sitter Language object for *lang*."""
    # Import lazily so that missing grammars only fail when actually used.
    if lang == SupportedLanguage.PYTHON:
        import tree_sitter_python as tspython

        return Language(tspython.language())
    if lang == SupportedLanguage.JAVASCRIPT:
        import tree_sitter_javascript as tsjs

        return Language(tsjs.language())
    if lang == SupportedLanguage.TYPESCRIPT:
        import tree_sitter_typescript as tsts

        return Language(tsts.language_typescript())
    raise ValueError(f"Unsupported language: {lang}")


def expand_hit_to_node(
    *,
    source: str,
    line_number: int,  # 1-based (matches ripgrep output)
    language: SupportedLanguage,
) -> ExpandedNode | None:
    """Expand a ripgrep hit line to the innermost enclosing function/class node.

    Returns ``None`` if the hit sits at module level with no enclosing context.
    """
    ts_lang = _get_language(language)
    parser = Parser(ts_lang)

    source_bytes = source.encode()
    tree = parser.parse(source_bytes)

    # tree-sitter lines are 0-based; ripgrep is 1-based
    target_row = line_number - 1

    # Walk down to the leaf node at *target_row*
    node = tree.root_node
    while True:
        child_at_row = None
        for child in node.children:
            if child.start_point[0] <= target_row <= child.end_point[0]:
                child_at_row = child
                break
        if child_at_row is None or child_at_row == node:
            break
        node = child_at_row

    # Climb up looking for an enclosing structural node
    candidate = node
    while candidate is not None:
        if candidate.type in _ENCLOSING_TYPES:
            text = source_bytes[candidate.start_byte : candidate.end_byte].decode(errors="replace")
            return ExpandedNode(
                text=text,
                start_byte=candidate.start_byte,
                end_byte=candidate.end_byte,
                start_line=candidate.start_point[0],
                end_line=candidate.end_point[0],
                node_type=candidate.type,
            )
        candidate = candidate.parent  # type: ignore[assignment]

    return None


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

_EXT_TO_LANG: dict[str, SupportedLanguage] = {
    ".py": SupportedLanguage.PYTHON,
    ".js": SupportedLanguage.JAVASCRIPT,
    ".mjs": SupportedLanguage.JAVASCRIPT,
    ".cjs": SupportedLanguage.JAVASCRIPT,
    ".ts": SupportedLanguage.TYPESCRIPT,
    ".tsx": SupportedLanguage.TYPESCRIPT,
}


def language_for_path(path: str) -> SupportedLanguage | None:
    """Return the SupportedLanguage for *path*, or None if unsupported."""
    from pathlib import Path as _Path

    return _EXT_TO_LANG.get(_Path(path).suffix.lower())


# ---------------------------------------------------------------------------
# Hierarchical summary helpers
# ---------------------------------------------------------------------------


@dataclass
class HierarchyNode:
    name: str
    node_type: str  # e.g. "class_definition", "function_definition"
    source_text: str
    start_line: int  # 0-based
    end_line: int  # 0-based, inclusive
    children: list[HierarchyNode] = dc_field(default_factory=list)
    summary: str | None = None  # filled in later by CodeSearch indexer


def build_hierarchy(source: str, language: SupportedLanguage) -> list[HierarchyNode]:
    """Return the top-level HierarchyNode list for *source*.

    Children of class nodes are nested inside their parent ``HierarchyNode``.
    Module-level functions and classes are top-level entries.
    """
    ts_lang = _get_language(language)
    parser = Parser(ts_lang)
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)

    return _collect_nodes(tree.root_node, source_bytes)


def _collect_nodes(node: object, source_bytes: bytes) -> list[HierarchyNode]:
    """Recursively collect enclosing-type nodes, nesting children inside parents."""
    results: list[HierarchyNode] = []
    for child in node.children:  # type: ignore[attr-defined]
        if child.type in _ENCLOSING_TYPES:
            name = _node_name(child, source_bytes)
            text = source_bytes[child.start_byte : child.end_byte].decode(errors="replace")
            h_node = HierarchyNode(
                name=name,
                node_type=child.type,
                source_text=text,
                start_line=child.start_point[0],
                end_line=child.end_point[0],
                children=_collect_nodes(child, source_bytes),
            )
            results.append(h_node)
        else:
            results.extend(_collect_nodes(child, source_bytes))
    return results


def _node_name(node: object, source_bytes: bytes) -> str:
    for child in node.children:  # type: ignore[attr-defined]
        if child.type in ("identifier", "name", "property_identifier"):
            return source_bytes[child.start_byte : child.end_byte].decode(errors="replace")
    return "<anonymous>"
