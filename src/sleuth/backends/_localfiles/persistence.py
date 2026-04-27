"""Persist and load IndexTree to/from <corpus>/.sleuth/index/<version>.json.

JSON is chosen over SQLite for the index tree because the tree is read
in its entirety at query time (to show the navigator LLM a TOC). A single
JSON file is simpler, faster to read, and avoids the schema migration burden
of SQLite for a blob-style document.

The file naming convention is: <version>.json where version is the 16-char
corpus hash from hasher.corpus_version(). Old versions are NOT auto-cleaned;
the LocalFiles class prunes stale files when rebuilding.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from sleuth.backends._localfiles.models import IndexNode, IndexTree, NodeKind

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _node_to_dict(node: IndexNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "title": node.title,
        "kind": node.kind,
        "summary": node.summary,
        "text": node.text,
        "source_path": node.source_path,
        "source_section": node.source_section,
        "page_or_line": node.page_or_line,
        "metadata": node.metadata,
        "children": [_node_to_dict(c) for c in node.children],
    }


def _node_from_dict(d: dict[str, Any]) -> IndexNode:
    return IndexNode(
        id=d["id"],
        title=d["title"],
        kind=NodeKind(d["kind"]),
        summary=d.get("summary"),
        text=d.get("text"),
        source_path=d["source_path"],
        source_section=d["source_section"],
        page_or_line=d["page_or_line"],
        metadata=d.get("metadata", {}),
        children=[_node_from_dict(c) for c in d.get("children", [])],
    )


def _tree_to_dict(tree: IndexTree) -> dict[str, Any]:
    return {
        "corpus_path": tree.corpus_path,
        "version": tree.version,
        "nodes": [_node_to_dict(n) for n in tree.nodes],
    }


def _tree_from_dict(d: dict[str, Any]) -> IndexTree:
    return IndexTree(
        corpus_path=d["corpus_path"],
        version=d["version"],
        nodes=[_node_from_dict(n) for n in d["nodes"]],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_tree(tree: IndexTree, *, index_dir: str) -> pathlib.Path:
    """Serialize tree to <index_dir>/<version>.json. Creates the directory."""
    dir_path = pathlib.Path(index_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    out = dir_path / f"{tree.version}.json"
    out.write_text(json.dumps(_tree_to_dict(tree), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_tree(corpus_path: str, *, version: str, index_dir: str) -> IndexTree | None:
    """Load tree from <index_dir>/<version>.json. Returns None if not found."""
    path = pathlib.Path(index_dir) / f"{version}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return _tree_from_dict(data)


def prune_stale(index_dir: str, keep_version: str) -> int:
    """Delete all .json files in index_dir except keep_version. Returns count deleted."""
    dir_path = pathlib.Path(index_dir)
    if not dir_path.exists():
        return 0
    count = 0
    for f in dir_path.glob("*.json"):
        if f.stem != keep_version:
            f.unlink()
            count += 1
    return count
