"""LocalFiles backend: PageIndex-style hierarchical tree-of-contents search.

Indexing (lazy by default, eager via warm_index()):
  - Walk the corpus directory, skip excluded patterns.
  - Parse each file with the appropriate format parser.
  - Build a node tree and ask the indexer LLM for branch summaries.
  - Persist under <corpus>/.sleuth/index/<corpus_version>.json.
  - Cache the in-memory tree; re-index only when corpus_version changes.

Query:
  - Load tree (from memory or disk).
  - Give the navigator LLM the compact TOC; it picks branches.
  - Collect leaf chunks under selected branches and return them.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Literal

import pathspec

from sleuth.backends._localfiles.hasher import corpus_version
from sleuth.backends._localfiles.models import IndexNode, IndexTree, ParsedDoc
from sleuth.backends._localfiles.navigator import navigate
from sleuth.backends._localfiles.parsers import get_parser
from sleuth.backends._localfiles.persistence import load_tree, prune_stale, save_tree
from sleuth.backends._localfiles.tree_builder import build_tree
from sleuth.backends.base import Capability
from sleuth.types import Chunk

if TYPE_CHECKING:
    from sleuth.llm.base import LLMClient

logger = logging.getLogger("sleuth.backends.localfiles")

DEFAULT_EXCLUDES: list[str] = [
    ".git/**",
    ".sleuth/**",
    ".venv/**",
    "node_modules/**",
    "dist/**",
    "build/**",
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
]

_INDEXABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".html",
        ".htm",
        ".pdf",
        ".py",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
    }
)


class LocalFiles:
    """PageIndex-style hierarchical backend for local document corpora."""

    name: str = "localfiles"
    capabilities: frozenset[Capability] = frozenset({Capability.DOCS})

    def __init__(
        self,
        path: str | pathlib.Path,
        indexer_llm: LLMClient | None = None,
        navigator_llm: LLMClient | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_branch_descent: int = 3,
        rebuild: Literal["mtime", "hash", "always"] = "mtime",
    ) -> None:
        self._path = pathlib.Path(path).resolve()
        self._indexer_llm = indexer_llm
        self._navigator_llm = navigator_llm
        self._include_patterns: list[str] = include or ["**/*"]
        self._exclude_patterns: list[str] = exclude if exclude is not None else DEFAULT_EXCLUDES
        self._max_branch_descent = max_branch_descent
        self._rebuild = rebuild

        self._tree: IndexTree | None = None  # in-memory cache
        self._tree_version: str | None = None  # corpus version of cached tree
        self._index_dir = self._path / ".sleuth" / "index"

    # ------------------------------------------------------------------
    # Backend protocol
    # ------------------------------------------------------------------

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        """Load or build index, then navigate and return up to k Chunks."""
        tree = await self._ensure_tree()
        nav_llm = self._navigator_llm or self._indexer_llm
        if nav_llm is None:
            raise RuntimeError(
                "LocalFiles requires an LLM (indexer_llm or navigator_llm). "
                "Pass one to LocalFiles(...) or set the Sleuth default LLM."
            )
        return await navigate(
            query=query,
            tree=tree,
            navigator_llm=nav_llm,
            k=k,
            max_branch_descent=self._max_branch_descent,
        )

    # ------------------------------------------------------------------
    # Summarization helper (called by Sleuth._agent when summarize() is used)
    # ------------------------------------------------------------------

    async def _get_summary(
        self,
        target: str,
        length: Literal["brief", "standard", "thorough"] = "standard",
    ) -> str:
        """Return a summary from the indexed tree for a given file path or corpus root.

        - "brief"     → root node summary (one sentence, already cached in the tree).
        - "standard"  → root + level-1 children summaries joined.
        - "thorough"  → all node summaries concatenated (full tree walk).
        """
        tree = await self._ensure_tree()
        root_node = self._find_node_for_target(target, tree)

        nodes_to_summarize: list[IndexNode] = tree.nodes if root_node is None else [root_node]

        parts: list[str] = []
        if length == "brief":
            parts = [n.summary or n.title for n in nodes_to_summarize]
        elif length == "standard":
            for n in nodes_to_summarize:
                parts.append(n.summary or n.title)
                for child in n.children:
                    if child.summary:
                        parts.append(f"  - {child.title}: {child.summary}")
        else:  # thorough

            def _collect(node: IndexNode, depth: int = 0) -> None:
                prefix = "  " * depth
                parts.append(f"{prefix}{node.title}: {node.summary or node.text or ''}")
                for child in node.children:
                    _collect(child, depth + 1)

            for n in nodes_to_summarize:
                _collect(n)

        return "\n".join(part for part in parts if part)

    # ------------------------------------------------------------------
    # Warm-up (eager indexing — called by Sleuth.warm_index())
    # ------------------------------------------------------------------

    async def warm_index(self) -> None:
        """Force (re)indexing. Useful for pre-warming before the first query."""
        self._tree = None  # clear in-memory cache to force rebuild
        await self._ensure_tree()

    # ------------------------------------------------------------------
    # Internal: path resolution (sync, avoids ASYNC240 in async methods)
    # ------------------------------------------------------------------

    def _find_node_for_target(self, target: str, tree: IndexTree) -> IndexNode | None:
        """Return the IndexNode whose source_path matches *target*, or None for the corpus root."""
        target_abs = pathlib.Path(target).resolve()
        if target_abs == self._path:
            return None  # caller should use tree.nodes (whole corpus)
        for node in tree.nodes:
            if pathlib.Path(node.source_path).resolve() == target_abs:
                return node
        return None

    # ------------------------------------------------------------------
    # Internal: index management
    # ------------------------------------------------------------------

    async def _ensure_tree(self) -> IndexTree:
        """Return the in-memory tree, loading from disk or rebuilding as needed."""
        version = corpus_version(str(self._path), mode=self._rebuild)

        if self._tree is not None and self._tree_version == version:
            return self._tree

        # Try loading from disk
        cached = load_tree(
            str(self._path),
            version=version,
            index_dir=str(self._index_dir),
        )
        if cached is not None:
            logger.debug("LocalFiles: loaded index from disk (version=%s)", version)
            self._tree = cached
            self._tree_version = version
            return self._tree

        # Build fresh index
        logger.info("LocalFiles: building index for %s (version=%s)", self._path, version)
        docs = self._collect_docs()
        indexer_llm = self._indexer_llm or self._navigator_llm
        if indexer_llm is None:
            raise RuntimeError(
                "LocalFiles requires an LLM to build the index. "
                "Pass indexer_llm= or navigator_llm= to LocalFiles(...)."
            )
        tree = await build_tree(
            docs,
            corpus_path=str(self._path),
            indexer_llm=indexer_llm,
            version=version,
        )
        save_tree(tree, index_dir=str(self._index_dir))
        prune_stale(str(self._index_dir), keep_version=version)

        self._tree = tree
        self._tree_version = version
        return self._tree

    def _collect_docs(self) -> list[ParsedDoc]:
        """Walk the corpus and parse all matching, non-excluded files."""
        exclude_spec = pathspec.PathSpec.from_lines("gitignore", self._exclude_patterns)
        docs: list[ParsedDoc] = []
        for file in sorted(self._path.rglob("*")):
            if not file.is_file():
                continue
            if file.suffix.lower() not in _INDEXABLE_EXTENSIONS:
                continue
            rel = str(file.relative_to(self._path))
            if exclude_spec.match_file(rel):
                continue
            try:
                parser = get_parser(str(file))
                doc = parser.parse(str(file))
                docs.append(doc)
            except Exception:
                logger.warning("LocalFiles: failed to parse %s — skipping", file, exc_info=True)
        return docs
