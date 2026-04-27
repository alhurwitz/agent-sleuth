"""CodeSearch backend: ripgrep + tree-sitter + optional embedding re-rank."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from sleuth.backends._codesearch._embedder import Embedder
from sleuth.backends._codesearch._ripgrep import run_ripgrep
from sleuth.backends._codesearch._symbol_index import SymbolIndex, SymbolRecord
from sleuth.backends._codesearch._treesitter import expand_hit_to_node, language_for_path
from sleuth.backends.base import Capability
from sleuth.types import Chunk, Source

logger = logging.getLogger("sleuth.backends.codesearch")

# Regex patterns for "where is X defined" style queries
_DEFINITION_QUERY_RE = re.compile(
    r"(?:where\s+is\s+|definition\s+of\s+|find\s+definition\s+of\s+|locate\s+)"
    r"[`'\"]?(\w+)[`'\"]?",
    re.IGNORECASE,
)


class CodeSearch:
    """Two-phase code retrieval backend.

    Phase 1: ripgrep lexical search for matching lines.
    Phase 2: tree-sitter expansion to enclosing function/class context.
    Optional: cosine re-rank via fastembed (default off).

    Symbol index: SQLite table of all definitions, keyed by name.  Queries
    matching the "where is X defined" pattern skip phase 1 entirely.
    """

    name: str = "codesearch"
    capabilities: frozenset[Capability] = frozenset({Capability.CODE})

    def __init__(
        self,
        path: str | Path,
        *,
        rerank: bool = False,
        rerank_model: str = "BAAI/bge-small-en-v1.5",
        rebuild: Literal["mtime", "always"] = "mtime",
    ) -> None:
        self._root = Path(path)
        self._db_path = self._root / ".sleuth" / "symbols.db"
        self._index = SymbolIndex(self._db_path)
        self._embedder = Embedder(rerank=rerank, model_name=rerank_model)
        self._rebuild = rebuild
        self._indexed = False

    async def _ensure_indexed(self) -> None:
        """Run the symbol index update if needed."""
        if not self._indexed or self._rebuild == "always":
            await self._index.update(self._root)
            self._indexed = True

    async def search(self, query: str, k: int = 10) -> list[Chunk]:
        """Return up to *k* Chunk objects relevant to *query*."""
        await self._ensure_indexed()

        # --- Symbol-index shortcut -------------------------------------------
        m = _DEFINITION_QUERY_RE.search(query)
        if m:
            symbol_name = m.group(1)
            records = await self._index.lookup(symbol_name)
            if records:
                chunks = [_record_to_chunk(r) for r in records[:k]]
                return await self._embedder.rerank(query, chunks)

        # --- Two-phase retrieval ---------------------------------------------
        seen: dict[tuple[Path, int], Chunk] = {}  # (path, start_line) → chunk

        async for hit in run_ripgrep(query, self._root, max_count=k * 5):
            if len(seen) >= k * 3:
                break

            lang = language_for_path(str(hit.path))
            if lang is None:
                # Unsupported file type: use raw hit line as chunk text
                key = (hit.path, hit.line_number)
                if key not in seen:
                    chunk = _make_chunk(
                        text=hit.line_text,
                        path=hit.path,
                        start_line=hit.line_number,
                        end_line=hit.line_number,
                    )
                    seen[key] = chunk
                continue

            try:
                source_text = hit.path.read_text(errors="replace")
            except OSError:
                continue

            expanded = expand_hit_to_node(
                source=source_text,
                line_number=hit.line_number,
                language=lang,
            )
            if expanded is not None:
                key = (hit.path, expanded.start_line)
                if key not in seen:
                    seen[key] = _make_chunk(
                        text=expanded.text,
                        path=hit.path,
                        start_line=expanded.start_line + 1,  # → 1-based
                        end_line=expanded.end_line + 1,
                    )
            else:
                key = (hit.path, hit.line_number)
                if key not in seen:
                    seen[key] = _make_chunk(
                        text=hit.line_text,
                        path=hit.path,
                        start_line=hit.line_number,
                        end_line=hit.line_number,
                    )

        chunks = list(seen.values())[:k]
        return await self._embedder.rerank(query, chunks)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    text: str,
    path: Path,
    start_line: int,
    end_line: int,
) -> Chunk:
    location = f"{path}:L{start_line}-L{end_line}"
    return Chunk(
        text=text,
        source=Source(kind="code", location=location, title=None, fetched_at=None),
        score=None,
        metadata={"start_line": start_line, "end_line": end_line},
    )


def _record_to_chunk(record: SymbolRecord) -> Chunk:
    location = f"{record.file_path}:L{record.line_number}-L{record.line_number}"
    try:
        lines = record.file_path.read_text(errors="replace").splitlines(keepends=True)
    except OSError:
        lines = []

    # Return a few lines of context around the symbol definition
    start_idx = max(0, record.line_number - 1)
    snippet_lines = lines[start_idx : start_idx + 20]
    text = "".join(snippet_lines).strip()

    return Chunk(
        text=text or f"# {record.symbol_name} at {location}",
        source=Source(kind="code", location=location, title=record.symbol_name, fetched_at=None),
        score=None,
        metadata={"symbol_name": record.symbol_name, "node_type": record.node_type},
    )
