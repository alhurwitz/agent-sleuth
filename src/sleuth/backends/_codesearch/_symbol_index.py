"""SQLite-backed symbol-definition index with incremental update."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from sleuth.backends._codesearch._treesitter import (
    _ENCLOSING_TYPES,
    SupportedLanguage,
    _get_language,
    language_for_path,
)

logger = logging.getLogger("sleuth.backends.codesearch.symbol_index")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_state (
    file_path TEXT PRIMARY KEY,
    mtime     REAL NOT NULL,
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS symbols (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    node_type   TEXT NOT NULL,
    FOREIGN KEY (file_path) REFERENCES file_state(file_path)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_symbol_name ON symbols(symbol_name);
"""

_SUPPORTED_EXTS = frozenset({".py", ".js", ".mjs", ".cjs", ".ts", ".tsx"})


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    symbol_name: str
    file_path: Path
    line_number: int  # 1-based
    node_type: str


class SymbolIndex:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def _connect(self) -> aiosqlite.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(self._db_path))
        conn.row_factory = aiosqlite.Row
        await conn.executescript(_SCHEMA)
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.commit()
        return conn

    async def update(self, root: Path) -> None:
        """Scan *root* and update the index for any new or changed files."""
        conn = await self._connect()
        try:
            for file_path in _walk_source_files(root):
                await self._update_file(conn, file_path)
            await conn.commit()
        finally:
            await conn.close()

    async def _update_file(self, conn: aiosqlite.Connection, file_path: Path) -> None:
        # Use run_in_executor for synchronous IO to avoid blocking the event loop
        import asyncio

        loop = asyncio.get_running_loop()
        stat, content = await loop.run_in_executor(
            None, lambda: (file_path.stat(), file_path.read_bytes())
        )
        mtime = stat.st_mtime
        content_hash = hashlib.sha256(content).hexdigest()

        async with conn.execute(
            "SELECT mtime, content_hash FROM file_state WHERE file_path = ?",
            (str(file_path),),
        ) as cur:
            row = await cur.fetchone()

        if row is not None and row["mtime"] == mtime and row["content_hash"] == content_hash:
            return  # unchanged — skip

        # Remove stale data for this file
        await conn.execute("DELETE FROM symbols WHERE file_path = ?", (str(file_path),))
        await conn.execute("DELETE FROM file_state WHERE file_path = ?", (str(file_path),))

        lang = language_for_path(str(file_path))
        if lang is None:
            return

        # Insert file_state first (symbols FK references it)
        await conn.execute(
            "INSERT INTO file_state (file_path, mtime, content_hash) VALUES (?,?,?)",
            (str(file_path), mtime, content_hash),
        )

        symbols = _extract_symbols(content.decode(errors="replace"), lang)
        _INSERT_SYM = (
            "INSERT INTO symbols (file_path, symbol_name, line_number, node_type) VALUES (?,?,?,?)"
        )
        for sym_name, line_no, node_type in symbols:
            await conn.execute(
                _INSERT_SYM,
                (str(file_path), sym_name, line_no, node_type),
            )

    async def lookup(self, symbol_name: str) -> list[SymbolRecord]:
        """Return all symbol definitions matching *symbol_name* (exact match)."""
        conn = await self._connect()
        try:
            _SELECT_SYM = (
                "SELECT file_path, symbol_name, line_number, node_type"
                " FROM symbols WHERE symbol_name = ?"
            )
            async with conn.execute(_SELECT_SYM, (symbol_name,)) as cur:
                rows = await cur.fetchall()
        finally:
            await conn.close()

        return [
            SymbolRecord(
                symbol_name=row["symbol_name"],
                file_path=Path(row["file_path"]),
                line_number=row["line_number"],
                node_type=row["node_type"],
            )
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_source_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in _SUPPORTED_EXTS and ".sleuth" not in p.parts
    ]


def _extract_symbols(source: str, lang: SupportedLanguage) -> list[tuple[str, int, str]]:
    """Return [(symbol_name, 1-based_line, node_type)] for top-/class-level defs."""
    from tree_sitter import Parser

    ts_lang = _get_language(lang)
    parser = Parser(ts_lang)
    tree = parser.parse(source.encode())

    results: list[tuple[str, int, str]] = []
    _walk_for_symbols(tree.root_node, source.encode(), results)
    return results


def _walk_for_symbols(
    node: object,  # tree_sitter.Node
    source_bytes: bytes,
    out: list[tuple[str, int, str]],
) -> None:
    if node.type in _ENCLOSING_TYPES:  # type: ignore[attr-defined]
        # The first named child that is an identifier is the name
        for child in node.children:  # type: ignore[attr-defined]
            if child.type == "identifier" or child.type == "name":
                name = source_bytes[child.start_byte : child.end_byte].decode(errors="replace")
                line_no = node.start_point[0] + 1  # type: ignore[attr-defined]  # 0-based → 1-based
                out.append((name, line_no, node.type))  # type: ignore[attr-defined]
                break

    for child in node.children:  # type: ignore[attr-defined]
        _walk_for_symbols(child, source_bytes, out)
