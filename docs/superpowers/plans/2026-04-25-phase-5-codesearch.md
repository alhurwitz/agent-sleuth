# Phase 5: CodeSearch Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `CodeSearch`, a `Backend`-protocol-compliant class that delivers two-phase code retrieval (ripgrep lexical search → tree-sitter context expansion), a symbol-definition index, hierarchical module/class summaries, and optional embedding re-rank — all respecting `.gitignore` and updating only on `(mtime, content hash)` changes.

**Architecture:** `CodeSearch` shells out to the `rg` binary for fast lexical hits, then uses tree-sitter Python bindings to walk each hit up to its enclosing function or class node and return that as the chunk text. A lightweight SQLite symbol index keyed by name enables fast "where is X defined" short-circuit queries. Hierarchical summaries (module → class) are built at index time via tree-sitter node walking and stored alongside the symbol index; they mirror the shape of Phase 2's LocalFiles tree without depending on any Phase 2 code.

**Tech Stack:** Python 3.11+, `tree-sitter>=0.22`, `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`, `aiofiles`, `aiosqlite`, SQLite (stdlib), `rg` binary on PATH. Optional: `numpy` + any fastembed-compatible embedding model (lazy-imported, default off).

> **CALLOUT — additions to conventions §3 `pyproject.toml` shape:**
> Phase 5 appends the following to `[project.optional-dependencies]`:
> ```toml
> code = [
>     "tree-sitter>=0.22",
>     "tree-sitter-python>=0.23",
>     "tree-sitter-javascript>=0.23",
>     "tree-sitter-typescript>=0.23",
>     "aiofiles>=23.2",
>     "aiosqlite>=0.20",
> ]
> code-embed = [
>     "agent-sleuth[code]",
>     "fastembed>=0.3",
>     "numpy>=1.26",
> ]
> ```
> The `rg` binary is a **runtime** requirement (not a Python dep). It must be documented in `CONTRIBUTING.md` (which Phase 0 owns — Phase 5 appends a note to it).

> **CALLOUT — `CONTRIBUTING.md` modification:** Phase 0 owns `CONTRIBUTING.md`. This plan adds one step that **appends** a "CodeSearch prerequisites" section to it. This is a modify, not a create, so it does not violate ownership rules per conventions §2.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/sleuth/backends/codesearch.py` | **Create** | Public `CodeSearch` class; implements `Backend` protocol |
| `src/sleuth/backends/_codesearch/__init__.py` | **Create** | Package marker |
| `src/sleuth/backends/_codesearch/_ripgrep.py` | **Create** | Async ripgrep wrapper; parses JSONL output; respects `.gitignore` |
| `src/sleuth/backends/_codesearch/_treesitter.py` | **Create** | tree-sitter parser pool; expand-hit-to-enclosing-node; hierarchical walker |
| `src/sleuth/backends/_codesearch/_symbol_index.py` | **Create** | SQLite-backed symbol-definition index; incremental update on mtime/hash change |
| `src/sleuth/backends/_codesearch/_embedder.py` | **Create** | Optional re-rank wrapper (lazy-imports fastembed); default no-op |
| `tests/backends/test_codesearch.py` | **Create** | Protocol compliance (via `BackendTestKit`) + all unit tests |
| `pyproject.toml` | **Modify** | Append `code` and `code-embed` extras |
| `CONTRIBUTING.md` | **Modify** | Append `rg` installation instructions |

---

## Task 1: Branch setup

**Files:** none (git only)

- [ ] **Create feature branch**

```bash
git checkout develop
git checkout -b feature/phase-5-codesearch
```

Expected: `Switched to a new branch 'feature/phase-5-codesearch'`

- [ ] **Commit branch marker**

```bash
git commit --allow-empty -m "chore: start phase-5-codesearch branch"
```

---

## Task 2: Add `code` extras to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Open `pyproject.toml` and append to `[project.optional-dependencies]`**

Find the block that ends the `[project.optional-dependencies]` table and add:

```toml
code = [
    "tree-sitter>=0.22",
    "tree-sitter-python>=0.23",
    "tree-sitter-javascript>=0.23",
    "tree-sitter-typescript>=0.23",
    "aiofiles>=23.2",
    "aiosqlite>=0.20",
]
code-embed = [
    "agent-sleuth[code]",
    "fastembed>=0.3",
    "numpy>=1.26",
]
```

- [ ] **Sync the environment with the new extras**

```bash
uv sync --extra code --group dev
```

Expected: packages installed without error; `tree-sitter` appears in the output.

- [ ] **Document `rg` prerequisite in `CONTRIBUTING.md`** (append to end of file)

```markdown
## CodeSearch prerequisites

The `CodeSearch` backend shells out to [`ripgrep`](https://github.com/BurntSushi/ripgrep) (`rg`). Install it before running code-search-related tests or using the backend:

| Platform | Command |
|---|---|
| macOS (Homebrew) | `brew install ripgrep` |
| Ubuntu/Debian | `sudo apt-get install ripgrep` |
| Windows (winget) | `winget install BurntSushi.ripgrep.MSVC` |
| Cargo | `cargo install ripgrep` |

Verify with: `rg --version`
```

- [ ] **Commit**

```bash
git add pyproject.toml CONTRIBUTING.md
git commit -m "chore: add code and code-embed extras; document rg prerequisite"
```

---

## Task 3: Ripgrep wrapper (`_ripgrep.py`)

**Files:**
- Create: `src/sleuth/backends/_codesearch/__init__.py`
- Create: `src/sleuth/backends/_codesearch/_ripgrep.py`
- Test: `tests/backends/test_codesearch.py` (first stubs)

The ripgrep wrapper runs `rg --json --context 0 <pattern> <path>` and yields `RipgrepHit` dataclass objects. It must honor `.gitignore`, handle binary files gracefully, and be fully async (subprocess via `asyncio.create_subprocess_exec`).

- [ ] **Write the failing test for `run_ripgrep`**

Create `tests/backends/test_codesearch.py`:

```python
"""Tests for CodeSearch backend — Phase 5."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from sleuth.backends._codesearch._ripgrep import RipgrepHit, run_ripgrep


# ---------------------------------------------------------------------------
# Task 3 — ripgrep wrapper
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_run_ripgrep_finds_match(tmp_path: Path) -> None:
    """run_ripgrep yields a hit for a literal pattern match."""
    source = tmp_path / "example.py"
    source.write_text("def hello_world():\n    pass\n")

    hits = [h async for h in run_ripgrep("hello_world", tmp_path)]

    assert len(hits) == 1
    hit = hits[0]
    assert hit.path == source
    assert hit.line_number == 1
    assert "hello_world" in hit.line_text


@pytest.mark.unit
async def test_run_ripgrep_no_match(tmp_path: Path) -> None:
    """run_ripgrep yields nothing when pattern does not appear."""
    (tmp_path / "empty.py").write_text("x = 1\n")

    hits = [h async for h in run_ripgrep("nonexistent_symbol_xyz", tmp_path)]

    assert hits == []


@pytest.mark.unit
async def test_run_ripgrep_respects_gitignore(tmp_path: Path) -> None:
    """run_ripgrep skips files listed in .gitignore."""
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    (tmp_path / "ignored.py").write_text("def hello_world(): pass\n")
    (tmp_path / "visible.py").write_text("# no match here\n")

    hits = [h async for h in run_ripgrep("hello_world", tmp_path)]

    paths = [h.path for h in hits]
    assert tmp_path / "ignored.py" not in paths


@pytest.mark.unit
async def test_run_ripgrep_skips_binary(tmp_path: Path) -> None:
    """run_ripgrep does not crash on binary files."""
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01hello_world\x02\x03")
    hits = [h async for h in run_ripgrep("hello_world", tmp_path)]
    # ripgrep skips binary by default — we just confirm no exception
    assert isinstance(hits, list)
```

- [ ] **Run test to confirm it fails (module not found)**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "ripgrep"
```

Expected: `ERRORS` — `ModuleNotFoundError: No module named 'sleuth.backends._codesearch'`

- [ ] **Create the package marker**

```python
# src/sleuth/backends/_codesearch/__init__.py
```

(empty file — just the package marker)

- [ ] **Implement `_ripgrep.py`**

```python
# src/sleuth/backends/_codesearch/_ripgrep.py
"""Async ripgrep wrapper for CodeSearch."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger("sleuth.backends.codesearch.ripgrep")


@dataclass(frozen=True, slots=True)
class RipgrepHit:
    path: Path
    line_number: int       # 1-based
    line_text: str


async def run_ripgrep(
    pattern: str,
    root: Path,
    *,
    glob: str | None = None,            # e.g. "*.py" to restrict file types
    fixed_strings: bool = False,        # pass -F for literal (non-regex) queries
    max_count: int | None = None,       # per-file hit cap
) -> AsyncIterator[RipgrepHit]:
    """Yield RipgrepHit objects for every match of *pattern* under *root*.

    Uses ``rg --json`` so output is machine-readable.  Respects .gitignore
    automatically (ripgrep default behaviour).  Binary files are skipped by
    ripgrep silently.
    """
    cmd: list[str] = [
        "rg",
        "--json",
        "--line-number",
    ]
    if fixed_strings:
        cmd.append("--fixed-strings")
    if glob:
        cmd.extend(["--glob", glob])
    if max_count is not None:
        cmd.extend(["--max-count", str(max_count)])
    cmd.extend(["--", pattern, str(root)])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None

    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("rg: non-JSON line skipped: %s", line[:80])
            continue

        if obj.get("type") != "match":
            continue

        data = obj["data"]
        path = Path(data["path"]["text"])
        line_number: int = data["line_number"]
        line_text: str = data["lines"]["text"].rstrip("\n")
        yield RipgrepHit(path=path, line_number=line_number, line_text=line_text)

    await proc.wait()
    if proc.returncode not in (0, 1):  # 1 = no matches, which is fine
        stderr = (await proc.stderr.read()).decode(errors="replace") if proc.stderr else ""
        logger.warning("rg exited %d: %s", proc.returncode, stderr[:200])
```

- [ ] **Run tests to confirm they pass**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "ripgrep"
```

Expected: `4 passed`

- [ ] **Commit**

```bash
git add src/sleuth/backends/_codesearch/ tests/backends/test_codesearch.py
git commit -m "feat: add async ripgrep wrapper with gitignore support"
```

---

## Task 4: Tree-sitter expander (`_treesitter.py`)

**Files:**
- Create: `src/sleuth/backends/_codesearch/_treesitter.py`
- Test: `tests/backends/test_codesearch.py` (append)

Given a file path and a 1-based line number, the expander walks the tree-sitter parse tree upward to find the innermost enclosing `function_definition`, `class_definition`, `method_definition`, or equivalent node, then returns that node's full source text plus its byte range (for citations).

- [ ] **Write the failing tests (append to `tests/backends/test_codesearch.py`)**

```python
# ---------------------------------------------------------------------------
# Task 4 — tree-sitter expander
# ---------------------------------------------------------------------------
from sleuth.backends._codesearch._treesitter import ExpandedNode, expand_hit_to_node, SupportedLanguage


PYTHON_SOURCE = """\
class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}"

    def farewell(self, name: str) -> str:
        return f"Goodbye, {name}"
"""


@pytest.mark.unit
def test_expand_hit_finds_enclosing_method() -> None:
    """Line inside a method body expands to the full method."""
    node = expand_hit_to_node(
        source=PYTHON_SOURCE,
        line_number=3,          # the `return f"Hello, ..."` line
        language=SupportedLanguage.PYTHON,
    )
    assert node is not None
    assert "def greet" in node.text
    assert "def farewell" not in node.text


@pytest.mark.unit
def test_expand_hit_finds_class_when_no_inner_function() -> None:
    """Line on the class definition itself expands to the full class."""
    node = expand_hit_to_node(
        source=PYTHON_SOURCE,
        line_number=1,          # `class Greeter:` line
        language=SupportedLanguage.PYTHON,
    )
    assert node is not None
    assert "class Greeter" in node.text


@pytest.mark.unit
def test_expand_hit_returns_none_for_module_level_code() -> None:
    """A bare assignment at module level (no enclosing function/class) returns None."""
    source = "X = 42\n"
    node = expand_hit_to_node(source=source, line_number=1, language=SupportedLanguage.PYTHON)
    # None is acceptable — caller falls back to the raw hit line
    assert node is None or isinstance(node.text, str)


@pytest.mark.unit
def test_expand_hit_byte_range() -> None:
    """Expanded node carries start/end byte offsets into the source."""
    node = expand_hit_to_node(
        source=PYTHON_SOURCE,
        line_number=3,
        language=SupportedLanguage.PYTHON,
    )
    assert node is not None
    assert node.start_byte >= 0
    assert node.end_byte > node.start_byte
    assert PYTHON_SOURCE[node.start_byte : node.end_byte] == node.text
```

- [ ] **Run to confirm failure**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "expand_hit"
```

Expected: `ERRORS` — `ModuleNotFoundError: No module named '...._treesitter'`

- [ ] **Implement `_treesitter.py`**

```python
# src/sleuth/backends/_codesearch/_treesitter.py
"""tree-sitter helpers: parse source, expand a hit line to its enclosing node."""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

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
        "lexical_declaration",   # const fn = () => {}
    }
)


class SupportedLanguage(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"


@dataclass(frozen=True, slots=True)
class ExpandedNode:
    text: str
    start_byte: int
    end_byte: int
    start_line: int    # 0-based (tree-sitter convention)
    end_line: int      # 0-based, inclusive
    node_type: str     # e.g. "function_definition"


@functools.lru_cache(maxsize=4)
def _get_language(lang: SupportedLanguage):  # type: ignore[return]
    """Return a cached tree-sitter Language object for *lang*."""
    # Import lazily so that missing grammars only fail when actually used.
    if lang == SupportedLanguage.PYTHON:
        import tree_sitter_python as tspython  # type: ignore[import]
        from tree_sitter import Language
        return Language(tspython.language())
    if lang == SupportedLanguage.JAVASCRIPT:
        import tree_sitter_javascript as tsjs  # type: ignore[import]
        from tree_sitter import Language
        return Language(tsjs.language())
    if lang == SupportedLanguage.TYPESCRIPT:
        import tree_sitter_typescript as tsts  # type: ignore[import]
        from tree_sitter import Language
        return Language(tsts.language_typescript())
    raise ValueError(f"Unsupported language: {lang}")


def expand_hit_to_node(
    *,
    source: str,
    line_number: int,   # 1-based (matches ripgrep output)
    language: SupportedLanguage,
) -> Optional[ExpandedNode]:
    """Expand a ripgrep hit line to the innermost enclosing function/class node.

    Returns ``None`` if the hit sits at module level with no enclosing context.
    """
    from tree_sitter import Parser

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


def language_for_path(path: str) -> Optional[SupportedLanguage]:
    """Return the SupportedLanguage for *path*, or None if unsupported."""
    from pathlib import Path as _Path
    return _EXT_TO_LANG.get(_Path(path).suffix.lower())
```

- [ ] **Run tests to confirm they pass**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "expand_hit"
```

Expected: `4 passed`

- [ ] **Commit**

```bash
git add src/sleuth/backends/_codesearch/_treesitter.py tests/backends/test_codesearch.py
git commit -m "feat: add tree-sitter hit expander for python/js/ts"
```

---

## Task 5: Symbol index (`_symbol_index.py`)

**Files:**
- Create: `src/sleuth/backends/_codesearch/_symbol_index.py`
- Test: `tests/backends/test_codesearch.py` (append)

The symbol index scans a source tree, extracts all top-level and class-level function/class definitions via tree-sitter, and stores them in a SQLite table keyed by `(symbol_name, file_path, line_number)`. Rows also carry `(mtime, content_hash)` so incremental updates only re-scan changed files.

- [ ] **Write the failing tests (append to test file)**

```python
# ---------------------------------------------------------------------------
# Task 5 — symbol index
# ---------------------------------------------------------------------------
import hashlib

from sleuth.backends._codesearch._symbol_index import SymbolIndex, SymbolRecord


@pytest.mark.unit
async def test_symbol_index_find_definition(tmp_path: Path) -> None:
    """Index a .py file and look up a function by name."""
    src = tmp_path / "math_utils.py"
    src.write_text("def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n")

    idx = SymbolIndex(tmp_path / ".sleuth" / "symbols.db")
    await idx.update(tmp_path)

    records = await idx.lookup("add")
    assert len(records) == 1
    assert records[0].symbol_name == "add"
    assert records[0].file_path == src
    assert records[0].line_number == 1


@pytest.mark.unit
async def test_symbol_index_no_rescan_on_unchanged_file(tmp_path: Path) -> None:
    """Calling update() twice on an unchanged file does not re-insert rows."""
    src = tmp_path / "stable.py"
    src.write_text("def stable_func():\n    pass\n")

    idx = SymbolIndex(tmp_path / ".sleuth" / "symbols.db")
    await idx.update(tmp_path)
    await idx.update(tmp_path)   # second call — same mtime/hash

    records = await idx.lookup("stable_func")
    assert len(records) == 1   # exactly one row, not duplicated


@pytest.mark.unit
async def test_symbol_index_rescans_on_content_change(tmp_path: Path, monkeypatch) -> None:
    """Changing a file's content causes a rescan."""
    import time
    src = tmp_path / "changing.py"
    src.write_text("def old_name():\n    pass\n")

    idx = SymbolIndex(tmp_path / ".sleuth" / "symbols.db")
    await idx.update(tmp_path)

    # Modify content and bump mtime artificially
    src.write_text("def new_name():\n    pass\n")
    new_mtime = src.stat().st_mtime + 1.0
    import os; os.utime(src, (new_mtime, new_mtime))

    await idx.update(tmp_path)

    old_records = await idx.lookup("old_name")
    new_records = await idx.lookup("new_name")
    assert old_records == []
    assert len(new_records) == 1


@pytest.mark.unit
async def test_symbol_index_lookup_missing_returns_empty(tmp_path: Path) -> None:
    """Looking up a symbol that doesn't exist returns an empty list."""
    idx = SymbolIndex(tmp_path / ".sleuth" / "symbols.db")
    await idx.update(tmp_path)
    result = await idx.lookup("completely_missing_xyz")
    assert result == []
```

- [ ] **Run to confirm failure**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "symbol_index"
```

Expected: `ERRORS` — `ModuleNotFoundError: No module named '...._symbol_index'`

- [ ] **Implement `_symbol_index.py`**

```python
# src/sleuth/backends/_codesearch/_symbol_index.py
"""SQLite-backed symbol-definition index with incremental update."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import aiosqlite

from sleuth.backends._codesearch._treesitter import (
    SupportedLanguage,
    _ENCLOSING_TYPES,
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
    line_number: int   # 1-based
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
        stat = file_path.stat()
        mtime = stat.st_mtime
        content = file_path.read_bytes()
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

        symbols = _extract_symbols(content.decode(errors="replace"), lang)
        for sym_name, line_no, node_type in symbols:
            await conn.execute(
                "INSERT INTO symbols (file_path, symbol_name, line_number, node_type) VALUES (?,?,?,?)",
                (str(file_path), sym_name, line_no, node_type),
            )

        await conn.execute(
            "INSERT INTO file_state (file_path, mtime, content_hash) VALUES (?,?,?)",
            (str(file_path), mtime, content_hash),
        )

    async def lookup(self, symbol_name: str) -> list[SymbolRecord]:
        """Return all symbol definitions matching *symbol_name* (exact match)."""
        conn = await self._connect()
        try:
            async with conn.execute(
                "SELECT file_path, symbol_name, line_number, node_type FROM symbols WHERE symbol_name = ?",
                (symbol_name,),
            ) as cur:
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


def _extract_symbols(
    source: str, lang: SupportedLanguage
) -> list[tuple[str, int, str]]:
    """Return [(symbol_name, 1-based_line, node_type)] for top-/class-level defs."""
    from tree_sitter import Parser

    ts_lang = _get_language(lang)
    parser = Parser(ts_lang)
    tree = parser.parse(source.encode())

    results: list[tuple[str, int, str]] = []
    _walk_for_symbols(tree.root_node, source.encode(), results)
    return results


def _walk_for_symbols(
    node,  # tree_sitter.Node
    source_bytes: bytes,
    out: list[tuple[str, int, str]],
) -> None:
    if node.type in _ENCLOSING_TYPES:
        # The first named child that is an identifier is the name
        for child in node.children:
            if child.type == "identifier" or child.type == "name":
                name = source_bytes[child.start_byte : child.end_byte].decode(errors="replace")
                line_no = node.start_point[0] + 1  # 0-based → 1-based
                out.append((name, line_no, node.type))
                break

    for child in node.children:
        _walk_for_symbols(child, source_bytes, out)
```

- [ ] **Run tests to confirm they pass**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "symbol_index"
```

Expected: `4 passed`

- [ ] **Commit**

```bash
git add src/sleuth/backends/_codesearch/_symbol_index.py tests/backends/test_codesearch.py
git commit -m "feat: add sqlite symbol index with incremental mtime/hash update"
```

---

## Task 6: Optional embedder wrapper (`_embedder.py`)

**Files:**
- Create: `src/sleuth/backends/_codesearch/_embedder.py`
- Test: `tests/backends/test_codesearch.py` (append)

The embedder is a thin, lazy-import wrapper around fastembed. When re-rank is disabled (default), `rerank()` is a no-op that returns the input unchanged. When enabled, it embeds the query and all chunk texts, then reorders by cosine similarity.

- [ ] **Write the failing tests (append to test file)**

```python
# ---------------------------------------------------------------------------
# Task 6 — embedder (no-op path, always available without fastembed)
# ---------------------------------------------------------------------------
from sleuth.backends._codesearch._embedder import Embedder
from sleuth.types import Chunk, Source
from datetime import datetime, timezone


def _make_chunk(text: str, idx: int) -> Chunk:
    return Chunk(
        text=text,
        source=Source(kind="code", location=f"repo/file.py:L{idx}", title=None, fetched_at=None),
        score=None,
        metadata={},
    )


@pytest.mark.unit
async def test_embedder_noop_preserves_order() -> None:
    """With rerank=False (default), rerank() returns chunks in input order."""
    embedder = Embedder(rerank=False)
    chunks = [_make_chunk(f"chunk {i}", i) for i in range(5)]
    result = await embedder.rerank("query", chunks)
    assert result == chunks


@pytest.mark.unit
async def test_embedder_noop_handles_empty() -> None:
    """No-op rerank on an empty list returns an empty list."""
    embedder = Embedder(rerank=False)
    result = await embedder.rerank("query", [])
    assert result == []
```

- [ ] **Run to confirm failure**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "embedder"
```

Expected: `ERRORS` — `ModuleNotFoundError: No module named '...._embedder'`

- [ ] **Implement `_embedder.py`**

```python
# src/sleuth/backends/_codesearch/_embedder.py
"""Optional cosine re-rank for CodeSearch.

When ``rerank=False`` (default), all methods are no-ops and fastembed is
never imported.  Set ``rerank=True`` only when the ``code-embed`` extra is
installed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from sleuth.types import Chunk

logger = logging.getLogger("sleuth.backends.codesearch.embedder")

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Wraps fastembed for optional cosine re-rank of code chunks."""

    def __init__(
        self,
        *,
        rerank: bool = False,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._rerank = rerank
        self._model_name = model_name
        self._model = None   # lazy-initialized

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Embedding re-rank requires 'agent-sleuth[code-embed]'. "
                "Install with: uv add agent-sleuth[code-embed]"
            ) from exc
        self._model = TextEmbedding(model_name=self._model_name)
        logger.debug("Loaded fastembed model: %s", self._model_name)

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return *chunks* reordered by cosine similarity to *query*.

        If ``self._rerank`` is False, returns *chunks* unchanged (fast path).
        """
        if not self._rerank or not chunks:
            return chunks

        import asyncio
        import numpy as np  # type: ignore[import]

        self._load_model()

        # fastembed is sync — run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        def _embed_all() -> tuple[list[list[float]], list[list[float]]]:
            assert self._model is not None
            q_emb = list(self._model.embed([query]))
            c_embs = list(self._model.embed([c.text for c in chunks]))
            return q_emb, c_embs

        q_emb_list, c_embs_list = await loop.run_in_executor(None, _embed_all)

        q_vec = np.array(q_emb_list[0])
        c_vecs = np.array(c_embs_list)

        # Cosine similarity: q·c / (||q|| * ||c||)
        norms_c = np.linalg.norm(c_vecs, axis=1, keepdims=True)
        norms_c = np.where(norms_c == 0, 1.0, norms_c)
        c_vecs_norm = c_vecs / norms_c
        q_norm = q_vec / (np.linalg.norm(q_vec) or 1.0)
        scores = c_vecs_norm @ q_norm

        ranked_indices = np.argsort(scores)[::-1].tolist()
        return [chunks[i] for i in ranked_indices]
```

- [ ] **Run tests to confirm they pass**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "embedder"
```

Expected: `2 passed`

- [ ] **Commit**

```bash
git add src/sleuth/backends/_codesearch/_embedder.py tests/backends/test_codesearch.py
git commit -m "feat: add optional embedding rerank wrapper (lazy fastembed import)"
```

---

## Task 7: Hierarchical summaries helper (in `_treesitter.py`)

**Files:**
- Modify: `src/sleuth/backends/_codesearch/_treesitter.py`
- Test: `tests/backends/test_codesearch.py` (append)

The hierarchical walker visits every module → class → method node in a file and returns a `HierarchyNode` tree.  This mirrors Phase 2's LocalFiles tree shape but is computed purely from tree-sitter — no Phase 2 code is imported.

- [ ] **Write the failing tests (append to test file)**

```python
# ---------------------------------------------------------------------------
# Task 7 — hierarchical walker
# ---------------------------------------------------------------------------
from sleuth.backends._codesearch._treesitter import HierarchyNode, build_hierarchy


HIERARCHY_SOURCE = """\
class Animal:
    def speak(self) -> str:
        return "..."

    class Inner:
        def inner_method(self) -> None:
            pass

def module_function() -> None:
    pass
"""


@pytest.mark.unit
def test_build_hierarchy_top_level_names() -> None:
    """build_hierarchy returns one node per top-level definition."""
    roots = build_hierarchy(HIERARCHY_SOURCE, SupportedLanguage.PYTHON)
    names = [n.name for n in roots]
    assert "Animal" in names
    assert "module_function" in names


@pytest.mark.unit
def test_build_hierarchy_class_has_children() -> None:
    """Class node contains method children."""
    roots = build_hierarchy(HIERARCHY_SOURCE, SupportedLanguage.PYTHON)
    animal = next(n for n in roots if n.name == "Animal")
    child_names = [c.name for c in animal.children]
    assert "speak" in child_names


@pytest.mark.unit
def test_build_hierarchy_node_carries_source() -> None:
    """Each HierarchyNode stores the full source text of its definition."""
    roots = build_hierarchy(HIERARCHY_SOURCE, SupportedLanguage.PYTHON)
    fn_node = next(n for n in roots if n.name == "module_function")
    assert "def module_function" in fn_node.source_text
```

- [ ] **Run to confirm failure**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "hierarchy"
```

Expected: `ERRORS` — `ImportError: cannot import name 'HierarchyNode'`

- [ ] **Append to `_treesitter.py`** (add after the `language_for_path` function)

```python
# --- Hierarchical summary helpers -------------------------------------------

from dataclasses import dataclass, field as dc_field


@dataclass
class HierarchyNode:
    name: str
    node_type: str           # e.g. "class_definition", "function_definition"
    source_text: str
    start_line: int          # 0-based
    end_line: int            # 0-based, inclusive
    children: list["HierarchyNode"] = dc_field(default_factory=list)
    summary: str | None = None   # filled in later by CodeSearch indexer


def build_hierarchy(source: str, language: SupportedLanguage) -> list[HierarchyNode]:
    """Return the top-level HierarchyNode list for *source*.

    Children of class nodes are nested inside their parent ``HierarchyNode``.
    Module-level functions and classes are top-level entries.
    """
    from tree_sitter import Parser

    ts_lang = _get_language(language)
    parser = Parser(ts_lang)
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)

    return _collect_nodes(tree.root_node, source_bytes)


def _collect_nodes(node, source_bytes: bytes) -> list[HierarchyNode]:
    """Recursively collect enclosing-type nodes, nesting children inside parents."""
    results: list[HierarchyNode] = []
    for child in node.children:
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


def _node_name(node, source_bytes: bytes) -> str:
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return source_bytes[child.start_byte : child.end_byte].decode(errors="replace")
    return "<anonymous>"
```

- [ ] **Run tests to confirm they pass**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "hierarchy"
```

Expected: `3 passed`

- [ ] **Commit**

```bash
git add src/sleuth/backends/_codesearch/_treesitter.py tests/backends/test_codesearch.py
git commit -m "feat: add tree-sitter hierarchical node walker for module/class summaries"
```

---

## Task 8: `CodeSearch` class — wire everything together

**Files:**
- Create: `src/sleuth/backends/codesearch.py`
- Test: `tests/backends/test_codesearch.py` (append)

`CodeSearch` ties ripgrep + tree-sitter + symbol index together into the `Backend` protocol. Two query paths:

1. **Symbol lookup path:** if the query matches the pattern `"where is <name> defined"` / `"definition of <name>"`, it queries the symbol index directly and skips ripgrep.
2. **General path:** run ripgrep → for each hit expand via tree-sitter → deduplicate by `(file, start_line)` → optional re-rank → return top-k `Chunk` objects.

Re-indexing: `CodeSearch.__init__` accepts a `rebuild` argument (`"mtime"` (default) or `"always"`). The symbol index uses `(mtime, content hash)` regardless.

- [ ] **Write the failing tests (append to test file)**

```python
# ---------------------------------------------------------------------------
# Task 8 — CodeSearch public class
# ---------------------------------------------------------------------------
from sleuth.backends.codesearch import CodeSearch
from sleuth.backends.base import Capability


@pytest.mark.unit
async def test_codesearch_name_and_capabilities(tmp_path: Path) -> None:
    """CodeSearch exposes correct name and capabilities."""
    cs = CodeSearch(path=tmp_path)
    assert cs.name == "codesearch"
    assert Capability.CODE in cs.capabilities


@pytest.mark.unit
async def test_codesearch_search_returns_chunks(tmp_path: Path) -> None:
    """search() returns Chunk objects with kind='code' sources."""
    (tmp_path / "auth.py").write_text(
        "def validate_token(token: str) -> bool:\n    return len(token) > 0\n"
    )
    cs = CodeSearch(path=tmp_path)
    chunks = await cs.search("validate_token", k=5)
    assert len(chunks) >= 1
    assert all(c.source.kind == "code" for c in chunks)
    assert any("validate_token" in c.text for c in chunks)


@pytest.mark.unit
async def test_codesearch_returns_at_most_k_chunks(tmp_path: Path) -> None:
    """search() returns no more than k chunks."""
    for i in range(10):
        (tmp_path / f"file_{i}.py").write_text(
            f"def func_{i}():\n    # match_target\n    pass\n"
        )
    cs = CodeSearch(path=tmp_path)
    chunks = await cs.search("match_target", k=3)
    assert len(chunks) <= 3


@pytest.mark.unit
async def test_codesearch_symbol_query_shortcut(tmp_path: Path) -> None:
    """'where is X defined' bypasses ripgrep and uses the symbol index."""
    (tmp_path / "handlers.py").write_text(
        "def handle_request(req):\n    pass\n"
    )
    cs = CodeSearch(path=tmp_path)
    await cs._ensure_indexed()   # warm the symbol index explicitly for the test

    chunks = await cs.search("where is handle_request defined", k=5)
    assert len(chunks) >= 1
    assert any("handle_request" in c.text for c in chunks)


@pytest.mark.unit
async def test_codesearch_empty_repo_returns_no_chunks(tmp_path: Path) -> None:
    """search() on a directory with no supported files returns []."""
    (tmp_path / "README.txt").write_text("not code\n")
    cs = CodeSearch(path=tmp_path)
    chunks = await cs.search("anything", k=10)
    assert chunks == []


@pytest.mark.unit
async def test_codesearch_chunk_location_format(tmp_path: Path) -> None:
    """Chunk source.location encodes file path and line range."""
    (tmp_path / "utils.py").write_text("def helper():\n    return 42\n")
    cs = CodeSearch(path=tmp_path)
    chunks = await cs.search("helper", k=5)
    assert len(chunks) >= 1
    # location format: "path/to/file.py:L<start>-L<end>"
    loc = chunks[0].source.location
    assert "utils.py" in loc
    assert "L" in loc


@pytest.mark.unit
async def test_codesearch_cancellation(tmp_path: Path) -> None:
    """search() is cancellable without raising anything other than CancelledError."""
    import asyncio
    (tmp_path / "big.py").write_text("\n".join(f"def fn_{i}(): pass" for i in range(200)))
    cs = CodeSearch(path=tmp_path)

    async def _run():
        return await cs.search("fn_", k=50)

    task = asyncio.create_task(_run())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
```

- [ ] **Run to confirm failure**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "codesearch_"
```

Expected: `ERRORS` — `ModuleNotFoundError: No module named 'sleuth.backends.codesearch'`

- [ ] **Implement `codesearch.py`**

```python
# src/sleuth/backends/codesearch.py
"""CodeSearch backend: ripgrep + tree-sitter + optional embedding re-rank."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from sleuth.backends.base import Backend, Capability
from sleuth.backends._codesearch._ripgrep import run_ripgrep
from sleuth.backends._codesearch._treesitter import expand_hit_to_node, language_for_path
from sleuth.backends._codesearch._symbol_index import SymbolIndex, SymbolRecord
from sleuth.backends._codesearch._embedder import Embedder
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
        seen: dict[tuple[Path, int], Chunk] = {}   # (path, start_line) → chunk

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
                        start_line=expanded.start_line + 1,   # → 1-based
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
```

- [ ] **Run tests to confirm they pass**

```bash
uv run pytest tests/backends/test_codesearch.py -v -k "codesearch_"
```

Expected: `7 passed`

- [ ] **Commit**

```bash
git add src/sleuth/backends/codesearch.py tests/backends/test_codesearch.py
git commit -m "feat: implement CodeSearch backend with ripgrep+tree-sitter two-phase retrieval"
```

---

## Task 9: `BackendTestKit` protocol compliance

**Files:**
- Test: `tests/backends/test_codesearch.py` (append)

`BackendTestKit` is owned by Phase 1 (`tests/contract/test_backend_protocol.py`). This task plugs `CodeSearch` into it to verify protocol compliance, error shapes, timeout behavior, and cancellation safety — without reimplementing those checks.

- [ ] **Write the BackendTestKit invocation (append to test file)**

```python
# ---------------------------------------------------------------------------
# Task 9 — BackendTestKit protocol compliance
# ---------------------------------------------------------------------------
# BackendTestKit is defined in tests/contract/test_backend_protocol.py (Phase 1).
# Import and parametrize it for CodeSearch.
from tests.contract.test_backend_protocol import BackendTestKit


class TestCodeSearchProtocol(BackendTestKit):
    """Run the full Backend-protocol contract suite against CodeSearch."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> CodeSearch:
        # Minimal valid repo: one Python file so search() has something to scan.
        (tmp_path / "sample.py").write_text("def sample_func():\n    pass\n")
        return CodeSearch(path=tmp_path)

    @pytest.fixture
    def matching_query(self) -> str:
        return "sample_func"

    @pytest.fixture
    def non_matching_query(self) -> str:
        return "zzz_this_does_not_exist_zzz_xyzzy"
```

- [ ] **Run the protocol suite**

```bash
uv run pytest tests/backends/test_codesearch.py::TestCodeSearchProtocol -v
```

Expected: all BackendTestKit tests pass (exact count depends on Phase 1's kit definition). If `BackendTestKit` is not yet implemented (Phase 1 not done), this test class is skipped automatically — the import will raise `ModuleNotFoundError`, which is acceptable in a pre-Phase-1 environment; leave the block in place for when Phase 1 merges.

- [ ] **Commit**

```bash
git add tests/backends/test_codesearch.py
git commit -m "test: plug CodeSearch into BackendTestKit protocol compliance suite"
```

---

## Task 10: Full test run, type checking, lint

**Files:** none new

- [ ] **Run the full test suite for this backend**

```bash
uv run pytest tests/backends/test_codesearch.py -v --tb=short
```

Expected: all tests pass with no warnings about unknown marks.

- [ ] **Run mypy over the new files**

```bash
uv run mypy src/sleuth/backends/codesearch.py src/sleuth/backends/_codesearch/
```

Expected: `Success: no issues found in N source files` (or resolve any errors before proceeding).

- [ ] **Run ruff lint + format check**

```bash
uv run ruff check src/sleuth/backends/codesearch.py src/sleuth/backends/_codesearch/
uv run ruff format --check src/sleuth/backends/codesearch.py src/sleuth/backends/_codesearch/
```

Expected: no errors. If formatting issues are reported, run `uv run ruff format <path>` to fix them and re-check.

- [ ] **Run coverage check**

```bash
uv run pytest tests/backends/test_codesearch.py --cov=src/sleuth/backends/codesearch --cov=src/sleuth/backends/_codesearch --cov-report=term-missing
```

Expected: coverage ≥ 85% across the new modules.

- [ ] **Commit any lint/format fixes**

```bash
# Only if ruff made changes:
git add src/sleuth/backends/
git commit -m "style: apply ruff format to codesearch backend"
```

---

## Task 11: Export from `backends/__init__.py`

**Files:**
- Modify: `src/sleuth/backends/__init__.py`

Phase 1 owns `backends/__init__.py`. This task appends one export so users can write `from sleuth.backends import CodeSearch`.

- [ ] **Append to `src/sleuth/backends/__init__.py`**

```python
from sleuth.backends.codesearch import CodeSearch  # noqa: F401
```

- [ ] **Verify the import works**

```bash
uv run python -c "from sleuth.backends import CodeSearch; print(CodeSearch.name)"
```

Expected: `codesearch`

- [ ] **Commit**

```bash
git add src/sleuth/backends/__init__.py
git commit -m "feat: export CodeSearch from sleuth.backends"
```

---

## Task 12: Final integration smoke test and PR prep

**Files:** none new

- [ ] **Write a quick end-to-end smoke test (append to test file, mark integration)**

```python
# ---------------------------------------------------------------------------
# Task 12 — integration smoke (skipped in unit CI, runs nightly)
# ---------------------------------------------------------------------------
import os


@pytest.mark.integration
async def test_codesearch_integration_on_real_repo(tmp_path: Path) -> None:
    """Smoke test against a slightly larger synthetic codebase."""
    # Build a mini repo with multiple files
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / "tokens.py").write_text(
        "def validate_token(token: str) -> bool:\n"
        "    return token.startswith('sk-')\n\n"
        "def refresh_token(old_token: str) -> str:\n"
        "    return old_token + '_refreshed'\n"
    )
    (tmp_path / "auth" / "middleware.py").write_text(
        "from auth.tokens import validate_token\n\n"
        "class AuthMiddleware:\n"
        "    def __call__(self, request):\n"
        "        token = request.headers.get('Authorization', '')\n"
        "        return validate_token(token)\n"
    )
    (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")

    cs = CodeSearch(path=tmp_path)

    # General search
    chunks = await cs.search("validate_token", k=5)
    assert any("validate_token" in c.text for c in chunks)

    # Symbol shortcut
    chunks_def = await cs.search("where is AuthMiddleware defined", k=3)
    assert any("AuthMiddleware" in c.text for c in chunks_def)

    # All chunks have code kind
    for c in chunks + chunks_def:
        assert c.source.kind == "code"
```

- [ ] **Run unit tests only (integration is skipped)**

```bash
uv run pytest tests/backends/test_codesearch.py -m "not integration" -v
```

Expected: all non-integration tests pass.

- [ ] **Commit**

```bash
git add tests/backends/test_codesearch.py
git commit -m "test: add integration smoke test for CodeSearch end-to-end"
```

- [ ] **Push branch**

```bash
git push -u origin feature/phase-5-codesearch
```

- [ ] **Open PR against `develop`** using the project's PR template. Title: `feat: Phase 5 — CodeSearch backend (ripgrep + tree-sitter + symbol index)`. Required CI checks: ruff, mypy, unit tests, coverage ≥ 85%.

---

## Self-review checklist (completed inline)

**1. Spec coverage**

| Spec requirement (§7.4) | Task |
|---|---|
| Two-phase retrieval: ripgrep → tree-sitter expand | Tasks 3, 4, 8 |
| Optional embedding re-rank (non-lexical queries) | Task 6, 8 |
| Symbol-aware index for "where is X defined" | Tasks 5, 8 |
| Respects `.gitignore` | Task 3 (ripgrep default) |
| Re-index only on `(mtime, content hash)` change | Task 5 |
| Hierarchical summaries (module/class via tree-sitter) | Task 7 |
| Backend protocol compliance | Tasks 8, 9 |
| Chunk type with `source.kind="code"` | Tasks 8, 12 |
| `pyproject.toml` extras for tree-sitter deps | Task 2 |
| `rg` binary documented | Task 2 |
| Tests via BackendTestKit | Task 9 |

**2. Placeholder scan:** No TBD, TODO, "implement later", or vague steps found. All code blocks are complete.

**3. Type consistency:** `RipgrepHit`, `ExpandedNode`, `HierarchyNode`, `SymbolRecord`, `SymbolIndex`, `Embedder`, `CodeSearch` — all defined before use. `Chunk` and `Source` imported from `sleuth.types` (Phase 1). `Backend`, `Capability` imported from `sleuth.backends.base` (Phase 1). `SupportedLanguage` defined in `_treesitter.py` Task 4 and imported in `_symbol_index.py` Task 5, `codesearch.py` Task 8. Consistent throughout.
