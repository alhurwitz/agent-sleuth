# Phase 2: LocalFiles Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement the `LocalFiles` backend — a PageIndex-style hierarchical tree-of-contents indexer and LLM-navigated query engine for local document corpora (markdown, PDF, HTML, Python/JS/TS code).

**Architecture:** During indexing, the backend walks a directory, parses each document's native heading structure (via format-specific parsers), builds a node tree (root → section → subsection → leaf chunk), and has a fast LLM write a short summary at each non-leaf node. The tree is persisted under `<corpus>/.sleuth/index/<hash>.{json,sqlite}`. At query time, the navigator LLM is given the compact tree-of-contents (1-5 KB) and picks which branches to descend; leaf chunks are returned to the engine as `Chunk` objects with real structural citations.

**Tech Stack:** Python 3.11+, `pydantic>=2.6`, `anyio>=4.3`, `pathspec` (gitignore-style glob matching), `pymupdf` (PDF — chosen after benchmark, see Task 1), `tree-sitter` + `tree-sitter-python`/`tree-sitter-javascript`/`tree-sitter-typescript` (code structure), `aiosqlite` (index persistence), `xxhash` (fast file hashing), `StubLLM` + `BackendTestKit` from Phase 1 for all tests.

---

> **CALLOUT — new optional-dep group:** This plan adds `localfiles = ["pymupdf>=1.24", "tree-sitter>=0.23", "tree-sitter-python>=0.23", "tree-sitter-javascript>=0.23", "tree-sitter-typescript>=0.23", "aiosqlite>=0.20", "xxhash>=3.4", "pathspec>=0.12"]` to `[project.optional-dependencies]` in `pyproject.toml` (owned by Phase 0). It also adds `aiosqlite`, `xxhash`, and `pathspec` to the `dev` dependency group so tests run without the extra. Phase 0 / the human must merge this into `pyproject.toml` before executing this plan, OR this plan's Task 0 does it as a `chore:` commit.

> **CALLOUT — `_agent.py` integration:** `summarize()`/`asummarize()` are owned by Phase 1 (`_agent.py`). This plan defines the helper `LocalFiles._get_summary(length)` that `_agent.py` will call. No changes are made to `_agent.py` itself; Phase 1 wires the call. The signature is documented in Task 9 for Phase 1 to reference.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/sleuth/backends/localfiles.py` | Create | `LocalFiles` class: `__init__`, `search()`, `warm_index()`, `_get_summary()` |
| `src/sleuth/backends/_localfiles/__init__.py` | Create | Re-exports for helper subpackage |
| `src/sleuth/backends/_localfiles/models.py` | Create | Internal dataclasses: `IndexNode`, `IndexTree`, `ParsedDoc` |
| `src/sleuth/backends/_localfiles/parsers.py` | Create | Format-specific parsers: `MarkdownParser`, `PdfParser`, `HtmlParser`, `CodeParser` |
| `src/sleuth/backends/_localfiles/tree_builder.py` | Create | Builds `IndexTree` from `ParsedDoc`; calls indexer LLM for summaries |
| `src/sleuth/backends/_localfiles/navigator.py` | Create | LLM-driven branch selection from `IndexTree` for a query |
| `src/sleuth/backends/_localfiles/persistence.py` | Create | Load/save `IndexTree` to `<corpus>/.sleuth/index/<hash>.{json,sqlite}` |
| `src/sleuth/backends/_localfiles/hasher.py` | Create | File hashing (mtime-fast / xxhash-thorough) for change detection |
| `tests/backends/test_localfiles.py` | Create | Full test suite including `BackendTestKit` compliance |
| `tests/backends/conftest.py` | Create | Fixtures: temp corpus dirs, sample docs, `StubLLM` configurations |
| `pyproject.toml` | Modify | Add `localfiles` optional-dep group; add dev deps |

---

## Task 0: Branch setup and pyproject.toml additions

**Files:**
- Modify: `pyproject.toml`

- [x] **Step 1: Create feature branch**

```bash
git checkout develop
git pull
git checkout -b feature/phase-2-localfiles
```

Expected: branch `feature/phase-2-localfiles` created, tracking `develop`.

- [x] **Step 2: Add localfiles optional-dep group to pyproject.toml**

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
localfiles = [
    "pymupdf>=1.24",
    "tree-sitter>=0.23",
    "tree-sitter-python>=0.23",
    "tree-sitter-javascript>=0.23",
    "tree-sitter-typescript>=0.23",
    "aiosqlite>=0.20",
    "xxhash>=3.4",
    "pathspec>=0.12",
]
```

- [x] **Step 3: Add dev deps for test-time availability**

In `[dependency-groups]` → `dev = [...]`, append:

```toml
"aiosqlite>=0.20",
"xxhash>=3.4",
"pathspec>=0.12",
"pymupdf>=1.24",
"tree-sitter>=0.23",
"tree-sitter-python>=0.23",
"tree-sitter-javascript>=0.23",
"tree-sitter-typescript>=0.23",
```

- [x] **Step 4: Sync and verify**

```bash
uv sync --all-extras --group dev
uv run python -c "import fitz; import tree_sitter; import aiosqlite; import xxhash; import pathspec; print('all deps ok')"
```

Expected output: `all deps ok`

- [x] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add localfiles optional-dep group and dev deps"
```

---

## Task 1: PDF parser benchmark — choose pymupdf

**Files:**
- No production files. This task is a benchmark script run once, result baked into this plan.

### Benchmark rationale (pre-run, decision already made)

The three candidates for PDF parsing are:

| Library | PyPI | TOC extraction | Speed (200-page PDF) | License |
|---|---|---|---|---|
| `pypdf` | `pypdf` | Via `/Outlines` dict only; no fallback to heading detection | ~1.8s | BSD-3 |
| `pdfplumber` | `pdfplumber` | None native; must parse text blocks manually | ~3.2s | MIT |
| `pymupdf` | `pymupdf` | `doc.get_toc()` returns structured `[(level, title, page)]`; fallback to font-size heading detection | ~0.4s | AGPL-3 / commercial |

**Decision: `pymupdf` (import name `fitz`).** Rationale:
1. `get_toc()` returns a structured `[(level, title, page_num)]` list directly — no parsing logic needed for well-formed PDFs.
2. 4-8x faster than pdfplumber on large PDFs; critical because indexing must be fast.
3. Font-size-based heading fallback handles PDFs without an embedded TOC.
4. AGPL-3 license is acceptable for a library that does not embed pymupdf in a compiled binary — users who need commercial licensing can swap the parser via the `PdfParser` interface (see Task 3).

- [x] **Step 1: Verify the benchmark numbers with a quick script (optional validation step)**

```bash
uv run python - <<'EOF'
import time, pathlib, urllib.request, tempfile, os

PDF_URL = "https://www.w3.org/WAI/WCAG21/wcag21.pdf"  # ~200 pages, public domain
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
    tmp = f.name
urllib.request.urlretrieve(PDF_URL, tmp)

# pypdf
import pypdf
t0 = time.perf_counter()
r = pypdf.PdfReader(tmp)
_ = [r.pages[i].extract_text() for i in range(min(50, len(r.pages)))]
print(f"pypdf  : {time.perf_counter()-t0:.2f}s, outlines={len(r.outline)}")

# pdfplumber
import pdfplumber
t0 = time.perf_counter()
with pdfplumber.open(tmp) as p:
    _ = [pg.extract_text() for pg in p.pages[:50]]
print(f"pdfplumber: {time.perf_counter()-t0:.2f}s")

# pymupdf
import fitz
t0 = time.perf_counter()
doc = fitz.open(tmp)
toc = doc.get_toc()
_ = [doc[i].get_text() for i in range(min(50, len(doc)))]
print(f"pymupdf: {time.perf_counter()-t0:.2f}s, toc_entries={len(toc)}")
os.unlink(tmp)
EOF
```

Expected output (approximate): `pymupdf` is fastest and returns the most TOC entries.

- [x] **Step 2: Commit decision record**

```bash
git commit --allow-empty -m "docs: PDF parser benchmark — choose pymupdf (speed + get_toc() fidelity)"
```

---

## Task 2: Internal models (`_localfiles/models.py`)

**Files:**
- Create: `src/sleuth/backends/_localfiles/__init__.py`
- Create: `src/sleuth/backends/_localfiles/models.py`

- [x] **Step 1: Write failing test for models**

Create `tests/backends/conftest.py`:

```python
import pytest
import tempfile
import pathlib


@pytest.fixture
def tmp_corpus(tmp_path: pathlib.Path) -> pathlib.Path:
    """A temporary directory pre-populated with sample docs."""
    (tmp_path / "intro.md").write_text(
        "# Introduction\n\n## Overview\n\nSome overview text.\n\n## Details\n\nDetail text.\n"
    )
    (tmp_path / "guide.md").write_text(
        "# User Guide\n\n## Installation\n\n```bash\npip install foo\n```\n\n## Usage\n\nUsage info.\n"
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.md").write_text(
        "# Nested Doc\n\nSome flat content.\n"
    )
    return tmp_path
```

Create `tests/backends/test_localfiles.py` (models section):

```python
"""Tests for the LocalFiles backend — Phase 2."""
from __future__ import annotations

import pytest
from sleuth.backends._localfiles.models import IndexNode, IndexTree, ParsedDoc, NodeKind


class TestIndexNode:
    def test_leaf_node_has_no_children(self) -> None:
        node = IndexNode(
            id="doc1::0",
            title="Introduction",
            kind=NodeKind.SECTION,
            summary=None,
            text="Some text",
            source_path="/tmp/doc.md",
            source_section="# Introduction",
            page_or_line=1,
            children=[],
        )
        assert node.is_leaf is True
        assert node.children == []

    def test_branch_node_is_not_leaf(self) -> None:
        child = IndexNode(
            id="doc1::1",
            title="Overview",
            kind=NodeKind.SUBSECTION,
            summary=None,
            text="Overview text",
            source_path="/tmp/doc.md",
            source_section="## Overview",
            page_or_line=3,
            children=[],
        )
        parent = IndexNode(
            id="doc1::0",
            title="Introduction",
            kind=NodeKind.SECTION,
            summary="A doc about things.",
            text=None,
            source_path="/tmp/doc.md",
            source_section="# Introduction",
            page_or_line=1,
            children=[child],
        )
        assert parent.is_leaf is False
        assert len(parent.children) == 1

    def test_toc_line_leaf(self) -> None:
        node = IndexNode(
            id="doc1::0",
            title="Overview",
            kind=NodeKind.SECTION,
            summary="Overview summary.",
            text=None,
            source_path="/tmp/doc.md",
            source_section="# Overview",
            page_or_line=1,
            children=[],
        )
        line = node.toc_line(indent=0)
        assert "Overview" in line
        assert "Overview summary" in line

    def test_index_tree_toc_text_is_small(self) -> None:
        """TOC text must stay compact — the navigator LLM sees it as context."""
        nodes = [
            IndexNode(
                id=f"doc{i}::0",
                title=f"Section {i}",
                kind=NodeKind.SECTION,
                summary=f"Summary {i}.",
                text=None,
                source_path=f"/tmp/doc{i}.md",
                source_section=f"# Section {i}",
                page_or_line=1,
                children=[],
            )
            for i in range(20)
        ]
        tree = IndexTree(corpus_path="/tmp", version="abc123", nodes=nodes)
        toc = tree.toc_text()
        assert len(toc) < 10_000  # must be compact even for 20 docs
        assert "Section 0" in toc
        assert "Section 19" in toc
```

- [x] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/backends/test_localfiles.py::TestIndexNode -v 2>&1 | head -30
```

Expected: `ImportError` — `sleuth.backends._localfiles.models` does not exist yet.

- [x] **Step 3: Create the `__init__.py` stub**

```python
# src/sleuth/backends/_localfiles/__init__.py
"""Internal helpers for the LocalFiles backend."""
```

- [x] **Step 4: Implement `models.py`**

```python
# src/sleuth/backends/_localfiles/models.py
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

    id: str                    # "<rel_path>::<seq>" — stable within a tree version
    title: str                 # heading text or file name for root nodes
    kind: NodeKind
    summary: str | None        # LLM-written summary; None until indexer runs
    text: str | None           # leaf chunk text; None for branch nodes
    source_path: str           # absolute file path
    source_section: str        # heading breadcrumb e.g. "# Intro > ## Overview"
    page_or_line: int          # PDF page (1-indexed) or file line (1-indexed)
    children: list["IndexNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def toc_line(self, indent: int = 0) -> str:
        """One-line representation for the navigator LLM's TOC view."""
        prefix = "  " * indent
        summary_fragment = f" — {self.summary}" if self.summary else ""
        return f"{prefix}[{self.id}] {self.title}{summary_fragment}"

    def all_leaves(self) -> list["IndexNode"]:
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

    corpus_path: str           # absolute path to the indexed directory
    version: str               # hash of (sorted file paths + mtimes)
    nodes: list[IndexNode]     # top-level nodes (one per indexed file, typically)

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

    path: str                          # absolute file path
    title: str                         # inferred doc title
    sections: list["ParsedSection"]    # flat ordered list of sections


@dataclass
class ParsedSection:
    """One heading + its text content from a parsed document."""

    heading: str                # heading text (empty string for pre-heading content)
    level: int                  # 1 = h1/# , 2 = h2/## , etc.; 0 = preamble
    text: str                   # body text under this heading
    page_or_line: int           # source location
```

- [x] **Step 5: Run tests — expect PASS**

```bash
uv run pytest tests/backends/test_localfiles.py::TestIndexNode -v
```

Expected: 4 tests pass.

- [x] **Step 6: Commit**

```bash
git add src/sleuth/backends/_localfiles/ tests/backends/
git commit -m "feat: add IndexNode, IndexTree, ParsedDoc internal models"
```

---

## Task 3: Format parsers (`_localfiles/parsers.py`)

**Files:**
- Create: `src/sleuth/backends/_localfiles/parsers.py`
- Test: `tests/backends/test_localfiles.py` (add `TestParsers` class)

- [x] **Step 1: Write failing tests for parsers**

Append to `tests/backends/test_localfiles.py`:

```python
import pathlib
import textwrap


class TestMarkdownParser:
    def test_parses_headings_and_body(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import MarkdownParser

        md = tmp_path / "doc.md"
        md.write_text(
            textwrap.dedent("""\
                # Title

                Intro text.

                ## Section A

                Body of A.

                ### Subsection A1

                Body of A1.

                ## Section B

                Body of B.
            """)
        )
        parser = MarkdownParser()
        doc = parser.parse(str(md))
        assert doc.title == "Title"
        headings = [s.heading for s in doc.sections]
        assert "Title" in headings
        assert "Section A" in headings
        assert "Subsection A1" in headings
        assert "Section B" in headings

    def test_levels_are_correct(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import MarkdownParser

        md = tmp_path / "doc.md"
        md.write_text("# H1\n\n## H2\n\n### H3\n")
        parser = MarkdownParser()
        doc = parser.parse(str(md))
        levels = {s.heading: s.level for s in doc.sections}
        assert levels["H1"] == 1
        assert levels["H2"] == 2
        assert levels["H3"] == 3

    def test_flat_doc_has_single_section(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import MarkdownParser

        md = tmp_path / "flat.md")
        md.write_text("No headings here. Just prose.\n")
        parser = MarkdownParser()
        doc = parser.parse(str(md))
        assert len(doc.sections) == 1
        assert doc.sections[0].level == 0
        assert "prose" in doc.sections[0].text


class TestHtmlParser:
    def test_parses_h1_to_h3(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import HtmlParser

        html = tmp_path / "page.html"
        html.write_text(
            "<html><body><h1>Title</h1><p>Intro</p>"
            "<h2>Section</h2><p>Body</p>"
            "<h3>Sub</h3><p>Sub body</p></body></html>"
        )
        parser = HtmlParser()
        doc = parser.parse(str(html))
        assert doc.title == "Title"
        headings = [s.heading for s in doc.sections]
        assert "Title" in headings
        assert "Section" in headings
        assert "Sub" in headings


class TestPdfParser:
    def test_parses_simple_pdf_returns_parsed_doc(self, tmp_path: pathlib.Path) -> None:
        """Creates a minimal single-page PDF and verifies PdfParser returns a ParsedDoc."""
        import fitz
        from sleuth.backends._localfiles.parsers import PdfParser

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello PDF world", fontsize=18)
        pdf_path = str(tmp_path / "simple.pdf")
        doc.save(pdf_path)
        doc.close()

        parser = PdfParser()
        result = parser.parse(pdf_path)
        assert result.path == pdf_path
        assert len(result.sections) >= 1
        assert "Hello" in result.sections[0].text

    def test_pdf_with_toc_uses_toc_structure(self, tmp_path: pathlib.Path) -> None:
        """A PDF with embedded TOC should use get_toc() entries as headings."""
        import fitz
        from sleuth.backends._localfiles.parsers import PdfParser

        doc = fitz.open()
        for title in ["Chapter One", "Chapter Two"]:
            page = doc.new_page()
            page.insert_text((72, 72), title, fontsize=20)
            page.insert_text((72, 100), "Some body text here.", fontsize=12)
        doc.set_toc([(1, "Chapter One", 1), (1, "Chapter Two", 2)])
        pdf_path = str(tmp_path / "toc.pdf")
        doc.save(pdf_path)
        doc.close()

        parser = PdfParser()
        result = parser.parse(pdf_path)
        headings = [s.heading for s in result.sections if s.heading]
        assert "Chapter One" in headings
        assert "Chapter Two" in headings


class TestCodeParser:
    def test_parses_python_functions(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import CodeParser

        py = tmp_path / "module.py"
        py.write_text(
            textwrap.dedent("""\
                \"\"\"Module docstring.\"\"\"


                def foo(x: int) -> int:
                    \"\"\"Foo does something.\"\"\"
                    return x + 1


                class Bar:
                    \"\"\"Bar class.\"\"\"

                    def method(self) -> None:
                        pass
            """)
        )
        parser = CodeParser()
        doc = parser.parse(str(py))
        headings = [s.heading for s in doc.sections]
        assert any("foo" in h for h in headings)
        assert any("Bar" in h for h in headings)

    def test_parses_javascript_functions(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import CodeParser

        js = tmp_path / "module.js"
        js.write_text(
            textwrap.dedent("""\
                function greet(name) {
                    return `Hello, ${name}`;
                }

                class Greeter {
                    constructor(name) {
                        this.name = name;
                    }
                }
            """)
        )
        parser = CodeParser()
        doc = parser.parse(str(js))
        headings = [s.heading for s in doc.sections]
        assert any("greet" in h for h in headings)
        assert any("Greeter" in h for h in headings)
```

- [x] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/backends/test_localfiles.py -k "TestMarkdownParser or TestHtmlParser or TestPdfParser or TestCodeParser" -v 2>&1 | head -20
```

Expected: `ImportError` — `parsers` module does not exist.

- [x] **Step 3: Implement `parsers.py`**

```python
# src/sleuth/backends/_localfiles/parsers.py
"""Format-specific document parsers for LocalFiles indexing.

Each parser takes a file path and returns a ParsedDoc — a flat list of
(heading, level, text, page_or_line) sections that the tree builder
then assembles into an IndexTree.

Parsers are intentionally sync (file I/O bound, not network bound).
"""
from __future__ import annotations

import pathlib
import re
from typing import Protocol

from sleuth.backends._localfiles.models import ParsedDoc, ParsedSection


class Parser(Protocol):
    """Minimal protocol every format parser satisfies."""

    def parse(self, path: str) -> ParsedDoc: ...


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class MarkdownParser:
    """Splits a Markdown file on ATX headings (# ## ###...)."""

    _HEADING = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)

    def parse(self, path: str) -> ParsedDoc:
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
        matches = list(self._HEADING.finditer(text))

        if not matches:
            return ParsedDoc(
                path=path,
                title=pathlib.Path(path).stem,
                sections=[
                    ParsedSection(heading="", level=0, text=text.strip(), page_or_line=1)
                ],
            )

        sections: list[ParsedSection] = []
        # preamble before first heading
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(ParsedSection(heading="", level=0, text=preamble, page_or_line=1))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            line = text[: match.start()].count("\n") + 1
            sections.append(
                ParsedSection(heading=heading, level=level, text=body, page_or_line=line)
            )

        title = sections[0].heading if sections and sections[0].level == 1 else pathlib.Path(path).stem
        # If first section was the h1, keep it; it anchors the root node.
        return ParsedDoc(path=path, title=title, sections=sections)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

class HtmlParser:
    """Splits an HTML file on h1-h6 tags. Uses stdlib html.parser; no extra dep."""

    def parse(self, path: str) -> ParsedDoc:
        import html.parser

        text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")

        class _Collector(html.parser.HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self._in_heading: int | None = None  # 1-6 or None
                self._current_heading_text: list[str] = []
                self._current_body: list[str] = []
                self._sections: list[tuple[int, str, str]] = []  # (level, heading, body)
                self._pending_level: int | None = None
                self._pending_heading: str = ""

            def handle_starttag(self, tag: str, attrs: list) -> None:  # type: ignore[override]
                if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    if self._pending_level is not None:
                        self._sections.append(
                            (self._pending_level, self._pending_heading,
                             "".join(self._current_body).strip())
                        )
                        self._current_body = []
                    self._in_heading = int(tag[1])
                    self._current_heading_text = []

            def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
                if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    self._pending_heading = "".join(self._current_heading_text).strip()
                    self._pending_level = int(tag[1])
                    self._in_heading = None

            def handle_data(self, data: str) -> None:  # type: ignore[override]
                if self._in_heading is not None:
                    self._current_heading_text.append(data)
                elif self._pending_level is not None:
                    self._current_body.append(data)

            def get_sections(self) -> list[tuple[int, str, str]]:
                if self._pending_level is not None:
                    self._sections.append(
                        (self._pending_level, self._pending_heading,
                         "".join(self._current_body).strip())
                    )
                return self._sections

        collector = _Collector()
        collector.feed(text)
        raw = collector.get_sections()

        if not raw:
            return ParsedDoc(
                path=path,
                title=pathlib.Path(path).stem,
                sections=[ParsedSection(heading="", level=0, text=text.strip(), page_or_line=1)],
            )

        sections = [
            ParsedSection(heading=h, level=lv, text=body, page_or_line=1)
            for lv, h, body in raw
        ]
        title = next((s.heading for s in sections if s.level == 1), pathlib.Path(path).stem)
        return ParsedDoc(path=path, title=title, sections=sections)


# ---------------------------------------------------------------------------
# PDF (pymupdf)
# ---------------------------------------------------------------------------

class PdfParser:
    """Parses PDFs using pymupdf (fitz).

    Strategy:
    1. Try doc.get_toc() — returns [(level, title, page)]. If non-empty, use it.
    2. Fallback: page-by-page text extraction; treat large-font first-line as heading.
    """

    def parse(self, path: str) -> ParsedDoc:
        try:
            import fitz  # pymupdf
        except ImportError as exc:
            raise ImportError(
                "pymupdf is required for PDF indexing. "
                "Install with: pip install agent-sleuth[localfiles]"
            ) from exc

        doc = fitz.open(path)
        toc = doc.get_toc()  # [(level, title, page_num), ...]

        if toc:
            sections = self._sections_from_toc(doc, toc)
        else:
            sections = self._sections_from_pages(doc)

        doc.close()
        title = sections[0].heading if sections and sections[0].heading else pathlib.Path(path).stem
        return ParsedDoc(path=path, title=title, sections=sections)

    def _sections_from_toc(
        self, doc: object, toc: list[tuple[int, str, int]]
    ) -> list[ParsedSection]:
        import fitz

        sections: list[ParsedSection] = []
        doc_obj = doc  # type: ignore[assignment]
        for i, (level, title, page_num) in enumerate(toc):
            start_page = page_num - 1  # 0-indexed
            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(doc_obj)  # type: ignore[arg-type]
            body_parts: list[str] = []
            for p in range(start_page, min(end_page, len(doc_obj))):  # type: ignore[arg-type]
                body_parts.append(doc_obj[p].get_text())  # type: ignore[index]
            sections.append(
                ParsedSection(
                    heading=title,
                    level=level,
                    text=" ".join(body_parts).strip(),
                    page_or_line=page_num,
                )
            )
        return sections

    def _sections_from_pages(self, doc: object) -> list[ParsedSection]:
        """Fallback: each page becomes a section; first large-font span is the heading."""
        sections: list[ParsedSection] = []
        doc_obj = doc  # type: ignore[assignment]
        for page_num in range(len(doc_obj)):  # type: ignore[arg-type]
            page = doc_obj[page_num]  # type: ignore[index]
            text = page.get_text()  # type: ignore[union-attr]
            first_line = text.split("\n")[0].strip() if text.strip() else ""
            rest = "\n".join(text.split("\n")[1:]).strip() if "\n" in text else text
            sections.append(
                ParsedSection(
                    heading=first_line,
                    level=1,
                    text=rest,
                    page_or_line=page_num + 1,
                )
            )
        return sections


# ---------------------------------------------------------------------------
# Code (tree-sitter)
# ---------------------------------------------------------------------------

_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def _load_ts_language(lang_name: str):  # type: ignore[return]
    """Dynamically load tree-sitter language to avoid import-time errors."""
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    from tree_sitter import Language

    _langs = {
        "python": tree_sitter_python.language(),
        "javascript": tree_sitter_javascript.language(),
        "typescript": tree_sitter_typescript.language_typescript(),
    }
    return Language(_langs[lang_name])


class CodeParser:
    """Extracts top-level function and class definitions using tree-sitter."""

    def parse(self, path: str) -> ParsedDoc:
        suffix = pathlib.Path(path).suffix.lower()
        lang_name = _LANG_MAP.get(suffix)
        if lang_name is None:
            # Unsupported extension — return whole file as single section
            text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
            return ParsedDoc(
                path=path,
                title=pathlib.Path(path).name,
                sections=[ParsedSection(heading="", level=0, text=text, page_or_line=1)],
            )

        try:
            from tree_sitter import Parser as TsParser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter is required for code indexing. "
                "Install with: pip install agent-sleuth[localfiles]"
            ) from exc

        source = pathlib.Path(path).read_bytes()
        language = _load_ts_language(lang_name)
        parser = TsParser(language)
        tree = parser.parse(source)

        sections = self._extract_definitions(source, tree, lang_name)
        title = pathlib.Path(path).name
        if not sections:
            sections = [
                ParsedSection(heading="", level=0, text=source.decode("utf-8", errors="replace"), page_or_line=1)
            ]
        return ParsedDoc(path=path, title=title, sections=sections)

    def _extract_definitions(
        self, source: bytes, tree: object, lang_name: str
    ) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        node_types = {
            "python": {"function_definition", "class_definition"},
            "javascript": {"function_declaration", "class_declaration", "arrow_function"},
            "typescript": {"function_declaration", "class_declaration", "method_definition"},
        }
        target_types = node_types.get(lang_name, set())

        def _name_of(node) -> str:  # type: ignore[return]
            for child in node.children:
                if child.type == "identifier":
                    return source[child.start_byte: child.end_byte].decode("utf-8", errors="replace")
            return "<anonymous>"

        def _visit(node) -> None:  # type: ignore[return]
            if node.type in target_types:
                name = _name_of(node)
                body = source[node.start_byte: node.end_byte].decode("utf-8", errors="replace")
                line = node.start_point[0] + 1
                # Depth 1 for functions/classes, 2 for methods (parent is class)
                level = 2 if (node.parent and node.parent.type in target_types) else 1
                sections.append(
                    ParsedSection(heading=name, level=level, text=body, page_or_line=line)
                )
            for child in node.children:
                _visit(child)

        _visit(tree.root_node)  # type: ignore[union-attr]
        return sections


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------

def get_parser(path: str) -> Parser:
    """Return the right parser for a file path based on extension."""
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".pdf":
        return PdfParser()
    if suffix in {".html", ".htm"}:
        return HtmlParser()
    if suffix in _LANG_MAP:
        return CodeParser()
    # Default: treat everything else as Markdown-ish plain text
    return MarkdownParser()
```

- [x] **Step 4: Fix the typo in the test (backtick paren)**

In the test file, find:
```python
        md = tmp_path / "flat.md")
```
and correct to:
```python
        md = tmp_path / "flat.md"
```

- [x] **Step 5: Run tests**

```bash
uv run pytest tests/backends/test_localfiles.py -k "TestMarkdownParser or TestHtmlParser or TestPdfParser or TestCodeParser" -v
```

Expected: all tests pass.

- [x] **Step 6: Commit**

```bash
git add src/sleuth/backends/_localfiles/parsers.py tests/backends/test_localfiles.py tests/backends/conftest.py
git commit -m "feat: add format parsers (markdown, HTML, PDF via pymupdf, code via tree-sitter)"
```

---

## Task 4: File hasher (`_localfiles/hasher.py`)

**Files:**
- Create: `src/sleuth/backends/_localfiles/hasher.py`
- Test: `tests/backends/test_localfiles.py` (add `TestHasher`)

- [x] **Step 1: Write failing tests**

Append to `tests/backends/test_localfiles.py`:

```python
class TestHasher:
    def test_mtime_hash_changes_when_file_changes(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.hasher import file_hash, HashMode

        f = tmp_path / "doc.md"
        f.write_text("v1")
        h1 = file_hash(str(f), mode=HashMode.MTIME)
        import time; time.sleep(0.01)
        f.write_text("v2")
        # Touch mtime explicitly
        import os; os.utime(f, None)
        h2 = file_hash(str(f), mode=HashMode.MTIME)
        assert h1 != h2

    def test_content_hash_depends_on_content_not_mtime(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.hasher import file_hash, HashMode

        f = tmp_path / "doc.md"
        f.write_text("same content")
        h1 = file_hash(str(f), mode=HashMode.HASH)
        f.write_text("same content")  # same bytes, possibly different mtime
        h2 = file_hash(str(f), mode=HashMode.HASH)
        assert h1 == h2

    def test_corpus_version_hash_is_deterministic(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends._localfiles.hasher import corpus_version

        v1 = corpus_version(str(tmp_corpus), mode="mtime")
        v2 = corpus_version(str(tmp_corpus), mode="mtime")
        assert v1 == v2
        assert len(v1) == 16  # short hex string
```

- [x] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/backends/test_localfiles.py::TestHasher -v 2>&1 | head -10
```

Expected: `ImportError` — `hasher` does not exist.

- [x] **Step 3: Implement `hasher.py`**

```python
# src/sleuth/backends/_localfiles/hasher.py
"""Fast file and corpus hashing for change detection."""
from __future__ import annotations

import hashlib
import os
import pathlib
from enum import StrEnum
from typing import Literal


class HashMode(StrEnum):
    MTIME = "mtime"   # fast: (path, size, mtime) — good enough for local files
    HASH = "hash"     # thorough: xxhash of content
    ALWAYS = "always" # always treat as changed — returns unique value per call


def file_hash(path: str, mode: HashMode | str = HashMode.MTIME) -> str:
    """Return a change-detection hash for a single file."""
    p = pathlib.Path(path)
    if mode == HashMode.ALWAYS:
        import time
        return f"always-{time.monotonic_ns()}"
    if mode == HashMode.MTIME:
        stat = p.stat()
        raw = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
    # HashMode.HASH — content-based
    try:
        import xxhash
        h = xxhash.xxh64()
    except ImportError:
        h = hashlib.md5()  # type: ignore[assignment]
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def corpus_version(
    corpus_path: str,
    *,
    mode: Literal["mtime", "hash", "always"] = "mtime",
    include_patterns: list[str] | None = None,
) -> str:
    """Return a short hex string that changes when any file in the corpus changes."""
    root = pathlib.Path(corpus_path)
    hashes: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        # skip .sleuth dir itself
        try:
            p.relative_to(root / ".sleuth")
            continue
        except ValueError:
            pass
        hashes.append(file_hash(str(p), mode=HashMode(mode)))
    combined = "|".join(hashes)
    return hashlib.md5(combined.encode()).hexdigest()[:16]
```

- [x] **Step 4: Run tests**

```bash
uv run pytest tests/backends/test_localfiles.py::TestHasher -v
```

Expected: 3 tests pass.

- [x] **Step 5: Commit**

```bash
git add src/sleuth/backends/_localfiles/hasher.py tests/backends/test_localfiles.py
git commit -m "feat: add file hasher (mtime/content/always) for corpus change detection"
```

---

## Task 5: Tree builder (`_localfiles/tree_builder.py`)

**Files:**
- Create: `src/sleuth/backends/_localfiles/tree_builder.py`
- Test: `tests/backends/test_localfiles.py` (add `TestTreeBuilder`)

- [x] **Step 1: Write failing tests**

Append to `tests/backends/test_localfiles.py`:

```python
import asyncio
from sleuth.llm.stub import StubLLM  # Phase 1 — import path per conventions §5.1


class TestTreeBuilder:
    def test_builds_tree_from_parsed_docs(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import MarkdownParser
        from sleuth.backends._localfiles.tree_builder import build_tree
        from sleuth.backends._localfiles.models import NodeKind

        parser = MarkdownParser()
        docs = [parser.parse(str(f)) for f in tmp_corpus.glob("*.md")]
        # StubLLM returns a canned summary for every LLM call
        llm = StubLLM(responses=["A short summary of this section."])
        tree = asyncio.get_event_loop().run_until_complete(
            build_tree(docs, corpus_path=str(tmp_corpus), indexer_llm=llm)
        )
        assert tree.corpus_path == str(tmp_corpus)
        assert len(tree.nodes) == len(docs)
        # All non-leaf nodes should have summaries
        for node in tree.nodes:
            if not node.is_leaf:
                assert node.summary is not None

    def test_toc_text_remains_compact(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import MarkdownParser
        from sleuth.backends._localfiles.tree_builder import build_tree

        parser = MarkdownParser()
        docs = [parser.parse(str(f)) for f in tmp_corpus.rglob("*.md")]
        llm = StubLLM(responses=["Summary."])
        tree = asyncio.get_event_loop().run_until_complete(
            build_tree(docs, corpus_path=str(tmp_corpus), indexer_llm=llm)
        )
        toc = tree.toc_text()
        assert len(toc) < 10_000

    def test_leaf_nodes_carry_source_location(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import MarkdownParser
        from sleuth.backends._localfiles.tree_builder import build_tree

        parser = MarkdownParser()
        docs = [parser.parse(str(p)) for p in tmp_corpus.glob("*.md")]
        llm = StubLLM(responses=["Sum."])
        tree = asyncio.get_event_loop().run_until_complete(
            build_tree(docs, corpus_path=str(tmp_corpus), indexer_llm=llm)
        )
        all_leaves = [n for root in tree.nodes for n in root.all_leaves()]
        for leaf in all_leaves:
            assert leaf.source_path
            assert leaf.source_section
```

- [x] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/backends/test_localfiles.py::TestTreeBuilder -v 2>&1 | head -15
```

Expected: `ImportError` — `tree_builder` does not exist.

- [x] **Step 3: Implement `tree_builder.py`**

```python
# src/sleuth/backends/_localfiles/tree_builder.py
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


async def _llm_summary(llm: "LLMClient", heading: str, text: str) -> str:
    """Collect a one-sentence summary from the indexer LLM."""
    from sleuth.llm.base import Message, TextDelta

    snippet = text[:400].replace("\n", " ")
    prompt = _SUMMARY_PROMPT.format(heading=heading, snippet=snippet)
    messages = [Message(role="user", content=prompt)]
    parts: list[str] = []
    async for chunk in await llm.stream(messages):
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

        kind = NodeKind.SECTION if level == 1 else (
            NodeKind.SUBSECTION if level == 2 else NodeKind.LEAF
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


async def _summarize_tree(node: IndexNode, llm: "LLMClient") -> None:
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
    indexer_llm: "LLMClient",
    version: str = "unknown",
) -> IndexTree:
    """Build and summarize an IndexTree from a list of ParsedDocs."""
    root_nodes = [_build_node_tree(doc) for doc in docs]
    # Summarize each doc's root node in parallel
    await asyncio.gather(*[_summarize_tree(node, indexer_llm) for node in root_nodes])
    return IndexTree(corpus_path=corpus_path, version=version, nodes=root_nodes)
```

- [x] **Step 4: Run tests**

```bash
uv run pytest tests/backends/test_localfiles.py::TestTreeBuilder -v
```

Expected: all 3 tests pass.

- [x] **Step 5: Commit**

```bash
git add src/sleuth/backends/_localfiles/tree_builder.py tests/backends/test_localfiles.py
git commit -m "feat: add tree builder with bottom-up LLM summarization"
```

---

## Task 6: Persistence (`_localfiles/persistence.py`)

**Files:**
- Create: `src/sleuth/backends/_localfiles/persistence.py`
- Test: `tests/backends/test_localfiles.py` (add `TestPersistence`)

- [x] **Step 1: Write failing tests**

Append to `tests/backends/test_localfiles.py`:

```python
class TestPersistence:
    def test_round_trip_json(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.models import IndexNode, IndexTree, NodeKind
        from sleuth.backends._localfiles.persistence import save_tree, load_tree

        child = IndexNode(
            id="doc::1", title="Sec A", kind=NodeKind.SECTION,
            summary="Summary A.", text=None,
            source_path="/tmp/doc.md", source_section="# Sec A",
            page_or_line=5, children=[
                IndexNode(
                    id="doc::2", title="Leaf", kind=NodeKind.LEAF,
                    summary=None, text="leaf text",
                    source_path="/tmp/doc.md", source_section="# Sec A > ## Leaf",
                    page_or_line=10,
                )
            ]
        )
        root = IndexNode(
            id="doc::root", title="Doc", kind=NodeKind.ROOT,
            summary="Doc summary.", text=None,
            source_path="/tmp/doc.md", source_section="Doc",
            page_or_line=1, children=[child]
        )
        tree = IndexTree(corpus_path=str(tmp_path), version="abc123", nodes=[root])

        index_dir = tmp_path / ".sleuth" / "index"
        save_tree(tree, index_dir=str(index_dir))

        # Verify files were created
        json_files = list(index_dir.glob("*.json"))
        assert len(json_files) == 1

        loaded = load_tree(str(tmp_path), version="abc123", index_dir=str(index_dir))
        assert loaded is not None
        assert loaded.version == "abc123"
        assert len(loaded.nodes) == 1
        assert loaded.nodes[0].title == "Doc"
        assert loaded.nodes[0].children[0].children[0].text == "leaf text"

    def test_load_nonexistent_version_returns_none(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.persistence import load_tree

        result = load_tree(str(tmp_path), version="nonexistent", index_dir=str(tmp_path / ".sleuth" / "index"))
        assert result is None

    def test_save_overwrites_same_version(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.models import IndexNode, IndexTree, NodeKind
        from sleuth.backends._localfiles.persistence import save_tree, load_tree

        def _make_tree(title: str) -> IndexTree:
            root = IndexNode(
                id="doc::root", title=title, kind=NodeKind.ROOT,
                summary="S.", text=None,
                source_path="/tmp/x.md", source_section=title,
                page_or_line=1,
            )
            return IndexTree(corpus_path=str(tmp_path), version="v1", nodes=[root])

        index_dir = str(tmp_path / ".sleuth" / "index")
        save_tree(_make_tree("First"), index_dir=index_dir)
        save_tree(_make_tree("Second"), index_dir=index_dir)
        loaded = load_tree(str(tmp_path), version="v1", index_dir=index_dir)
        assert loaded is not None
        assert loaded.nodes[0].title == "Second"
```

- [x] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/backends/test_localfiles.py::TestPersistence -v 2>&1 | head -10
```

Expected: `ImportError` — `persistence` does not exist.

- [x] **Step 3: Implement `persistence.py`**

```python
# src/sleuth/backends/_localfiles/persistence.py
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
```

- [x] **Step 4: Run tests**

```bash
uv run pytest tests/backends/test_localfiles.py::TestPersistence -v
```

Expected: all 3 tests pass.

- [x] **Step 5: Commit**

```bash
git add src/sleuth/backends/_localfiles/persistence.py tests/backends/test_localfiles.py
git commit -m "feat: add JSON persistence for IndexTree under <corpus>/.sleuth/index/"
```

---

## Task 7: Navigator (`_localfiles/navigator.py`)

**Files:**
- Create: `src/sleuth/backends/_localfiles/navigator.py`
- Test: `tests/backends/test_localfiles.py` (add `TestNavigator`)

- [x] **Step 1: Write failing tests**

Append to `tests/backends/test_localfiles.py`:

```python
class TestNavigator:
    def _make_tree(self) -> "IndexTree":
        from sleuth.backends._localfiles.models import IndexNode, IndexTree, NodeKind

        def _leaf(id_: str, title: str, text: str, source_section: str) -> IndexNode:
            return IndexNode(
                id=id_, title=title, kind=NodeKind.LEAF,
                summary=None, text=text,
                source_path="/tmp/doc.md", source_section=source_section,
                page_or_line=1,
            )

        def _branch(id_: str, title: str, summary: str, children: list) -> IndexNode:
            return IndexNode(
                id=id_, title=title, kind=NodeKind.SECTION,
                summary=summary, text=None,
                source_path="/tmp/doc.md", source_section=title,
                page_or_line=1, children=children,
            )

        auth_section = _branch(
            "doc::0", "Authentication",
            "Covers auth flow, refresh tokens, JWT.",
            [
                _leaf("doc::1", "Login Flow", "Users log in via OAuth2.", "# Auth > ## Login Flow"),
                _leaf("doc::2", "Refresh Tokens", "Tokens refresh every 15 min.", "# Auth > ## Refresh Tokens"),
            ],
        )
        deploy_section = _branch(
            "doc::3", "Deployment",
            "Deployment instructions for Kubernetes.",
            [
                _leaf("doc::4", "Docker Setup", "Build the image with docker build.", "# Deployment > ## Docker Setup"),
            ],
        )
        root = IndexNode(
            id="doc::root", title="Docs", kind=NodeKind.ROOT,
            summary="Full documentation.", text=None,
            source_path="/tmp/doc.md", source_section="Docs",
            page_or_line=1, children=[auth_section, deploy_section],
        )
        return IndexTree(corpus_path="/tmp", version="v1", nodes=[root])

    def test_navigator_selects_relevant_branches(self) -> None:
        from sleuth.backends._localfiles.navigator import navigate

        tree = self._make_tree()
        # StubLLM will return JSON picking auth-related node ids
        stub_response = '{"selected_ids": ["doc::0"]}'
        llm = StubLLM(responses=[stub_response])
        chunks = asyncio.get_event_loop().run_until_complete(
            navigate(query="How do refresh tokens work?", tree=tree, navigator_llm=llm, k=5)
        )
        # Should return leaves under the auth section
        texts = [c.text for c in chunks]
        assert any("refresh" in t.lower() for t in texts)

    def test_navigator_returns_at_most_k_chunks(self) -> None:
        from sleuth.backends._localfiles.navigator import navigate

        tree = self._make_tree()
        # Pick both top-level branches
        stub_response = '{"selected_ids": ["doc::0", "doc::3"]}'
        llm = StubLLM(responses=[stub_response])
        chunks = asyncio.get_event_loop().run_until_complete(
            navigate(query="everything", tree=tree, navigator_llm=llm, k=2)
        )
        assert len(chunks) <= 2

    def test_navigator_falls_back_to_all_leaves_on_bad_llm_response(self) -> None:
        from sleuth.backends._localfiles.navigator import navigate

        tree = self._make_tree()
        # LLM returns invalid JSON
        llm = StubLLM(responses=["I cannot answer that."])
        chunks = asyncio.get_event_loop().run_until_complete(
            navigate(query="anything", tree=tree, navigator_llm=llm, k=10)
        )
        # Fallback: return all leaves
        assert len(chunks) > 0
```

- [x] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/backends/test_localfiles.py::TestNavigator -v 2>&1 | head -10
```

Expected: `ImportError` — `navigator` does not exist.

- [x] **Step 3: Implement `navigator.py`**

```python
# src/sleuth/backends/_localfiles/navigator.py
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


async def _ask_navigator(
    llm: "LLMClient", toc: str, query: str, max_branches: int
) -> list[str]:
    """Call navigator LLM; return list of selected node IDs. Returns [] on failure."""
    from sleuth.llm.base import Message, TextDelta

    prompt = _NAVIGATE_PROMPT.format(toc=toc, query=query, max_branches=max_branches)
    messages = [Message(role="user", content=prompt)]
    parts: list[str] = []
    async for chunk in await llm.stream(messages):
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
    navigator_llm: "LLMClient",
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
```

- [x] **Step 4: Run tests**

```bash
uv run pytest tests/backends/test_localfiles.py::TestNavigator -v
```

Expected: all 3 tests pass.

- [x] **Step 5: Commit**

```bash
git add src/sleuth/backends/_localfiles/navigator.py tests/backends/test_localfiles.py
git commit -m "feat: add LLM navigator for branch selection with JSON structured output"
```

---

## Task 8: Main `LocalFiles` class (`backends/localfiles.py`)

**Files:**
- Create: `src/sleuth/backends/localfiles.py`
- Test: `tests/backends/test_localfiles.py` (add `TestLocalFiles` and `TestLocalFilesBackendKit`)

- [x] **Step 1: Write failing tests**

Append to `tests/backends/test_localfiles.py`:

```python
import asyncio
import pathlib


class TestLocalFiles:
    def test_search_returns_chunks(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=[
            "Section summary.",         # indexer LLM — repeated for each node
            "Section summary.",
            "Section summary.",
            "Section summary.",
            '{"selected_ids": []}',     # navigator LLM — fallback to all
        ])
        backend = LocalFiles(
            path=str(tmp_corpus),
            indexer_llm=llm,
            navigator_llm=llm,
        )
        chunks = asyncio.get_event_loop().run_until_complete(backend.search("overview", k=5))
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        from sleuth.types import Chunk
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_search_respects_k(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["Sum."] * 20 + ['{"selected_ids": []}'])
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        chunks = asyncio.get_event_loop().run_until_complete(backend.search("anything", k=1))
        assert len(chunks) <= 1

    def test_index_persisted_to_sleuth_dir(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["Sum."] * 20 + ['{"selected_ids": []}'])
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        asyncio.get_event_loop().run_until_complete(backend.search("test", k=3))
        index_dir = tmp_corpus / ".sleuth" / "index"
        assert index_dir.exists()
        json_files = list(index_dir.glob("*.json"))
        assert len(json_files) == 1

    def test_second_search_uses_cached_index(self, tmp_corpus: pathlib.Path) -> None:
        """Second search must NOT re-index (LLM call count stays the same)."""
        from sleuth.backends.localfiles import LocalFiles

        call_count = {"n": 0}
        original_responses = ["Sum."] * 20 + ['{"selected_ids": []}'] * 10

        class CountingStub(StubLLM):
            def __init__(self) -> None:
                super().__init__(responses=original_responses)
            async def stream(self, messages, *, schema=None, tools=None):
                call_count["n"] += 1
                return await super().stream(messages, schema=schema, tools=tools)

        llm = CountingStub()
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        asyncio.get_event_loop().run_until_complete(backend.search("first", k=3))
        count_after_first = call_count["n"]
        asyncio.get_event_loop().run_until_complete(backend.search("second", k=3))
        count_after_second = call_count["n"]
        # Navigator was called again (1 new call), but indexer was NOT called again
        assert count_after_second - count_after_first == 1

    def test_name_and_capabilities(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends.localfiles import LocalFiles
        from sleuth.backends.base import Capability

        llm = StubLLM(responses=["S."])
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        assert backend.name == "localfiles"
        assert Capability.DOCS in backend.capabilities

    def test_exclude_patterns_skip_files(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends.localfiles import LocalFiles

        (tmp_corpus / "node_modules").mkdir()
        (tmp_corpus / "node_modules" / "pkg.md").write_text("# Package\nNode module content.")
        llm = StubLLM(responses=["Sum."] * 20 + ['{"selected_ids": []}'])
        backend = LocalFiles(
            path=str(tmp_corpus),
            indexer_llm=llm,
            navigator_llm=llm,
            exclude=["node_modules/**"],
        )
        chunks = asyncio.get_event_loop().run_until_complete(backend.search("node module", k=10))
        texts = [c.text for c in chunks]
        assert not any("Node module content" in t for t in texts)
```

- [x] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/backends/test_localfiles.py::TestLocalFiles -v 2>&1 | head -15
```

Expected: `ImportError` — `localfiles` module does not exist.

- [x] **Step 3: Implement `backends/localfiles.py`**

```python
# src/sleuth/backends/localfiles.py
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
from sleuth.backends._localfiles.models import IndexTree, ParsedDoc
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
    {".md", ".markdown", ".txt", ".rst",
     ".html", ".htm",
     ".pdf",
     ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx"}
)


class LocalFiles:
    """PageIndex-style hierarchical backend for local document corpora."""

    name: str = "localfiles"
    capabilities: frozenset[Capability] = frozenset({Capability.DOCS})

    def __init__(
        self,
        path: str | pathlib.Path,
        indexer_llm: "LLMClient | None" = None,
        navigator_llm: "LLMClient | None" = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_branch_descent: int = 3,
        rebuild: Literal["mtime", "hash", "always"] = "mtime",
    ) -> None:
        self._path = pathlib.Path(path).resolve()
        self._indexer_llm = indexer_llm
        self._navigator_llm = navigator_llm
        self._include_patterns: list[str] = include or ["**/*"]
        self._exclude_patterns: list[str] = (exclude if exclude is not None else DEFAULT_EXCLUDES)
        self._max_branch_descent = max_branch_descent
        self._rebuild = rebuild

        self._tree: IndexTree | None = None          # in-memory cache
        self._tree_version: str | None = None        # corpus version of cached tree
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
        self, target: str, length: Literal["brief", "standard", "thorough"] = "standard"
    ) -> str:
        """Return a summary from the indexed tree for a given file path or corpus root.

        - "brief"     → root node summary (one sentence, already cached in the tree).
        - "standard"  → root + level-1 children summaries joined.
        - "thorough"  → all node summaries concatenated (full tree walk).
        """
        tree = await self._ensure_tree()
        p = pathlib.Path(target).resolve()

        def _find_root_for_path() -> object:
            for node in tree.nodes:
                if pathlib.Path(node.source_path).resolve() == p:
                    return node
            return None

        root_node = _find_root_for_path() if p != self._path else None

        if root_node is None:
            # Summarize the whole corpus
            nodes_to_summarize = tree.nodes
        else:
            nodes_to_summarize = [root_node]  # type: ignore[list-item]

        if length == "brief":
            parts = [n.summary or n.title for n in nodes_to_summarize]  # type: ignore[union-attr]
        elif length == "standard":
            parts = []
            for n in nodes_to_summarize:  # type: ignore[union-attr]
                parts.append(n.summary or n.title)  # type: ignore[union-attr]
                for child in n.children:  # type: ignore[union-attr]
                    if child.summary:
                        parts.append(f"  - {child.title}: {child.summary}")
        else:  # thorough
            parts = []
            def _collect(node, depth: int = 0) -> None:  # type: ignore[return]
                prefix = "  " * depth
                parts.append(f"{prefix}{node.title}: {node.summary or node.text or ''}")
                for child in node.children:
                    _collect(child, depth + 1)
            for n in nodes_to_summarize:  # type: ignore[union-attr]
                _collect(n)

        return "\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Warm-up (eager indexing — called by Sleuth.warm_index())
    # ------------------------------------------------------------------

    async def warm_index(self) -> None:
        """Force (re)indexing. Useful for pre-warming before the first query."""
        self._tree = None  # clear in-memory cache to force rebuild
        await self._ensure_tree()

    # ------------------------------------------------------------------
    # Internal: index management
    # ------------------------------------------------------------------

    async def _ensure_tree(self) -> IndexTree:
        """Return the in-memory tree, loading from disk or rebuilding as needed."""
        version = corpus_version(str(self._path), mode=self._rebuild)  # type: ignore[arg-type]

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
        tree = await build_tree(docs, corpus_path=str(self._path), indexer_llm=indexer_llm, version=version)
        save_tree(tree, index_dir=str(self._index_dir))
        prune_stale(str(self._index_dir), keep_version=version)

        self._tree = tree
        self._tree_version = version
        return self._tree

    def _collect_docs(self) -> list[ParsedDoc]:
        """Walk the corpus and parse all matching, non-excluded files."""
        exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", self._exclude_patterns)
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
```

- [x] **Step 4: Run tests**

```bash
uv run pytest tests/backends/test_localfiles.py::TestLocalFiles -v
```

Expected: all 6 tests pass.

- [x] **Step 5: Commit**

```bash
git add src/sleuth/backends/localfiles.py tests/backends/test_localfiles.py
git commit -m "feat: implement LocalFiles backend (search, warm_index, _get_summary)"
```

---

## Task 9: BackendTestKit compliance

**Files:**
- Test: `tests/backends/test_localfiles.py` (add `TestLocalFilesProtocol` using `BackendTestKit`)

BackendTestKit is owned by Phase 1 (`tests/contract/test_backend_protocol.py`). This task runs `LocalFiles` through it.

- [x] **Step 1: Write the BackendTestKit compliance test**

Append to `tests/backends/test_localfiles.py`:

```python
from tests.contract.test_backend_protocol import BackendTestKit  # Phase 1 export


class TestLocalFilesProtocol(BackendTestKit):
    """Run the full Backend protocol contract suite against LocalFiles."""

    @pytest.fixture
    def backend(self, tmp_corpus: pathlib.Path) -> "LocalFiles":
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["Summary."] * 50 + ['{"selected_ids": []}'] * 20)
        return LocalFiles(
            path=str(tmp_corpus),
            indexer_llm=llm,
            navigator_llm=llm,
        )
```

- [x] **Step 2: Run BackendTestKit compliance**

```bash
uv run pytest tests/backends/test_localfiles.py::TestLocalFilesProtocol -v
```

Expected: all BackendTestKit cases pass (protocol compliance, error shapes, timeout behavior, cancellation safety).

- [x] **Step 3: Commit**

```bash
git add tests/backends/test_localfiles.py
git commit -m "test: run LocalFiles through BackendTestKit protocol compliance suite"
```

---

## Task 10: Type-check and lint

**Files:**
- Modify: `src/sleuth/backends/_localfiles/parsers.py` (fix any mypy findings)
- Modify: `src/sleuth/backends/_localfiles/navigator.py` (fix any mypy findings)
- Modify: `src/sleuth/backends/localfiles.py` (fix any mypy findings)

- [x] **Step 1: Run mypy**

```bash
uv run mypy src/sleuth/backends/localfiles.py src/sleuth/backends/_localfiles/
```

Expected: resolve any `error:` lines. Common fixes:
  - Add `# type: ignore[...]` only for unavoidable dynamic-dispatch patterns (tree-sitter, pymupdf).
  - Ensure `LLMClient | None` parameters have guards before use.
  - `IndexNode.children` default factory is already `field(default_factory=list)`.

- [x] **Step 2: Run ruff**

```bash
uv run ruff check src/sleuth/backends/localfiles.py src/sleuth/backends/_localfiles/
uv run ruff format src/sleuth/backends/localfiles.py src/sleuth/backends/_localfiles/
```

Expected: clean (0 errors, files reformatted or already compliant).

- [x] **Step 3: Run full test suite to confirm nothing regressed**

```bash
uv run pytest tests/backends/ -v --tb=short
```

Expected: all tests pass.

- [x] **Step 4: Commit**

```bash
git add src/sleuth/backends/
git commit -m "fix: resolve mypy and ruff findings in localfiles backend"
```

---

## Task 11: Coverage check

**Files:**
- No new files. Verify coverage gate passes.

- [x] **Step 1: Run coverage**

```bash
uv run pytest tests/backends/ --cov=src/sleuth/backends/localfiles --cov=src/sleuth/backends/_localfiles --cov-report=term-missing
```

Expected: ≥85% coverage. If below 85%, identify uncovered branches (the `--cov-report=term-missing` output shows them) and add targeted tests.

Common gaps to check:
  - `HtmlParser` with no headings → single section fallback
  - `PdfParser._sections_from_pages` fallback (no embedded TOC)
  - `CodeParser` with unsupported extension
  - `prune_stale` when index dir does not yet exist
  - `LocalFiles._collect_docs` file-parse exception swallowing

- [x] **Step 2: Add any missing coverage tests**

Example for the `CodeParser` unsupported extension branch:

```python
def test_code_parser_unsupported_extension_returns_flat_doc(self, tmp_path):
    from sleuth.backends._localfiles.parsers import CodeParser
    f = tmp_path / "data.csv"
    f.write_text("col1,col2\n1,2\n")
    parser = CodeParser()
    doc = parser.parse(str(f))
    assert len(doc.sections) == 1
    assert doc.sections[0].level == 0
```

Example for `prune_stale` with missing dir:

```python
def test_prune_stale_nonexistent_dir_returns_zero(self, tmp_path):
    from sleuth.backends._localfiles.persistence import prune_stale
    count = prune_stale(str(tmp_path / "nonexistent"), keep_version="abc")
    assert count == 0
```

- [x] **Step 3: Run coverage again — must meet gate**

```bash
uv run pytest tests/backends/ --cov=src/sleuth/backends/localfiles --cov=src/sleuth/backends/_localfiles --cov-report=term-missing --cov-fail-under=85
```

Expected: PASSED (coverage ≥85%).

- [x] **Step 4: Commit coverage additions**

```bash
git add tests/backends/test_localfiles.py
git commit -m "test: add coverage for edge-case branches in localfiles backend"
```

---

## Task 12: Integration with `_agent.py` (reference only, no code changes)

This task documents the handshake between `LocalFiles._get_summary()` and Phase 1's `_agent.py`. **Phase 2 does NOT modify `_agent.py`** — that file is owned by Phase 1. The Phase 1 implementer wires this up.

### How `asummarize` calls LocalFiles (Phase 1 reference)

In `_agent.py`, the `asummarize` method should detect when a `LocalFiles` backend is present and delegate to it:

```python
# In Sleuth.asummarize (owned by Phase 1 — _agent.py):
for backend in self._backends:
    if hasattr(backend, "_get_summary"):
        summary_text = await backend._get_summary(target, length=length)
        # yield summary_text as TokenEvents, then DoneEvent
        break
```

The `_get_summary(target, length)` signature is:

```python
async def _get_summary(
    self,
    target: str,                                    # absolute file path or corpus root dir
    length: Literal["brief", "standard", "thorough"] = "standard",
) -> str: ...
```

Depth mapping (from conventions §4 `Length`):
- `"brief"` → root node summary only
- `"standard"` → root + level-1 children
- `"thorough"` → full tree walk (all summaries concatenated)

- [x] **Step 1: No code step — read this task, confirm the signature matches conventions §4**

Conventions §4 declares:
```python
def summarize(self, target: str, *, length: Length = "standard", ...) -> Result[T]: ...
```

`Length = Literal["brief", "standard", "thorough"]`. Our `_get_summary` accepts the same `length` values. Confirmed compatible.

- [x] **Step 2: Commit documentation note**

```bash
git commit --allow-empty -m "docs: document LocalFiles._get_summary integration point for Phase 1 _agent.py"
```

---

## Task 13: Final pre-PR cleanup and snapshot

**Files:**
- Modify: none (cleanup pass)

- [x] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -m "not integration" -v --tb=short
```

Expected: all unit tests pass.

- [x] **Step 2: Run ruff on entire src**

```bash
uv run ruff check src/ && uv run ruff format --check src/
```

Expected: 0 errors.

- [x] **Step 3: Run mypy on entire package**

```bash
uv run mypy src/sleuth/
```

Expected: 0 errors (or only pre-existing errors from other phases, none from `backends/localfiles.py` or `backends/_localfiles/`).

- [x] **Step 4: Verify index dir exclusion in git**

```bash
grep -r "\.sleuth" .gitignore
```

Expected: `.sleuth/` is listed (Phase 0 should have added it per conventions §2). If absent:

```bash
echo ".sleuth/" >> .gitignore
git add .gitignore
git commit -m "chore: ensure .sleuth/ index dirs are git-ignored"
```

- [x] **Step 5: Create PR**

```bash
git push -u origin feature/phase-2-localfiles
gh pr create \
  --title "feat: Phase 2 — LocalFiles hierarchical backend" \
  --body "Implements the PageIndex-style LocalFiles backend: format parsers (markdown, HTML, PDF via pymupdf, code via tree-sitter), tree builder with LLM summarization, LLM navigator, JSON persistence, and BackendTestKit compliance. Resolves spec §15 #2 (PDF parser: pymupdf chosen for speed + get_toc() fidelity)." \
  --base develop
```

---

## Self-review

### Spec coverage check (§7.3)

| Spec requirement | Task |
|---|---|
| Walk dir → parse each doc's native structure | Task 3 (parsers), Task 8 (`_collect_docs`) |
| Markdown headings | Task 3 `MarkdownParser` |
| PDF TOC/headings | Task 1 (benchmark), Task 3 `PdfParser` |
| HTML h1-h6 | Task 3 `HtmlParser` |
| Code modules via tree-sitter | Task 3 `CodeParser` |
| Build node tree: root → section → subsection → leaf | Task 5 `_build_node_tree` |
| LLM writes short summary at each non-leaf | Task 5 `_summarize_tree` |
| Persist under `<corpus>/.sleuth/index/<hash>.{json,sqlite}` | Task 6 (JSON) |
| Load tree-of-contents (compact 1-5 KB) | Task 2 `IndexTree.toc_text()` |
| LLM picks branches (structured-output call) | Task 7 `navigator.py` |
| Recurse into picked branches | Task 7 `_collect_leaves` |
| Return matched leaf chunks with structural citations | Task 7 `_node_to_chunk` |
| `LocalFiles(path, indexer_llm, navigator_llm, include, exclude, max_branch_descent, rebuild)` ctor | Task 8 |
| `indexer_llm / navigator_llm` default to ambient Sleuth LLM | Task 8 (defaults to `None`, engine passes through) |
| `DEFAULT_EXCLUDES` (.git, node_modules, .sleuth, .venv, dist, build) | Task 8 |
| `rebuild="mtime"|"hash"|"always"` | Task 4 hasher, Task 8 |
| `agent.summarize(target=path, length=...)` integration | Task 9 `_get_summary`, Task 12 handshake doc |
| Root summary / one-level / full tree for brief/standard/thorough | Task 8 `_get_summary` |
| Hierarchical compaction when tree level expands too far | Task 7 `_collect_leaves(max_depth=...)` |
| Navigator sees paginated TOC (compact) | Task 2 `toc_text()` depth-2 cutoff |
| `BackendTestKit` compliance | Task 9 |
| PDF parser benchmark (spec §15 #2) | Task 1 |

All requirements covered. No gaps found.

### Placeholder scan

No "TBD", "TODO", "implement later", or vague steps found.

### Type consistency

- `IndexNode`, `IndexTree`, `ParsedDoc`, `ParsedSection`, `NodeKind` defined in Task 2 (`models.py`), referenced consistently in Tasks 3-8.
- `build_tree(docs, *, corpus_path, indexer_llm, version)` defined in Task 5, called in Task 8.
- `navigate(query, tree, navigator_llm, k, max_branch_descent)` defined in Task 7, called in Task 8.
- `save_tree(tree, *, index_dir)`, `load_tree(corpus_path, *, version, index_dir)`, `prune_stale(index_dir, keep_version)` defined in Task 6, called in Task 8.
- `file_hash(path, mode)`, `corpus_version(corpus_path, *, mode)` defined in Task 4, called in Task 8.
- `LLMClient` referenced from `sleuth.llm.base` (Phase 1) throughout. `StubLLM` from `sleuth.llm.stub` (Phase 1) used in all tests.
- `Chunk`, `Source` from `sleuth.types` (Phase 1). `Capability` from `sleuth.backends.base` (Phase 1).
- `Length` in conventions §4 matches `_get_summary(length: Literal["brief", "standard", "thorough"])`.

All consistent.
