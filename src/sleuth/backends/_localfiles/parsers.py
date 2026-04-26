"""Format-specific document parsers for LocalFiles indexing.

Each parser takes a file path and returns a ParsedDoc — a flat list of
(heading, level, text, page_or_line) sections that the tree builder
then assembles into an IndexTree.

Parsers are intentionally sync (file I/O bound, not network bound).
"""

from __future__ import annotations

import pathlib
import re
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass

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
                sections=[ParsedSection(heading="", level=0, text=text.strip(), page_or_line=1)],
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

        title = (
            sections[0].heading if sections and sections[0].level == 1 else pathlib.Path(path).stem
        )
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

            def handle_starttag(self, tag: str, attrs: list[Any]) -> None:
                if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    if self._pending_level is not None:
                        self._sections.append(
                            (
                                self._pending_level,
                                self._pending_heading,
                                "".join(self._current_body).strip(),
                            )
                        )
                        self._current_body = []
                    self._in_heading = int(tag[1])
                    self._current_heading_text = []

            def handle_endtag(self, tag: str) -> None:
                if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    self._pending_heading = "".join(self._current_heading_text).strip()
                    self._pending_level = int(tag[1])
                    self._in_heading = None

            def handle_data(self, data: str) -> None:
                if self._in_heading is not None:
                    self._current_heading_text.append(data)
                elif self._pending_level is not None:
                    self._current_body.append(data)

            def get_sections(self) -> list[tuple[int, str, str]]:
                if self._pending_level is not None:
                    self._sections.append(
                        (
                            self._pending_level,
                            self._pending_heading,
                            "".join(self._current_body).strip(),
                        )
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
            ParsedSection(heading=h, level=lv, text=body, page_or_line=1) for lv, h, body in raw
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

        doc: Any = fitz.open(path)
        toc: list[tuple[int, str, int]] = doc.get_toc()  # [(level, title, page_num), ...]

        sections = self._sections_from_toc(doc, toc) if toc else self._sections_from_pages(doc)

        doc.close()
        title = sections[0].heading if sections and sections[0].heading else pathlib.Path(path).stem
        return ParsedDoc(path=path, title=title, sections=sections)

    def _sections_from_toc(self, doc: Any, toc: list[tuple[int, str, int]]) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        for i, (level, title, page_num) in enumerate(toc):
            start_page = page_num - 1  # 0-indexed
            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(doc)
            body_parts: list[str] = []
            for p in range(start_page, min(end_page, len(doc))):
                body_parts.append(doc[p].get_text())
            sections.append(
                ParsedSection(
                    heading=title,
                    level=level,
                    text=" ".join(body_parts).strip(),
                    page_or_line=page_num,
                )
            )
        return sections

    def _sections_from_pages(self, doc: Any) -> list[ParsedSection]:
        """Fallback: each page becomes a section with the full page text."""
        sections: list[ParsedSection] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text: str = page.get_text()
            first_line = text.split("\n")[0].strip() if text.strip() else ""
            sections.append(
                ParsedSection(
                    heading=first_line,
                    level=1,
                    text=text.strip(),
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


def _load_ts_language(lang_name: str) -> object:
    """Dynamically load tree-sitter language to avoid import-time errors."""
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_typescript
    from tree_sitter import Language

    _langs: dict[str, Any] = {
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
        language: Any = _load_ts_language(lang_name)
        parser = TsParser(language)
        tree: Any = parser.parse(source)

        sections = self._extract_definitions(source, tree, lang_name)
        title = pathlib.Path(path).name
        if not sections:
            sections = [
                ParsedSection(
                    heading="",
                    level=0,
                    text=source.decode("utf-8", errors="replace"),
                    page_or_line=1,
                )
            ]
        return ParsedDoc(path=path, title=title, sections=sections)

    def _extract_definitions(self, source: bytes, tree: Any, lang_name: str) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        node_types = {
            "python": {"function_definition", "class_definition"},
            "javascript": {"function_declaration", "class_declaration", "arrow_function"},
            "typescript": {"function_declaration", "class_declaration", "method_definition"},
        }
        target_types = node_types.get(lang_name, set())

        def _name_of(node: Any) -> str:
            for child in node.children:
                if child.type == "identifier":
                    return source[child.start_byte : child.end_byte].decode(
                        "utf-8", errors="replace"
                    )
            return "<anonymous>"

        def _visit(node: Any) -> None:
            if node.type in target_types:
                name = _name_of(node)
                body = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
                line: int = node.start_point[0] + 1
                # Depth 1 for functions/classes, 2 for methods (parent is class)
                level = 2 if (node.parent and node.parent.type in target_types) else 1
                sections.append(
                    ParsedSection(heading=name, level=level, text=body, page_or_line=line)
                )
            for child in node.children:
                _visit(child)

        _visit(tree.root_node)
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
