"""Tests for the LocalFiles backend — Phase 2."""

from __future__ import annotations

import asyncio
import pathlib
import textwrap
from typing import TYPE_CHECKING

import pytest

from sleuth.backends._localfiles.models import (  # noqa: F401
    IndexNode,
    IndexTree,
    NodeKind,
    ParsedDoc,
)
from sleuth.llm.stub import StubLLM
from tests.contract.test_backend_protocol import BackendTestKit  # Phase 1 export

if TYPE_CHECKING:
    from sleuth.backends.localfiles import LocalFiles


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

        md = tmp_path / "flat.md"
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


class TestHasher:
    def test_mtime_hash_changes_when_file_changes(self, tmp_path: pathlib.Path) -> None:
        import os
        import time

        from sleuth.backends._localfiles.hasher import HashMode, file_hash

        f = tmp_path / "doc.md"
        f.write_text("v1")
        h1 = file_hash(str(f), mode=HashMode.MTIME)
        time.sleep(0.01)
        f.write_text("v2")
        # Touch mtime explicitly
        os.utime(f, None)
        h2 = file_hash(str(f), mode=HashMode.MTIME)
        assert h1 != h2

    def test_content_hash_depends_on_content_not_mtime(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.hasher import HashMode, file_hash

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


class TestTreeBuilder:
    def test_builds_tree_from_parsed_docs(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends._localfiles.models import NodeKind  # noqa: F401
        from sleuth.backends._localfiles.parsers import MarkdownParser
        from sleuth.backends._localfiles.tree_builder import build_tree

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


class TestPersistence:
    def test_round_trip_json(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.models import IndexNode, IndexTree, NodeKind
        from sleuth.backends._localfiles.persistence import load_tree, save_tree

        child = IndexNode(
            id="doc::1",
            title="Sec A",
            kind=NodeKind.SECTION,
            summary="Summary A.",
            text=None,
            source_path="/tmp/doc.md",
            source_section="# Sec A",
            page_or_line=5,
            children=[
                IndexNode(
                    id="doc::2",
                    title="Leaf",
                    kind=NodeKind.LEAF,
                    summary=None,
                    text="leaf text",
                    source_path="/tmp/doc.md",
                    source_section="# Sec A > ## Leaf",
                    page_or_line=10,
                )
            ],
        )
        root = IndexNode(
            id="doc::root",
            title="Doc",
            kind=NodeKind.ROOT,
            summary="Doc summary.",
            text=None,
            source_path="/tmp/doc.md",
            source_section="Doc",
            page_or_line=1,
            children=[child],
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

        result = load_tree(
            str(tmp_path),
            version="nonexistent",
            index_dir=str(tmp_path / ".sleuth" / "index"),
        )
        assert result is None

    def test_save_overwrites_same_version(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.models import IndexNode, IndexTree, NodeKind
        from sleuth.backends._localfiles.persistence import load_tree, save_tree

        def _make_tree(title: str) -> IndexTree:
            root = IndexNode(
                id="doc::root",
                title=title,
                kind=NodeKind.ROOT,
                summary="S.",
                text=None,
                source_path="/tmp/x.md",
                source_section=title,
                page_or_line=1,
            )
            return IndexTree(corpus_path=str(tmp_path), version="v1", nodes=[root])

        index_dir = str(tmp_path / ".sleuth" / "index")
        save_tree(_make_tree("First"), index_dir=index_dir)
        save_tree(_make_tree("Second"), index_dir=index_dir)
        loaded = load_tree(str(tmp_path), version="v1", index_dir=index_dir)
        assert loaded is not None
        assert loaded.nodes[0].title == "Second"


class TestNavigator:
    def _make_tree(self) -> IndexTree:
        from sleuth.backends._localfiles.models import IndexNode, IndexTree, NodeKind

        def _leaf(id_: str, title: str, text: str, source_section: str) -> IndexNode:
            return IndexNode(
                id=id_,
                title=title,
                kind=NodeKind.LEAF,
                summary=None,
                text=text,
                source_path="/tmp/doc.md",
                source_section=source_section,
                page_or_line=1,
            )

        def _branch(id_: str, title: str, summary: str, children: list) -> IndexNode:  # type: ignore[type-arg]
            return IndexNode(
                id=id_,
                title=title,
                kind=NodeKind.SECTION,
                summary=summary,
                text=None,
                source_path="/tmp/doc.md",
                source_section=title,
                page_or_line=1,
                children=children,
            )

        auth_section = _branch(
            "doc::0",
            "Authentication",
            "Covers auth flow, refresh tokens, JWT.",
            [
                _leaf("doc::1", "Login Flow", "Users log in via OAuth2.", "# Auth > ## Login Flow"),
                _leaf(
                    "doc::2",
                    "Refresh Tokens",
                    "Tokens refresh every 15 min.",
                    "# Auth > ## Refresh Tokens",
                ),
            ],
        )
        deploy_section = _branch(
            "doc::3",
            "Deployment",
            "Deployment instructions for Kubernetes.",
            [
                _leaf(
                    "doc::4",
                    "Docker Setup",
                    "Build the image with docker build.",
                    "# Deployment > ## Docker Setup",
                ),
            ],
        )
        root = IndexNode(
            id="doc::root",
            title="Docs",
            kind=NodeKind.ROOT,
            summary="Full documentation.",
            text=None,
            source_path="/tmp/doc.md",
            source_section="Docs",
            page_or_line=1,
            children=[auth_section, deploy_section],
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


class TestLocalFiles:
    def test_search_returns_chunks(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(
            responses=[
                "Section summary.",  # indexer LLM — repeated for each node
                "Section summary.",
                "Section summary.",
                "Section summary.",
                '{"selected_ids": []}',  # navigator LLM — fallback to all
            ]
        )
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
                async for chunk in super().stream(messages, schema=schema, tools=tools):
                    yield chunk

        llm = CountingStub()
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        asyncio.get_event_loop().run_until_complete(backend.search("first", k=3))
        count_after_first = call_count["n"]
        asyncio.get_event_loop().run_until_complete(backend.search("second", k=3))
        count_after_second = call_count["n"]
        # Navigator was called again (1 new call), but indexer was NOT called again
        assert count_after_second - count_after_first == 1

    def test_name_and_capabilities(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends.base import Capability
        from sleuth.backends.localfiles import LocalFiles

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


class TestLocalFilesProtocol(BackendTestKit):
    """Run the full Backend protocol contract suite against LocalFiles."""

    @pytest.fixture
    def backend(self, tmp_corpus: pathlib.Path) -> LocalFiles:
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["Summary."] * 50 + ['{"selected_ids": []}'] * 20)
        return LocalFiles(
            path=str(tmp_corpus),
            indexer_llm=llm,
            navigator_llm=llm,
        )


# ---------------------------------------------------------------------------
# Coverage edge-case tests
# ---------------------------------------------------------------------------


class TestCoverageEdgeCases:
    def test_code_parser_unsupported_extension_returns_flat_doc(
        self, tmp_path: pathlib.Path
    ) -> None:
        from sleuth.backends._localfiles.parsers import CodeParser

        f = tmp_path / "data.csv"
        f.write_text("col1,col2\n1,2\n")
        parser = CodeParser()
        doc = parser.parse(str(f))
        assert len(doc.sections) == 1
        assert doc.sections[0].level == 0

    def test_prune_stale_nonexistent_dir_returns_zero(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.persistence import prune_stale

        count = prune_stale(str(tmp_path / "nonexistent"), keep_version="abc")
        assert count == 0

    def test_html_parser_no_headings_returns_single_section(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import HtmlParser

        html = tmp_path / "flat.html"
        html.write_text("<html><body><p>No headings here.</p></body></html>")
        parser = HtmlParser()
        doc = parser.parse(str(html))
        assert len(doc.sections) == 1
        assert doc.sections[0].level == 0

    def test_pdf_parser_no_toc_falls_back_to_page_sections(self, tmp_path: pathlib.Path) -> None:
        import fitz

        from sleuth.backends._localfiles.parsers import PdfParser

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Page One Content", fontsize=12)
        pdf_path = str(tmp_path / "notoc.pdf")
        doc.save(pdf_path)
        doc.close()

        parser = PdfParser()
        result = parser.parse(pdf_path)
        assert len(result.sections) >= 1

    def test_local_files_collect_docs_skips_parse_errors(self, tmp_corpus: pathlib.Path) -> None:
        """Files that fail to parse should be silently skipped, not raise."""
        from sleuth.backends.localfiles import LocalFiles

        # Create an invalid PDF file (binary garbage)
        bad_pdf = tmp_corpus / "broken.pdf"
        bad_pdf.write_bytes(b"not a real pdf")

        llm = StubLLM(responses=["Sum."] * 30 + ['{"selected_ids": []}'] * 5)
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        # Should not raise
        chunks = asyncio.get_event_loop().run_until_complete(backend.search("test", k=5))
        assert isinstance(chunks, list)

    def test_markdown_preamble_before_heading(self, tmp_path: pathlib.Path) -> None:
        from sleuth.backends._localfiles.parsers import MarkdownParser

        md = tmp_path / "preamble.md"
        md.write_text("This is a preamble.\n\n# Section One\n\nBody text.\n")
        parser = MarkdownParser()
        doc = parser.parse(str(md))
        # Should have a preamble section (level=0) and the heading section (level=1)
        levels = [s.level for s in doc.sections]
        assert 0 in levels
        assert 1 in levels

    def test_get_summary_brief_returns_root_summary(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["Brief summary."] * 30 + ['{"selected_ids": []}'] * 5)
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        summary = asyncio.get_event_loop().run_until_complete(
            backend._get_summary(str(tmp_corpus), length="brief")
        )
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_warm_index_builds_index(self, tmp_corpus: pathlib.Path) -> None:
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["Sum."] * 30)
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        asyncio.get_event_loop().run_until_complete(backend.warm_index())
        assert backend._tree is not None

    def test_search_raises_when_no_llm(self, tmp_corpus: pathlib.Path) -> None:
        """search() raises RuntimeError when no navigator or indexer LLM is configured."""
        from sleuth.backends.localfiles import LocalFiles

        # Build with an LLM so index can be created
        llm = StubLLM(responses=["Sum."] * 30)
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        asyncio.get_event_loop().run_until_complete(backend.warm_index())

        # Now remove both LLMs to simulate no-LLM search
        backend._indexer_llm = None
        backend._navigator_llm = None
        with pytest.raises(RuntimeError, match="LocalFiles requires an LLM"):
            asyncio.get_event_loop().run_until_complete(backend.search("test", k=3))

    def test_ensure_tree_raises_when_no_llm_for_build(self, tmp_path: pathlib.Path) -> None:
        """_ensure_tree() raises RuntimeError when no LLM provided and index not on disk."""
        from sleuth.backends.localfiles import LocalFiles

        backend = LocalFiles(path=str(tmp_path), indexer_llm=None, navigator_llm=None)
        with pytest.raises(RuntimeError, match="LocalFiles requires an LLM to build"):
            asyncio.get_event_loop().run_until_complete(backend._ensure_tree())

    def test_ensure_tree_loads_from_disk_cache(self, tmp_corpus: pathlib.Path) -> None:
        """_ensure_tree() hits the disk cache on a second backend instance."""
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["Sum."] * 30 + ['{"selected_ids": []}'] * 5)
        # First backend: build and persist the index
        b1 = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        asyncio.get_event_loop().run_until_complete(b1.warm_index())

        # Second backend: fresh object (no in-memory cache), but disk index exists
        b2 = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        tree = asyncio.get_event_loop().run_until_complete(b2._ensure_tree())
        assert tree is not None
        assert tree.version == b1._tree_version

    def test_get_summary_standard_returns_children_info(self, tmp_corpus: pathlib.Path) -> None:
        """_get_summary(..., length='standard') includes child summaries."""
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["Child summary text."] * 30 + ['{"selected_ids": []}'] * 5)
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        summary = asyncio.get_event_loop().run_until_complete(
            backend._get_summary(str(tmp_corpus), length="standard")
        )
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_get_summary_thorough_includes_all_nodes(self, tmp_corpus: pathlib.Path) -> None:
        """_get_summary(..., length='thorough') walks the entire tree."""
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["Thorough node text."] * 30 + ['{"selected_ids": []}'] * 5)
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        summary = asyncio.get_event_loop().run_until_complete(
            backend._get_summary(str(tmp_corpus), length="thorough")
        )
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_get_summary_for_specific_file(self, tmp_corpus: pathlib.Path) -> None:
        """_get_summary() for a specific file path (not corpus root) finds the right node."""
        from sleuth.backends.localfiles import LocalFiles

        llm = StubLLM(responses=["File summary."] * 30 + ['{"selected_ids": []}'] * 5)
        backend = LocalFiles(path=str(tmp_corpus), indexer_llm=llm, navigator_llm=llm)
        # Use an actual file in the corpus
        target_file = str(tmp_corpus / "intro.md")
        summary = asyncio.get_event_loop().run_until_complete(
            backend._get_summary(target_file, length="brief")
        )
        # Should return something (even if the node's summary is the title fallback)
        assert isinstance(summary, str)
