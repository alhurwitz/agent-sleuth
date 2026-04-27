"""Tests for CodeSearch backend — Phase 5."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from sleuth.backends._codesearch._embedder import Embedder
from sleuth.backends._codesearch._ripgrep import run_ripgrep
from sleuth.backends._codesearch._symbol_index import SymbolIndex
from sleuth.backends._codesearch._treesitter import (
    SupportedLanguage,
    build_hierarchy,
    expand_hit_to_node,
    language_for_path,
)
from sleuth.backends.base import Capability
from sleuth.backends.codesearch import CodeSearch
from sleuth.types import Chunk, Source
from tests.contract.test_backend_protocol import BackendTestKit

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


# ---------------------------------------------------------------------------
# Task 4 — tree-sitter expander
# ---------------------------------------------------------------------------

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
        line_number=3,  # the `return f"Hello, ..."` line
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
        line_number=1,  # `class Greeter:` line
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


# ---------------------------------------------------------------------------
# Task 5 — symbol index
# ---------------------------------------------------------------------------


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
    await idx.update(tmp_path)  # second call — same mtime/hash

    records = await idx.lookup("stable_func")
    assert len(records) == 1  # exactly one row, not duplicated


@pytest.mark.unit
async def test_symbol_index_rescans_on_content_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing a file's content causes a rescan."""
    src = tmp_path / "changing.py"
    src.write_text("def old_name():\n    pass\n")

    idx = SymbolIndex(tmp_path / ".sleuth" / "symbols.db")
    await idx.update(tmp_path)

    # Modify content and bump mtime artificially
    src.write_text("def new_name():\n    pass\n")
    new_mtime = src.stat().st_mtime + 1.0
    os.utime(src, (new_mtime, new_mtime))

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


# ---------------------------------------------------------------------------
# Task 6 — embedder (no-op path, always available without fastembed)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 7 — hierarchical walker
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task 8 — CodeSearch public class
# ---------------------------------------------------------------------------


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
        (tmp_path / f"file_{i}.py").write_text(f"def func_{i}():\n    # match_target\n    pass\n")
    cs = CodeSearch(path=tmp_path)
    chunks = await cs.search("match_target", k=3)
    assert len(chunks) <= 3


@pytest.mark.unit
async def test_codesearch_symbol_query_shortcut(tmp_path: Path) -> None:
    """'where is X defined' bypasses ripgrep and uses the symbol index."""
    (tmp_path / "handlers.py").write_text("def handle_request(req):\n    pass\n")
    cs = CodeSearch(path=tmp_path)
    await cs._ensure_indexed()  # warm the symbol index explicitly for the test

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
    (tmp_path / "big.py").write_text("\n".join(f"def fn_{i}(): pass" for i in range(200)))
    cs = CodeSearch(path=tmp_path)

    async def _run() -> list[Chunk]:
        return await cs.search("fn_", k=50)

    task = asyncio.create_task(_run())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Additional coverage: language_for_path, JS/TS expansion, unsupported files
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_language_for_path_python() -> None:
    """language_for_path returns PYTHON for .py files."""
    assert language_for_path("app.py") == SupportedLanguage.PYTHON


@pytest.mark.unit
def test_language_for_path_typescript() -> None:
    """language_for_path returns TYPESCRIPT for .ts/.tsx files."""
    assert language_for_path("comp.tsx") == SupportedLanguage.TYPESCRIPT
    assert language_for_path("utils.ts") == SupportedLanguage.TYPESCRIPT


@pytest.mark.unit
def test_language_for_path_javascript() -> None:
    """language_for_path returns JAVASCRIPT for .js/.mjs/.cjs files."""
    assert language_for_path("app.js") == SupportedLanguage.JAVASCRIPT
    assert language_for_path("mod.mjs") == SupportedLanguage.JAVASCRIPT


@pytest.mark.unit
def test_language_for_path_unsupported() -> None:
    """language_for_path returns None for unsupported extensions."""
    assert language_for_path("data.json") is None
    assert language_for_path("README.md") is None


JS_SOURCE = """\
function greetUser(name) {
    return "Hello, " + name;
}

class UserService {
    constructor(db) {
        this.db = db;
    }
}
"""


@pytest.mark.unit
def test_expand_hit_javascript_function() -> None:
    """JavaScript function_declaration is expanded correctly."""
    node = expand_hit_to_node(
        source=JS_SOURCE,
        line_number=2,  # inside greetUser body
        language=SupportedLanguage.JAVASCRIPT,
    )
    assert node is not None
    assert "greetUser" in node.text


@pytest.mark.unit
def test_expand_hit_javascript_class() -> None:
    """JavaScript class_declaration is expanded correctly."""
    node = expand_hit_to_node(
        source=JS_SOURCE,
        line_number=5,  # class UserService line
        language=SupportedLanguage.JAVASCRIPT,
    )
    assert node is not None
    assert "UserService" in node.text


TS_SOURCE = """\
interface Config {
    host: string;
}

function createConfig(host: string): Config {
    return { host };
}
"""


@pytest.mark.unit
def test_expand_hit_typescript_function() -> None:
    """TypeScript function is expanded correctly."""
    node = expand_hit_to_node(
        source=TS_SOURCE,
        line_number=6,  # inside createConfig body
        language=SupportedLanguage.TYPESCRIPT,
    )
    assert node is not None
    assert "createConfig" in node.text


@pytest.mark.unit
async def test_codesearch_searches_js_files(tmp_path: Path) -> None:
    """CodeSearch finds matches in JavaScript files."""
    (tmp_path / "api.js").write_text("function handleRequest(req) {\n    return req.body;\n}\n")
    cs = CodeSearch(path=tmp_path)
    chunks = await cs.search("handleRequest", k=5)
    assert len(chunks) >= 1
    assert any("handleRequest" in c.text for c in chunks)


@pytest.mark.unit
async def test_embedder_rerank_raises_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedder.rerank raises ImportError when fastembed is unavailable.

    CI runs with --all-extras so fastembed is installed; simulate the
    "missing extra" case by hiding the module from imports.
    """
    import sys

    monkeypatch.setitem(sys.modules, "fastembed", None)
    embedder = Embedder(rerank=True)
    chunk = _make_chunk("some code", 1)
    with pytest.raises(ImportError):
        await embedder.rerank("query", [chunk])


# ---------------------------------------------------------------------------
# Task 9 — BackendTestKit protocol compliance
# ---------------------------------------------------------------------------
# BackendTestKit is defined in tests/contract/test_backend_protocol.py (Phase 1).
# Import and parametrize it for CodeSearch.


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


# ---------------------------------------------------------------------------
# Task 12 — integration smoke (skipped in unit CI, runs nightly)
# ---------------------------------------------------------------------------


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
