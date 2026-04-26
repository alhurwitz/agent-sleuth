# Phase 0: Bootstrap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `agent-sleuth` repo fully developable — uv project, pre-commit hooks, GitHub Actions CI/CD, branch structure, package skeleton with empty `__init__.py` stubs for every module, and test infrastructure. No business logic.

**Architecture:** uv manages the Python environment and lockfile; hatchling builds the wheel; pre-commit enforces code quality and commit message style locally; GitHub Actions runs four separate workflows (CI, integration, perf, release). All `src/sleuth/` subpackages exist as empty stubs so later phases can import from them without import errors.

**Tech Stack:** Python 3.13 (pinned), uv, hatchling, ruff, mypy, pre-commit, commitizen, pytest + pytest-asyncio + pytest-cov + pytest-xdist + pytest-benchmark + syrupy + hypothesis + respx, detect-secrets, pip-audit, GitHub Actions.

---

> **No callouts needed.** Everything required for Phase 0 is already specified in conventions §1–3, §8–9 and spec §9, §16. The only notable finding: `.gitignore` already contains `.sleuth/` from the initial commit; Task 1 still adds `.venv/` explicitly (it is commented-out in the current file).

---

## Task 1: Branch setup + `.gitignore` update

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Create feature branch off `main`**

```bash
git checkout main
git checkout -b feature/phase-0-bootstrap
```

Expected: `Switched to a new branch 'feature/phase-0-bootstrap'`

- [ ] **Step 2: Uncomment `.venv/` in `.gitignore`**

Open `.gitignore`. Find the block:

```
# Environments
.env
.envrc
.venv
```

`.venv` is already listed (not commented). Verify by running:

```bash
grep -n "\.venv" .gitignore
```

Expected: a line like `155:.venv`. If it is already uncommented, nothing to change. If it appears only as `# .venv`, remove the `#`.

Also confirm `.sleuth/` is present (already added in initial commit):

```bash
grep -n "\.sleuth" .gitignore
```

Expected: a line like `227:.sleuth/`

- [ ] **Step 3: Commit (only if `.gitignore` was modified)**

If Step 2 required removing a `#` to uncomment `.venv`:

```bash
git add .gitignore
git commit -m "chore: uncomment .venv/ in .gitignore"
```

If `.venv` was already uncommented (confirmed in Step 2), skip this step — nothing to commit.

---

## Task 2: `pyproject.toml` — full skeleton

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Create `pyproject.toml` (verbatim from conventions §3)**

```toml
[project]
name = "agent-sleuth"
version = "0.1.0"
description = "Plug-and-play agentic search with reasoning, planning, citations, and observability."
requires-python = ">=3.11"
license = "MIT"
authors = [{name = "agent-sleuth maintainers"}]
dependencies = [
    "pydantic>=2.6",
    "httpx>=0.27",
    "anyio>=4.3",
]

[project.optional-dependencies]
anthropic     = ["anthropic>=0.40"]
openai        = ["openai>=1.40"]
langchain     = ["langchain-core>=0.3"]
langgraph     = ["langgraph>=0.2"]
llamaindex    = ["llama-index-core>=0.11"]
openai-agents = ["openai-agents>=0.1"]
claude-agent  = ["claude-agent-sdk>=0.1"]
pydantic-ai   = ["pydantic-ai>=0.0.13"]
crewai        = ["crewai>=0.80"]
autogen       = ["pyautogen>=0.3"]
mcp           = ["mcp>=1.0"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "pytest-xdist>=3.5",
    "pytest-benchmark>=4.0",
    "syrupy>=4.6",
    "hypothesis>=6.100",
    "respx>=0.21",
    "ruff>=0.6",
    "mypy>=1.11",
    "pre-commit>=3.7",
    "commitizen>=3.27",
    "detect-secrets>=1.5",
    "pip-audit>=2.7",
]

[project.scripts]
# Phase 8 adds: sleuth-mcp = "sleuth.mcp.__main__:main"

[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sleuth"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "RUF", "ASYNC"]

[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]
mypy_path = "src"

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "-ra --strict-markers --strict-config"
testpaths = ["tests"]
markers = [
    "unit: default — runs every push",
    "integration: env-gated, nightly",
    "perf: regression suite",
    "adapter: per-framework smoke",
]

[tool.coverage.run]
source = ["src/sleuth"]
branch = true

[tool.coverage.report]
fail_under = 85
exclude_also = ["if TYPE_CHECKING:", "@overload"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject.toml skeleton (uv + hatchling, all extras, dev group)"
```

---

## Task 3: Pin Python version and initialise uv

**Files:**
- Create: `.python-version`
- Create: `uv.lock` (generated)

- [ ] **Step 1: Pin Python 3.13 with uv**

```bash
uv python pin 3.13
```

Expected: creates `.python-version` containing `3.13` (or `cpython-3.13.x-...` format depending on uv version). Verify:

```bash
cat .python-version
```

Expected: a line starting with `3.13` or `cpython-3.13`.

- [ ] **Step 2: Sync the dev environment (generates `uv.lock`)**

```bash
uv sync --group dev
```

This installs core deps + dev group. Framework extras are not installed in bootstrap (they need the actual library wheels to exist). Expected: resolves and locks all core + dev dependencies; `uv.lock` is created/updated.

- [ ] **Step 3: Verify lock file is present**

```bash
ls -lh uv.lock
```

Expected: file exists, non-zero size.

- [ ] **Step 4: Commit**

```bash
git add .python-version uv.lock
git commit -m "chore: pin Python 3.13 and add uv.lock"
```

---

## Task 4: Package skeleton — all `__init__.py` stubs

**Files:**
- Create: `src/sleuth/__init__.py`
- Create: `src/sleuth/_version.py`
- Create: `src/sleuth/engine/__init__.py`
- Create: `src/sleuth/backends/__init__.py`
- Create: `src/sleuth/memory/__init__.py`
- Create: `src/sleuth/llm/__init__.py`
- Create: `src/sleuth/langchain/__init__.py`
- Create: `src/sleuth/langgraph/__init__.py`
- Create: `src/sleuth/llamaindex/__init__.py`
- Create: `src/sleuth/openai_agents/__init__.py`
- Create: `src/sleuth/claude_agent/__init__.py`
- Create: `src/sleuth/pydantic_ai/__init__.py`
- Create: `src/sleuth/crewai/__init__.py`
- Create: `src/sleuth/autogen/__init__.py`
- Create: `src/sleuth/mcp/__init__.py`

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p src/sleuth/engine
mkdir -p src/sleuth/backends
mkdir -p src/sleuth/memory
mkdir -p src/sleuth/llm
mkdir -p src/sleuth/langchain
mkdir -p src/sleuth/langgraph
mkdir -p src/sleuth/llamaindex
mkdir -p src/sleuth/openai_agents
mkdir -p src/sleuth/claude_agent
mkdir -p src/sleuth/pydantic_ai
mkdir -p src/sleuth/crewai
mkdir -p src/sleuth/autogen
mkdir -p src/sleuth/mcp
```

- [ ] **Step 2: Create `src/sleuth/_version.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create `src/sleuth/__init__.py`**

```python
"""
sleuth — plug-and-play agentic search with reasoning, planning, citations, and observability.

Public re-exports are added by Phase 1 (Core MVP). This stub lets the package be imported
without ImportError during bootstrap.
"""

from sleuth._version import __version__

__all__ = ["__version__"]
```

- [ ] **Step 4: Create all remaining empty `__init__.py` stubs**

Each file below should contain exactly one line:

```
# stub — populated by the phase that owns this module
```

Files to create with this content:
- `src/sleuth/engine/__init__.py`
- `src/sleuth/backends/__init__.py`
- `src/sleuth/memory/__init__.py`
- `src/sleuth/llm/__init__.py`
- `src/sleuth/langchain/__init__.py`
- `src/sleuth/langgraph/__init__.py`
- `src/sleuth/llamaindex/__init__.py`
- `src/sleuth/openai_agents/__init__.py`
- `src/sleuth/claude_agent/__init__.py`
- `src/sleuth/pydantic_ai/__init__.py`
- `src/sleuth/crewai/__init__.py`
- `src/sleuth/autogen/__init__.py`
- `src/sleuth/mcp/__init__.py`

Run this to create them all at once:

```bash
for pkg in engine backends memory llm langchain langgraph llamaindex openai_agents claude_agent pydantic_ai crewai autogen mcp; do
  echo "# stub — populated by the phase that owns this module" > src/sleuth/${pkg}/__init__.py
done
```

- [ ] **Step 5: Verify the package is importable**

```bash
uv run python -c "import sleuth; print(sleuth.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add src/
git commit -m "feat: add src/sleuth package skeleton with empty __init__.py stubs"
```

---

## Task 5: Test directory skeleton + `conftest.py`

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/contract/__init__.py`
- Create: `tests/engine/__init__.py`
- Create: `tests/backends/__init__.py`
- Create: `tests/memory/__init__.py`
- Create: `tests/llm/__init__.py`
- Create: `tests/snapshots/` (directory only — syrupy auto-populates)
- Create: `tests/adapters/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/perf/__init__.py`

- [ ] **Step 1: Write the failing test for `respx_mock` fixture**

Create `tests/test_conftest_bootstrap.py`:

```python
"""Smoke tests for Phase 0 conftest fixtures (removed after Phase 1 adds real tests)."""
import pytest


def test_tmp_corpus_fixture_creates_directory(tmp_corpus):
    """tmp_corpus must be a writable directory path."""
    assert tmp_corpus.is_dir()


async def test_respx_mock_fixture_is_active(respx_mock):
    """respx_mock must patch httpx transport so unknown hosts raise ConnectError."""
    import httpx
    import respx

    # register one route
    respx_mock.get("https://example.com/ok").respond(200, text="hello")

    async with httpx.AsyncClient() as client:
        response = await client.get("https://example.com/ok")
        assert response.status_code == 200
        assert response.text == "hello"
```

- [ ] **Step 2: Run the test — expected FAIL (fixtures not defined yet)**

```bash
uv run pytest tests/test_conftest_bootstrap.py -v
```

Expected output contains:
```
ERRORS
fixture 'tmp_corpus' not found
```

- [ ] **Step 3: Create the test directory tree**

```bash
mkdir -p tests/contract tests/engine tests/backends tests/memory tests/llm tests/snapshots tests/adapters tests/integration tests/perf
for d in "" contract engine backends memory llm adapters integration perf; do
  touch tests/${d:+$d/}__init__.py 2>/dev/null || touch tests/__init__.py
done
```

Or more explicitly:

```bash
touch tests/__init__.py
touch tests/contract/__init__.py
touch tests/engine/__init__.py
touch tests/backends/__init__.py
touch tests/memory/__init__.py
touch tests/llm/__init__.py
touch tests/adapters/__init__.py
touch tests/integration/__init__.py
touch tests/perf/__init__.py
```

- [ ] **Step 4: Create `tests/conftest.py` with `tmp_corpus` and `respx_mock` fixtures**

```python
"""
Cross-cutting pytest fixtures for agent-sleuth.

Fixture inventory (Phase 0):
  - tmp_corpus:   a tmp_path subdirectory pre-created as an empty corpus root
  - respx_mock:   respx mock transport active for the test (async-safe)

Phase 1 adds:
  - stub_llm:     a StubLLM instance with a default "hello" response
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import respx as respx_module


@pytest.fixture
def tmp_corpus(tmp_path: Path) -> Path:
    """Return a fresh empty directory suitable for use as a LocalFiles corpus root.

    Each test gets an isolated directory under pytest's tmp_path mechanism.
    The directory is created and ready to write files into.

    Usage::

        def test_something(tmp_corpus):
            (tmp_corpus / "doc.md").write_text("# Hello")
            backend = LocalFiles(path=tmp_corpus)
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return corpus


@pytest.fixture
def respx_mock() -> Generator[respx_module.MockRouter, None, None]:
    """Activate respx mock transport for the duration of the test.

    All httpx requests made during the test are intercepted. Unmatched
    requests raise ``respx.errors.AllMockedError`` so tests cannot make
    accidental real HTTP calls.

    Usage::

        async def test_fetch(respx_mock):
            respx_mock.get("https://api.example.com/search").respond(
                200, json={"results": []}
            )
            # ... code under test that uses httpx ...

    Note: ``respx_mock`` is synchronous-fixture-friendly even in async tests
    because respx patches the transport layer, not the event loop.
    """
    with respx_module.mock(assert_all_mocked=True) as mock:
        yield mock
```

- [ ] **Step 5: Run the test — expected PASS**

```bash
uv run pytest tests/test_conftest_bootstrap.py -v
```

Expected:
```
PASSED tests/test_conftest_bootstrap.py::test_tmp_corpus_fixture_creates_directory
PASSED tests/test_conftest_bootstrap.py::test_respx_mock_fixture_is_active
2 passed
```

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: add tests/conftest.py with tmp_corpus and respx_mock fixtures"
```

---

## Task 6: Pre-commit configuration

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

```yaml
# .pre-commit-config.yaml
# Run: uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
# Update hooks: uv run pre-commit autoupdate
default_language_version:
  python: python3.13

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: end-of-file-fixer
      - id: trailing-whitespace

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.15.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.6
          - pydantic[mypy]
        args: [--config-file=pyproject.toml]

  - repo: https://github.com/commitizen-tools/commitizen
    rev: v4.6.0
    hooks:
      - id: commitizen
        stages: [commit-msg]

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: [--baseline, .secrets.baseline]
        exclude: uv\.lock
```

- [ ] **Step 2: Generate `detect-secrets` baseline (required before first run)**

```bash
uv run detect-secrets scan > .secrets.baseline
```

Expected: creates `.secrets.baseline` (JSON file). Inspect it:

```bash
cat .secrets.baseline | head -20
```

- [ ] **Step 3: Install pre-commit hooks**

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Expected:
```
pre-commit installed at .git/hooks/pre-commit
pre-commit installed at .git/hooks/commit-msg
```

- [ ] **Step 4: Run pre-commit against all files (dry-run validation)**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass (ruff, ruff-format, mypy, check-yaml, check-toml, end-of-file-fixer, trailing-whitespace, detect-secrets). If `end-of-file-fixer` modifies files, re-run until clean.

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml .secrets.baseline
git commit -m "chore: add pre-commit config (ruff, mypy, commitizen, detect-secrets)"
```

---

## Task 7: GitHub Actions — `ci.yml`

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/` directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

```yaml
# .github/workflows/ci.yml
# Triggers: every push and PR. Runs ruff, mypy, unit tests, coverage.
# Matrix: Python 3.11/3.12/3.13 × ubuntu-latest/macos-latest
name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["**"]

jobs:
  lint:
    name: Lint & type-check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --frozen --group dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src/sleuth

  test:
    name: Unit tests (Python ${{ matrix.python-version }} / ${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
        os: [ubuntu-latest, macos-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          python-version: ${{ matrix.python-version }}
      - run: uv sync --frozen --group dev
      - name: Run unit tests with coverage
        run: |
          uv run pytest -m "not integration and not perf and not adapter" \
            --cov=src/sleuth \
            --cov-report=xml \
            --cov-report=term-missing \
            -n auto \
            -v
      - name: Upload coverage
        uses: codecov/codecov-action@v5
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.13'
        with:
          files: coverage.xml
          fail_ci_if_error: false

  audit:
    name: Dependency security audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --frozen --group dev
      - run: uv run pip-audit --strict
```

- [ ] **Step 3: Validate YAML syntax**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

Expected: no output (no error).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add CI workflow (lint, unit tests matrix, security audit)"
```

---

## Task 8: GitHub Actions — `integration.yml`

**Files:**
- Create: `.github/workflows/integration.yml`

- [ ] **Step 1: Create `.github/workflows/integration.yml`**

```yaml
# .github/workflows/integration.yml
# Triggers: nightly cron + manual dispatch.
# Runs pytest -m integration against real APIs using repo secrets.
name: Integration tests

on:
  schedule:
    - cron: "0 3 * * *"   # 03:00 UTC nightly
  workflow_dispatch:

jobs:
  integration:
    name: Integration tests
    runs-on: ubuntu-latest
    environment: integration
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --frozen --group dev
      - name: Run integration tests
        env:
          TAVILY_API_KEY: ${{ secrets.TAVILY_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          uv run pytest -m integration -v --tb=short
```

- [ ] **Step 2: Validate YAML syntax**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/integration.yml'))"
```

Expected: no output (no error).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/integration.yml
git commit -m "ci: add nightly integration test workflow"
```

---

## Task 9: GitHub Actions — `perf.yml`

**Files:**
- Create: `.github/workflows/perf.yml`

- [ ] **Step 1: Create `.github/workflows/perf.yml`**

```yaml
# .github/workflows/perf.yml
# Triggers: pull_request to develop or main.
# Measures end-to-end latency + RunStats.first_token_ms.
# Fails if first-token median > 1500 ms on fast path or p50/p95 regress >10%.
name: Performance regression

on:
  pull_request:
    branches: [develop, main]

jobs:
  perf:
    name: Performance regression suite
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          # fetch develop baseline for comparison
          fetch-depth: 0
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --frozen --group dev
      - name: Run perf suite
        run: |
          uv run pytest -m perf \
            --benchmark-json=perf-results.json \
            --benchmark-min-rounds=5 \
            -v
      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: perf-results-${{ github.sha }}
          path: perf-results.json
```

- [ ] **Step 2: Validate YAML syntax**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/perf.yml'))"
```

Expected: no output (no error).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/perf.yml
git commit -m "ci: add performance regression workflow"
```

---

## Task 10: GitHub Actions — `release.yml`

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Create `.github/workflows/release.yml`**

```yaml
# .github/workflows/release.yml
# Triggers: signed v* tag pushed to main.
# Builds via uv build, publishes to PyPI via Trusted Publisher (OIDC),
# creates GitHub release with auto-generated CHANGELOG via git-cliff.
name: Release

on:
  push:
    tags:
      - "v*"

permissions:
  contents: write       # create GitHub release
  id-token: write       # OIDC for PyPI Trusted Publisher

jobs:
  release:
    name: Build and publish
    runs-on: ubuntu-latest
    environment: release
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0   # needed for git-cliff to walk full history

      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - run: uv sync --frozen --group dev

      - name: Build wheel and sdist
        run: uv build

      - name: Generate CHANGELOG for this release
        uses: orhun/git-cliff-action@v4
        with:
          config: cliff.toml
          args: --latest --strip header
        env:
          OUTPUT: RELEASE_NOTES.md
          GITHUB_REPO: ${{ github.repository }}

      - name: Publish to PyPI (OIDC — no token required)
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/

      - name: Create GitHub release
        uses: softprops/action-gh-release@v2
        with:
          body_path: RELEASE_NOTES.md
          files: dist/*
          make_latest: true
```

- [ ] **Step 2: Create minimal `cliff.toml` for git-cliff**

```toml
# cliff.toml — git-cliff configuration for CHANGELOG generation
[changelog]
header = """
# Changelog\n
All notable changes to this project will be documented in this file.\n
"""
body = """
{% if version %}\
## [{{ version | trim_start_matches(pat="v") }}] - {{ timestamp | date(format="%Y-%m-%d") }}
{% else %}\
## [Unreleased]
{% endif %}\
{% for group, commits in commits | group_by(attribute="group") %}
### {{ group | strtitle }}
{% for commit in commits %}
- {% if commit.breaking %}[**breaking**] {% endif %}{{ commit.message | upper_first }}\
{% endfor %}
{% endfor %}\n
"""
trim = true

[git]
conventional_commits = true
filter_unconventional = true
commit_parsers = [
  { message = "^feat", group = "Features" },
  { message = "^fix", group = "Bug Fixes" },
  { message = "^perf", group = "Performance" },
  { message = "^refactor", group = "Refactoring" },
  { message = "^docs", group = "Documentation" },
  { message = "^test", group = "Testing" },
  { message = "^chore", skip = true },
  { message = "^ci", skip = true },
]
filter_commits = false
tag_pattern = "v[0-9]*"
```

- [ ] **Step 3: Validate YAML syntax**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))"
```

Expected: no output (no error).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/release.yml cliff.toml
git commit -m "ci: add release workflow (uv build, PyPI OIDC, git-cliff CHANGELOG)"
```

---

## Task 11: Dependabot configuration

**Files:**
- Create: `.github/dependabot.yml`

- [ ] **Step 1: Create `.github/dependabot.yml`**

```yaml
# .github/dependabot.yml
# Weekly updates for Python deps, GitHub Actions, and pre-commit hooks.
# Patch/minor updates auto-merge if CI passes (configure via branch protection + auto-merge).
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    labels: ["dependencies", "python"]

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    labels: ["dependencies", "github-actions"]
```

- [ ] **Step 2: Validate YAML syntax**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/dependabot.yml'))"
```

Expected: no output (no error).

- [ ] **Step 3: Commit**

```bash
git add .github/dependabot.yml
git commit -m "chore: add Dependabot config (weekly Python + Actions updates)"
```

---

## Task 12: `CONTRIBUTING.md`

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Create `CONTRIBUTING.md`**

```markdown
# Contributing to agent-sleuth

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/<org>/agent-sleuth.git
cd agent-sleuth

# 2. Install everything (uv required — https://docs.astral.sh/uv/getting-started/installation/)
uv sync --all-extras --group dev

# 3. Install git hooks
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

# 4. Verify the environment
uv run pytest -m "not integration and not perf and not adapter" -v
uv run mypy src/sleuth
uv run ruff check .
```

## Branch model (Gitflow)

| Branch | Purpose |
| --- | --- |
| `main` | Released versions only. Each merge is a tagged release. **Protected.** |
| `develop` | Integration branch for the next release. Feature branches merge here. **Protected.** |
| `feature/<name>` | Branched from `develop`; merged back via PR. |
| `release/<version>` | Stabilisation off `develop`; merges into `main` (tagged) + `develop`. |
| `hotfix/<version>` | Off `main` for emergencies; merges into `main` (tagged) + `develop`. |

All work goes through a PR. No direct commits to `main` or `develop`.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/).
The `commitizen` pre-commit hook will reject non-conforming messages.

Examples:
```
feat: add Exa web backend adapter
fix: handle empty chunk list in synthesizer
docs: update LocalFiles README section
chore: bump ruff to 0.7.0
refactor: extract PlanStep validation into its own function
test: add hypothesis property tests for Backend protocol
perf: short-circuit navigator when tree has single node
breaking!: rename Sleuth.ask() session kwarg to thread
```

## Running tests

```bash
# Fast tests (unit only — no API keys needed)
uv run pytest -m "not integration and not perf and not adapter" -n auto -v

# Integration tests (needs API keys in env)
TAVILY_API_KEY=... uv run pytest -m integration -v

# Performance regression suite
uv run pytest -m perf --benchmark-json=results.json -v

# Framework adapter smoke tests
uv run pytest -m adapter -v

# Full coverage report
uv run pytest --cov=src/sleuth --cov-report=html
open htmlcov/index.html
```

## Type-checking and linting

```bash
uv run mypy src/sleuth      # strict; must pass clean
uv run ruff check .
uv run ruff format .
```

## Adding a dependency

```bash
# Runtime dependency
uv add <pkg>

# Dev-only
uv add --group dev <pkg>

# Optional extra (framework adapter)
# Edit pyproject.toml [project.optional-dependencies] by hand, then:
uv sync --group dev
```

## Release process (maintainers only)

1. Merge the `release/<version>` branch into `main` via PR.
2. After CI passes, create and push a **signed** tag:
   ```bash
   git checkout main && git pull
   git tag -s v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```
3. The `release.yml` workflow triggers automatically: builds the wheel, publishes to PyPI via OIDC, and creates a GitHub release with the auto-generated CHANGELOG.

---

## Manual setup checklist (one-time, per maintainer)

The following steps cannot be automated — a human with repo admin rights must perform them.

### Branch protection

- [ ] On GitHub → Settings → Branches → Add branch protection rule:
  - Pattern: `main`
  - Require a pull request before merging: ✓ (1 approval)
  - Require status checks to pass: ✓ → select `lint`, `test (3.13 / ubuntu-latest)`
  - Require branches to be up to date before merging: ✓
  - Include administrators: ✓
  - Restrict who can push: ✓ → only maintainers
- [ ] Repeat for pattern: `develop` (same settings)

### PyPI Trusted Publisher (OIDC — no long-lived tokens)

- [ ] On PyPI → Publishing → Add a new pending publisher:
  - PyPI project name: `agent-sleuth`
  - Owner: `<github-org>`
  - Repository name: `agent-sleuth`
  - Workflow filename: `release.yml`
  - Environment name: `release`
- [ ] Confirm there are **no** `PYPI_TOKEN` secrets in GitHub — OIDC replaces them.

### GPG / SSH signing key for release tags

- [ ] Each maintainer who will sign release tags must:
  1. Generate a GPG key: `gpg --full-generate-key` (RSA 4096 or Ed25519)
  2. Export the public key: `gpg --armor --export <key-id>`
  3. Upload to GitHub → Settings → SSH and GPG keys → New GPG key
  4. Configure git locally: `git config --global user.signingkey <key-id>` and `git config --global commit.gpgsign true`
- [ ] Verify a test tag signs cleanly: `git tag -s vtest -m "test" && git tag -v vtest && git tag -d vtest`

### GitHub Secret Scanning + push protection

- [ ] On GitHub → Settings → Security → Code security and analysis:
  - Secret scanning: Enable ✓
  - Push protection: Enable ✓

### Dependabot auto-merge (optional)

- [ ] On GitHub → Settings → General → Allow auto-merge: Enable ✓
- [ ] Create a Dependabot auto-merge GitHub Actions workflow (or use the merge queue) to auto-approve and merge Dependabot PRs with only patch-level version bumps after CI passes.
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md with setup, branch model, and manual checklist"
```

---

## Task 13: Create `develop` branch

**Files:** none (git operation only)

- [ ] **Step 1: Push the feature branch to remote**

```bash
git push -u origin feature/phase-0-bootstrap
```

Expected: branch pushed to GitHub; CI workflow triggers.

- [ ] **Step 2: Merge feature branch into `main` via PR**

On GitHub, open a PR from `feature/phase-0-bootstrap` → `main`. Title: `feat: Phase 0 — Bootstrap (uv project, pre-commit, CI/CD, package skeleton)`. Wait for CI to pass, then merge.

After merge, pull locally:

```bash
git checkout main
git pull
```

- [ ] **Step 3: Create `develop` branch off the newly merged `main`**

```bash
git checkout -b develop
git push -u origin develop
```

Expected:
```
Branch 'develop' set up to track remote branch 'develop' from 'origin'.
```

- [ ] **Step 4: Verify branches on remote**

```bash
git branch -r
```

Expected output includes:
```
  origin/main
  origin/develop
  origin/feature/phase-0-bootstrap
```

- [ ] **Step 5: Complete the manual checklist**

At this point a maintainer must perform the manual steps documented in `CONTRIBUTING.md`:
- Configure branch protection on `main` and `develop`.
- Set up PyPI Trusted Publisher (OIDC).
- Register GPG signing keys.
- Enable GitHub Secret Scanning + push protection.

These are not automated; see the "Manual setup checklist" section in `CONTRIBUTING.md`.

---

## Task 14: Final validation

**Files:** none

- [ ] **Step 1: Verify full test suite passes on `develop`**

```bash
git checkout develop
uv run pytest -m "not integration and not perf and not adapter" -v
```

Expected: all bootstrap smoke tests pass (the `test_conftest_bootstrap.py` tests from Task 5).

- [ ] **Step 2: Verify ruff and mypy are clean**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src/sleuth
```

Expected: no errors, no diffs.

- [ ] **Step 3: Verify package is buildable**

```bash
uv build
ls dist/
```

Expected: `dist/agent_sleuth-0.1.0-py3-none-any.whl` and `dist/agent_sleuth-0.1.0.tar.gz` (or equivalent).

- [ ] **Step 4: Clean up bootstrap smoke-test file**

The `tests/test_conftest_bootstrap.py` file was only needed to drive TDD for Task 5's conftest fixtures. Remove it now that the fixtures are proven:

```bash
git rm tests/test_conftest_bootstrap.py
git commit -m "test: remove transient conftest bootstrap smoke test"
```

- [ ] **Step 5: Confirm `develop` is the active integration branch**

```bash
git log --oneline -10
git branch -vv | grep develop
```

Expected: `develop` is ahead of `main` by 0 commits (or exactly the commits from the merged feature branch), and tracks `origin/develop`.

---

## Summary of files created by Phase 0

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Project metadata, deps, all extras, dev group, ruff/mypy/pytest/coverage config |
| `.python-version` | Pins Python 3.13 for uv |
| `uv.lock` | Reproducible lockfile for dev environment |
| `.pre-commit-config.yaml` | ruff, ruff-format, mypy, commitizen, detect-secrets, yaml/toml/merge-conflict checks |
| `.secrets.baseline` | detect-secrets baseline (empty at bootstrap) |
| `cliff.toml` | git-cliff CHANGELOG template |
| `.github/workflows/ci.yml` | Lint + unit tests matrix + pip-audit (every push/PR) |
| `.github/workflows/integration.yml` | Nightly integration tests against real APIs |
| `.github/workflows/perf.yml` | Performance regression suite (on PR) |
| `.github/workflows/release.yml` | Build + PyPI publish + GitHub release (on v* tag) |
| `.github/dependabot.yml` | Weekly dep + Actions updates |
| `CONTRIBUTING.md` | Dev setup, branch model, release process, manual checklist |
| `src/sleuth/__init__.py` | Package root; re-exports `__version__` |
| `src/sleuth/_version.py` | `__version__ = "0.1.0"` |
| `src/sleuth/{engine,backends,memory,llm,langchain,langgraph,llamaindex,openai_agents,claude_agent,pydantic_ai,crewai,autogen,mcp}/__init__.py` | Empty stubs for all subpackages |
| `tests/conftest.py` | `tmp_corpus` + `respx_mock` fixtures |
| `tests/{contract,engine,backends,memory,llm,adapters,integration,perf}/__init__.py` | Empty test namespace packages |
