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

## CodeSearch prerequisites

The `CodeSearch` backend shells out to [`ripgrep`](https://github.com/BurntSushi/ripgrep) (`rg`). Install it before running code-search-related tests or using the backend:

| Platform | Command |
|---|---|
| macOS (Homebrew) | `brew install ripgrep` |
| Ubuntu/Debian | `sudo apt-get install ripgrep` |
| Windows (winget) | `winget install BurntSushi.ripgrep.MSVC` |
| Cargo | `cargo install ripgrep` |

Verify with: `rg --version`
