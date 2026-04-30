# Design specification

The full design spec — the single source of truth for `agent-sleuth`'s architecture — is checked into the repository at [`docs/superpowers/specs/2026-04-25-sleuth-design.md`](https://github.com/alhurwitz/agent-sleuth/blob/develop/docs/superpowers/specs/2026-04-25-sleuth-design.md).

Read it when:

- A documentation page disagrees with the source — the spec wins.
- You need protocol shapes verbatim (`LLMClient`, `Backend`, `Cache`, `Embedder`, event types, `Result[T]`).
- You're considering a feature change — start with §15 ("Open questions") to see what's intentional vs. what's pending.

Companion documents (also in the repo):

- [`docs/superpowers/plans/_conventions.md`](https://github.com/alhurwitz/agent-sleuth/blob/develop/docs/superpowers/plans/_conventions.md) — file-tree freeze, naming, pyproject.toml shape, frozen protocols.
- [`docs/superpowers/plans/_reconciliations.md`](https://github.com/alhurwitz/agent-sleuth/blob/develop/docs/superpowers/plans/_reconciliations.md) — cross-plan deltas captured during the v0.1.0 implementation.
- [`docs/superpowers/plans/2026-04-25-phase-{0..11}-*.md`](https://github.com/alhurwitz/agent-sleuth/tree/develop/docs/superpowers/plans) — the per-phase implementation plans the codebase was built from.
