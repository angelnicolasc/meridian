# Contributing to Meridian

Thank you for considering a contribution. Meridian is an inference-time compute
scheduler for reasoning-model serving — correctness, performance and clarity of
contracts all matter at the same time. This document describes the bar.

## Code of Conduct

Participation is governed by the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
Report violations to nick.dicerutti@gmail.com.

## Developer Certificate of Origin (DCO)

Meridian uses the DCO instead of a CLA. Every commit must be signed off:

```bash
git commit -s -m "feat(core): add RPDI ratio computation"
```

This appends `Signed-off-by: Your Name <you@example.com>`, which certifies that
you have the right to submit the change under the project license. PRs without
sign-off will not merge — the [DCO check](https://github.com/apps/dco) blocks them.

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) with the
following scopes:

| Scope     | Used for                                                      |
|-----------|---------------------------------------------------------------|
| `core`    | `crates/meridian-core/`                                       |
| `kernels` | `crates/meridian-kernels/` (CUDA + FFI)                       |
| `python`  | `crates/meridian-python/` and `python/meridian/`              |
| `ci`      | `.github/workflows/`, devcontainer, scripts                   |
| `docs`    | `docs/`, README, NOTICE                                       |
| `adr`     | new or modified ADRs only                                     |
| `deps`    | dependency bumps                                              |
| `bench`   | `benchmarks/`                                                 |

Title must be under 72 characters. Breaking changes use `feat!:` / `fix!:` and
include a `BREAKING CHANGE:` footer.

## Branch protection

`main` is protected. PRs require:

1. Linear history (rebase, no merge commits).
2. CI green: `cargo fmt --check`, `clippy -D warnings`, `cargo nextest`, `ruff`,
   `mypy --strict`, `pytest -m "not gpu"`, `mdbook build`.
3. DCO sign-off on every commit.
4. Conventional Commit title.
5. At least one approving review.

## Local development

```bash
./scripts/dev-up.sh                  # devcontainer + sanity checks
./scripts/ci-local.sh                # mirrors CI matrix locally
```

Pre-commit hooks (rustfmt, clippy, ruff, mypy, commitlint) are configured in
`.pre-commit-config.yaml`. Install with `pre-commit install --install-hooks`.

## Test strategy

- Pure Rust logic lives in `meridian-core` and must be covered by unit tests
  plus, where state machines are involved, `proptest`-based property tests.
- CUDA correctness lives in `meridian-kernels` and is verified against the
  reference CPU implementation in Python.
- Anything that crosses the FFI boundary is exercised from Python tests
  (`python/tests/`).
- GPU-dependent tests are marked `@pytest.mark.gpu`; CI runs them only on the
  GPU job.

## What lands in `main`

A change is mergeable when:

- Tests cover the new code path and existing tests still pass.
- Public API additions have rustdoc / docstrings with examples.
- Behavior changes that affect operators (config defaults, metric names,
  exposed traits) are recorded in an ADR.
- `CHANGELOG.md` is updated under `## [Unreleased]` (handled automatically by
  `release-plz` for routine changes — manually for breaking ones).

## Getting help

Open a [GitHub Discussion](https://github.com/angelnicolasc/meridian/discussions)
for design questions, an issue for bugs.
