<!--
Thanks for contributing to Meridian. Before submitting:

1. Title follows Conventional Commits: `<type>(<scope>): <summary>` (≤72 chars).
   Scopes: core, kernels, python, ci, docs, adr, deps, bench.
2. Every commit is DCO-signed: `git commit -s ...`.
3. CI is green locally (`./scripts/ci-local.sh`).
4. Behavioral changes are recorded in `CHANGELOG.md` under `[Unreleased]`.
-->

## Summary

<!-- One-paragraph description: what changed and why. Link issues with `Fixes #N`. -->

## Motivation

<!-- What problem does this solve? What evidence motivated it? -->

## Changes

- [ ] Rust core (`crates/meridian-core/`)
- [ ] CUDA kernels (`crates/meridian-kernels/`)
- [ ] Python facade / vLLM plugin (`python/meridian/`)
- [ ] Configuration / model defaults
- [ ] Docs / ADRs
- [ ] CI / governance

## Test plan

<!-- Concrete steps you ran. Example:
- `cargo nextest run -p meridian-core` — all tests pass including new property test.
- `uv run pytest python/tests/test_entropy_cpu.py` — passes.
-->

## Risk / rollback

<!-- What is the blast radius if this regresses? How would you roll back? -->

## Checklist

- [ ] Conventional Commit title
- [ ] DCO sign-off on every commit
- [ ] `CHANGELOG.md` updated if user-visible
- [ ] Public API changes documented (rustdoc / docstring)
- [ ] Behavior changes have an ADR if architectural
