# ADR-0002: Workspace tri-crate layout (`core` / `kernels` / `python`)

- **Status**: Accepted
- **Date**: 2026-05-20
- **Authors**: angelnicolasc

## Context

Meridian spans three execution domains: a pure-Rust scheduler core, CUDA
kernels behind an FFI boundary, and pyo3 bindings that the Python vLLM
plugin consumes. The natural layouts are:

1. **Single crate**. All code in one place, gated by `cfg(feature = ...)`.
2. **Tri-crate Cargo workspace.** `meridian-core` (Rust only),
   `meridian-kernels` (CUDA + FFI), `meridian-python` (pyo3, built via
   maturin). All members of one workspace, sharing lockfile and lints.
3. **Polyrepo.** Each layer in its own repository, glued by published
   versions.

## Decision

**Tri-crate workspace.** All three crates live under `crates/` in this
repository.

## Consequences

### Positive

- `cargo test -p meridian-core` runs on any host — no CUDA, no Python, no
  `nvcc`. The CI matrix can validate the core invariants on the cheapest
  GitHub-hosted runner and only spin up a GPU runner for the CUDA layer.
- The `unsafe` surface area is *visibly* contained. Auditors looking at
  `meridian-core` see `#![forbid(unsafe_code)]` at the crate root. The
  unsafe code lives in exactly one crate (`meridian-kernels`) and at
  exactly one boundary (the FFI declarations in `src/ffi.rs`).
- Workspace `[workspace.lints]` is applied uniformly across all three
  crates — one place to change a clippy lint, no drift.
- A future fork that wants only the scheduler core (e.g. for a non-CUDA
  inference framework) can depend on `meridian-core` directly without
  pulling pyo3 or CUDA artifacts.

### Negative / risks

- Three `Cargo.toml`s to keep in sync. Mitigated by `workspace.dependencies`
  and `workspace.package` inheritance — versioned dependencies are
  declared once.
- The `meridian-kernels` crate has `links = "meridian_kernels"`. Cargo
  enforces uniqueness so we cannot accidentally link two implementations
  of the same native library — a small but real guard.

### Neutral

- Build artifacts grow by one extra `target/` directory per crate.
  Negligible in practice.

## Alternatives considered

### Single crate

Rejected because: any Python binding gate requires pyo3 in the dep graph,
which pulls a non-trivial transitive closure. Anyone wanting to depend on
just the scheduler core would be forced to compile it.

### Polyrepo

Rejected because: in a pre-1.0 project where the three layers co-evolve,
splitting them into separate repositories introduces version-skew bugs
without a corresponding benefit. Once the public API stabilises post-1.0
this can be revisited.

## References

- [Rust API Guidelines — C-FEATURE](https://rust-lang.github.io/api-guidelines/).
- [Cargo Book — Workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html).
