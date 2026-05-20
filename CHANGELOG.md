# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Routine entries are appended automatically by
[`release-plz`](https://release-plz.dev) from Conventional Commit titles;
breaking changes and ADR-driven entries are written manually.

## [Unreleased]

## [0.1.0] - 2026-05-20

### Added

- Initial scaffolding: tri-crate Cargo workspace (`meridian-core`,
  `meridian-kernels`, `meridian-python`).
- `PhaseRouter` state machine with full token-stream transitions, EAT
  convergence detection, RPDI overthinking detection, and `force_in_progress`
  idempotency guard.
- `PhaseRouter::phase_of_kind` (stable string labels) for cross-language consumers.
- `PhaseRouter::reap_stale` and `reap_stale_older_than(Duration)` — heartbeat
  reaper for orphaned requests; the Duration variant converts wall-clock seconds
  to ticks internally.
- `PhaseRouter::touch(req_id)` to refresh idle clock from outside the token stream.
- `AtomicUsize` counter for `tracked_requests` — exact, not eventually consistent.
- `MeridianScheduler::schedule_batch` dual-queue dispatcher with
  `think_batch_multiplier` expansion and memory-cap saturation.
- `PhaseAwareBlockManager::allocate` + `evict_for` with tier-ordered LRU eviction
  and MLA-aware accounting; exposed via pyo3 as `BlockManager`.
- `BlockManager` trait gained `offload_block`, `ingest_block`, and
  `block_location`. Default implementations return `DisaggUnavailable` and
  `BlockLocation::Local` respectively, preserving source compatibility.
- `meridian-kernels` `nixl` cargo feature: wire-protocol-compatible disaggregated
  KV transfer layer with `MRDN` v1 framing (Blake3-128 checksum, versioned header).
- `SyntheticNixlFabric` — in-process `Fabric` implementation for integration tests
  and hardware-free deployments.
- `NixlBackedBlockManager` — wraps `PhaseAwareBlockManager` with a fabric handle;
  `block_location` is correct after offload.
- `DisaggConfig` (`[disagg]` section) in both Rust config and Python Pydantic mirror.
- `KvConfig` gains `block_size_bytes` and `capacity_bytes`; the Python side
  auto-resolves `"auto"` against `torch.cuda.mem_get_info`.
- Real CUDA kernels for Shannon entropy and EAT signal (templated over f32/bf16/f16),
  log-sum-exp two-pass numerically stable.
- `EntropyProbe` Python facade with switchable `cpu` (NumPy) and `cuda` backends;
  `compute_batch` vectorised over the full decode batch.
- Transparent `MeridianSchedulerPlugin` for vLLM ≥ 0.9.0: reorders dispatch,
  injects `</think>` on budget-force, drives disagg offload on `ExitThink`,
  and uses `reap_stale_older_than(60s)` for orphan cleanup.
- `BudgetForceReason` (`Converged` / `Overthinking` / `HardCap`) propagated
  on `PhaseEvent::ForceBudget` and emitted as a metric label.
- CUDA kernel correctness tests (`tests/kernel_correctness.rs`) — entropy + EAT
  kernels verified against CPU reference within `1e-4` over representative vocab sizes.
- Benchmark harness (`synthetic-replay` + `real-vllm`) with phase-differentiated
  metrics (TTOT, output ITL, budget force rate) in JSON + Markdown.
- `StockSchedulerBaseline` and `ABComparisonReport` for side-by-side A/B evaluation.
- `benchmarks/datasets.py` — ShareGPT and MATH-500 loaders with local cache.
- `meridian-kernels/tests/default_mode_unavailable.rs` pinning the no-CUDA stub contract.
- `meridian-kernels/tests/nixl_integration.rs` exercising the full disagg path.
- ADR-0001 through ADR-0007 documenting all major architectural decisions.
- Apache-2.0 license, DCO governance, Contributor Covenant 2.1, security
  disclosure policy.
- CI matrix (Rust + Python + CUDA + mdbook), release-plz scaffold, SBOM
  generation, SLSA Level 2 provenance attestation on every `v*` tag.

### Changed

- `BlockTier` exposes a stable `as_label()` for telemetry tagging.
- `Error` enum gained `Disagg`, `DisaggUnavailable`, and `DisaggChecksum` variants.
- `PhaseEvent::ForceBudget` is now `{ inject_token, reason }`.
  **BREAKING** — tracked under the pre-1.0 stability policy per ADR-0007.

[Unreleased]: https://github.com/angelnicolasc/meridian/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/angelnicolasc/meridian/releases/tag/v0.1.0
