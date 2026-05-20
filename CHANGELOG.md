# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Routine entries are appended automatically by
[`release-plz`](https://release-plz.dev) from Conventional Commit titles;
breaking changes and ADR-driven entries are written manually.

## [Unreleased]

## [0.1.0] - 2026-05-23

### Added (Sprint 3 — disagg-native)

- `BlockManager` trait gained `offload_block`, `ingest_block`, and
  `block_location`. Default implementations return `DisaggUnavailable`
  and `BlockLocation::Local` respectively, so existing implementations
  remain source-compatible.
- `meridian-kernels` ships a new `nixl` cargo feature that compiles a
  wire-protocol-compatible disaggregated KV transfer layer. The
  `MRDN` v1 framing (Blake3-128 checksum, versioned header) is the
  canonical Meridian wire format documented in ADR-0006.
- `SyntheticNixlFabric` — an in-process implementation of the
  `Fabric` trait that exercises the full disagg call path without
  requiring a libnixl runtime. Used for integration tests and for
  hardware-free portfolio demonstrations.
- `NixlBackedBlockManager` wraps a local `PhaseAwareBlockManager`
  with a fabric handle and records the local→remote ownership
  mapping so `block_location` is correct after offload.
- `DisaggConfig` (`[disagg]` section) added to both the Rust core
  config and the Python Pydantic mirror.
- `CapacitySpec` accepts integer byte counts *or* the literal
  `"auto"` string. `KvConfig` now carries `block_size_bytes` and
  `capacity_bytes`, with auto-resolution against
  `torch.cuda.mem_get_info` on the Python side.
- `PhaseRouter::reap_stale_older_than(Duration)` lets operators
  express the heartbeat threshold in wall-clock seconds; the router
  converts to ticks internally using a measured tick rate.
- vLLM plugin reads block manager sizing from config, queues
  `ThinkComplete` offloads on `ExitThink` once the configured
  threshold is reached, and uses `reap_stale_older_than(60s)` for
  the orphan reaper.
- `benchmarks/baselines.py` ships `StockSchedulerBaseline` — the
  priority-weight single-queue scheduler that vLLM ≤ 0.8 deployed.
- `benchmarks/metrics.py` ships `ABComparisonReport` with a
  side-by-side `Stock | Meridian | Δ (%)` table and a textual
  semaphore (`WIN | win | FLAT | loss | LOSS`).
- `benchmarks/datasets.py` loads ShareGPT and MATH-500 from
  HuggingFace, caches them locally under
  `~/.cache/meridian/datasets/`, and blends them at the configured
  reasoning ratio for the `mix` workload.
- `benchmarks/meridian_bench.py` learns `--baseline {none,stock}`
  and `--workload {synthetic,sharegpt,math500,mix}` flags.
- `meridian-kernels` ships `tests/default_mode_unavailable.rs` (D28)
  pinning the no-CUDA stub contract.
- `meridian-kernels` ships `tests/nixl_integration.rs` exercising
  the full disagg path through the synthetic fabric.
- ADR-0006 documents the disagg wire protocol and trigger points.
- ADR-0007 documents the release and versioning policy, including
  the SLSA Level 2 provenance commitment.

### Changed

- `BlockTier` exposes a stable `as_label()` for telemetry tagging
  (previously a free function inside `block_manager`).
- `Error` enum gained `Disagg`, `DisaggUnavailable`, and
  `DisaggChecksum` variants.
- Release pipeline emits SLSA Level 2 attestations and uploads them
  as GitHub release assets on every tag matching `v*`.

### Added (Sprint 2)

- `PhaseRouter::phase_of_kind` (stable `"prefill" | "think_decode" |
  "output_decode" | "complete"` labels) for cross-language consumers.
- `PhaseRouter::reap_stale(older_than_ticks)` heartbeat reaper for
  orphaned requests whose `Complete` event was never observed.
- `PhaseRouter::touch(req_id)` to refresh idle clock from outside the
  token stream.
- `AtomicUsize` counter for `tracked_requests` — exact, not eventually
  consistent.
- `PhaseAwareBlockManager` exposed via pyo3 as `BlockManager`, with
  `available_blocks()` for accurate scheduler budget queries.
- `EntropyProbe.compute_batch(req_ids, logits_2d)` vectorised over the
  full decode batch.
- Plugin now uses `phase_of_kind` for reorder ranking, `available_blocks`
  for accurate scheduling budget, `compute_batch` for batched entropy
  signals, and periodic `reap_stale` heartbeats.
- CUDA kernel correctness test (`tests/kernel_correctness.rs`) compares
  entropy + EAT kernels against the CPU reference within `1e-4` over
  realistic vocab sizes.
- `benchmarks/` harness with `synthetic-replay` (CI-friendly) and
  `real-vllm` (GPU-only) modes. Phase-differentiated metric report in
  JSON + Markdown.
- ADR-0005 documents the benchmark methodology.

### Added (Sprint 1)

- `MeridianScheduler::schedule_batch` dual-queue dispatcher with
  `think_batch_multiplier` expansion and memory-cap saturation.
- `PhaseAwareBlockManager::allocate` + `evict_for` with tier-ordered LRU
  eviction and MLA-aware accounting.
- Real CUDA kernels for Shannon entropy and EAT signal (templated over
  f32/bf16/f16) with log-sum-exp two-pass numerical stability.
- vLLM plugin now reorders the schedule and propagates `</think>`
  injection events back to the worker.
- `BudgetForceReason` (`Converged` / `Overthinking` / `HardCap`) propagated
  on `PhaseEvent::ForceBudget` and emitted as a metric label.
- `eat_probe_interval_tokens` honoured by the plugin via gated probe calls.
- Native `MeridianScheduler` exposed through pyo3.
- ADR-0003: DashMap rationale. ADR-0004: KV tier promotion policy.
- CUDA workflow targeting a self-hosted GPU runner; Rust↔Python
  integration test (`test_native_bindings.py`).

### Changed

- `PhaseEvent::ForceBudget` is now `{ inject_token, reason }` — breaking
  change tracked under the pre-1.0 stability policy.

### Added

- Sprint 0 scaffolding: tri-crate Cargo workspace (`meridian-core`,
  `meridian-kernels`, `meridian-python`).
- `PhaseRouter` state machine with full token-stream transitions, EAT
  convergence detection, RPDI overthinking detection and `force_in_progress`
  idempotency guard.
- `EntropyProbe` Python facade with switchable backend (`cpu` operational,
  `cuda` stubbed to delegate to CPU).
- Transparent `MeridianSchedulerPlugin` for vLLM ≥ 0.9.0.
- ADR-0001: dual-queue versus priority weights.
- ADR-0002: workspace tri-crate rationale.
- Apache-2.0 license, DCO governance, Contributor Covenant 2.1, security
  disclosure policy.
- CI matrix (Rust + Python + CUDA + mdbook), Renovate config, release-plz
  scaffold, SBOM generation.

[Unreleased]: https://github.com/angelnicolasc/meridian/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/angelnicolasc/meridian/releases/tag/v0.1.0
