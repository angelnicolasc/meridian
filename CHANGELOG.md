# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Routine entries are appended automatically by
[`release-plz`](https://release-plz.dev) from Conventional Commit titles;
breaking changes and ADR-driven entries are written manually.

## [Unreleased]

### Added

- `meridian_core::dspark_bridge` — phase-conditioned speculative decoding.
  Extends Meridian's think/output phase thesis to draft-model scheduling,
  motivated by DeepSpec's documented training of its released Qwen3 drafters on
  non-thinking-mode generations only. **Ships with no measured acceptance rate
  and cannot produce one**; see ADR-0009.
  - `confidence_model` — formal model of DSpark's confidence head and Meridian's
    EAT/RPDI signal in shared terms, with four proved relations: the point-mass
    identity, EAT as the confidence head's supervision target for a specific
    drafter, EAT as a ceiling on boundary-supported drafters, and the
    entropy bracket `e^{-H} ≤ max_v pᵗ(v) ≤ p*(H, V)` whose upper half ceilings
    any deterministic drafter's single-step acceptance.
  - `hook::PhaseConditioningHook` — scheduler-facing draft-depth recommendation.
    With an uncalibrated prior it may only ever *reduce* the configured
    baseline; raising it requires a measured, provenance-tagged prior.
  - `ledger::AcceptanceLedger` — phase-segmented acceptance accounting with
    DeepSpec-compatible metric definitions, explicit boundary-straddle policy,
    Welch unequal-variance comparison, and a pre-registered
    `HypothesisVerdict` decision rule.
  - `provenance::Provenance` — synthetic-vs-measured gate. Promotion to a
    publishable claim is a typed operation that fails on synthetic data.
  - `stats` — Welch's *t*-test, Welch–Satterthwaite df, Cohen's *d*, and a
    Student-*t* CDF verified against closed forms exact at `df = 1` and `df = 2`.
- `[speculation]` config section (Rust + Python Pydantic mirror) and
  `[speculation.acceptance_prior]`, whose schema cannot express acceptance rates
  without the harness, checkpoint, target model, thinking-mode flag and date of
  the run that produced them.
- pyo3 `PhaseConditioningHook` and `AcceptanceLedger`.
- Metrics: `meridian.speculation.policy_basis{phase,basis}`,
  `meridian.speculation.proposal_len{phase}`,
  `meridian.speculation.accepted_length{phase}`,
  `meridian.speculation.straddling_steps`.
- ADR-0009 (phase-conditioned speculation) and three research notes: the signal
  comparison, the DeepSpec harness instrumentation gap analysis, and the
  deferred Phase 1 experiment protocol.
- `tests/dspark_bridge_synthetic.rs` — synthetic trace fixtures covering clean,
  ambiguous, all-think and all-output phase boundaries, plus property-based
  invariants over arbitrary traces.

### Changed

- `Error` gained `SyntheticProvenance`, `ProvenanceMismatch`,
  `StraddlePolicyMismatch`, and `InsufficientObservations` variants.

## [0.2.0] - 2026-05-21

### Added

- `BlockManager` trait gained `free_block_by_id(block_id)` and
  `blocks_for_request(req_id)`; `PhaseAwareBlockManager::is_resident(block_id)`.
  These let fabric layers reclaim a single offloaded slot and enumerate a
  request's resident blocks by exact id.
- `NixlBackedBlockManager::offload_block` now reclaims the local slot after a
  successful fabric push, and reports `Local` again once a reclaimed slot is
  reused (closes the disagg offload loop end-to-end).
- pyo3 `BlockManager` exposes `blocks_for_request` and `free_block_by_id`.
- `MooncakeAdapter` — a `Fabric` implementation that re-frames the v1 wire
  payload inside a Mooncake transfer envelope, enabling Mooncake protocol
  compatibility over the same offload/ingest path.
- Static-budget A/B baseline (`StaticBudgetBaseline`) modelling vLLM 0.9's
  `thinking_token_budget`; `--baseline` now accepts `static-budget` and `all`.
- N-way `ABComparisonReport`: compares any number of baselines against the run
  under test, one delta column per baseline.
- OpenTelemetry OTLP export: `meridian-core` `otel` cargo feature with a
  `telemetry::install` helper (tracing spans), and a Python `meridian.telemetry`
  module exporting the plugin counters; gated by the `[telemetry]` config
  section and the `otel` Python extra.
- `meridian_disagg_blocks_offloaded_total{fabric}` and
  `meridian_vocab_fallback_total` Prometheus counters in the vLLM plugin.
- `[model.qwen35]` preset and `[telemetry]` section in `meridian.toml.example`.
- ADR-0008 (request preemption policy) — documents the deferral and design.
- Manual, gated `publish` job in `release.yml` (dry-run by default, token-gated
  live publish). mdBook now renders mermaid architecture and sequence diagrams.
- Property-based tests for block-manager invariants and wire-format round-trips.

### Changed

- The vLLM plugin offload path enumerates exact block ids when the manager owns
  them, replacing the `tokens_used // 16` heuristic with a named fallback used
  only when the manager is a pure capacity model.

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

[Unreleased]: https://github.com/angelnicolasc/meridian/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/angelnicolasc/meridian/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/angelnicolasc/meridian/releases/tag/v0.1.0
