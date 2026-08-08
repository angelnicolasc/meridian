//! Metrics and tracing surface.
//!
//! Every meaningful scheduler event emits both a `tracing` span (for
//! OpenTelemetry export) and a `metrics` counter / histogram (for Prometheus
//! scrape). Operators can disable Prometheus by building without the
//! `prometheus` feature; the `tracing` side is always on.
//!
//! ## Metric catalog (stable names)
//!
//! | Name                                       | Type      | Meaning                                                       |
//! |--------------------------------------------|-----------|---------------------------------------------------------------|
//! | `meridian.think_tokens_per_request`        | histogram | Think tokens consumed before `ExitThink`.                     |
//! | `meridian.budget_force_triggered`          | counter   | Times the router emitted `ForceBudget`.                       |
//! | `meridian.budget_force_reason{reason=...}` | counter   | One of `converged`, `overthinking`, `hard_cap`.               |
//! | `meridian.output_critical_eviction`        | counter   | Block manager had to evict an `OutputCritical` block.         |
//! | `meridian.phase_router.tracked_requests`   | gauge     | Number of requests currently in `PhaseRouter` state.          |
//! | `meridian.schedule_batch.duration_ns`      | histogram | Wall time spent inside `MeridianScheduler::schedule_batch`.   |
//! | `meridian.queue_depth{queue=...}`          | gauge     | Depth of `output_queue` / `think_queue`.                      |
//! | `meridian.block_manager.used_bytes`        | gauge     | Total bytes currently allocated across all tiers.             |
//! | `meridian.block_manager.evictions{tier=...}` | counter | Eviction events per source tier.                              |
//! | `meridian.scheduler.batch_size{phase=...}` | histogram | Slot count per scheduled batch, split by phase.               |
//! | `meridian.speculation.policy_basis{phase,basis}` | counter | Phase-conditioning decisions, by what authority they rest on. |
//! | `meridian.speculation.proposal_len{phase=...}` | histogram | Recommended draft depth per decision.                       |
//! | `meridian.speculation.accepted_length{phase=...}` | histogram | Observed accepted draft length per verification step.     |
//! | `meridian.speculation.straddling_steps`    | counter   | Verification steps spanning the `</think>` boundary.          |
//!
//! Re-exports [`BudgetForceReason`](crate::types::BudgetForceReason) for
//! ergonomic access at metric call sites.

pub use crate::types::BudgetForceReason;

/// Stable metric-name constants. Use these instead of stringly-typed names
/// at call sites so renames are caught by the compiler.
pub mod names {
    /// Histogram of think-token counts per request.
    pub const THINK_TOKENS_PER_REQUEST: &str = "meridian.think_tokens_per_request";

    /// Counter of `ForceBudget` events emitted (untagged).
    pub const BUDGET_FORCE_TRIGGERED: &str = "meridian.budget_force_triggered";

    /// Counter of `ForceBudget` events tagged with [`BudgetForceReason`].
    pub const BUDGET_FORCE_REASON: &str = "meridian.budget_force_reason";

    /// Counter of `OutputCritical` evictions — every increment is a user-visible
    /// degradation event and should page if it sustains over a threshold.
    pub const OUTPUT_CRITICAL_EVICTION: &str = "meridian.output_critical_eviction";

    /// Gauge of tracked requests in the phase router.
    pub const PHASE_ROUTER_TRACKED_REQUESTS: &str = "meridian.phase_router.tracked_requests";

    /// Histogram of `schedule_batch` durations (nanoseconds).
    pub const SCHEDULE_BATCH_DURATION_NS: &str = "meridian.schedule_batch.duration_ns";

    /// Gauge of per-queue depth. Tagged with `queue=output|think`.
    pub const QUEUE_DEPTH: &str = "meridian.queue_depth";

    /// Gauge of bytes currently allocated by the block manager.
    pub const BLOCK_MANAGER_USED_BYTES: &str = "meridian.block_manager.used_bytes";

    /// Counter of evictions per source tier. Tagged with `tier=think_complete|think_active|output_critical`.
    pub const BLOCK_MANAGER_EVICTIONS: &str = "meridian.block_manager.evictions";

    /// Histogram of slot count per scheduled batch. Tagged with `phase=output|think`.
    pub const SCHEDULER_BATCH_SIZE: &str = "meridian.scheduler.batch_size";

    /// Counter of phase-conditioning decisions, tagged with
    /// `phase=think|output|prefill|complete` and
    /// `basis=baseline|not_decoding|entropy_ceiling|measured_prior|measured_prior_capped_by_entropy`.
    ///
    /// Until Phase 1 of the phase-conditioned-speculation work has run, a
    /// healthy deployment shows **zero** samples with a `measured_*` basis —
    /// see [`crate::dspark_bridge::hook`].
    pub const SPECULATION_POLICY_BASIS: &str = "meridian.speculation.policy_basis";

    /// Histogram of recommended draft proposal length, tagged with `phase=...`.
    pub const SPECULATION_PROPOSAL_LEN: &str = "meridian.speculation.proposal_len";

    /// Histogram of observed accepted draft length per verification step,
    /// tagged with `phase=think|output`.
    pub const SPECULATION_ACCEPTED_LENGTH: &str = "meridian.speculation.accepted_length";

    /// Counter of verification steps whose committed span crossed the
    /// `</think>` boundary. A rising share of these means phase attribution is
    /// too coarse to trust.
    pub const SPECULATION_STRADDLING_STEPS: &str = "meridian.speculation.straddling_steps";
}
