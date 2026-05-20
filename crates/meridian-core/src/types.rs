//! Shared domain types.
//!
//! These are the contracts that the rest of the workspace and downstream
//! consumers depend on. They are intentionally small, `Copy` where it is sound,
//! and `serde`-aware where useful for snapshot/telemetry use cases.

use std::ops::Range;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Budget-force diagnostic enum
// ---------------------------------------------------------------------------

/// Reason the router emitted [`PhaseEvent::ForceBudget`].
///
/// Carried inline on the event so the scheduler can attribute the cause to a
/// metric label and operators can answer "are we forcing because the model
/// converged, because it was overthinking, or because we hit the wall?".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BudgetForceReason {
    /// EAT EMA variance fell below the convergence threshold and
    /// `tokens_so_far > min_think_tokens`.
    Converged,
    /// RPDI local/global ratio exceeded the overthinking threshold.
    Overthinking,
    /// `max_think_tokens` reached regardless of any entropy signal.
    HardCap,
}

impl BudgetForceReason {
    /// Stable string label for telemetry export. **Do not rename** — these
    /// values are part of the public metric contract documented in
    /// [`crate::metrics::names`].
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::Overthinking => "overthinking",
            Self::HardCap => "hard_cap",
        }
    }
}

// ---------------------------------------------------------------------------
// Phase state machine types
// ---------------------------------------------------------------------------

/// Per-request reasoning phase.
///
/// The state machine is monotonic: a request advances
/// `Prefill -> ThinkDecode -> OutputDecode -> Complete`. Requests for models
/// that do not emit `<think>` skip `ThinkDecode` and transition directly to
/// `OutputDecode` on the first non-prefill token.
///
/// # Why an enum with payload rather than a flat struct?
///
/// Rust's type system makes the *valid combinations of state and payload*
/// unrepresentable when they shouldn't coexist. `eat_ema` only makes sense
/// inside `ThinkDecode`; encoding that statically removes a class of bugs.
#[derive(Debug, Clone, PartialEq)]
pub enum ThinkPhase {
    /// Request is still consuming prompt tokens; no decode has begun.
    Prefill,

    /// Request has emitted `<think>` and is producing reasoning tokens.
    ThinkDecode {
        /// Number of think tokens emitted so far in this phase.
        tokens_so_far: u32,

        /// Exponential moving average of the EAT (Entropy After `</think>`) signal.
        ///
        /// EAT measures the softmax probability mass the model places on the
        /// `</think>` token at each step. When this EMA stabilises (low
        /// variance), the model is signalling that it is willing to stop —
        /// safe to force the budget. See `arXiv:2509.26522`.
        eat_ema: f32,

        /// EMA of the squared EAT signal. Combined with `eat_ema` this gives a
        /// running variance estimate without storing a window of samples.
        eat_ema_sq: f32,

        /// RPDI local component: EMA of high-entropy transition-token frequency
        /// over a short recent window.
        rpdi_local: f32,

        /// RPDI global component: cumulative average of transition-token
        /// frequency since `ThinkDecode` began.
        rpdi_global: f32,

        /// True once a budget-force injection has been scheduled for this
        /// request. Prevents double-injection when the next token arrives
        /// before the previous injection has been observed in the stream.
        force_in_progress: bool,
    },

    /// Request has emitted `</think>` and is producing user-visible output.
    OutputDecode {
        /// Total think tokens that were consumed before this phase began.
        /// Useful for post-hoc analysis and emitted on the `meridian.think_tokens_per_request`
        /// histogram.
        think_tokens_used: u32,
    },

    /// Request has terminated (EOS or stop).
    Complete,
}

/// Event emitted by the [`crate::phase_router::PhaseRouter`] for each decoded
/// token. The scheduler consumes these to move requests between queues and
/// signal the token injector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseEvent {
    /// Nothing of interest happened on this token.
    None,

    /// The request transitioned `Prefill -> ThinkDecode`.
    EnterThink,

    /// The request transitioned `ThinkDecode -> OutputDecode`.
    ExitThink {
        /// Number of think tokens consumed in the just-completed phase.
        tokens_used: u32,
    },

    /// The router has decided to force `</think>`. The scheduler must arrange
    /// for `inject_token` to appear as the next sampled token for this request.
    ForceBudget {
        /// The token id to inject (typically the canonical `</think>` id for
        /// the model).
        inject_token: u32,
        /// Why the router decided to force the budget. Used as a metric label
        /// (`meridian.budget_force_reason{reason=...}`) and for postmortem
        /// analysis.
        reason: BudgetForceReason,
    },

    /// The request reached EOS.
    Complete,
}

// ---------------------------------------------------------------------------
// KV block tiers
// ---------------------------------------------------------------------------

/// Three-tier eviction priority for KV blocks.
///
/// Lower-numbered tiers are evicted first. A request whose think phase has
/// completed donates all of its think-phase blocks to `ThinkComplete`
/// immediately, so under memory pressure the next eviction round prefers them
/// over any block still on the critical path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum BlockTier {
    /// Think-phase blocks of a request that has already transitioned to
    /// `OutputDecode`. **First to evict.**
    ThinkComplete = 0,

    /// Think-phase blocks belonging to a request that is still reasoning.
    ThinkActive = 1,

    /// Output-phase blocks. Evicted only as a last resort; doing so triggers a
    /// `meridian.output_critical_eviction` counter increment and a warning
    /// span, since this is user-visible degradation.
    OutputCritical = 2,
}

impl BlockTier {
    /// Iterate tiers in eviction order (`ThinkComplete -> ThinkActive -> OutputCritical`).
    #[must_use]
    pub const fn eviction_order() -> [Self; 3] {
        [Self::ThinkComplete, Self::ThinkActive, Self::OutputCritical]
    }

    /// Stable telemetry label. Part of the public metric contract.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::ThinkComplete => "think_complete",
            Self::ThinkActive => "think_active",
            Self::OutputCritical => "output_critical",
        }
    }
}

/// Physical residency of a KV block.
///
/// The default in-process manager only ever returns [`BlockLocation::Local`];
/// disaggregated deployments (see ADR-0006) may report
/// [`BlockLocation::Remote`] for blocks that have been offloaded across a
/// fabric such as NVIDIA NIXL or Mooncake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum BlockLocation {
    /// The block lives in the prefill-decode worker that produced it.
    Local,
    /// The block has been offloaded to a remote KV fabric.
    Remote {
        /// Static label of the fabric (`"nixl"`, `"mooncake"`, ...).
        fabric: &'static str,
    },
}

impl BlockLocation {
    /// Stable telemetry label for the location's kind.
    #[must_use]
    pub const fn kind_label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Remote { .. } => "remote",
        }
    }
}

/// A single KV block tracked by [`crate::block_manager::PhaseAwareBlockManager`].
#[derive(Debug, Clone)]
pub struct KvBlock {
    /// Stable identifier within the block manager.
    pub block_id: u32,
    /// The request that owns this block.
    pub req_id: u64,
    /// Current eviction tier; mutated by `demote_think_blocks` on phase transition.
    pub tier: BlockTier,
    /// Half-open sequence-position range this block covers, in tokens.
    pub token_range: Range<u32>,
    /// Memory footprint in bytes. With MLA (`MlaBlockConfig`) this is
    /// `latent_dim`-derived, not the full KV.
    pub size_bytes: u32,
    /// Monotonic-clock timestamp of the last access. Used for LRU ordering
    /// inside a tier.
    pub last_access_ns: u64,
}

/// Configuration for Multi-Head Latent Attention block sizing.
///
/// MLA-aware models (DeepSeek-V3, Qwen3) store a low-rank latent representation
/// instead of full per-head KV, dramatically reducing KV footprint. The
/// scheduler is MLA-aware so its memory accounting reflects what the model
/// actually allocates on device.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MlaBlockConfig {
    /// Latent dimension stored per token (e.g. 512 for DeepSeek-V3).
    pub latent_dim: u32,
    /// Bytes per latent element on device (2 for bf16/fp16, 1 for int8).
    pub bytes_per_element: u8,
    /// Tokens per block (usual vLLM convention: 16).
    pub tokens_per_block: u32,
}

impl MlaBlockConfig {
    /// Returns the size in bytes of a single KV block under this MLA layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use meridian_core::types::MlaBlockConfig;
    /// let mla = MlaBlockConfig { latent_dim: 512, bytes_per_element: 2, tokens_per_block: 16 };
    /// assert_eq!(mla.block_size_bytes(), 512 * 2 * 16);
    /// ```
    #[must_use]
    pub const fn block_size_bytes(&self) -> u32 {
        self.latent_dim * self.bytes_per_element as u32 * self.tokens_per_block
    }
}

// ---------------------------------------------------------------------------
// Entropy signal exposed by the probe
// ---------------------------------------------------------------------------

/// Per-token entropy signal computed by the entropy probe on the secondary
/// CUDA stream (or its CPU equivalent in tests).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntropySignal {
    /// Shannon entropy of the sampled token distribution, in nats.
    /// A high value indicates a "transition" token where multiple continuations
    /// are roughly equiprobable.
    pub token_entropy: f32,

    /// EAT raw signal: total softmax probability mass on the configured set of
    /// `</think>` token ids at this decode step.
    pub eat: f32,

    /// EMA of `eat`, smoothed with `entropy.ema_alpha`.
    pub eat_ema: f32,

    /// Variance of the EAT EMA. When this drops below
    /// `entropy.eat_ema_variance_threshold` the model is judged to have
    /// converged.
    pub eat_ema_variance: f32,
}

// ---------------------------------------------------------------------------
// Scheduler queue types
// ---------------------------------------------------------------------------

/// Scheduling priority assigned to a queued request slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Default priority for fresh requests.
    Normal,
    /// Elevated priority after `ExitThink`; output tokens are user-visible.
    High,
}

/// A request as seen by the scheduler queues.
#[derive(Debug, Clone, Copy)]
pub struct RequestSlot {
    /// Request identifier (matches the id used by the [`crate::PhaseRouter`]).
    pub req_id: u64,
    /// Effective scheduling priority.
    pub priority: Priority,
}

/// A batch produced by [`crate::scheduler::MeridianScheduler::schedule_batch`].
#[derive(Debug, Default, Clone)]
pub struct ScheduledBatch {
    /// Output-phase slots dispatched this iteration.
    pub output_slots: Vec<RequestSlot>,
    /// Think-phase slots dispatched this iteration.
    pub think_slots: Vec<RequestSlot>,
}

impl ScheduledBatch {
    /// Total number of slots dispatched.
    #[must_use]
    pub fn len(&self) -> usize {
        self.output_slots.len() + self.think_slots.len()
    }

    /// `true` if no slots were dispatched.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.output_slots.is_empty() && self.think_slots.is_empty()
    }
}
