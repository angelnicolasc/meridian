//! Configuration types loaded from `meridian.toml`.
//!
//! Every field has a documented rationale for its default — see
//! [`meridian.toml.example`](https://github.com/angelnicolasc/meridian/blob/main/meridian.toml.example)
//! for the user-facing version with prose comments. Defaults are tuned for
//! a DeepSeek-R1-class model on an A100/H100 class GPU; downstream operators
//! tune via TOML overlay.

use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Root configuration parsed from `meridian.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeridianConfig {
    /// Scheduler-wide SLO and batching policy.
    #[serde(default)]
    pub scheduler: SchedulerConfig,

    /// Entropy probe / convergence-detection parameters.
    #[serde(default)]
    pub entropy: EntropyConfig,

    /// KV memory tiering and eviction policy.
    #[serde(default)]
    pub kv_memory: KvConfig,

    /// Disaggregated KV transfer (ADR-0006). Off by default — set
    /// `[disagg] enabled = true` to opt into NIXL / Mooncake offload.
    #[serde(default)]
    pub disagg: DisaggConfig,

    /// Phase-conditioned speculative decoding (ADR-0009). Off by default.
    #[serde(default)]
    pub speculation: SpeculationConfig,

    /// Per-model token-boundary and parser configuration. Keyed by model name
    /// (e.g. `"deepseek_r1"`).
    #[serde(default)]
    pub model: std::collections::BTreeMap<String, ModelConfig>,
}

impl MeridianConfig {
    /// Load and validate a config from a TOML file on disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_toml_str(&text)
    }

    /// Parse a config from an in-memory TOML string.
    ///
    /// Equivalent to `<Self as std::str::FromStr>::from_str` but kept under a
    /// distinct name so call sites do not need to import `FromStr`.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let cfg: Self = toml::from_str(s).map_err(Error::from)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Run cross-field validation. Returns the first violation as
    /// [`Error::ConfigValidation`].
    pub fn validate(&self) -> Result<()> {
        self.scheduler.validate()?;
        self.entropy.validate()?;
        self.kv_memory.validate()?;
        self.disagg.validate()?;
        self.speculation.validate()?;
        Ok(())
    }
}

impl Default for MeridianConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            entropy: EntropyConfig::default(),
            kv_memory: KvConfig::default(),
            disagg: DisaggConfig::default(),
            speculation: SpeculationConfig::default(),
            model: std::collections::BTreeMap::new(),
        }
    }
}

impl FromStr for MeridianConfig {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_toml_str(s)
    }
}

// ---------------------------------------------------------------------------
// Scheduler section
// ---------------------------------------------------------------------------

/// Dual-queue scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerConfig {
    /// TPOT budget for think tokens (ms). Loose: user does not see inter-token
    /// latency during reasoning.
    pub think_tpot_budget_ms: f32,

    /// TPOT budget for output tokens (ms). Tight: this is the user-visible
    /// streaming latency.
    pub output_tpot_budget_ms: f32,

    /// Think-phase batch can fill this multiple of the output-phase token budget.
    /// Higher = more GPU utilization during think phase at the cost of output
    /// latency variance.
    pub think_batch_multiplier: f32,

    /// Absolute hard cap on think tokens regardless of entropy signals.
    pub max_think_tokens: u32,

    /// No budget forcing is allowed before this many think tokens have been
    /// emitted. Prevents premature termination of short-but-correct chains.
    pub min_think_tokens: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            think_tpot_budget_ms: 80.0,
            output_tpot_budget_ms: 20.0,
            think_batch_multiplier: 2.5,
            max_think_tokens: 32_768,
            min_think_tokens: 512,
        }
    }
}

impl SchedulerConfig {
    fn validate(&self) -> Result<()> {
        if self.think_tpot_budget_ms <= 0.0 {
            return Err(Error::ConfigValidation {
                field: "scheduler.think_tpot_budget_ms",
                reason: "must be > 0".into(),
            });
        }
        if self.output_tpot_budget_ms <= 0.0 {
            return Err(Error::ConfigValidation {
                field: "scheduler.output_tpot_budget_ms",
                reason: "must be > 0".into(),
            });
        }
        if self.think_batch_multiplier < 1.0 {
            return Err(Error::ConfigValidation {
                field: "scheduler.think_batch_multiplier",
                reason: "must be >= 1.0 (else think phase is starved)".into(),
            });
        }
        if self.min_think_tokens >= self.max_think_tokens {
            return Err(Error::ConfigValidation {
                field: "scheduler.min_think_tokens",
                reason: "must be < scheduler.max_think_tokens".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Entropy section
// ---------------------------------------------------------------------------

/// Entropy probe and convergence-detection thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntropyConfig {
    /// Whether the entropy probe is active. When `false`, the system falls back
    /// to pure token-count budget forcing with the same hard cap.
    pub enabled: bool,

    /// EMA decay for all entropy signals. Small alpha = long memory.
    pub ema_alpha: f32,

    /// Ratio above which the RPDI local/global signal declares overthinking.
    pub rpdi_threshold: f32,

    /// EAT EMA variance below which the model is judged converged.
    pub eat_ema_variance_threshold: f32,

    /// Token entropy (nats) above which a token is classified as a "transition"
    /// for RPDI accounting.
    pub transition_entropy_threshold: f32,

    /// Run the EAT kernel every N think tokens. 1 = every token; higher values
    /// trade signal latency for kernel-launch overhead.
    pub eat_probe_interval_tokens: u32,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ema_alpha: 0.05,
            rpdi_threshold: 3.0,
            eat_ema_variance_threshold: 0.001,
            transition_entropy_threshold: 2.5,
            eat_probe_interval_tokens: 32,
        }
    }
}

impl EntropyConfig {
    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.ema_alpha) {
            return Err(Error::ConfigValidation {
                field: "entropy.ema_alpha",
                reason: "must be in [0, 1]".into(),
            });
        }
        if self.rpdi_threshold <= 1.0 {
            return Err(Error::ConfigValidation {
                field: "entropy.rpdi_threshold",
                reason: "must be > 1.0 (local must exceed global to be meaningful)".into(),
            });
        }
        if self.eat_ema_variance_threshold < 0.0 {
            return Err(Error::ConfigValidation {
                field: "entropy.eat_ema_variance_threshold",
                reason: "variance is non-negative".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KV memory section
// ---------------------------------------------------------------------------

/// Capacity specification for the KV pool.
///
/// `Bytes(n)` pins the budget at exactly `n` bytes. `Auto` defers the
/// decision to the runtime — Python callers may query `torch.cuda.mem_get_info`
/// and pass the resolved number back into the Rust manager. Rust-side
/// validation accepts both forms; the Python side prefers `"auto"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacitySpec {
    /// Use the given absolute capacity in bytes.
    Bytes(u64),
    /// Resolve at runtime to a hardware-aware default.
    Auto,
}

impl Serialize for CapacitySpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Bytes(n) => s.serialize_u64(*n),
            Self::Auto => s.serialize_str("auto"),
        }
    }
}

impl<'de> Deserialize<'de> for CapacitySpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        use serde::de::Error;
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Int(u64),
            Str(String),
        }
        match Raw::deserialize(d)? {
            Raw::Int(n) => Ok(Self::Bytes(n)),
            Raw::Str(s) if s.eq_ignore_ascii_case("auto") => Ok(Self::Auto),
            Raw::Str(s) => Err(D::Error::custom(format!(
                "expected an integer byte count or the literal \"auto\", got {s:?}",
            ))),
        }
    }
}

impl CapacitySpec {
    /// Resolve to a concrete byte budget, falling back to `default_bytes`
    /// when the spec is `Auto`. The caller is responsible for providing a
    /// hardware-aware default.
    #[must_use]
    pub fn resolve(self, default_bytes: u64) -> u64 {
        match self {
            Self::Bytes(n) => n,
            Self::Auto => default_bytes,
        }
    }
}

/// KV block manager configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvConfig {
    /// Free think-phase blocks immediately on phase transition. Safer to keep
    /// `false` until cross-attention backref behavior is fully audited.
    #[serde(default)]
    pub aggressive_think_eviction: bool,

    /// Fraction of KV memory budget reserved for think-phase blocks.
    #[serde(default = "default_think_fraction")]
    pub think_phase_memory_fraction: f32,

    /// Per-block size in bytes. Defaults to 16 KiB which matches the canonical
    /// vLLM block layout for fp16/bf16 KV at 16 tokens per block.
    #[serde(default = "default_block_size_bytes")]
    pub block_size_bytes: u32,

    /// Total KV memory budget. `"auto"` defers the decision to the runtime —
    /// the Python plugin resolves it to `device_total * 0.85` when constructing
    /// the manager.
    #[serde(default = "default_capacity_spec")]
    pub capacity_bytes: CapacitySpec,
}

const fn default_think_fraction() -> f32 {
    0.40
}
const fn default_block_size_bytes() -> u32 {
    16_384
}
const fn default_capacity_spec() -> CapacitySpec {
    CapacitySpec::Auto
}

impl Default for KvConfig {
    fn default() -> Self {
        Self {
            aggressive_think_eviction: false,
            think_phase_memory_fraction: 0.40,
            block_size_bytes: 16_384,
            capacity_bytes: CapacitySpec::Auto,
        }
    }
}

impl KvConfig {
    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.think_phase_memory_fraction) {
            return Err(Error::ConfigValidation {
                field: "kv_memory.think_phase_memory_fraction",
                reason: "must be in [0, 1]".into(),
            });
        }
        if self.block_size_bytes == 0 {
            return Err(Error::ConfigValidation {
                field: "kv_memory.block_size_bytes",
                reason: "must be > 0".into(),
            });
        }
        if self.capacity_bytes == CapacitySpec::Bytes(0) {
            return Err(Error::ConfigValidation {
                field: "kv_memory.capacity_bytes",
                reason: "must be > 0 (or use \"auto\")".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Disagg section — ADR-0006
// ---------------------------------------------------------------------------

/// Disaggregated KV transfer (offload / ingest) settings.
///
/// Off by default. When enabled, the scheduler issues `offload_block`
/// calls at [`crate::types::PhaseEvent::ExitThink`] for any block whose tier
/// has been demoted to `ThinkComplete`, amortising transfer over batches of
/// at least `offload_threshold_blocks` blocks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DisaggConfig {
    /// Master switch. When `false`, the scheduler treats every block as
    /// `BlockLocation::Local` and never issues offload calls.
    pub enabled: bool,

    /// Fabric to route offloads through.
    pub fabric: DisaggFabric,

    /// Minimum number of think-complete blocks to accumulate before flushing
    /// to the fabric. Bigger batches amortise per-transfer fixed costs at
    /// the price of extra device memory pressure.
    pub offload_threshold_blocks: u32,
}

impl Default for DisaggConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fabric: DisaggFabric::None,
            offload_threshold_blocks: 4,
        }
    }
}

impl DisaggConfig {
    fn validate(self) -> Result<()> {
        if self.enabled && matches!(self.fabric, DisaggFabric::None) {
            return Err(Error::ConfigValidation {
                field: "disagg.fabric",
                reason: "must be `nixl` or `mooncake` when disagg.enabled = true".into(),
            });
        }
        if self.offload_threshold_blocks == 0 {
            return Err(Error::ConfigValidation {
                field: "disagg.offload_threshold_blocks",
                reason: "must be >= 1 (zero would offload on every demote)".into(),
            });
        }
        Ok(())
    }
}

/// Disaggregated transport selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisaggFabric {
    /// No fabric — disabled even if `disagg.enabled = true`.
    None,
    /// NVIDIA NIXL (CUDA-blessed, Python API).
    Nixl,
    /// Mooncake protocol-compatible adapter.
    Mooncake,
}

impl DisaggFabric {
    /// Stable telemetry label.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Nixl => "nixl",
            Self::Mooncake => "mooncake",
        }
    }
}

// ---------------------------------------------------------------------------
// Speculation section — ADR-0009
// ---------------------------------------------------------------------------

/// Phase-conditioned speculative decoding settings.
///
/// Off by default, and inert even when enabled until
/// `[speculation.acceptance_prior]` is supplied — see
/// [`crate::dspark_bridge::hook`] for why that posture is deliberate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SpeculationConfig {
    /// Master switch. When `false` the serving stack never consults the hook.
    #[serde(default)]
    pub enabled: bool,

    /// Draft depth the serving stack would use without the hook. Also the
    /// ceiling on what an uncalibrated hook may return.
    #[serde(default = "default_baseline_proposal_len")]
    pub baseline_proposal_len: u32,

    /// Floor on recommended draft depth.
    #[serde(default = "default_min_proposal_len")]
    pub min_proposal_len: u32,

    /// Ceiling on recommended draft depth, bounded by the drafter's block size.
    #[serde(default = "default_baseline_proposal_len")]
    pub max_proposal_len: u32,

    /// Target-model vocabulary size, needed for the entropy bound.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: u32,

    /// Whether to derive a draft-depth ceiling from the entropy probe.
    #[serde(default = "default_true")]
    pub use_entropy_ceiling: bool,

    /// Marginal microseconds to draft one additional token.
    #[serde(default = "default_draft_token_us")]
    pub draft_token_us: f32,

    /// Fixed microseconds per verification pass.
    #[serde(default = "default_verify_fixed_us")]
    pub verify_fixed_us: f32,

    /// Marginal microseconds to verify one additional position.
    #[serde(default = "default_verify_token_us")]
    pub verify_token_us: f32,

    /// Measured per-phase acceptance rates. Absent until a Phase 1 run exists.
    ///
    /// There is deliberately **no way to express an unmeasured prior in
    /// configuration**: the schema requires the full provenance of a real run
    /// alongside the numbers, so an operator cannot hand-tune their way into
    /// phase conditioning on a hunch.
    #[serde(default)]
    pub acceptance_prior: Option<MeasuredPriorSpec>,
}

const fn default_baseline_proposal_len() -> u32 {
    // DeepSpec's released DSpark Qwen3 checkpoints are `block7`.
    7
}
const fn default_min_proposal_len() -> u32 {
    1
}
const fn default_vocab_size() -> u32 {
    // Qwen3.
    151_936
}
const fn default_true() -> bool {
    true
}
const fn default_draft_token_us() -> f32 {
    40.0
}
const fn default_verify_fixed_us() -> f32 {
    900.0
}
const fn default_verify_token_us() -> f32 {
    25.0
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            baseline_proposal_len: default_baseline_proposal_len(),
            min_proposal_len: default_min_proposal_len(),
            max_proposal_len: default_baseline_proposal_len(),
            vocab_size: default_vocab_size(),
            use_entropy_ceiling: true,
            draft_token_us: default_draft_token_us(),
            verify_fixed_us: default_verify_fixed_us(),
            verify_token_us: default_verify_token_us(),
            acceptance_prior: None,
        }
    }
}

impl SpeculationConfig {
    /// Build the runtime hook configuration this section describes.
    ///
    /// # Errors
    ///
    /// [`Error::ConfigValidation`] if any field or cross-field invariant is
    /// violated, propagated from
    /// [`PhaseConditioningConfig::validate`](crate::dspark_bridge::PhaseConditioningConfig::validate).
    pub fn to_hook_config(&self) -> Result<crate::dspark_bridge::PhaseConditioningConfig> {
        use crate::dspark_bridge::{
            AcceptancePrior, DraftCostModel, PhaseConditioningConfig, Provenance,
        };

        let prior = match &self.acceptance_prior {
            None => AcceptancePrior::Uncalibrated,
            Some(spec) => AcceptancePrior::measured(
                spec.think,
                spec.output,
                Provenance::Measured {
                    harness: spec.harness.clone(),
                    draft_checkpoint: spec.draft_checkpoint.clone(),
                    target_model: spec.target_model.clone(),
                    thinking_mode: spec.thinking_mode,
                    recorded_on: spec.recorded_on.clone(),
                },
            )?,
        };

        let config = PhaseConditioningConfig {
            baseline_proposal_len: self.baseline_proposal_len,
            min_proposal_len: self.min_proposal_len,
            max_proposal_len: self.max_proposal_len,
            vocab_size: self.vocab_size,
            cost: DraftCostModel {
                draft_token_us: self.draft_token_us,
                verify_fixed_us: self.verify_fixed_us,
                verify_token_us: self.verify_token_us,
            },
            prior,
            use_entropy_ceiling: self.use_entropy_ceiling,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if let Some(spec) = &self.acceptance_prior {
            if spec.harness.trim().is_empty()
                || spec.draft_checkpoint.trim().is_empty()
                || spec.target_model.trim().is_empty()
                || spec.recorded_on.trim().is_empty()
            {
                return Err(Error::ConfigValidation {
                    field: "speculation.acceptance_prior",
                    reason: "every provenance field must be a non-empty description of the run \
                             that produced the numbers"
                        .into(),
                });
            }
        }
        self.to_hook_config().map(|_| ())
    }
}

/// Measured per-phase acceptance rates, with the provenance of the run that
/// produced them.
///
/// Every field is required. Numbers without a run description are not
/// admissible input — see [`crate::dspark_bridge::provenance`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MeasuredPriorSpec {
    /// Measured mean token acceptance rate during the think phase.
    pub think: f32,
    /// Measured mean token acceptance rate during the output phase.
    pub output: f32,
    /// Evaluation harness and version, e.g. `"DeepSpec@<sha>"`.
    pub harness: String,
    /// Draft checkpoint the rates were measured against.
    pub draft_checkpoint: String,
    /// Target model the rates were measured against.
    pub target_model: String,
    /// Whether the target ran with thinking mode enabled.
    pub thinking_mode: bool,
    /// ISO-8601 date the run was recorded.
    pub recorded_on: String,
}

// ---------------------------------------------------------------------------
// Per-model config
// ---------------------------------------------------------------------------

/// Per-model token-boundary configuration. Used by the phase router to detect
/// `<think>` / `</think>` transitions for any given served model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Tokenizer-encoded ids that signal "begin reasoning". Some models
    /// (DeepSeek-R1) tokenize `"\n<think>\n"` as a single id; others tokenize
    /// `"<think>"` directly. Configure both if both can occur.
    #[serde(default)]
    pub think_start_token_ids: Vec<u32>,

    /// Tokenizer-encoded ids that signal "end reasoning".
    #[serde(default)]
    pub think_end_token_ids: Vec<u32>,

    /// Reasoning-parser key understood by vLLM (`deepseek_r1`, `qwen3`, ...).
    #[serde(default)]
    pub reasoning_parser: Option<String>,

    /// Whether the model supports `think_disable` via prompt control.
    #[serde(default)]
    pub supports_think_disable: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            think_start_token_ids: Vec::new(),
            think_end_token_ids: Vec::new(),
            reasoning_parser: None,
            supports_think_disable: false,
        }
    }
}
