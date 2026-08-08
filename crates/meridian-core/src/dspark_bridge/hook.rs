//! Phase-conditioning hook — the scheduler-facing surface (**MER-P0.5**).
//!
//! Given a phase label and an optional entropy sample, return a recommended
//! draft-parameter adjustment. That is the whole contract.
//!
//! # The safety property
//!
//! The hypothesis this project exists to test — that draft acceptance is lower
//! during the think phase because DeepSpec's released Qwen3 drafters were
//! trained only on non-thinking-mode generations — **has not been measured**.
//! Shipping a scheduler that acts on an unmeasured hypothesis would be the
//! exact failure mode the blueprint's §11 forbids.
//!
//! So the hook is built around one invariant, pinned by the unit test
//! `uncalibrated_hook_never_drafts_deeper_than_baseline` in this module and by
//! `an_uncalibrated_hook_never_exceeds_its_baseline_across_any_fixture` in
//! `tests/dspark_bridge_synthetic.rs`:
//!
//! > **With an [`AcceptancePrior::Uncalibrated`] prior, the returned proposal
//! > length is never greater than the operator's configured baseline.**
//!
//! Uncalibrated, the hook has exactly one justified move: shrink the draft
//! budget when Corollary 5 of [`super::confidence_model`] *proves* that a
//! deeper draft cannot pay off at the observed entropy. It can never grow the
//! budget, because nothing it knows would justify that. Only a
//! [`AcceptancePrior::Measured`] prior — which by construction carries a
//! [`Provenance::Measured`] tag, obtainable only from a real run — unlocks
//! upward adjustment.
//!
//! The result is that merging this module changes production behaviour by
//! default only in the direction that is provably correct, and the interesting
//! behaviour switches on the day Phase 1 produces data.
//!
//! # Decision rule
//!
//! 1. Establish a planning acceptance rate `a`:
//!    - measured prior for the phase, if calibrated;
//!    - otherwise the Corollary 5 ceiling from the entropy sample;
//!    - in either case capped by the ceiling, because a per-step proof
//!      dominates an aggregate average.
//! 2. Maximise the throughput proxy
//!    `Θ(γ) = τ(a, γ) / (draft_us·γ + verify_fixed_us + verify_marginal_us·γ)`
//!    over `γ ∈ [min_proposal_len, max_proposal_len]`, where `τ` is
//!    [`expected_accepted_length`]. This is a deliberately simplified stand-in
//!    for DSpark's `Θ = τ · SPS(B)` objective: same shape, no cost table, no
//!    GPU.
//! 3. If the prior is uncalibrated, clamp the result to the baseline.

use crate::dspark_bridge::confidence_model::{AcceptanceBounds, expected_accepted_length};
use crate::dspark_bridge::provenance::Provenance;
use crate::metrics::names;
use crate::types::{EntropySignal, ThinkPhase};

// ---------------------------------------------------------------------------
// Phase label
// ---------------------------------------------------------------------------

/// The phase a request is in, as far as speculative decoding cares.
///
/// A coarsening of [`ThinkPhase`]: the speculation path does not need the
/// entropy accumulators, only which side of the `</think>` boundary a
/// verification step landed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeculationPhase {
    /// Prompt is still being consumed; no draft/verify cycles yet.
    Prefill,
    /// Inside the reasoning span. The phase the hypothesis is about.
    Think,
    /// Producing user-visible output — the distribution DeepSpec's released
    /// checkpoints were actually trained on.
    Output,
    /// Request has terminated.
    Complete,
}

impl SpeculationPhase {
    /// Stable telemetry label. Part of the public metric contract; matches the
    /// vocabulary [`crate::phase_router::PhaseRouter::phase_of_kind`] already
    /// exports, so dashboards can join on it.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Think => "think",
            Self::Output => "output",
            Self::Complete => "complete",
        }
    }

    /// `true` for the two phases that actually run draft/verify cycles.
    #[must_use]
    pub const fn is_decoding(self) -> bool {
        matches!(self, Self::Think | Self::Output)
    }
}

impl From<&ThinkPhase> for SpeculationPhase {
    fn from(phase: &ThinkPhase) -> Self {
        match phase {
            ThinkPhase::Prefill => Self::Prefill,
            ThinkPhase::ThinkDecode { .. } => Self::Think,
            ThinkPhase::OutputDecode { .. } => Self::Output,
            ThinkPhase::Complete => Self::Complete,
        }
    }
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/// Per-phase acceptance rates, and whether anyone has actually measured them.
///
/// The default is [`AcceptancePrior::Uncalibrated`] and it is expected to stay
/// that way until Phase 1 runs. See [`Provenance`].
#[derive(Debug, Clone, PartialEq)]
pub enum AcceptancePrior {
    /// No phase-segmented measurement exists. Phase conditioning is inert.
    Uncalibrated,

    /// Calibrated from a real phase-segmented run.
    Measured {
        /// Mean per-token acceptance rate observed during think-phase
        /// verification steps.
        think: f32,
        /// Mean per-token acceptance rate observed during output-phase
        /// verification steps.
        output: f32,
        /// Evidence the numbers came from hardware.
        provenance: Provenance,
    },
}

impl Default for AcceptancePrior {
    fn default() -> Self {
        Self::Uncalibrated
    }
}

impl AcceptancePrior {
    /// Build a measured prior, rejecting anything not backed by a real run.
    ///
    /// # Errors
    ///
    /// [`crate::Error::SyntheticProvenance`] if `provenance` is synthetic;
    /// [`crate::Error::ConfigValidation`] if either rate is outside `[0, 1]`.
    pub fn measured(think: f32, output: f32, provenance: Provenance) -> crate::Result<Self> {
        for (label, value) in [("think", think), ("output", output)] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(crate::Error::ConfigValidation {
                    field: "speculation.acceptance_prior",
                    reason: format!("{label} acceptance rate must be a probability, got {value}"),
                });
            }
        }
        // Re-tagging proves the provenance is a real run without consuming it.
        let run = provenance.clone().into_measured()?;
        debug_assert!(!run.harness.is_empty());
        Ok(Self::Measured {
            think,
            output,
            provenance,
        })
    }

    /// Rate for a phase, or `None` when uncalibrated.
    #[must_use]
    pub const fn for_phase(&self, phase: SpeculationPhase) -> Option<f32> {
        match (self, phase) {
            (Self::Measured { think, .. }, SpeculationPhase::Think) => Some(*think),
            (Self::Measured { output, .. }, SpeculationPhase::Output) => Some(*output),
            _ => None,
        }
    }

    /// `true` when a measurement is present.
    #[must_use]
    pub const fn is_calibrated(&self) -> bool {
        matches!(self, Self::Measured { .. })
    }

    /// Signed think-minus-output gap, when calibrated. Negative is the
    /// direction the Section 5 hypothesis predicts.
    #[must_use]
    pub const fn phase_gap(&self) -> Option<f32> {
        match self {
            Self::Measured { think, output, .. } => Some(*think - *output),
            Self::Uncalibrated => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Microsecond costs of one draft/verify cycle.
///
/// Deliberately three scalars rather than DSpark's measured cost table: the
/// hook must run without a GPU, and the optimiser only needs the *shape* of
/// the cost curve to place `γ*`. Operators who have profiled their deployment
/// can set these from measurement; the defaults are placeholders that produce
/// sane relative ordering, not a claim about any particular hardware.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftCostModel {
    /// Marginal cost of drafting one additional token.
    pub draft_token_us: f32,
    /// Fixed cost of a verification pass, independent of depth.
    pub verify_fixed_us: f32,
    /// Marginal cost of verifying one additional position.
    pub verify_token_us: f32,
}

impl Default for DraftCostModel {
    fn default() -> Self {
        Self {
            draft_token_us: 40.0,
            verify_fixed_us: 900.0,
            verify_token_us: 25.0,
        }
    }
}

impl DraftCostModel {
    /// Total microseconds for a cycle that drafts `proposal_len` tokens.
    #[must_use]
    pub fn cycle_cost_us(&self, proposal_len: u32) -> f32 {
        let gamma = proposal_len as f32;
        self.verify_fixed_us + gamma * (self.draft_token_us + self.verify_token_us)
    }

    /// Throughput proxy `Θ(γ) = τ(a, γ) / cost(γ)`, in accepted tokens per
    /// microsecond.
    #[must_use]
    pub fn throughput(&self, acceptance: f32, proposal_len: u32) -> f32 {
        let cost = self.cycle_cost_us(proposal_len);
        if cost <= 0.0 {
            return 0.0;
        }
        expected_accepted_length(acceptance, proposal_len) / cost
    }

    fn validate(&self) -> crate::Result<()> {
        let fields = [
            ("draft_token_us", self.draft_token_us),
            ("verify_fixed_us", self.verify_fixed_us),
            ("verify_token_us", self.verify_token_us),
        ];
        for (name, value) in fields {
            if !value.is_finite() || value < 0.0 {
                return Err(crate::Error::ConfigValidation {
                    field: "speculation.cost",
                    reason: format!("{name} must be finite and >= 0, got {value}"),
                });
            }
        }
        if self.verify_fixed_us <= 0.0 {
            return Err(crate::Error::ConfigValidation {
                field: "speculation.cost",
                reason: "verify_fixed_us must be > 0 (a free verify pass makes γ unbounded)".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Hook configuration
// ---------------------------------------------------------------------------

/// Configuration for [`PhaseConditioningHook`].
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseConditioningConfig {
    /// Proposal length the serving stack would use without this hook. Also the
    /// ceiling on what an uncalibrated hook may return.
    pub baseline_proposal_len: u32,
    /// Floor on the returned proposal length. `1` keeps speculation on; `0`
    /// would let the hook disable it entirely, which is deliberately allowed
    /// but not the default.
    pub min_proposal_len: u32,
    /// Ceiling on the returned proposal length. Bounded by the drafter's block
    /// size — DeepSpec's released DSpark checkpoints use `block7`.
    pub max_proposal_len: u32,
    /// Vocabulary size of the target model, needed for the Corollary 5 bound.
    pub vocab_size: u32,
    /// Cycle cost model driving the throughput objective.
    pub cost: DraftCostModel,
    /// Measured per-phase acceptance rates, if any.
    pub prior: AcceptancePrior,
    /// Whether to use the entropy-derived ceiling at all. Turning this off
    /// makes an uncalibrated hook a strict no-op, which is the most
    /// conservative possible posture.
    pub use_entropy_ceiling: bool,
}

impl Default for PhaseConditioningConfig {
    fn default() -> Self {
        Self {
            // DeepSpec's released DSpark Qwen3 checkpoints are `block7`.
            baseline_proposal_len: 7,
            min_proposal_len: 1,
            max_proposal_len: 7,
            // Qwen3's vocabulary.
            vocab_size: 151_936,
            cost: DraftCostModel::default(),
            prior: AcceptancePrior::Uncalibrated,
            use_entropy_ceiling: true,
        }
    }
}

impl PhaseConditioningConfig {
    /// Validate cross-field invariants.
    ///
    /// # Errors
    ///
    /// [`crate::Error::ConfigValidation`] on the first violation.
    pub fn validate(&self) -> crate::Result<()> {
        if self.min_proposal_len > self.max_proposal_len {
            return Err(crate::Error::ConfigValidation {
                field: "speculation.min_proposal_len",
                reason: "must be <= speculation.max_proposal_len".into(),
            });
        }
        if self.baseline_proposal_len > self.max_proposal_len
            || self.baseline_proposal_len < self.min_proposal_len
        {
            return Err(crate::Error::ConfigValidation {
                field: "speculation.baseline_proposal_len",
                reason: "must lie within [min_proposal_len, max_proposal_len]".into(),
            });
        }
        if self.vocab_size < 2 {
            return Err(crate::Error::ConfigValidation {
                field: "speculation.vocab_size",
                reason: "must be >= 2".into(),
            });
        }
        self.cost.validate()
    }
}

// ---------------------------------------------------------------------------
// Policy output
// ---------------------------------------------------------------------------

/// Why the hook returned the proposal length it did.
///
/// Emitted as a metric label so an operator can tell at a glance whether the
/// hook is doing anything, and if so on what authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicyBasis {
    /// No entropy sample and no measurement: baseline returned untouched.
    Baseline,
    /// The request is not in a decoding phase; speculation does not apply.
    NotDecoding,
    /// Corollary 5's entropy ceiling justified a shallower draft.
    EntropyCeiling,
    /// A measured phase prior set the depth.
    MeasuredPrior,
    /// A measured prior, further capped by the entropy ceiling at this step.
    MeasuredPriorCappedByEntropy,
}

impl PolicyBasis {
    /// Stable telemetry label.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::NotDecoding => "not_decoding",
            Self::EntropyCeiling => "entropy_ceiling",
            Self::MeasuredPrior => "measured_prior",
            Self::MeasuredPriorCappedByEntropy => "measured_prior_capped_by_entropy",
        }
    }

    /// `true` when the basis rests on measured data rather than a bound.
    #[must_use]
    pub const fn is_measured(self) -> bool {
        matches!(
            self,
            Self::MeasuredPrior | Self::MeasuredPriorCappedByEntropy
        )
    }
}

/// The hook's recommendation for one request at one decode step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftPolicy {
    /// Recommended number of tokens to draft before verifying.
    pub proposal_len: u32,
    /// Acceptance rate used to plan, when one could be justified. `None` means
    /// the hook had no defensible number and fell back to the baseline —
    /// deliberately not a default value, so callers cannot mistake absence of
    /// evidence for an estimate of zero.
    pub planning_acceptance: Option<f32>,
    /// Expected tokens committed per cycle at [`Self::proposal_len`] under
    /// [`Self::planning_acceptance`], when the latter exists.
    pub expected_accepted_length: Option<f32>,
    /// What authority the recommendation rests on.
    pub basis: PolicyBasis,
}

impl DraftPolicy {
    /// The untouched-baseline policy.
    #[must_use]
    pub const fn baseline(proposal_len: u32, basis: PolicyBasis) -> Self {
        Self {
            proposal_len,
            planning_acceptance: None,
            expected_accepted_length: None,
            basis,
        }
    }
}

// ---------------------------------------------------------------------------
// The hook
// ---------------------------------------------------------------------------

/// Phase-conditioning hook.
///
/// Stateless and `Sync`: one instance is shared across decode workers and
/// consulted per request per step. Construction validates the configuration so
/// the hot path has no failure mode.
///
/// # Examples
///
/// An uncalibrated hook with no entropy sample is a no-op:
///
/// ```
/// use meridian_core::dspark_bridge::{
///     PhaseConditioningConfig, PhaseConditioningHook, PolicyBasis, SpeculationPhase,
/// };
///
/// let hook = PhaseConditioningHook::new(PhaseConditioningConfig::default())?;
/// let policy = hook.policy_for(SpeculationPhase::Think, None);
///
/// assert_eq!(policy.proposal_len, 7);
/// assert_eq!(policy.basis, PolicyBasis::Baseline);
/// assert!(policy.planning_acceptance.is_none());
/// # Ok::<(), meridian_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct PhaseConditioningHook {
    config: PhaseConditioningConfig,
}

impl PhaseConditioningHook {
    /// Construct a hook from a configuration.
    ///
    /// # Errors
    ///
    /// [`crate::Error::ConfigValidation`] if the configuration is
    /// self-inconsistent.
    pub fn new(config: PhaseConditioningConfig) -> crate::Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Read-only view of the configuration in effect.
    #[must_use]
    pub const fn config(&self) -> &PhaseConditioningConfig {
        &self.config
    }

    /// `true` when a measured prior is installed. Operators should expect
    /// `false` until Phase 1 of the blueprint has run.
    #[must_use]
    pub const fn is_calibrated(&self) -> bool {
        self.config.prior.is_calibrated()
    }

    /// Recommend draft parameters for a request in `phase`, optionally using a
    /// fresh entropy sample.
    ///
    /// The entropy sample is optional for the same reason it is optional in
    /// [`crate::phase_router::PhaseRouter::on_token`]: the probe runs on its
    /// own stream and may not have a sample for every step.
    #[must_use]
    pub fn policy_for(
        &self,
        phase: SpeculationPhase,
        entropy_signal: Option<&EntropySignal>,
    ) -> DraftPolicy {
        let policy = self.decide(phase, entropy_signal);
        metrics::counter!(
            names::SPECULATION_POLICY_BASIS,
            "phase" => phase.as_label(),
            "basis" => policy.basis.as_label(),
        )
        .increment(1);
        metrics::histogram!(
            names::SPECULATION_PROPOSAL_LEN,
            "phase" => phase.as_label(),
        )
        .record(f64::from(policy.proposal_len));
        policy
    }

    fn decide(
        &self,
        phase: SpeculationPhase,
        entropy_signal: Option<&EntropySignal>,
    ) -> DraftPolicy {
        let baseline = self.config.baseline_proposal_len;

        if !phase.is_decoding() {
            return DraftPolicy::baseline(baseline, PolicyBasis::NotDecoding);
        }

        // Corollary 5: an entropy sample proves an upper bound on the
        // single-step acceptance rate of any deterministic drafter.
        let ceiling = if self.config.use_entropy_ceiling {
            entropy_signal.map(|signal| {
                AcceptanceBounds::from_signal(signal, self.config.vocab_size).deterministic_ceiling
            })
        } else {
            None
        };

        let prior = self.config.prior.for_phase(phase);

        let (planning_acceptance, basis) = match (prior, ceiling) {
            // A per-step proof dominates an aggregate average: if the prior
            // claims more acceptance than this step can support, believe the
            // proof.
            (Some(p), Some(c)) if c < p => (c, PolicyBasis::MeasuredPriorCappedByEntropy),
            (Some(p), _) => (p, PolicyBasis::MeasuredPrior),
            (None, Some(c)) => (c, PolicyBasis::EntropyCeiling),
            (None, None) => return DraftPolicy::baseline(baseline, PolicyBasis::Baseline),
        };

        let optimal = self.argmax_throughput(planning_acceptance);

        // The safety property: without a measurement the hook may only shrink.
        let proposal_len = if self.config.prior.is_calibrated() {
            optimal
        } else {
            optimal.min(baseline)
        };

        DraftPolicy {
            proposal_len,
            planning_acceptance: Some(planning_acceptance),
            expected_accepted_length: Some(expected_accepted_length(
                planning_acceptance,
                proposal_len,
            )),
            basis,
        }
    }

    /// `argmax_γ Θ(γ)` over the configured range.
    ///
    /// Linear scan. The range is single digits — DeepSpec's released
    /// checkpoints draft seven tokens — so a scan is both faster and more
    /// obviously correct than anything cleverer.
    fn argmax_throughput(&self, acceptance: f32) -> u32 {
        let mut best_len = self.config.min_proposal_len;
        let mut best_throughput = f32::NEG_INFINITY;
        for gamma in self.config.min_proposal_len..=self.config.max_proposal_len {
            let throughput = self.config.cost.throughput(acceptance, gamma);
            if throughput > best_throughput {
                best_throughput = throughput;
                best_len = gamma;
            }
        }
        best_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dspark_bridge::provenance::Provenance;

    fn signal(entropy_nats: f32, eat: f32) -> EntropySignal {
        EntropySignal {
            token_entropy: entropy_nats,
            eat,
            eat_ema: eat,
            eat_ema_variance: 0.0,
        }
    }

    fn measured_provenance() -> Provenance {
        Provenance::Measured {
            harness: "DeepSpec@0000000".into(),
            draft_checkpoint: "deepseek-ai/dspark_qwen3_4b_block7".into(),
            target_model: "Qwen/Qwen3-4B".into(),
            thinking_mode: true,
            recorded_on: "2026-08-07".into(),
        }
    }

    fn hook_with(config: PhaseConditioningConfig) -> PhaseConditioningHook {
        PhaseConditioningHook::new(config).expect("valid config")
    }

    // -- Phase mapping ---------------------------------------------------

    #[test]
    fn think_phase_maps_from_the_router_state_machine() {
        assert_eq!(
            SpeculationPhase::from(&ThinkPhase::Prefill),
            SpeculationPhase::Prefill,
        );
        assert_eq!(
            SpeculationPhase::from(&ThinkPhase::ThinkDecode {
                tokens_so_far: 12,
                eat_ema: 0.5,
                eat_ema_sq: 0.25,
                rpdi_local: 0.0,
                rpdi_global: 0.0,
                force_in_progress: false,
            }),
            SpeculationPhase::Think,
        );
        assert_eq!(
            SpeculationPhase::from(&ThinkPhase::OutputDecode {
                think_tokens_used: 12,
            }),
            SpeculationPhase::Output,
        );
        assert_eq!(
            SpeculationPhase::from(&ThinkPhase::Complete),
            SpeculationPhase::Complete,
        );
    }

    #[test]
    fn only_decode_phases_run_speculation() {
        assert!(SpeculationPhase::Think.is_decoding());
        assert!(SpeculationPhase::Output.is_decoding());
        assert!(!SpeculationPhase::Prefill.is_decoding());
        assert!(!SpeculationPhase::Complete.is_decoding());
    }

    // -- The safety property ---------------------------------------------

    /// The invariant the module is built around. Swept over the full entropy
    /// range and both decoding phases.
    #[test]
    fn uncalibrated_hook_never_drafts_deeper_than_baseline() {
        let config = PhaseConditioningConfig {
            baseline_proposal_len: 4,
            max_proposal_len: 16,
            ..PhaseConditioningConfig::default()
        };
        let hook = hook_with(config);

        for phase in [SpeculationPhase::Think, SpeculationPhase::Output] {
            assert!(hook.policy_for(phase, None).proposal_len <= 4);
            for step in 0..=120 {
                let entropy = step as f32 * 0.1;
                for eat in [0.0_f32, 0.3, 0.9] {
                    let policy = hook.policy_for(phase, Some(&signal(entropy, eat)));
                    assert!(
                        policy.proposal_len <= 4,
                        "H={entropy}, eat={eat}, phase={phase:?} → γ={}",
                        policy.proposal_len,
                    );
                    assert!(!policy.basis.is_measured());
                }
            }
        }
    }

    #[test]
    fn uncalibrated_hook_with_no_signal_is_a_strict_no_op() {
        let hook = hook_with(PhaseConditioningConfig::default());
        let policy = hook.policy_for(SpeculationPhase::Think, None);
        assert_eq!(policy.proposal_len, 7);
        assert_eq!(policy.basis, PolicyBasis::Baseline);
        assert!(policy.planning_acceptance.is_none());
        assert!(policy.expected_accepted_length.is_none());
    }

    #[test]
    fn disabling_the_entropy_ceiling_makes_the_hook_inert_even_with_signals() {
        let hook = hook_with(PhaseConditioningConfig {
            use_entropy_ceiling: false,
            ..PhaseConditioningConfig::default()
        });
        let policy = hook.policy_for(SpeculationPhase::Think, Some(&signal(9.0, 0.01)));
        assert_eq!(policy.proposal_len, 7);
        assert_eq!(policy.basis, PolicyBasis::Baseline);
    }

    #[test]
    fn non_decoding_phases_return_the_baseline_untouched() {
        let hook = hook_with(PhaseConditioningConfig::default());
        for phase in [SpeculationPhase::Prefill, SpeculationPhase::Complete] {
            let policy = hook.policy_for(phase, Some(&signal(9.0, 0.01)));
            assert_eq!(policy.proposal_len, 7);
            assert_eq!(policy.basis, PolicyBasis::NotDecoding);
        }
    }

    // -- Entropy-driven shrinking ----------------------------------------

    #[test]
    fn high_entropy_shrinks_the_draft_budget() {
        let hook = hook_with(PhaseConditioningConfig::default());
        // Near-uniform over Qwen3's vocabulary: acceptance is provably tiny,
        // so drafting seven tokens cannot pay.
        let policy = hook.policy_for(SpeculationPhase::Think, Some(&signal(11.5, 0.0)));
        assert_eq!(policy.basis, PolicyBasis::EntropyCeiling);
        assert!(policy.proposal_len < 7, "γ={}", policy.proposal_len);
        assert!(
            policy.planning_acceptance.unwrap() < 0.1,
            "a={:?}",
            policy.planning_acceptance,
        );
    }

    #[test]
    fn low_entropy_leaves_the_baseline_alone() {
        let hook = hook_with(PhaseConditioningConfig::default());
        // A near-deterministic step: the ceiling is ~1, so the optimiser wants
        // the deepest draft available and the baseline clamp binds.
        let policy = hook.policy_for(SpeculationPhase::Output, Some(&signal(0.001, 0.0)));
        assert_eq!(policy.proposal_len, 7);
        assert!(policy.planning_acceptance.unwrap() > 0.99);
    }

    #[test]
    fn proposal_length_is_monotone_non_increasing_in_entropy() {
        let hook = hook_with(PhaseConditioningConfig {
            baseline_proposal_len: 16,
            max_proposal_len: 16,
            ..PhaseConditioningConfig::default()
        });
        let mut previous = u32::MAX;
        for step in 0..=100 {
            let entropy = step as f32 * 0.12;
            let policy = hook.policy_for(SpeculationPhase::Think, Some(&signal(entropy, 0.0)));
            assert!(
                policy.proposal_len <= previous,
                "γ rose from {previous} to {} at H={entropy}",
                policy.proposal_len,
            );
            previous = policy.proposal_len;
        }
    }

    #[test]
    fn proposal_length_always_respects_the_configured_bounds() {
        let hook = hook_with(PhaseConditioningConfig {
            min_proposal_len: 2,
            baseline_proposal_len: 5,
            max_proposal_len: 9,
            ..PhaseConditioningConfig::default()
        });
        for step in 0..=200 {
            let entropy = step as f32 * 0.06;
            let policy = hook.policy_for(SpeculationPhase::Think, Some(&signal(entropy, 0.2)));
            assert!(
                (2..=5).contains(&policy.proposal_len),
                "γ={}",
                policy.proposal_len
            );
        }
    }

    // -- Calibrated behaviour --------------------------------------------

    #[test]
    fn a_measured_prior_unlocks_upward_adjustment() {
        let uncalibrated = hook_with(PhaseConditioningConfig {
            baseline_proposal_len: 2,
            max_proposal_len: 12,
            ..PhaseConditioningConfig::default()
        });
        assert_eq!(
            uncalibrated
                .policy_for(SpeculationPhase::Output, Some(&signal(0.01, 0.0)))
                .proposal_len,
            2,
            "uncalibrated hook must not exceed baseline",
        );

        let calibrated = hook_with(PhaseConditioningConfig {
            baseline_proposal_len: 2,
            max_proposal_len: 12,
            prior: AcceptancePrior::measured(0.55, 0.95, measured_provenance()).unwrap(),
            ..PhaseConditioningConfig::default()
        });
        let policy = calibrated.policy_for(SpeculationPhase::Output, Some(&signal(0.01, 0.0)));
        assert!(
            policy.proposal_len > 2,
            "calibrated hook should exceed baseline at high measured acceptance, got {}",
            policy.proposal_len,
        );
        assert!(policy.basis.is_measured());
    }

    /// The behaviour the hypothesis predicts, exercised against a *synthetic*
    /// prior: a lower think-phase acceptance rate yields a shallower think
    /// draft than output draft. Nothing here asserts the hypothesis is true.
    #[test]
    fn a_lower_think_rate_produces_a_shallower_think_draft() {
        let hook = hook_with(PhaseConditioningConfig {
            baseline_proposal_len: 7,
            max_proposal_len: 12,
            prior: AcceptancePrior::measured(0.45, 0.92, measured_provenance()).unwrap(),
            ..PhaseConditioningConfig::default()
        });
        let think = hook.policy_for(SpeculationPhase::Think, None);
        let output = hook.policy_for(SpeculationPhase::Output, None);
        assert!(
            think.proposal_len < output.proposal_len,
            "think γ={} not shallower than output γ={}",
            think.proposal_len,
            output.proposal_len,
        );
    }

    #[test]
    fn entropy_caps_an_overconfident_measured_prior() {
        let hook = hook_with(PhaseConditioningConfig {
            prior: AcceptancePrior::measured(0.9, 0.9, measured_provenance()).unwrap(),
            ..PhaseConditioningConfig::default()
        });
        // A high-entropy step cannot support a 0.9 acceptance rate.
        let policy = hook.policy_for(SpeculationPhase::Think, Some(&signal(11.0, 0.0)));
        assert_eq!(policy.basis, PolicyBasis::MeasuredPriorCappedByEntropy);
        assert!(policy.planning_acceptance.unwrap() < 0.9);
    }

    #[test]
    fn phase_gap_reports_the_hypothesis_direction() {
        let prior = AcceptancePrior::measured(0.45, 0.92, measured_provenance()).unwrap();
        assert!(prior.phase_gap().unwrap() < 0.0);
        assert!(AcceptancePrior::Uncalibrated.phase_gap().is_none());
        assert!(
            AcceptancePrior::Uncalibrated
                .for_phase(SpeculationPhase::Think)
                .is_none()
        );
    }

    // -- Guardrails ------------------------------------------------------

    #[test]
    fn synthetic_provenance_cannot_build_a_measured_prior() {
        let err = AcceptancePrior::measured(0.5, 0.9, Provenance::synthetic("fixture"))
            .expect_err("synthetic provenance must be rejected");
        assert!(matches!(err, crate::Error::SyntheticProvenance { .. }));
    }

    #[test]
    fn out_of_range_rates_are_rejected() {
        for (think, output) in [(-0.1_f32, 0.5_f32), (0.5, 1.5), (f32::NAN, 0.5)] {
            assert!(AcceptancePrior::measured(think, output, measured_provenance()).is_err());
        }
    }

    #[test]
    fn invalid_configs_are_rejected_at_construction() {
        let cases = [
            PhaseConditioningConfig {
                min_proposal_len: 8,
                max_proposal_len: 4,
                ..PhaseConditioningConfig::default()
            },
            PhaseConditioningConfig {
                baseline_proposal_len: 99,
                ..PhaseConditioningConfig::default()
            },
            PhaseConditioningConfig {
                vocab_size: 1,
                ..PhaseConditioningConfig::default()
            },
            PhaseConditioningConfig {
                cost: DraftCostModel {
                    verify_fixed_us: 0.0,
                    ..DraftCostModel::default()
                },
                ..PhaseConditioningConfig::default()
            },
            PhaseConditioningConfig {
                cost: DraftCostModel {
                    draft_token_us: -1.0,
                    ..DraftCostModel::default()
                },
                ..PhaseConditioningConfig::default()
            },
        ];
        for case in cases {
            assert!(
                PhaseConditioningHook::new(case.clone()).is_err(),
                "config should have been rejected: {case:?}",
            );
        }
    }

    #[test]
    fn default_config_is_valid_and_uncalibrated() {
        let hook = hook_with(PhaseConditioningConfig::default());
        assert!(!hook.is_calibrated());
        assert_eq!(hook.config().baseline_proposal_len, 7);
    }

    // -- Cost model ------------------------------------------------------

    #[test]
    fn cycle_cost_grows_with_depth() {
        let cost = DraftCostModel::default();
        assert!(cost.cycle_cost_us(1) < cost.cycle_cost_us(7));
        assert!((cost.cycle_cost_us(0) - cost.verify_fixed_us).abs() < 1e-3);
    }

    #[test]
    fn throughput_peaks_deeper_as_acceptance_rises() {
        let hook = hook_with(PhaseConditioningConfig {
            max_proposal_len: 32,
            ..PhaseConditioningConfig::default()
        });
        let low = hook.argmax_throughput(0.3);
        let high = hook.argmax_throughput(0.95);
        assert!(low <= high, "γ*(0.3)={low} exceeded γ*(0.95)={high}");
    }
}
