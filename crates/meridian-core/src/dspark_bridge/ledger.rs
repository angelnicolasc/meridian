//! Phase-segmented acceptance ledger — the consumer side of Phase 1.
//!
//! DeepSpec's evaluation harness already produces, per verification step, the
//! three quantities this ledger needs (`acceptance_lengths`,
//! `proposal_lengths`, `accepted_draft_lengths`). What it does not do is
//! preserve *where in the response* each step landed: the per-step lists are
//! summed into scalars before anything is written out. The gap analysis in
//! [`docs/src/notes/deepspec-harness-instrumentation.md`][gap] shows the
//! segmentation is reconstructible from data the harness already returns.
//!
//! This module is the other half of that: given phase-labelled observations,
//! it computes the Section 5 statistic and renders a report that cannot be
//! mistaken for a measurement when it is not one.
//!
//! # What is measured
//!
//! Two quantities per phase, both matching DeepSpec's own definitions so the
//! numbers are directly comparable to its published tables:
//!
//! - **mean accepted length** — `acceptance_length_sum / proposal_count`,
//!   including the target-generated bonus token, the metric DSpark reports;
//! - **token acceptance rate** — `acceptance_length_sum /
//!   (proposal_length_sum + proposal_count)`, DeepSpec's `verify_rate`.
//!
//! # The straddle problem
//!
//! A verification step commits several tokens at once, so a step can span the
//! `</think>` boundary — part of its committed span is reasoning, part is
//! output. Such a step belongs to neither phase cleanly.
//! [`StraddlePolicy`] makes the choice explicit and the count of straddling
//! steps is always reported, because a high straddle rate is itself evidence
//! that the segmentation is too coarse to trust.
//!
//! [gap]: https://github.com/angelnicolasc/meridian/blob/main/docs/src/notes/deepspec-harness-instrumentation.md

use std::fmt;

use crate::dspark_bridge::hook::{AcceptancePrior, SpeculationPhase};
use crate::dspark_bridge::provenance::{MeasuredRun, Provenance};
use crate::dspark_bridge::stats::{Moments, WelchResult, welch_t_test};
use crate::error::{Error, Result};
use crate::metrics::names;

// ---------------------------------------------------------------------------
// Observations
// ---------------------------------------------------------------------------

/// One verification step, as the harness reports it.
///
/// Field names mirror DeepSpec's `generate_decoding_sample` return payload so
/// the mapping from harness output to ledger input needs no translation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptanceObservation {
    /// Which side of the `</think>` boundary the committed span landed on.
    pub phase: SpeculationPhase,
    /// Tokens committed by this step, including the bonus token
    /// (`acceptance_lengths[i]` in the harness).
    pub accepted_length: u32,
    /// Tokens the drafter proposed for this step
    /// (`proposal_lengths[i]` in the harness).
    pub proposal_length: u32,
    /// `true` when the committed span contains the phase boundary itself.
    pub straddles_boundary: bool,
}

impl AcceptanceObservation {
    /// A step wholly inside one phase.
    #[must_use]
    pub const fn new(phase: SpeculationPhase, accepted_length: u32, proposal_length: u32) -> Self {
        Self {
            phase,
            accepted_length,
            proposal_length,
            straddles_boundary: false,
        }
    }

    /// A step whose committed span crosses the `</think>` boundary.
    #[must_use]
    pub const fn straddling(accepted_length: u32, proposal_length: u32) -> Self {
        Self {
            // Nominal label; [`StraddlePolicy`] decides what actually happens.
            phase: SpeculationPhase::Think,
            accepted_length,
            proposal_length,
            straddles_boundary: true,
        }
    }
}

/// How to attribute a verification step that spans the phase boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StraddlePolicy {
    /// Drop the step from both phases. The default: a straddling step is
    /// contaminated by construction, and discarding it biases neither arm.
    #[default]
    Exclude,
    /// Count it as a think-phase step.
    AttributeToThink,
    /// Count it as an output-phase step.
    AttributeToOutput,
}

impl StraddlePolicy {
    /// Stable telemetry label.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Exclude => "exclude",
            Self::AttributeToThink => "attribute_to_think",
            Self::AttributeToOutput => "attribute_to_output",
        }
    }
}

// ---------------------------------------------------------------------------
// Per-phase accumulation
// ---------------------------------------------------------------------------

/// Accumulated statistics for one phase.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PhaseStats {
    accepted_length: Moments,
    accepted_length_sum: u64,
    proposal_length_sum: u64,
}

impl PhaseStats {
    /// An empty accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            accepted_length: Moments::new(),
            accepted_length_sum: 0,
            proposal_length_sum: 0,
        }
    }

    fn push(&mut self, observation: &AcceptanceObservation) {
        self.accepted_length
            .push(f64::from(observation.accepted_length));
        self.accepted_length_sum += u64::from(observation.accepted_length);
        self.proposal_length_sum += u64::from(observation.proposal_length);
    }

    /// Merge another accumulator, e.g. from a different rank or shard.
    pub fn merge(&mut self, other: &Self) {
        self.accepted_length.merge(&other.accepted_length);
        self.accepted_length_sum += other.accepted_length_sum;
        self.proposal_length_sum += other.proposal_length_sum;
    }

    /// Number of verification steps recorded.
    #[must_use]
    pub const fn step_count(&self) -> u64 {
        self.accepted_length.count()
    }

    /// Mean accepted length per verification step, including the bonus token.
    /// `None` when no steps were recorded.
    #[must_use]
    pub fn mean_accepted_length(&self) -> Option<f64> {
        (self.step_count() > 0).then(|| self.accepted_length.mean())
    }

    /// Standard deviation of accepted length. `None` below two steps.
    #[must_use]
    pub fn std_dev_accepted_length(&self) -> Option<f64> {
        self.accepted_length.std_dev()
    }

    /// DeepSpec's `verify_rate`: accepted tokens as a fraction of verified
    /// positions, counting the bonus slot.
    #[must_use]
    pub fn token_acceptance_rate(&self) -> Option<f64> {
        let denominator = self.proposal_length_sum + self.step_count();
        (denominator > 0).then(|| self.accepted_length_sum as f64 / denominator as f64)
    }

    /// Underlying moments, for callers running their own tests.
    #[must_use]
    pub const fn moments(&self) -> &Moments {
        &self.accepted_length
    }
}

// ---------------------------------------------------------------------------
// The ledger
// ---------------------------------------------------------------------------

/// Phase-segmented accumulator over verification steps.
///
/// # Examples
///
/// ```
/// use meridian_core::dspark_bridge::{
///     AcceptanceLedger, AcceptanceObservation, Provenance, SpeculationPhase,
/// };
///
/// let mut ledger = AcceptanceLedger::new(Provenance::synthetic("doc-example"));
/// ledger.record(&AcceptanceObservation::new(SpeculationPhase::Think, 3, 7));
/// ledger.record(&AcceptanceObservation::new(SpeculationPhase::Output, 6, 7));
///
/// let report = ledger.report();
/// // Synthetic data can be inspected, but never promoted to a claim.
/// assert!(report.to_string().contains("SYNTHETIC"));
/// assert!(report.into_measured_claim().is_err());
/// ```
#[derive(Debug, Clone)]
pub struct AcceptanceLedger {
    provenance: Provenance,
    straddle_policy: StraddlePolicy,
    think: PhaseStats,
    output: PhaseStats,
    straddling_steps: u64,
    discarded_steps: u64,
}

impl AcceptanceLedger {
    /// A ledger tagged with where its data comes from.
    #[must_use]
    pub fn new(provenance: Provenance) -> Self {
        Self {
            provenance,
            straddle_policy: StraddlePolicy::default(),
            think: PhaseStats::new(),
            output: PhaseStats::new(),
            straddling_steps: 0,
            discarded_steps: 0,
        }
    }

    /// Override the straddling-step attribution policy.
    #[must_use]
    pub const fn with_straddle_policy(mut self, policy: StraddlePolicy) -> Self {
        self.straddle_policy = policy;
        self
    }

    /// Fold one verification step in.
    pub fn record(&mut self, observation: &AcceptanceObservation) {
        let phase = if observation.straddles_boundary {
            self.straddling_steps += 1;
            match self.straddle_policy {
                StraddlePolicy::Exclude => {
                    self.discarded_steps += 1;
                    metrics::counter!(names::SPECULATION_STRADDLING_STEPS).increment(1);
                    return;
                }
                StraddlePolicy::AttributeToThink => SpeculationPhase::Think,
                StraddlePolicy::AttributeToOutput => SpeculationPhase::Output,
            }
        } else {
            observation.phase
        };

        match phase {
            SpeculationPhase::Think => self.think.push(observation),
            SpeculationPhase::Output => self.output.push(observation),
            // A verification step outside a decode phase is a caller bug, but
            // dropping it silently would corrupt the comparison invisibly.
            SpeculationPhase::Prefill | SpeculationPhase::Complete => {
                self.discarded_steps += 1;
                return;
            }
        }

        if observation.straddles_boundary {
            metrics::counter!(names::SPECULATION_STRADDLING_STEPS).increment(1);
        }
        metrics::histogram!(
            names::SPECULATION_ACCEPTED_LENGTH,
            "phase" => phase.as_label(),
        )
        .record(f64::from(observation.accepted_length));
    }

    /// Fold in a batch of steps.
    pub fn record_all(&mut self, observations: &[AcceptanceObservation]) {
        for observation in observations {
            self.record(observation);
        }
    }

    /// Merge another ledger — the multi-rank aggregation path.
    ///
    /// Both the provenance and the straddle policy must match. Combining a
    /// synthetic trace into a measured one is exactly the mistake this module
    /// exists to prevent; combining arms built under different straddle
    /// policies is subtler and worse, because it corrupts the headline
    /// statistic without any visible symptom.
    ///
    /// # Errors
    ///
    /// [`Error::ProvenanceMismatch`] when the two ledgers disagree about where
    /// their data came from; [`Error::StraddlePolicyMismatch`] when they
    /// attributed boundary-spanning steps differently.
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        if self.provenance != other.provenance {
            return Err(Error::ProvenanceMismatch {
                left: self.provenance.as_label(),
                right: other.provenance.as_label(),
            });
        }
        if self.straddle_policy != other.straddle_policy {
            return Err(Error::StraddlePolicyMismatch {
                left: self.straddle_policy.as_label(),
                right: other.straddle_policy.as_label(),
            });
        }
        self.think.merge(&other.think);
        self.output.merge(&other.output);
        self.straddling_steps += other.straddling_steps;
        self.discarded_steps += other.discarded_steps;
        Ok(())
    }

    /// Think-phase statistics.
    #[must_use]
    pub const fn think(&self) -> &PhaseStats {
        &self.think
    }

    /// Output-phase statistics.
    #[must_use]
    pub const fn output(&self) -> &PhaseStats {
        &self.output
    }

    /// Where this ledger's data came from.
    #[must_use]
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Number of steps that spanned the phase boundary.
    #[must_use]
    pub const fn straddling_steps(&self) -> u64 {
        self.straddling_steps
    }

    /// Fraction of recorded steps that spanned the boundary. A large value
    /// means the phase attribution is coarse relative to the draft block size
    /// and the comparison should be treated with suspicion.
    #[must_use]
    pub fn straddle_rate(&self) -> f64 {
        let total = self.think.step_count() + self.output.step_count() + self.discarded_steps;
        if total == 0 {
            return 0.0;
        }
        self.straddling_steps as f64 / total as f64
    }

    /// Compute the phase-gap report.
    #[must_use]
    pub fn report(&self) -> PhaseGapReport {
        PhaseGapReport {
            provenance: self.provenance.clone(),
            straddle_policy: self.straddle_policy,
            think: self.think,
            output: self.output,
            straddling_steps: self.straddling_steps,
            straddle_rate: self.straddle_rate(),
            welch: welch_t_test(self.think.moments(), self.output.moments()),
        }
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// The verdict on the Section 5 hypothesis.
///
/// Section 5 asks for two things at once: that think-phase accepted length is
/// *lower*, and that the gap *exceeds the variation already observed between
/// task domains*. Both are encoded here so Phase 1 requires no further
/// decisions about what counts as a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypothesisVerdict {
    /// Think-phase accepted length is lower, the interval excludes zero, and
    /// the gap exceeds the supplied between-domain baseline.
    Supported,
    /// The gap is in the predicted direction and statistically resolvable, but
    /// smaller than variation already seen between task domains — so it is not
    /// evidence of a *phase* effect specifically.
    WithinDomainVariation,
    /// The interval includes zero: no resolvable gap. A legitimate, publishable
    /// outcome under this project's own rules.
    NoResolvableGap,
    /// A resolvable gap in the *opposite* direction to the prediction.
    OppositeDirection,
    /// Not enough data on one or both arms to compute the statistic.
    InsufficientData,
}

impl HypothesisVerdict {
    /// Stable telemetry label.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::WithinDomainVariation => "within_domain_variation",
            Self::NoResolvableGap => "no_resolvable_gap",
            Self::OppositeDirection => "opposite_direction",
            Self::InsufficientData => "insufficient_data",
        }
    }
}

/// A phase-segmented comparison, ready to render or publish.
#[derive(Debug, Clone)]
pub struct PhaseGapReport {
    /// Where the underlying data came from. Rendered on every line of output.
    pub provenance: Provenance,
    /// Straddling-step policy in force when the ledger was accumulated.
    pub straddle_policy: StraddlePolicy,
    /// Think-phase statistics.
    pub think: PhaseStats,
    /// Output-phase statistics.
    pub output: PhaseStats,
    /// Steps that spanned the phase boundary.
    pub straddling_steps: u64,
    /// Straddling steps as a fraction of all recorded steps.
    pub straddle_rate: f64,
    /// Welch test of think-minus-output mean accepted length. `None` when
    /// either arm is too small or degenerate.
    pub welch: Option<WelchResult>,
}

impl PhaseGapReport {
    /// Evaluate the Section 5 hypothesis.
    ///
    /// `between_domain_gap` is the magnitude of accepted-length variation
    /// already observed *between task domains* at a fixed confidence
    /// threshold — the bar Section 5 sets for the phase effect to be
    /// interesting rather than merely present. Pass `0.0` to test only for a
    /// resolvable directional gap.
    #[must_use]
    pub fn verdict(&self, between_domain_gap: f64) -> HypothesisVerdict {
        let Some(welch) = self.welch else {
            return HypothesisVerdict::InsufficientData;
        };
        if !welch.ci95_excludes_zero() {
            return HypothesisVerdict::NoResolvableGap;
        }
        if welch.mean_difference > 0.0 {
            return HypothesisVerdict::OppositeDirection;
        }
        if welch.mean_difference.abs() <= between_domain_gap.abs() {
            return HypothesisVerdict::WithinDomainVariation;
        }
        HypothesisVerdict::Supported
    }

    /// Derive a calibrated [`AcceptancePrior`] for
    /// [`super::hook::PhaseConditioningHook`] from this report.
    ///
    /// This is the loop closing: Phase 1 measures, the hook consumes. It fails
    /// on synthetic data, so the only way to calibrate a production hook is
    /// with real numbers.
    ///
    /// # Errors
    ///
    /// [`Error::SyntheticProvenance`] if the report is not from a measured
    /// run; [`Error::InsufficientObservations`] if either arm is empty.
    pub fn to_acceptance_prior(&self) -> Result<AcceptancePrior> {
        let think = self
            .think
            .token_acceptance_rate()
            .ok_or(Error::InsufficientObservations { phase: "think" })?;
        let output = self
            .output
            .token_acceptance_rate()
            .ok_or(Error::InsufficientObservations { phase: "output" })?;
        AcceptancePrior::measured(think as f32, output as f32, self.provenance.clone())
    }

    /// Promote this report to a publishable claim.
    ///
    /// The single supported path from a statistic to something that may appear
    /// in a paper, a post, or a README.
    ///
    /// # Errors
    ///
    /// [`Error::SyntheticProvenance`] for anything not measured on hardware;
    /// [`Error::InsufficientObservations`] when the statistic is undefined.
    pub fn into_measured_claim(self) -> Result<MeasuredPhaseGap> {
        // Provenance is checked first on purpose. Both failures are real, but
        // only one is fixable: thin data can be cured by running longer, while
        // synthetic data can never become a measurement. Reporting the curable
        // problem first would send a caller off to collect more of exactly the
        // wrong thing.
        let run = self.provenance.into_measured()?;
        let welch = self.welch.ok_or(Error::InsufficientObservations {
            phase: "think|output",
        })?;
        Ok(MeasuredPhaseGap {
            run,
            think_mean_accepted_length: self.think.mean_accepted_length().unwrap_or_default(),
            output_mean_accepted_length: self.output.mean_accepted_length().unwrap_or_default(),
            welch,
            straddle_rate: self.straddle_rate,
        })
    }
}

impl fmt::Display for PhaseGapReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.provenance.banner())?;
        writeln!(
            f,
            "phase-segmented draft acceptance  (straddle policy: {}, straddle rate: {:.2}%)",
            self.straddle_policy.as_label(),
            self.straddle_rate * 100.0,
        )?;
        writeln!(
            f,
            "  {:<8} {:>10} {:>18} {:>10} {:>16}",
            "phase", "steps", "mean accept len", "sd", "token accept rate",
        )?;
        for (label, stats) in [("think", &self.think), ("output", &self.output)] {
            writeln!(
                f,
                "  {:<8} {:>10} {:>18} {:>10} {:>16}",
                label,
                stats.step_count(),
                stats
                    .mean_accepted_length()
                    .map_or_else(|| "-".to_owned(), |v| format!("{v:.4}")),
                stats
                    .std_dev_accepted_length()
                    .map_or_else(|| "-".to_owned(), |v| format!("{v:.4}")),
                stats
                    .token_acceptance_rate()
                    .map_or_else(|| "-".to_owned(), |v| format!("{v:.4}")),
            )?;
        }
        match self.welch {
            Some(welch) => writeln!(
                f,
                "  think - output = {:+.4}  95% CI [{:+.4}, {:+.4}]  t = {:.3}  df = {:.1}  p = {:.3e}  d = {:+.3}",
                welch.mean_difference,
                welch.ci95_lower(),
                welch.ci95_upper(),
                welch.t_statistic,
                welch.degrees_of_freedom,
                welch.p_value,
                welch.cohens_d,
            ),
            None => writeln!(f, "  think - output = (insufficient data)"),
        }
    }
}

/// A phase gap that has been proven to come from a real run.
///
/// Only obtainable via [`PhaseGapReport::into_measured_claim`].
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredPhaseGap {
    /// Description of the run that produced the numbers.
    pub run: MeasuredRun,
    /// Mean accepted length during think-phase verification steps.
    pub think_mean_accepted_length: f64,
    /// Mean accepted length during output-phase verification steps.
    pub output_mean_accepted_length: f64,
    /// The Welch comparison.
    pub welch: WelchResult,
    /// Fraction of steps that spanned the boundary.
    pub straddle_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured_provenance() -> Provenance {
        Provenance::Measured {
            harness: "DeepSpec@0000000".into(),
            draft_checkpoint: "deepseek-ai/dspark_qwen3_4b_block7".into(),
            target_model: "Qwen/Qwen3-4B".into(),
            thinking_mode: true,
            recorded_on: "2026-08-07".into(),
        }
    }

    /// Deterministic alternating pattern around a mean — enough spread for the
    /// variance to be defined without needing a random source.
    fn fill(
        ledger: &mut AcceptanceLedger,
        phase: SpeculationPhase,
        steps: usize,
        base: u32,
        proposal: u32,
    ) {
        for i in 0..steps {
            let accepted = base + u32::from(i % 2 == 0);
            ledger.record(&AcceptanceObservation::new(phase, accepted, proposal));
        }
    }

    // -- Accumulation ----------------------------------------------------

    #[test]
    fn records_are_routed_to_the_right_phase() {
        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("routing"));
        fill(&mut ledger, SpeculationPhase::Think, 10, 2, 7);
        fill(&mut ledger, SpeculationPhase::Output, 6, 5, 7);

        assert_eq!(ledger.think().step_count(), 10);
        assert_eq!(ledger.output().step_count(), 6);
        assert!(ledger.think().mean_accepted_length().unwrap() < 3.0);
        assert!(ledger.output().mean_accepted_length().unwrap() > 5.0);
    }

    #[test]
    fn token_acceptance_rate_matches_deepspec_verify_rate() {
        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("verify-rate"));
        // Three steps: accepted 4, 4, 4 out of proposal 7 each.
        for _ in 0..3 {
            ledger.record(&AcceptanceObservation::new(SpeculationPhase::Output, 4, 7));
        }
        // DeepSpec: acceptance_length_sum / (proposal_length_sum + proposal_count)
        //         = 12 / (21 + 3) = 0.5
        let rate = ledger.output().token_acceptance_rate().unwrap();
        assert!((rate - 0.5).abs() < 1e-12, "rate={rate}");
    }

    #[test]
    fn empty_arms_report_none_rather_than_zero() {
        let ledger = AcceptanceLedger::new(Provenance::synthetic("empty"));
        assert!(ledger.think().mean_accepted_length().is_none());
        assert!(ledger.think().token_acceptance_rate().is_none());
        assert!(ledger.report().welch.is_none());
        assert_eq!(
            ledger.report().verdict(0.0),
            HypothesisVerdict::InsufficientData,
        );
    }

    #[test]
    fn steps_outside_a_decode_phase_are_discarded_not_misfiled() {
        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("out-of-phase"));
        ledger.record(&AcceptanceObservation::new(SpeculationPhase::Prefill, 3, 7));
        ledger.record(&AcceptanceObservation::new(
            SpeculationPhase::Complete,
            3,
            7,
        ));
        assert_eq!(ledger.think().step_count(), 0);
        assert_eq!(ledger.output().step_count(), 0);
    }

    // -- Straddling ------------------------------------------------------

    #[test]
    fn exclude_is_the_default_straddle_policy() {
        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("straddle-default"));
        fill(&mut ledger, SpeculationPhase::Think, 4, 2, 7);
        ledger.record(&AcceptanceObservation::straddling(5, 7));

        assert_eq!(ledger.think().step_count(), 4);
        assert_eq!(ledger.output().step_count(), 0);
        assert_eq!(ledger.straddling_steps(), 1);
        assert!((ledger.straddle_rate() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn straddle_attribution_policies_route_to_the_named_phase() {
        for (policy, think_steps, output_steps) in [
            (StraddlePolicy::AttributeToThink, 1, 0),
            (StraddlePolicy::AttributeToOutput, 0, 1),
            (StraddlePolicy::Exclude, 0, 0),
        ] {
            let mut ledger = AcceptanceLedger::new(Provenance::synthetic("straddle-policy"))
                .with_straddle_policy(policy);
            ledger.record(&AcceptanceObservation::straddling(4, 7));
            assert_eq!(ledger.think().step_count(), think_steps, "{policy:?}");
            assert_eq!(ledger.output().step_count(), output_steps, "{policy:?}");
            assert_eq!(ledger.straddling_steps(), 1, "{policy:?}");
        }
    }

    #[test]
    fn straddle_rate_is_zero_for_an_empty_ledger() {
        let ledger = AcceptanceLedger::new(Provenance::synthetic("empty"));
        assert!((ledger.straddle_rate() - 0.0).abs() < 1e-12);
    }

    // -- Merging ---------------------------------------------------------

    /// Record an explicit sequence, so a merge can be compared against the
    /// concatenation of exactly the same observations.
    fn record_values(
        ledger: &mut AcceptanceLedger,
        phase: SpeculationPhase,
        accepted: &[u32],
        proposal: u32,
    ) {
        for &value in accepted {
            ledger.record(&AcceptanceObservation::new(phase, value, proposal));
        }
    }

    #[test]
    fn merging_matching_provenance_is_equivalent_to_recording_both() {
        let first = [3_u32, 1, 4, 1, 5];
        let second = [9_u32, 2, 6, 5, 3, 5, 8];

        let mut left = AcceptanceLedger::new(Provenance::synthetic("merge"));
        record_values(&mut left, SpeculationPhase::Think, &first, 7);

        let mut right = AcceptanceLedger::new(Provenance::synthetic("merge"));
        record_values(&mut right, SpeculationPhase::Think, &second, 7);

        let mut combined = AcceptanceLedger::new(Provenance::synthetic("merge"));
        let concatenated: Vec<u32> = first.iter().chain(second.iter()).copied().collect();
        record_values(&mut combined, SpeculationPhase::Think, &concatenated, 7);

        left.merge(&right).unwrap();
        assert_eq!(left.think().step_count(), combined.think().step_count());
        assert!(
            (left.think().mean_accepted_length().unwrap()
                - combined.think().mean_accepted_length().unwrap())
            .abs()
                < 1e-12,
        );
        assert!(
            (left.think().std_dev_accepted_length().unwrap()
                - combined.think().std_dev_accepted_length().unwrap())
            .abs()
                < 1e-12,
        );
        assert!(
            (left.think().token_acceptance_rate().unwrap()
                - combined.think().token_acceptance_rate().unwrap())
            .abs()
                < 1e-12,
        );
    }

    #[test]
    fn merging_across_provenance_is_refused() {
        let mut synthetic = AcceptanceLedger::new(Provenance::synthetic("a"));
        let measured = AcceptanceLedger::new(measured_provenance());
        let err = synthetic.merge(&measured).unwrap_err();
        assert!(matches!(err, Error::ProvenanceMismatch { .. }));
    }

    /// Two ranks configured with different straddle policies would produce
    /// arms built under incompatible attribution rules. Silently combining
    /// them would corrupt the headline statistic with no visible symptom.
    #[test]
    fn merging_across_straddle_policies_is_refused() {
        let mut excluding = AcceptanceLedger::new(measured_provenance());
        let attributing = AcceptanceLedger::new(measured_provenance())
            .with_straddle_policy(StraddlePolicy::AttributeToThink);
        let err = excluding.merge(&attributing).unwrap_err();
        assert!(matches!(err, Error::StraddlePolicyMismatch { .. }));
    }

    // -- Verdicts --------------------------------------------------------

    #[test]
    fn a_clear_gap_in_the_predicted_direction_is_supported() {
        let mut ledger = AcceptanceLedger::new(measured_provenance());
        fill(&mut ledger, SpeculationPhase::Think, 500, 2, 7);
        fill(&mut ledger, SpeculationPhase::Output, 500, 5, 7);

        let report = ledger.report();
        assert_eq!(report.verdict(0.0), HypothesisVerdict::Supported);
        let welch = report.welch.unwrap();
        assert!(welch.mean_difference < 0.0);
        assert!(welch.ci95_excludes_zero());
    }

    #[test]
    fn a_gap_smaller_than_between_domain_variation_is_not_a_phase_effect() {
        let mut ledger = AcceptanceLedger::new(measured_provenance());
        fill(&mut ledger, SpeculationPhase::Think, 500, 4, 7);
        fill(&mut ledger, SpeculationPhase::Output, 500, 5, 7);

        let report = ledger.report();
        // The observed gap is about one token; a two-token between-domain
        // baseline swallows it.
        assert_eq!(
            report.verdict(2.0),
            HypothesisVerdict::WithinDomainVariation,
        );
        assert_eq!(report.verdict(0.5), HypothesisVerdict::Supported);
    }

    #[test]
    fn identical_arms_yield_no_resolvable_gap() {
        let mut ledger = AcceptanceLedger::new(measured_provenance());
        fill(&mut ledger, SpeculationPhase::Think, 400, 4, 7);
        fill(&mut ledger, SpeculationPhase::Output, 400, 4, 7);
        assert_eq!(
            ledger.report().verdict(0.0),
            HypothesisVerdict::NoResolvableGap,
        );
    }

    #[test]
    fn a_reversed_gap_is_reported_as_such_not_silently_accepted() {
        let mut ledger = AcceptanceLedger::new(measured_provenance());
        fill(&mut ledger, SpeculationPhase::Think, 400, 6, 7);
        fill(&mut ledger, SpeculationPhase::Output, 400, 3, 7);
        assert_eq!(
            ledger.report().verdict(0.0),
            HypothesisVerdict::OppositeDirection,
        );
    }

    #[test]
    fn one_sided_data_is_insufficient() {
        let mut ledger = AcceptanceLedger::new(measured_provenance());
        fill(&mut ledger, SpeculationPhase::Think, 50, 3, 7);
        assert_eq!(
            ledger.report().verdict(0.0),
            HypothesisVerdict::InsufficientData,
        );
    }

    // -- The publication gate --------------------------------------------

    #[test]
    fn synthetic_reports_cannot_become_claims() {
        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("no-publish"));
        fill(&mut ledger, SpeculationPhase::Think, 100, 2, 7);
        fill(&mut ledger, SpeculationPhase::Output, 100, 5, 7);

        let report = ledger.report();
        // The statistic computes fine — this is not about hiding the number.
        assert!(report.welch.is_some());
        assert_eq!(report.verdict(0.0), HypothesisVerdict::Supported);
        // It just cannot be promoted.
        assert!(matches!(
            report.clone().into_measured_claim().unwrap_err(),
            Error::SyntheticProvenance { .. },
        ));
        assert!(matches!(
            report.to_acceptance_prior().unwrap_err(),
            Error::SyntheticProvenance { .. },
        ));
    }

    #[test]
    fn measured_reports_become_claims_and_carry_the_run_description() {
        let mut ledger = AcceptanceLedger::new(measured_provenance());
        fill(&mut ledger, SpeculationPhase::Think, 200, 2, 7);
        fill(&mut ledger, SpeculationPhase::Output, 200, 5, 7);

        let claim = ledger.report().into_measured_claim().unwrap();
        assert_eq!(claim.run.target_model, "Qwen/Qwen3-4B");
        assert!(claim.run.thinking_mode);
        assert!(claim.think_mean_accepted_length < claim.output_mean_accepted_length);
        assert!(claim.welch.mean_difference < 0.0);
    }

    #[test]
    fn a_measured_report_closes_the_loop_into_a_hook_prior() {
        let mut ledger = AcceptanceLedger::new(measured_provenance());
        fill(&mut ledger, SpeculationPhase::Think, 200, 2, 7);
        fill(&mut ledger, SpeculationPhase::Output, 200, 6, 7);

        let prior = ledger.report().to_acceptance_prior().unwrap();
        assert!(prior.is_calibrated());
        assert!(
            prior.phase_gap().unwrap() < 0.0,
            "prior should carry the measured direction",
        );
    }

    #[test]
    fn measured_report_with_an_empty_arm_refuses_to_produce_a_prior() {
        let mut ledger = AcceptanceLedger::new(measured_provenance());
        fill(&mut ledger, SpeculationPhase::Think, 20, 2, 7);
        assert!(matches!(
            ledger.report().to_acceptance_prior().unwrap_err(),
            Error::InsufficientObservations { phase: "output" },
        ));
    }

    #[test]
    fn insufficient_data_cannot_become_a_claim() {
        let ledger = AcceptanceLedger::new(measured_provenance());
        assert!(matches!(
            ledger.report().into_measured_claim().unwrap_err(),
            Error::InsufficientObservations { .. },
        ));
    }

    /// When a report is both synthetic *and* statistically degenerate, the
    /// caller must be told about the unfixable problem, not the fixable one.
    #[test]
    fn provenance_is_reported_before_thin_data() {
        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("degenerate"));
        // Constant values on both arms: zero variance, so the Welch statistic
        // is undefined.
        for _ in 0..50 {
            ledger.record(&AcceptanceObservation::new(SpeculationPhase::Think, 2, 7));
            ledger.record(&AcceptanceObservation::new(SpeculationPhase::Output, 5, 7));
        }
        assert!(ledger.report().welch.is_none());
        assert!(matches!(
            ledger.report().into_measured_claim().unwrap_err(),
            Error::SyntheticProvenance { .. },
        ));
    }

    // -- Rendering -------------------------------------------------------

    #[test]
    fn synthetic_reports_render_an_unmissable_banner() {
        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("render"));
        fill(&mut ledger, SpeculationPhase::Think, 20, 2, 7);
        fill(&mut ledger, SpeculationPhase::Output, 20, 5, 7);

        let rendered = ledger.report().to_string();
        assert!(rendered.contains("SYNTHETIC"));
        assert!(rendered.contains("NOT A MEASUREMENT"));
        assert!(rendered.contains("think"));
        assert!(rendered.contains("output"));
        assert!(rendered.contains("95% CI"));
    }

    #[test]
    fn measured_reports_render_the_run_description() {
        let mut ledger = AcceptanceLedger::new(measured_provenance());
        fill(&mut ledger, SpeculationPhase::Think, 20, 2, 7);
        fill(&mut ledger, SpeculationPhase::Output, 20, 5, 7);

        let rendered = ledger.report().to_string();
        assert!(rendered.contains("MEASURED"));
        assert!(rendered.contains("thinking-mode ON"));
        assert!(!rendered.contains("SYNTHETIC"));
    }

    #[test]
    fn empty_report_renders_without_panicking() {
        let rendered = AcceptanceLedger::new(Provenance::synthetic("empty"))
            .report()
            .to_string();
        assert!(rendered.contains("insufficient data"));
    }
}
