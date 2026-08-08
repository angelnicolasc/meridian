//! Integration tests for `dspark_bridge` against synthetic traces (**MER-P0.6**).
//!
//! No GPU, no live model, no drafter. Every trace here is generated in-process
//! and tagged [`Provenance::Synthetic`], so nothing this file produces can be
//! promoted to a measurement — a property the tests themselves assert.
//!
//! The fixture families are the four the blueprint calls for:
//!
//! 1. **Clean boundary** — `</think>` lands exactly on a verification-step
//!    edge, so every step belongs to one phase.
//! 2. **Ambiguous boundary** — `</think>` lands *inside* a committed span, so
//!    one step contains both reasoning and output tokens. This is not a
//!    contrived case: DeepSpec's released DSpark checkpoints commit up to
//!    eight tokens per step, so a boundary lands mid-span most of the time.
//! 3. **Degenerate all-think** — the model never emits `</think>` within the
//!    budget. The output arm is empty.
//! 4. **Degenerate all-output** — a non-reasoning request that never emits
//!    `<think>`. The think arm is empty.
//!
//! The replay harness below is deliberately shaped like the Phase 1
//! instrumentation patch described in
//! `docs/src/notes/deepspec-harness-instrumentation.md`: walk the committed
//! spans, attribute each to a phase, emit one observation per verification
//! step. If that patch is written correctly, it produces observations this
//! ledger already knows how to consume.

use meridian_core::dspark_bridge::{
    AcceptanceLedger, AcceptanceObservation, AcceptancePrior, HypothesisVerdict,
    PhaseConditioningConfig, PhaseConditioningHook, PolicyBasis, Provenance, SpeculationPhase,
    StraddlePolicy,
};
use meridian_core::phase_router::{PhaseRouter, PhaseRouterConfig};
use meridian_core::types::{EntropySignal, PhaseEvent};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Trace construction
// ---------------------------------------------------------------------------

const THINK_START: u32 = 1;
const THINK_END: u32 = 2;
const EOS: u32 = 3;
const FILLER: u32 = 100;

/// Vocabulary size used throughout — Qwen3's, so the entropy bounds land in a
/// realistic regime rather than a toy one.
const VOCAB: u32 = 151_936;

/// One decoded token plus the probe sample that accompanied it.
#[derive(Debug, Clone, Copy)]
struct TokenSample {
    id: u32,
    signal: EntropySignal,
}

fn sample(id: u32, entropy_nats: f32, eat: f32) -> TokenSample {
    TokenSample {
        id,
        signal: EntropySignal {
            token_entropy: entropy_nats,
            eat,
            eat_ema: eat,
            eat_ema_variance: 0.0,
        },
    }
}

/// Build a reasoning trace: `<think>` · N think tokens · `</think>` · M output
/// tokens · EOS.
///
/// Think tokens carry higher entropy than output tokens. That asymmetry is a
/// *modelling assumption of the fixture*, not a measured fact — it exists so
/// the tests exercise both sides of the hook's decision surface, and no test
/// in this file treats it as evidence for anything.
fn reasoning_trace(think_tokens: usize, output_tokens: usize) -> Vec<TokenSample> {
    let mut trace = Vec::with_capacity(think_tokens + output_tokens + 3);
    trace.push(sample(THINK_START, 3.0, 0.01));
    for _ in 0..think_tokens {
        trace.push(sample(FILLER, 3.2, 0.02));
    }
    trace.push(sample(THINK_END, 1.0, 0.90));
    for _ in 0..output_tokens {
        trace.push(sample(FILLER, 0.8, 0.0));
    }
    trace.push(sample(EOS, 0.2, 0.0));
    trace
}

/// A request that never emits `<think>` — the non-reasoning path.
fn output_only_trace(output_tokens: usize) -> Vec<TokenSample> {
    let mut trace = Vec::with_capacity(output_tokens + 1);
    for _ in 0..output_tokens {
        trace.push(sample(FILLER, 0.8, 0.0));
    }
    trace.push(sample(EOS, 0.2, 0.0));
    trace
}

/// A request that enters the think phase and never leaves it.
fn think_only_trace(think_tokens: usize) -> Vec<TokenSample> {
    let mut trace = Vec::with_capacity(think_tokens + 1);
    trace.push(sample(THINK_START, 3.0, 0.01));
    for _ in 0..think_tokens {
        trace.push(sample(FILLER, 3.2, 0.02));
    }
    trace
}

fn router() -> PhaseRouter {
    PhaseRouter::new(PhaseRouterConfig {
        // Large enough that budget forcing never preempts a fixture; these
        // tests are about phase attribution, not about the force path, which
        // `phase_router_state_machine.rs` already covers.
        min_think_tokens: u32::MAX,
        max_think_tokens: u32::MAX,
        ..PhaseRouterConfig::with_boundary_ids(vec![THINK_START], vec![THINK_END], vec![EOS])
    })
}

fn hook_with_baseline(baseline: u32) -> PhaseConditioningHook {
    PhaseConditioningHook::new(PhaseConditioningConfig {
        baseline_proposal_len: baseline,
        min_proposal_len: 1,
        max_proposal_len: baseline.max(1),
        vocab_size: VOCAB,
        ..PhaseConditioningConfig::default()
    })
    .expect("valid hook config")
}

// ---------------------------------------------------------------------------
// Replay harness
// ---------------------------------------------------------------------------

/// Outcome of replaying one trace.
#[derive(Debug, Default)]
struct Replay {
    observations: Vec<AcceptanceObservation>,
    /// Proposal lengths the hook recommended, in order.
    recommended: Vec<u32>,
}

/// Walk a trace as a sequence of verification steps, asking the hook for a
/// draft depth at each one and emitting a phase-attributed observation.
///
/// `accepted_of` maps `(phase, span_len)` to how many of the span's tokens the
/// drafter got right — the one quantity a CPU-only test cannot know and must
/// therefore be told.
fn replay(
    trace: &[TokenSample],
    hook: &PhaseConditioningHook,
    accepted_of: impl Fn(SpeculationPhase, usize) -> u32,
) -> Replay {
    let router = router();
    let req_id = 1;
    router.register(req_id);

    let mut out = Replay::default();

    // Prefill samples and commits the first token *before* the draft/verify
    // loop begins — see DeepSpec's `generate_decoding_sample`, which writes
    // `output_ids[num_input_tokens]` from the prefill logits and only then
    // enters its `while start < max_length` loop. That first token is not a
    // verification step and must not be recorded as one.
    let mut cursor = 0;
    if let Some(first) = trace.first() {
        router.on_token(req_id, first.id, Some(&first.signal));
        cursor = 1;
    }

    while cursor < trace.len() {
        let phase_before = router
            .phase_of(req_id)
            .as_ref()
            .map_or(SpeculationPhase::Complete, SpeculationPhase::from);

        let policy = hook.policy_for(phase_before, Some(&trace[cursor].signal));
        out.recommended.push(policy.proposal_len);

        // A verification step commits at most `γ + 1` tokens: the accepted
        // draft prefix plus the target's bonus token.
        let span_len = ((policy.proposal_len as usize) + 1).min(trace.len() - cursor);
        let span = &trace[cursor..cursor + span_len];

        let mut straddles = false;
        for (offset, token) in span.iter().enumerate() {
            let event = router.on_token(req_id, token.id, Some(&token.signal));
            // The span straddles the boundary only if tokens from *both*
            // phases landed in it — a boundary on the final token leaves the
            // span wholly within the think phase.
            if matches!(event, PhaseEvent::ExitThink { .. }) && offset + 1 < span.len() {
                straddles = true;
            }
        }

        let accepted = accepted_of(phase_before, span_len).min(span_len as u32);
        out.observations.push(if straddles {
            AcceptanceObservation::straddling(accepted, policy.proposal_len)
        } else {
            AcceptanceObservation::new(phase_before, accepted, policy.proposal_len)
        });

        cursor += span_len;
    }

    out
}

/// A fixed acceptance model: think steps land fewer tokens than output steps.
/// A *stipulation of the fixture*, never a finding.
fn stipulated_acceptance(phase: SpeculationPhase, span_len: usize) -> u32 {
    let cap = span_len as u32;
    match phase {
        SpeculationPhase::Think => cap.saturating_sub(3).max(1),
        _ => cap,
    }
}

// ---------------------------------------------------------------------------
// Fixture 1: clean phase boundary
// ---------------------------------------------------------------------------

#[test]
fn clean_boundary_attributes_every_step_to_exactly_one_phase() {
    let hook = hook_with_baseline(3);
    // γ = 3 → 4-token spans, and prefill absorbs `<think>`, so the loop starts
    // at index 1. With 7 think tokens `</think>` sits at index 8, the final
    // slot of the span [5..8] — a boundary exactly on a step edge.
    let trace = reasoning_trace(7, 12);
    let replayed = replay(&trace, &hook, stipulated_acceptance);

    let mut ledger = AcceptanceLedger::new(Provenance::synthetic("clean-phase-boundary"));
    ledger.record_all(&replayed.observations);

    assert_eq!(
        ledger.straddling_steps(),
        0,
        "boundary should sit on a span edge"
    );
    assert!(ledger.straddle_rate() < f64::EPSILON);
    assert!(ledger.think().step_count() > 0);
    assert!(ledger.output().step_count() > 0);

    let report = ledger.report();
    assert!(
        report.welch.is_some(),
        "both arms populated → statistic defined"
    );
    assert!(
        report.think.mean_accepted_length() < report.output.mean_accepted_length(),
        "fixture stipulates a lower think-phase acceptance",
    );
}

#[test]
fn clean_boundary_conserves_every_recorded_step() {
    let hook = hook_with_baseline(3);
    let trace = reasoning_trace(7, 12);
    let replayed = replay(&trace, &hook, stipulated_acceptance);

    let mut ledger = AcceptanceLedger::new(Provenance::synthetic("clean-phase-boundary"));
    ledger.record_all(&replayed.observations);

    let accounted = ledger.think().step_count() + ledger.output().step_count();
    assert_eq!(
        accounted as usize,
        replayed.observations.len(),
        "no step may vanish when nothing straddles",
    );
}

// ---------------------------------------------------------------------------
// Fixture 2: ambiguous (straddling) boundary
// ---------------------------------------------------------------------------

#[test]
fn ambiguous_boundary_produces_exactly_one_straddling_step() {
    let hook = hook_with_baseline(4);
    // γ = 4 → 5-token spans; 7 think tokens puts `</think>` mid-span.
    let trace = reasoning_trace(7, 12);
    let replayed = replay(&trace, &hook, stipulated_acceptance);

    let mut ledger = AcceptanceLedger::new(Provenance::synthetic("ambiguous-boundary"));
    ledger.record_all(&replayed.observations);

    assert_eq!(
        ledger.straddling_steps(),
        1,
        "exactly one step should span the boundary",
    );
    assert!(ledger.straddle_rate() > 0.0);
}

#[test]
fn straddle_policy_changes_attribution_without_changing_the_straddle_count() {
    let hook = hook_with_baseline(4);
    let trace = reasoning_trace(7, 12);
    let observations = replay(&trace, &hook, stipulated_acceptance).observations;

    let mut totals = Vec::new();
    for policy in [
        StraddlePolicy::Exclude,
        StraddlePolicy::AttributeToThink,
        StraddlePolicy::AttributeToOutput,
    ] {
        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("ambiguous-boundary"))
            .with_straddle_policy(policy);
        ledger.record_all(&observations);
        assert_eq!(ledger.straddling_steps(), 1, "{policy:?}");
        totals.push((
            policy,
            ledger.think().step_count(),
            ledger.output().step_count(),
        ));
    }

    let (_, exclude_think, exclude_output) = totals[0];
    let (_, to_think, to_think_output) = totals[1];
    let (_, to_output_think, to_output) = totals[2];

    assert_eq!(to_think, exclude_think + 1, "think arm gains the straddler");
    assert_eq!(to_think_output, exclude_output);
    assert_eq!(
        to_output,
        exclude_output + 1,
        "output arm gains the straddler"
    );
    assert_eq!(to_output_think, exclude_think);
}

/// A high straddle rate is the signal that phase attribution is too coarse to
/// trust. It must be visible in the rendered report, not buried.
#[test]
fn straddle_rate_is_surfaced_in_the_rendered_report() {
    let hook = hook_with_baseline(4);
    let trace = reasoning_trace(7, 12);
    let mut ledger = AcceptanceLedger::new(Provenance::synthetic("ambiguous-boundary"));
    ledger.record_all(&replay(&trace, &hook, stipulated_acceptance).observations);

    let rendered = ledger.report().to_string();
    assert!(rendered.contains("straddle rate"));
    assert!(rendered.contains("straddle policy: exclude"));
}

// ---------------------------------------------------------------------------
// Fixture 3: degenerate all-think
// ---------------------------------------------------------------------------

#[test]
fn all_think_trace_leaves_the_output_arm_empty_and_the_verdict_undecided() {
    let hook = hook_with_baseline(3);
    let replayed = replay(&think_only_trace(40), &hook, stipulated_acceptance);

    let mut ledger = AcceptanceLedger::new(Provenance::synthetic("degenerate-all-think"));
    ledger.record_all(&replayed.observations);

    assert!(ledger.think().step_count() > 0);
    assert_eq!(ledger.output().step_count(), 0);
    assert!(ledger.output().mean_accepted_length().is_none());

    let report = ledger.report();
    assert!(report.welch.is_none());
    assert_eq!(report.verdict(0.0), HypothesisVerdict::InsufficientData);
    // Rendering must not panic on a half-empty ledger.
    assert!(report.to_string().contains("insufficient data"));
}

// ---------------------------------------------------------------------------
// Fixture 4: degenerate all-output
// ---------------------------------------------------------------------------

#[test]
fn all_output_trace_leaves_the_think_arm_empty() {
    let hook = hook_with_baseline(3);
    let replayed = replay(&output_only_trace(40), &hook, stipulated_acceptance);

    let mut ledger = AcceptanceLedger::new(Provenance::synthetic("degenerate-all-output"));
    ledger.record_all(&replayed.observations);

    assert_eq!(ledger.think().step_count(), 0);
    assert!(ledger.output().step_count() > 0);
    assert_eq!(ledger.straddling_steps(), 0);
    assert_eq!(
        ledger.report().verdict(0.0),
        HypothesisVerdict::InsufficientData,
    );
}

#[test]
fn an_empty_trace_produces_an_empty_ledger_without_panicking() {
    let hook = hook_with_baseline(3);
    let replayed = replay(&[], &hook, stipulated_acceptance);
    assert!(replayed.observations.is_empty());

    let ledger = AcceptanceLedger::new(Provenance::synthetic("empty-trace"));
    assert_eq!(
        ledger.report().verdict(0.0),
        HypothesisVerdict::InsufficientData,
    );
}

// ---------------------------------------------------------------------------
// The publication gate, end to end
// ---------------------------------------------------------------------------

#[test]
fn no_synthetic_fixture_in_this_file_can_produce_a_publishable_claim() {
    let hook = hook_with_baseline(4);
    for (label, trace) in [
        ("clean-phase-boundary", reasoning_trace(7, 12)),
        ("ambiguous-boundary", reasoning_trace(9, 15)),
        ("degenerate-all-think", think_only_trace(40)),
        ("degenerate-all-output", output_only_trace(40)),
    ] {
        let mut ledger = AcceptanceLedger::new(Provenance::synthetic(label));
        ledger.record_all(&replay(&trace, &hook, stipulated_acceptance).observations);

        let report = ledger.report();
        assert!(report.to_string().contains("NOT A MEASUREMENT"), "{label}");
        assert!(report.clone().into_measured_claim().is_err(), "{label}");
        assert!(report.to_acceptance_prior().is_err(), "{label}");
    }
}

// ---------------------------------------------------------------------------
// The hook's safety property, end to end
// ---------------------------------------------------------------------------

#[test]
fn an_uncalibrated_hook_never_exceeds_its_baseline_across_any_fixture() {
    for baseline in 1_u32..=8 {
        let hook = hook_with_baseline(baseline);
        assert!(!hook.is_calibrated());

        for trace in [
            reasoning_trace(7, 12),
            reasoning_trace(31, 3),
            think_only_trace(40),
            output_only_trace(40),
        ] {
            for depth in replay(&trace, &hook, stipulated_acceptance).recommended {
                assert!(depth <= baseline, "γ={depth} exceeded baseline {baseline}");
                assert!(depth >= 1, "speculation should not be disabled outright");
            }
        }
    }
}

/// The loop closing: a *measured* ledger yields a prior, the prior calibrates a
/// hook, and the calibrated hook drafts differently by phase. This is the only
/// path by which phase conditioning ever becomes active.
#[test]
fn a_measured_ledger_calibrates_a_hook_that_then_conditions_on_phase() {
    let provenance = Provenance::Measured {
        harness: "synthetic-integration-test".into(),
        draft_checkpoint: "fixture/drafter".into(),
        target_model: "fixture/target".into(),
        thinking_mode: true,
        recorded_on: "2026-08-07".into(),
    };

    let mut ledger = AcceptanceLedger::new(provenance);
    for step in 0..400 {
        let jitter = u32::from(step % 2 == 0);
        ledger.record(&AcceptanceObservation::new(
            SpeculationPhase::Think,
            2 + jitter,
            7,
        ));
        ledger.record(&AcceptanceObservation::new(
            SpeculationPhase::Output,
            6 + jitter,
            7,
        ));
    }

    let report = ledger.report();
    assert_eq!(report.verdict(0.5), HypothesisVerdict::Supported);
    // A measured report is the only thing that can become a claim.
    assert!(report.clone().into_measured_claim().is_ok());

    let prior = report.to_acceptance_prior().expect("measured → prior");
    assert!(matches!(prior, AcceptancePrior::Measured { .. }));

    let hook = PhaseConditioningHook::new(PhaseConditioningConfig {
        baseline_proposal_len: 4,
        min_proposal_len: 1,
        max_proposal_len: 12,
        vocab_size: VOCAB,
        prior,
        ..PhaseConditioningConfig::default()
    })
    .expect("valid calibrated config");

    assert!(hook.is_calibrated());
    let think = hook.policy_for(SpeculationPhase::Think, None);
    let output = hook.policy_for(SpeculationPhase::Output, None);

    assert!(think.basis.is_measured() && output.basis.is_measured());
    assert!(
        think.proposal_len < output.proposal_len,
        "calibrated hook should draft shallower in the phase with lower measured acceptance \
         (think γ={}, output γ={})",
        think.proposal_len,
        output.proposal_len,
    );
    assert_eq!(output.basis, PolicyBasis::MeasuredPrior);
}

// ---------------------------------------------------------------------------
// Property-based invariants
// ---------------------------------------------------------------------------

proptest! {
    /// Whatever the trace shape or draft depth, an uncalibrated hook never
    /// drafts deeper than the operator asked for.
    #[test]
    fn uncalibrated_depth_is_bounded_for_arbitrary_traces(
        think_tokens in 0_usize..120,
        output_tokens in 0_usize..120,
        baseline in 1_u32..8,
    ) {
        let hook = hook_with_baseline(baseline);
        let trace = reasoning_trace(think_tokens, output_tokens);
        for depth in replay(&trace, &hook, stipulated_acceptance).recommended {
            prop_assert!(depth <= baseline);
            prop_assert!(depth >= 1);
        }
    }

    /// Every observation lands in exactly one bucket: a phase arm, the
    /// straddle set, or neither — and the totals always reconcile.
    #[test]
    fn ledger_accounting_reconciles_for_arbitrary_traces(
        think_tokens in 0_usize..120,
        output_tokens in 0_usize..120,
        baseline in 1_u32..8,
    ) {
        let hook = hook_with_baseline(baseline);
        let trace = reasoning_trace(think_tokens, output_tokens);
        let observations = replay(&trace, &hook, stipulated_acceptance).observations;

        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("proptest"));
        ledger.record_all(&observations);

        let arms = ledger.think().step_count() + ledger.output().step_count();
        prop_assert_eq!(
            arms + ledger.straddling_steps(),
            observations.len() as u64,
            "steps must be conserved under the default Exclude policy",
        );
        prop_assert!(ledger.straddle_rate() >= 0.0 && ledger.straddle_rate() <= 1.0);
    }

    /// A reasoning trace crossing the boundary produces at most one straddling
    /// step, because the phase transition happens exactly once.
    #[test]
    fn at_most_one_step_straddles_a_single_boundary(
        think_tokens in 1_usize..120,
        output_tokens in 1_usize..120,
        baseline in 1_u32..8,
    ) {
        let hook = hook_with_baseline(baseline);
        let trace = reasoning_trace(think_tokens, output_tokens);
        let observations = replay(&trace, &hook, stipulated_acceptance).observations;

        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("proptest"));
        ledger.record_all(&observations);
        prop_assert!(ledger.straddling_steps() <= 1);
    }

    /// Synthetic provenance survives every trace shape: no fixture, however
    /// well-populated, can be promoted to a measurement.
    #[test]
    fn synthetic_provenance_is_never_promotable(
        think_tokens in 0_usize..120,
        output_tokens in 0_usize..120,
    ) {
        let hook = hook_with_baseline(4);
        let trace = reasoning_trace(think_tokens, output_tokens);

        let mut ledger = AcceptanceLedger::new(Provenance::synthetic("proptest"));
        ledger.record_all(&replay(&trace, &hook, stipulated_acceptance).observations);

        prop_assert!(ledger.report().into_measured_claim().is_err());
    }
}
