# ADR-0009: The phase-conditioning hook ships inert until measured

- **Status**: Accepted
- **Date**: 2026-08-07
- **Authors**: angelnicolasc
- **Reviewers**: sole-maintainer decision record

## Context

DeepSpec, DeepSeek's open-source speculative-decoding toolkit, ships trained
drafters for Qwen3. Its README states that each released checkpoint was trained
on data its target model generated **in non-thinking mode** — while Qwen3
exposes an `enable_thinking` toggle that production reasoning traffic uses.

That is a documented distribution mismatch between how a drafter was trained and
how it is used. It leads to a plausible hypothesis: draft acceptance is lower
during a reasoning span than during the final answer, making speculative
decoding's speedup silently phase-dependent for anyone using these checkpoints
as shipped. Meridian already routes on exactly that phase boundary, so it is the
natural place to act on the hypothesis.

**Nobody has measured it.** Doing so requires a GPU and is not currently
scheduled. So we are implementing a scheduler behaviour whose motivating premise
is unverified — which is a well-known way to ship a regression with a good story
attached.

We also established something that *is* proved (see the
[companion note](../notes/phase-conditioned-speculation.md)): for any
deterministic drafter, single-step acceptance is bounded above by the target's
mode mass, which is in turn bounded above by a function of the target's Shannon
entropy — a quantity Meridian's probe already computes. Critically, that bound
is **one-sided**: it can prove a draft is too deep, never that it is too shallow.

Three options were on the table:

1. **Ship nothing until Phase 1 runs.** Zero risk, zero progress; the code that
   would consume the measurement does not exist when the measurement arrives.
2. **Ship the full phase-conditioning behaviour with plausible default rates.**
   Maximum apparent completeness. The defaults would be invented, and an
   operator enabling the feature would be running an unvalidated policy that
   *looks* measured.
3. **Ship the mechanism, gate the unproven half.**

## Decision

**The hook ships with its unproven half gated, and the gate is enforced by the
type system rather than by documentation.**

Concretely:

- With an `AcceptancePrior::Uncalibrated` prior — the default, and the state the
  project ships in — the hook may only ever return a proposal length **less than
  or equal to** the operator's configured baseline. It acts solely on the
  proved, one-sided entropy ceiling.
- Raising draft depth, and conditioning on phase at all, requires an
  `AcceptancePrior::Measured`, which can only be constructed from a
  `Provenance::Measured` value.
- Every acceptance statistic carries a `Provenance`. The single function that
  promotes a statistic to a publishable claim, `into_measured_claim`, returns
  `Error::SyntheticProvenance` for anything synthetic.
- The TOML schema cannot express acceptance rates without the harness,
  checkpoint, target model, thinking-mode flag and date of the run that produced
  them. There is no field for a hand-tuned rate.

The invariant is pinned by tests at both levels: a unit test sweeping the full
entropy range against both phases, and an integration test sweeping every
synthetic fixture and baseline.

## Consequences

### Positive

- **Merging cannot regress serving throughput on a hunch.** The only behaviour
  change available to an uncalibrated deployment is a *shallower* draft, taken
  only where a theorem says the deeper draft cannot pay. Nothing about that
  depends on the hypothesis being true.
- **The publication rule is mechanical.** "Do not publish synthetic numbers as
  measurements" is a rule that gets broken under deadline pressure. Here it is a
  compile-time and run-time error. A synthetic trace cannot reach a paper
  without someone deleting code, which is a reviewable act.
- **Phase 1 is a data-plug-in, not a project.** The ledger, the statistic, the
  decision rule and the config path all exist and are tested. The experiment
  produces a JSONL file; everything downstream already runs.
- **Enabling the feature is an honest signal.** `speculation.acceptance_prior`
  being present in a config means someone measured something and said where. Its
  absence means nobody has.
- **The entropy ceiling has standalone value.** Corollary 5 holds regardless of
  how Phase 1 turns out. Even under a null result, capping draft depth at
  high-entropy steps remains correct.

### Negative / risks

- **The shipped default is close to a no-op.** With `use_entropy_ceiling = true`
  it trims depth only where entropy is genuinely high; with it `false` it does
  nothing at all. Someone evaluating the module on behaviour alone will find
  little happening. That is the intended trade and it is documented at every
  entry point, but it is a real cost in perceived value.
- **The cost model is unvalidated.** `γ*` is chosen by maximising a throughput
  proxy whose three cost constants are placeholders. A deployment with a very
  different cost profile could see the ceiling bind at the wrong depth.
  *Detection:* `meridian.speculation.proposal_len{phase}` diverging from
  `baseline_proposal_len` far more or less often than an operator expects.
  *Mitigation:* the constants are configurable and documented as requiring
  profiling.
- **`τ(a, γ)` assumes i.i.d. per-position acceptance, which is false.** Real
  drafters decay along the block. The model will place `γ*` slightly too deep.
  *Detection:* Phase 1's per-position acceptance rates. *Mitigation:* the
  ceiling is one-sided, so the error is bounded by the baseline clamp while
  uncalibrated.
- **A calibrated hook inherits its prior's staleness.** Acceptance rates measured
  against one checkpoint and target do not transfer to another.
  *Detection:* the prior carries `draft_checkpoint` and `target_model`; a
  deployment serving something else is visibly mismatched.
  *Mitigation:* none automated. This is a known gap, not a solved problem.

### Neutral

- `SpeculationPhase` duplicates part of `ThinkPhase` rather than reusing it. The
  speculation path needs the phase label and not the entropy accumulators, and
  `From<&ThinkPhase>` keeps them in sync at one site.
- The hook is stateless. Per-request state stays in `PhaseRouter`, which already
  owns it.

## Alternatives considered

### Ship nothing until Phase 1 runs

Rejected. The structural work — the formal comparison, the proved bounds, the
statistic, the harness gap analysis — is complete, CPU-only, and independently
useful. Withholding it buys no safety, because none of it changes runtime
behaviour, and it guarantees that when the measurement arrives, nothing exists to
consume it.

### Ship full phase conditioning with plausible defaults

Rejected, and this was the tempting option. Numbers like "think 0.45, output
0.90" would have made the module look finished and demoed well. They would also
have been invented, and an operator enabling the feature would have been running
an unvalidated policy indistinguishable, from the outside, from a measured one.

The failure is not that the guesses might be wrong — it is that nothing
downstream could tell they were guesses. Provenance tagging exists specifically
so that difference is legible.

### Gate by a runtime flag instead of the type system

Rejected. A boolean `allow_uncalibrated_phase_conditioning` is the same rule
enforced by convention, and conventions decay. Making the calibrated path
unreachable without a `Provenance::Measured` value means the gate cannot be
bypassed by setting a flag under pressure — it has to be bypassed by editing
code, in a diff a reviewer will see.

### Let the ledger emit measurements regardless of provenance

Rejected for the same reason, one layer up. The ledger *does* compute the full
statistic on synthetic data — hiding the number would make the fixtures
untestable. What it refuses is the promotion step. Analysis stays cheap;
publication stays gated.

## References

- [Phase-conditioned speculative decoding](../notes/phase-conditioned-speculation.md)
  — the formal comparison and the five results this ADR rests on.
- [DeepSpec harness instrumentation gap](../notes/deepspec-harness-instrumentation.md)
  — MER-P0.4.
- [Phase 1 protocol](../notes/phase-1-protocol.md) — the deferred experiment.
- DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
  Generation, [arXiv:2607.05147](https://arxiv.org/abs/2607.05147).
- ADR-0007 — release and versioning policy; `[speculation]` is additive and
  defaults off, so this is a minor-version change.
