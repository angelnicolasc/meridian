# Phase 1 Experiment Protocol

| | |
|---|---|
| **Status** | Specified, **not scheduled** — requires GPU |
| **Work item** | MER-P0.7 |
| **Author** | angelnicolasc |
| **Date** | 2026-08-07 |
| **Prerequisite** | [Instrumentation gap analysis](deepspec-harness-instrumentation.md) |

This document exists so that Phase 1 requires **no design decisions** — only
hardware and time. Everything below is fixed in advance: hypotheses, models,
datasets, sample sizes, statistic, decision rule, and what gets published under
each outcome. Fixing the decision rule before seeing data is the point; it is
what makes a null result publishable rather than embarrassing.

## 1. Hypotheses

### H1 — primary

> Using a DeepSpec-released, non-thinking-trained DSpark checkpoint against a
> Qwen3 target running **in thinking mode**, mean accepted draft length during
> the think-phase span is lower than during the output-phase span of the *same*
> responses, and the gap exceeds the accepted-length variation already observed
> **between task domains** at a fixed configuration.

Directional and pre-registered. The between-domain clause is what makes it a
claim about *phase* rather than a rediscovery of the domain sensitivity DeepSpec
already reports.

### H2 — secondary, free

> The confidence head's calibration error is higher on think-phase positions
> than on output-phase positions.

DeepSpec's `confidence_head.py` already computes per-position ECE, AUROC and
Brier scores. Segmenting its existing accumulators by phase costs no new metric
code. If H1 holds, H2 tells you *why* it matters operationally: a drafter that
is merely worse can be scheduled around, but a drafter whose **confidence is
miscalibrated** breaks DSpark's prefix scheduler, which trusts `∏ c_i` to choose
verification depth. H2 is the more consequential of the two.

## 2. Design

A **within-response, between-condition** design.

- **Within-response** (primary): compare think-phase steps against output-phase
  steps of the same responses, thinking mode ON. Controls for prompt, dataset,
  sampling parameters and checkpoint — everything except phase.
- **Between-condition** (control): the same prompts with thinking mode OFF.
  Establishes the baseline the checkpoint was trained for, and detects a
  confound in which think-phase text is merely *longer* or *later* rather than
  distributionally different.

### Factors

| Factor | Levels | Rationale |
|---|---|---|
| Thinking mode | on, off | The manipulation. |
| Phase | think, output | The within-response contrast. |
| Task domain | `math500`, `gsm8k`, `mt-bench` | One reasoning-heavy, one reasoning-light, one open-ended. **Held fixed within each comparison**; never pooled across domains for H1. |
| Model size | Qwen3-4B, then 8B | 4B first. Only run 8B if 4B shows an effect, or to rule out a size artefact if it does not. |

Domain is held fixed rather than pooled because DeepSpec's own results show
accepted length varies substantially by domain. Pooling would let a domain effect
masquerade as a phase effect — the single largest threat to the result's
validity, and the reason the risk register flags it.

### Fixed across all runs

Checkpoint (`deepseek-ai/dspark_qwen3_4b_block7`), target
(`Qwen/Qwen3-4B`), temperature, seed schedule, `max_proposal_tokens`, and
confidence threshold. Anything not listed as a factor is pinned.

## 3. Procedure

1. Apply the four changes in §5 of the
   [gap analysis](deepspec-harness-instrumentation.md).
2. For each domain, run the harness twice — thinking on, thinking off — over the
   same prompt set with the same seed schedule.
3. Reconstruct per-step spans from the JSONL records by cumulative sum of
   `acceptance_lengths`; attribute each step to a phase against the `</think>`
   index; mark straddling steps.
4. Feed the observations into `AcceptanceLedger`, tagged
   `Provenance::Measured { .. }` with the harness commit, checkpoint, target,
   thinking-mode flag and date.
5. Run the report. Apply the §5 decision rule. Publish.

Step 4 is the only place a mistake is irreversible: a run recorded with the wrong
`thinking_mode` flag is uninterpretable and has to be repeated. Everything else
can be recomputed from the JSONL.

### Sample size

Target **≥ 10,000 verification steps per arm per domain**. At `block7`, a
1,000-token response yields roughly 150–300 steps, so 100 responses per domain
per condition is comfortably sufficient.

This is not a power calculation, because a power calculation is not the binding
constraint here — at 10⁴ steps per arm, Welch's test resolves differences far
smaller than anything operationally interesting. **Statistical significance will
be easy and nearly meaningless. The effect size and the between-domain
comparison are what decide the outcome**, which is why the decision rule in §5
is not a `p`-value threshold.

## 4. Statistic

Welch's unequal-variance *t*-test on mean accepted length, think minus output.
Welch rather than Student's because the two arms have neither equal variance nor
equal size, and thinking mode changes both.

Reported per comparison:

| Quantity | Why |
|---|---|
| mean accepted length, per phase | The Section 5 metric; DeepSpec's definition, bonus token included |
| token acceptance rate, per phase | DeepSpec's `verify_rate`, so numbers are comparable to its tables |
| `think - output`, with 95 % CI | The effect, with its uncertainty |
| Welch `t`, `df`, `p` | Conventional reporting |
| Cohen's `d` | The quantity that decides, per §3 |
| straddle rate | Segmentation quality; **>5 % invalidates the comparison** |
| step counts per arm | Lets a reader recompute everything above |

Implemented in `meridian_core::dspark_bridge::stats`, verified against closed
forms exact for `df = 1` and `df = 2` and against published critical values.

## 5. Decision rule

Fixed in advance, and implemented as
`PhaseGapReport::verdict(between_domain_gap)`:

| Verdict | Condition | What gets published |
|---|---|---|
| `Supported` | 95 % CI excludes zero, gap is negative, and \|gap\| exceeds the between-domain baseline | H1 confirmed, with effect size and the domain baseline it cleared |
| `WithinDomainVariation` | CI excludes zero, gap negative, but ≤ the between-domain baseline | A real but unremarkable gap — *not* evidence of a phase effect. Published as such. |
| `NoResolvableGap` | CI includes zero | H1 falsified at this sample size. Published. |
| `OppositeDirection` | CI excludes zero, gap positive | H1 falsified, direction reversed — the most interesting outcome, and published loudest |
| `InsufficientData` | An arm is empty or degenerate | Not a result. Rerun. |

`between_domain_gap` is computed from this experiment's own runs: the spread in
mean accepted length across the three domains at a fixed configuration. It is not
taken from DeepSpec's published tables, since those use a different metric
aggregation and a different confidence threshold.

**All five outcomes are published.** That is a rule this project has already
committed to and it is not renegotiable after seeing the data.

## 6. Cost

| Item | Estimate |
|---|---|
| Hardware | One consumer/prosumer GPU with ≥ 24 GB — Qwen3-4B target plus a `block7` drafter fits comfortably in bf16 |
| Compute | Six harness runs (3 domains × 2 conditions), ~100 responses each |
| Wall clock | Hours, not days |
| Multi-GPU node | **Not required** |

The 8B follow-up doubles the memory requirement and remains single-GPU.

## 7. Threats to validity

| Threat | Control |
|---|---|
| Domain confound | Domain held fixed within each comparison; never pooled for H1 |
| Length confound — think spans are simply longer | The thinking-off control condition; plus per-position acceptance rates, which separate "worse everywhere" from "decays faster" |
| Position confound — later tokens are harder regardless of phase | Compare output-phase steps under thinking-on against output-phase steps under thinking-off at matched sequence positions |
| Straddling contamination | Excluded by default; rate reported; >5 % is a stop condition |
| Tokenizer mismatch on `</think>` | Resolve the id through the tokenizer, never hard-coded |
| Checkpoint idiosyncrasy | Repeat on the Eagle3 and DFlash checkpoints for the same target — same harness, no new code |
| Analyst degrees of freedom | Every choice above is fixed in this document, before data exists |

## 8. Feeding the result back

A measured report closes the loop with no glue code:

```rust
let prior = report.to_acceptance_prior()?;   // fails on synthetic data
let hook = PhaseConditioningHook::new(PhaseConditioningConfig {
    prior,
    ..config
})?;
```

or equivalently in `meridian.toml`:

```toml
[speculation.acceptance_prior]
think            = 0.42   # from the run
output           = 0.88   # from the run
harness          = "DeepSpec@<commit-sha>"
draft_checkpoint = "deepseek-ai/dspark_qwen3_4b_block7"
target_model     = "Qwen/Qwen3-4B"
thinking_mode    = true
recorded_on      = "<date>"
```

Supplying that section is what switches the hook from "may only reduce draft
depth" to "conditions on phase". Until it exists, the hook stays inert in the
upward direction by construction — see
[ADR-0009](../adr/0009-phase-conditioned-speculation.md).

## 9. If H1 is falsified

The most likely single outcome, and worth planning for rather than dreading.

A null result would say something genuinely useful: that speculative drafters
trained on non-thinking-mode text generalise across the thinking-mode boundary
well enough that the documented training mismatch does not cost throughput. That
is a finding practitioners can act on — it retires a plausible worry about every
released Qwen3 drafter — and it is worth publishing on its own terms.

The structural results in the
[companion note](phase-conditioned-speculation.md) are unaffected either way.
Corollary 5 is a theorem; it does not depend on how this experiment turns out.
