# DeepSpec Harness: Phase-Segmentation Gap Analysis

| | |
|---|---|
| **Status** | Complete — implementable without further investigation |
| **Work item** | MER-P0.4 |
| **Author** | angelnicolasc |
| **Date** | 2026-08-07 |
| **Subject** | [`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec) `@main`, 2026-08-07 |

The question this note answers: **what exactly has to change in DeepSpec's
evaluation harness to measure draft acceptance separately for the think and
output phases of the same response?**

The answer is smaller than expected, and the reason is worth stating up front.

> **The decode loop already records everything needed. The information is
> destroyed at the aggregation boundary, one function later.**

## 1. What the harness already does

Speculative decoding runs in `generate_decoding_sample`
(`deepspec/eval/base_evaluator.py`). Each iteration of its `while start <
max_length` loop appends to three parallel per-step lists:

| List | Appended | Meaning |
|---|---|---|
| `proposal_lengths` | `verification.effective_proposal_length` | tokens the drafter proposed |
| `accepted_draft_lengths` | `verification.accepted_draft_tokens` | draft tokens that survived |
| `acceptance_lengths` | `accepted_draft_tokens + 1` | tokens committed, including the bonus token |

The loop advances `start` by exactly `acceptance_lengths[i]` each iteration, and
the terminal iteration appends `accepted_draft_tokens` without the bonus token
and advances by that.

All three lists, plus the full `output_ids` tensor and `num_input_tokens`, are
returned on the per-response `SimpleNamespace`.

**Consequence:** the absolute sequence position of every verification step is
exactly reconstructible from the returned payload, with no new instrumentation
inside the decode loop:

```python
start_i = num_input_tokens + sum(acceptance_lengths[:i])
committed_span_i = output_ids[0, start_i + 1 : start_i + 1 + acceptance_lengths[i]]
```

Locate `</think>` once in `output_ids` and every step's phase follows.

## 2. Where the information is lost

`BaseEvaluator.allreduce_response_metrics` folds the per-step lists into five
scalars and two per-position histograms:

```python
metric_summary["proposal_count"]       += 1
metric_summary["acceptance_length_sum"] += int(acceptance_length)
metric_summary["proposal_length_sum"]   += int(proposal_length)
proposals_at_pos[pos_idx] += 1
accepted_at_pos[pos_idx]  += 1
```

These are what `dist.all_reduce` moves across ranks and what
`build_metrics_row` turns into the reported `acceptance_length`,
`draft_tokens_per_proposal`, `verify_rate` and `accept_rates_by_position`.

The per-step lists are never written to disk and never leave the process. By the
time anything is persisted, position information is gone.

**This is the gap.** It is not a missing measurement — it is a discarded one.

## 3. The blocking issue: thinking mode is hard-coded off

`BaseEvaluator.run_dataset` builds its prompts with:

```python
input_ids = encode_chat_messages(
    self.tokenizer,
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    # enable_thinking=True,
)
```

`enable_thinking=False`, with the `True` variant present as a commented-out line
directly beneath it. `encode_chat_messages` (`deepspec/data/parser.py`) forwards
the flag into `tokenizer.apply_chat_template` only when it is not `None`, so the
plumbing is complete — the value is simply pinned, and not exposed as a CLI
argument.

This is consistent with the README's statement that the released checkpoints were
trained on non-thinking-mode generations: the evaluation matches the training
distribution. It also means that **as shipped, the harness cannot produce a
thinking-mode trace at all**, which has to be fixed before anything else matters.

## 4. The straddle problem

A verification step commits up to `γ + 1` tokens at once. The released DSpark
Qwen3 checkpoints are `block7`, so a step commits up to eight tokens. The
`</think>` boundary will usually fall *inside* a committed span rather than on a
step edge, making that step partly reasoning and partly output.

Such a step belongs to neither phase. There are three defensible policies —
exclude it, attribute it to think, attribute it to output — and the choice is an
empirical question, not a design one. The consumer side
(`meridian_core::dspark_bridge::ledger`) therefore implements all three, defaults
to excluding, and reports the straddle rate on every report, because a high
straddle rate means the segmentation is too coarse relative to the block size to
be trusted at all.

At most one step per response can straddle, since the phase transition happens
once. With `block7` and reasoning spans in the hundreds to thousands of tokens,
the expected straddle rate is well under 1 % — but it must be reported, not
assumed.

## 5. Required changes, in full

Four changes. Nothing else is needed.

| # | File | Change | Size |
|---|---|---|---|
| 1 | `deepspec/eval/base_evaluator.py` | Thread `enable_thinking` from an `argparse` flag through `run_dataset` instead of pinning it to `False`. | ~5 lines |
| 2 | `eval.py` | Add `--enable-thinking` / `--no-enable-thinking`. | ~3 lines |
| 3 | `deepspec/eval/base_evaluator.py` | Emit one JSONL record per response — `num_input_tokens`, `acceptance_lengths`, `proposal_lengths`, `accepted_draft_lengths`, and the index of `</think>` in `output_ids` (or `-1`) — *before* `allreduce_response_metrics` discards the lists. One file per rank. | ~20 lines |
| 4 | *(analysis, out of harness)* | Reconstruct spans by cumulative sum, attribute each to a phase, feed `AcceptanceLedger`. | already implemented |

Change 3 is the only one that needs care, and only in one respect: it must
happen **per rank before the all-reduce**, since the all-reduce is where the
data dies.

Recording the `</think>` index rather than the whole `output_ids` tensor keeps
the record small and avoids persisting generated text — see §7.

## 6. What does *not* need to change

Worth stating explicitly, because each of these was a candidate and each turned
out to be unnecessary:

- **No change inside the decode loop.** `generate_decoding_sample` already
  records per-step acceptance. Touching the hot loop was the expected cost of
  this work; it is not needed.
- **No new metric definitions.** `mean accepted length` and `verify_rate` are
  computed per phase using DeepSpec's exact formulas, so the resulting numbers
  are directly comparable to its published tables:
  `verify_rate = acceptance_length_sum / (proposal_length_sum + proposal_count)`.
- **No change to the confidence-head calibration harness.**
  `deepspec/eval/dspark/confidence_head.py` already computes per-position ECE,
  AUROC and Brier scores over cumulative-product predictions. Running it
  separately over think-phase and output-phase steps yields a **second**
  hypothesis for free (see [Phase 1 protocol](phase-1-protocol.md), H2) with no
  new metric code — only segmentation.
- **No new evaluation datasets.** `gsm8k`, `math500`, `aime25`, `humaneval`,
  `mbpp`, `livecodebench`, `mt-bench`, `alpaca` and `arena-hard-v2` ship with the
  repository, and holding the dataset fixed while varying only the thinking-mode
  flag is exactly the confound control the experiment needs.

## 7. Licensing and redistribution

DeepSpec is **MIT** licensed. Its `NOTICE` records that portions are adapted
from third-party projects under their own terms, and this matters for a patch:

- **Apache-2.0** portions adapted from
  [SpecForge](https://github.com/sgl-project/SpecForge) cover the Eagle3
  modeling, loss, optimizer, attention and evaluation code. The listed files
  include `deepspec/eval/eagle3/evaluator.py`, and the list is explicitly
  **non-exhaustive** — files that incorporate adapted code carry an in-file
  comment pointing upstream.
- **MIT** design input from [DFlash](https://github.com/z-lab/dflash) informs
  the DSpark/DFlash configurations and modeling.

`deepspec/eval/base_evaluator.py` and `eval.py` — the two files this patch
touches — are not in the enumerated SpecForge list. Because that list is
non-exhaustive, **check both files for an upstream-pointer comment before
redistributing a modified copy**, and if either carries one, retain the Apache-2.0
notice and state the modification, as §4 of that licence requires.

The cleanest route, and the recommended one, is to publish the patch as a diff
against an upstream commit rather than as a redistributed fork, which sidesteps
the question entirely.

Data handling: the JSONL record in change 3 deliberately stores the `</think>`
index rather than the token stream. Generated text is not needed for the
measurement, and not persisting it avoids redistributing model output derived
from third-party evaluation prompts.

## 8. Residual risks

| Risk | Assessment |
|---|---|
| `</think>` is emitted as a token id that differs from the one searched for | Qwen3's chat template makes the delimiter unambiguous, but the patch must resolve the id **through the tokenizer**, not hard-code it. Same failure mode Meridian's `[model.*]` config already exists to handle. |
| The target never emits `</think>` within `max_new_tokens` | Produces an all-think response with an empty output arm. The ledger handles it — the arm is empty, the statistic is `InsufficientData`, no silent zero. Covered by the `degenerate-all-think` fixture. |
| Thinking mode changes response length enough to change step count per arm | Expected and fine: Welch's test does not assume equal sample sizes. It is the reason the test is Welch's rather than Student's. |
| Straddle rate turns out to be high | Would invalidate the segmentation, not just weaken it. Reported on every report for exactly this reason; treat >5 % as a stop condition. |

## 9. Verification

Everything in §1–§3 was read from the repository at `main` on 2026-08-07:
`eval.py`, `deepspec/eval/base_evaluator.py`, `deepspec/data/parser.py`,
`deepspec/eval/dspark/confidence_head.py`, `LICENSE` and `NOTICE`. Line-level
claims describe that revision; re-verify before implementing, since the
`enable_thinking` line in particular is the kind of thing that changes.
