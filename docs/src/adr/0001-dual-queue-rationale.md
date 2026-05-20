# ADR-0001: Dual-queue scheduling vs. priority weights on a single queue

- **Status**: Accepted
- **Date**: 2026-05-20
- **Authors**: angelnicolasc
- **Reviewers**: sole-maintainer decision record

## Context

Meridian's central thesis is that **think-decode** and **output-decode** are
two structurally different workloads inside a single reasoning-model request:

- Output tokens are *user-visible streaming*. They must hit a tight TTOT
  (time-to-output-token) target — ~20 ms is the perceptual threshold for
  fluid streaming at typical display rates.
- Think tokens are *user-invisible reasoning*. The user is already waiting
  for the answer; inter-token latency during reasoning is irrelevant.
  Throughput (tokens/sec/GPU) is what matters here.

Given that, the scheduler needs to give **output-phase requests absolute
priority** while letting **think-phase requests fill any remaining batch
capacity** with a larger effective batch size to maximise GPU utilisation.

There are two plausible structural shapes to implement this:

1. **Single queue with priority weights.** One queue of all
   "decode-eligible" requests; each request carries a priority numeric.
   The scheduler picks the highest-weighted requests every iteration and
   the eviction policy reads block tier from the *request*'s phase.
2. **Two independent queues, one per phase.** An `output_queue` drained
   first to its budget, then a `think_queue` drained to a larger budget
   capped by remaining KV memory.

Both can produce equivalent dispatch ordering. They diverge in observability,
in the failure modes they expose, and in how cleanly they compose with KV
tier management.

## Decision

**Meridian uses two independent queues — `output_queue` and `think_queue` —
sharing the same GPU workers, with output drained first every iteration.**

The scheduler exposes per-queue depth as a separate metric label, applies
per-queue SLO budgets, and the block manager's eviction tiers are indexed
on the block's phase membership rather than on the owning request's priority
number.

## Consequences

### Positive

- **SLO isolation by construction.** TTOT and TPOT live on different queues
  and cannot interfere through priority arithmetic. We never need to reason
  about whether a priority of `5` vs. `7` is enough to keep an output token
  from being preempted by a think token — the queues are physically separate.
- **Reasoning about starvation is local.** With priority weights, you have
  to argue globally about the joint distribution of priorities under load to
  prove that think requests are not starved. With two queues, the worst-case
  is "output_queue saturates the budget → think_queue waits its turn." That
  is a one-line argument and a single bounded scalar (`think_batch_multiplier
  × output_budget`) to tune.
- **Block manager tiering is structurally aligned.** The eviction policy
  iterates `BlockTier::ThinkComplete → ThinkActive → OutputCritical`. The
  scheduler's queues map 1:1 onto two of those tiers (`ThinkActive`,
  `OutputCritical`), and the `ThinkComplete` tier appears precisely when a
  request transitions queues. The pipeline of state transitions is uniform
  end-to-end.
- **Observability is honest.** `meridian.queue_depth{queue="output"}` and
  `…{queue="think"}` are operationally meaningful — they correspond to
  things an oncall engineer can act on. A single
  `meridian.queue_depth_p95_priority` would obscure the failure mode.
- **Future disaggregation is cheap.** When we add a separate decode pool
  for think (a natural extension co-located with prefill-decode disagg
  systems like Mooncake / NIXL), the seam already exists.

### Negative / risks

- **Two queue data structures instead of one.** Marginal memory cost
  (`crossbeam::SegQueue` is small) and a second `O(log n)` insert path.
  Not material against the per-token compute budget.
- **Risk that think queue is permanently starved under sustained output
  pressure.** Mitigation: the scheduler enforces a minimum think-batch
  reservation when `output_queue.len() < output_budget`. Detection:
  `meridian.queue_depth{queue="think"}` rising while
  `meridian.budget_force_triggered` stays flat — alert at p95 depth > 4×
  baseline for 5 minutes.
- **Edge cases at phase transition.** A request that emits `</think>` and
  the next token in the same decode step transitions queues *mid-iteration*.
  This is handled by [`MeridianScheduler::on_phase_event`](https://github.com/angelnicolasc/meridian/blob/main/crates/meridian-core/src/scheduler.rs)
  taking the request out of the think queue and pushing it into the output
  queue before the next `schedule_batch` call. Tests for this case live in
  `tests/phase_router_state_machine.rs`.

### Neutral

- The number of tunables stays the same. A single-queue design with
  priority weights requires `output_priority`, `think_priority`, and a
  `priority_gap_min`; the dual-queue design requires `output_tpot_budget_ms`,
  `think_tpot_budget_ms`, and `think_batch_multiplier`. Both surfaces are
  three scalars.

## Alternatives considered

### Single queue with continuous priority weights

`RequestSlot { priority: f32 }`, dispatch is `argmax(priority)` with
preemption. Output requests carry `priority ≈ 10`, think requests
`priority ≈ 1`. Rejected because:

- The dispatch order under heavy load depends on the *distribution* of
  weighted requests, not on a per-class budget — you can no longer write
  down a one-line invariant like "output never waits more than `K` ms."
- The block manager would need to read priorities to decide eviction order,
  coupling two subsystems that we want orthogonal.
- Operators tuning the system find priority weights opaque — "is `8.5`
  enough?" is not a question with a principled answer.

### Single queue with phase-stratified preemption

One queue, but a hard rule that any output-phase request preempts any
think-phase one. This is structurally equivalent to two queues but
expressed differently. Rejected for code-clarity reasons only: the dual-queue
shape makes the invariant ("output drains first") *the structure*, rather
than an invariant we have to police in the dispatcher.

### Per-tenant queues with phase tags

Considered for multi-tenant SaaS deployments. Not rejected outright, but
deferred — it is an orthogonal axis we can layer on top of the two-queue
shape. Captured as a future ADR placeholder.

## References

- Playbook §3.3 — Dual-Queue Scheduler.
- vLLM v0.9 scheduler internals: `vllm/core/scheduler.py` (single-queue
  priority-weighted implementation we are improving on).
- Mooncake disagg paper — separates prefill from decode; orthogonal axis.
- DUCHESS (arXiv:2509.24957) — intra-request branch orchestration; operates
  below the queue layer.
