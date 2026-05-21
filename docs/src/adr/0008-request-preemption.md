# ADR-0008: Request preemption policy

- **Status**: Accepted
- **Date**: 2026-05-20
- **Authors**: angelnicolasc
- **Reviewers**: sole-maintainer decision record

## Context

Meridian reorders the batch vLLM hands it so output-phase requests are
dispatched ahead of think-phase requests (ADR-0001). Reordering is a
*non-destructive* operation: it changes the rank of requests that vLLM has
already decided to run this step, but it never removes a request from the
running set and never reclaims KV from a request mid-flight.

The next lever — and the one every mature serving scheduler eventually
reaches for — is **preemption**: evicting a request that is already
dispatched so its KV blocks can be reused by a higher-priority request.
vLLM has its own preemption (recompute / swap), but it is phase-blind: it
preempts on global memory pressure without knowing that a think-phase
request holding 14 GB of KV is a far better victim than an output-phase
request streaming to a waiting user.

A phase-aware preemption policy is therefore a natural Meridian feature.
The question this ADR answers is **not** "how do we build it" in isolation,
but "do we build it before 1.0, and if not, what is the design and the risk
that justify deferring it". Shipping preemption is the single highest-risk
change available to the scheduler: it can deadlock, it can thrash, and it
can corrupt a request's KV if the reconstruction path is wrong. A wrong
preemption decision is user-visible as a stalled or restarted generation.

## Decision

**Meridian will not preempt already-dispatched requests before 1.0.** The
plugin's influence on vLLM remains advisory — reordering and budget forcing
only. This ADR records the intended design and its risk analysis so the
deferral is a deliberate, documented choice rather than a gap.

### Intended design (post-1.0)

When preemption is implemented it will follow this shape:

- **Victim selection by phase, then LRU.** The victim search walks the
  block manager's tier order — `ThinkComplete` first, then `ThinkActive`,
  and `OutputCritical` only under the same severe-pressure warning that
  `evict_for` already emits. Within a tier, the least-recently-touched
  request is the victim. This reuses the existing eviction ordering rather
  than introducing a second policy.
- **Recompute, not swap, as the default reclamation path.** A preempted
  think-phase request is cheaper to recompute from its prompt than to swap
  its KV to host and back, because think-phase KV is exactly the data we are
  willing to discard (ADR-0004). Swap remains available for output-phase
  victims, which must never lose progress.
- **A preemption budget.** At most a configurable fraction of the running
  set may be preempted per scheduler step (`preempt_max_fraction`, default
  small). This caps thrash: a pressure spike cannot evict the whole batch in
  one step.
- **A re-admission guard.** A preempted request is parked with a
  monotonically increasing priority floor so it cannot be preempted again
  immediately after re-admission. This is the anti-livelock invariant.
- **Disagg interaction.** When a fabric is configured (ADR-0006), a
  `ThinkComplete` victim is *offloaded* rather than discarded, so its KV is
  recoverable from the fabric instead of recomputed. Preemption and offload
  share the same victim search.

### API shape (sketch, not committed)

The scheduler would gain a single entry point that returns victims for the
caller to actuate against vLLM, keeping Meridian advisory rather than
reaching into vLLM's running set directly:

```text
fn select_preemption_victims(
    &self,
    needed_bytes: u64,
    running: &[RequestId],
) -> Vec<PreemptionVictim>   // { req_id, reason, reclaim: Recompute | Swap | Offload }
```

The plugin translates each victim into the vLLM preemption call appropriate
for that vLLM version, isolated in the same `_extract_*` / `_reorder` shim
layer that already absorbs vLLM API drift.

## Consequences

### Positive

- The pre-1.0 scheduler stays advisory and therefore safe: the worst case
  of a wrong Meridian decision is a sub-optimal dispatch order, never a lost
  or corrupted generation.
- The design is written down, so when preemption lands it starts from a
  reviewed risk analysis rather than a blank page.
- Victim selection reuses the existing tier ordering and disagg offload
  path, so the eventual implementation is additive, not a rewrite.

### Negative / risks (the reason for deferral)

- **Without phase-aware preemption, Meridian leaves throughput on the
  table** under heavy memory pressure: vLLM's phase-blind preemption will
  sometimes evict an output-phase request when a think-phase victim was
  available. This is the cost we accept pre-1.0.
- The risks that justify waiting:
  - **Deadlock**: a preempted request needs memory to be re-admitted that
    only its own preemption could free. Mitigated by the recompute default
    and the re-admission priority floor — both unproven until implemented.
  - **Thrash / livelock**: oscillating preempt/re-admit under sustained
    pressure. Mitigated by the per-step preemption budget.
  - **KV correctness**: a swap-and-restore bug silently corrupts a
    request's context. This needs a dedicated correctness harness on real
    hardware before it can ship — which is precisely the validation we do
    not yet have.

### Neutral

- This ADR is revisited once Meridian has a real-hardware benchmark
  baseline. Preemption that cannot be measured against stock vLLM under
  memory pressure cannot be justified, so the work is gated on that
  measurement capability existing first.

## Alternatives considered

**Implement preemption now, behind a default-off flag.** Considered, so the
code path exists for early adopters. Rejected: a default-off feature with no
real-hardware correctness harness is untested code that rots, and the
risk analysis above shows the failure modes are the kind that only surface
under real load.

**Delegate entirely to vLLM's preemption forever.** Considered as the
permanent answer. Rejected as a *permanent* policy because phase-blind
preemption contradicts Meridian's entire thesis; accepted as the
*pre-1.0* policy because the safety bar for taking over preemption is high.

## References

- ADR-0001 (Dual-queue vs. priority weights) — the advisory-reordering
  baseline this ADR declines to extend pre-1.0.
- ADR-0004 (KV tier promotion policy) — the tier ordering victim selection
  reuses.
- ADR-0006 (Disagg KV transfer protocol) — the offload path a
  `ThinkComplete` victim takes when a fabric is configured.
