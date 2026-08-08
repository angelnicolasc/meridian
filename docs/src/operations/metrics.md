# Metrics

Meridian emits Prometheus metrics and OpenTelemetry traces. **All metric names
are stable contracts** — renames or semantic changes trigger a minor-version
bump per [ADR-0007](../adr/0007-release-versioning-policy.md).

## Metric catalog

### `meridian.think_tokens_per_request`

| Property | Value |
|----------|-------|
| Type | histogram |
| Unit | tokens |
| Cardinality | 1 series (no labels) |
| Source | `PhaseRouter` on `ExitThink` or `ForceBudget` |
| Why | Tracks the distribution of reasoning-chain lengths. Long tails here indicate the entropy probe is deferring too late; a spike in the P99 bucket means hard-cap forcing is dominating. |

**Operator action**: if P99 frequently hits `max_think_tokens`, lower `max_think_tokens`
or tighten `eat_ema_variance_threshold` / `rpdi_threshold` to fire forcing earlier.

---

### `meridian.budget_force_triggered`

| Property | Value |
|----------|-------|
| Type | counter |
| Unit | events |
| Cardinality | 1 series |
| Source | `PhaseRouter` |
| Why | Measures how often the router fires `</think>` injection. A counter that never moves means the entropy probe is never converging (check `eat_ema_variance_threshold`). |

---

### `meridian.budget_force_reason{reason=...}`

| Property | Value |
|----------|-------|
| Type | counter |
| Unit | events |
| Labels | `reason` ∈ `{converged, overthinking, hard_cap}` |
| Cardinality | 3 series |
| Source | `PhaseRouter` |
| Why | Breaks down *why* forcing fired. `converged` and `overthinking` mean the entropy probe is working. Sustained `hard_cap` dominance means the probe is failing to detect convergence. |

**Operator action**: monitor the ratio `hard_cap / (converged + overthinking)` over
a 1-hour window. A ratio above 0.5 is a signal to investigate probe thresholds or
inspect sample EAT traces.

---

### `meridian.output_critical_eviction`

| Property | Value |
|----------|-------|
| Type | counter |
| Unit | events |
| Cardinality | 1 series |
| Source | `PhaseAwareBlockManager` |
| Why | Every increment is a user-visible degradation event — a KV block backing the live output stream was evicted under memory pressure. Zero is the target. |

**Operator action**: alert at `rate(...) > 0` sustained for 5 minutes. Mitigate
by lowering `think_phase_memory_fraction`, `think_batch_multiplier`, or
`max_think_tokens`.

---

### `meridian.phase_router.tracked_requests`

| Property | Value |
|----------|-------|
| Type | gauge |
| Unit | requests |
| Cardinality | 1 series |
| Source | `PhaseRouter` |
| Why | Requests that complete but are never reaped leak entries. Monotonically growing gauge means `reap_stale_older_than` is not being called, or the vLLM plugin's `post_step` is not receiving EOS events. |

**Operator action**: if the gauge grows without a corresponding growth in active
concurrent requests, check that `post_step` receives EOS and that the reap
period (`60 s` default) is shorter than request lifetime.

---

### `meridian.schedule_batch.duration_ns`

| Property | Value |
|----------|-------|
| Type | histogram |
| Unit | nanoseconds |
| Cardinality | 1 series |
| Source | `MeridianScheduler::schedule_batch` |
| Why | Measures scheduling overhead on the hot path. Should be in the microsecond range; millisecond-range values indicate contention inside the scheduler lock. |

**Operator action**: P99 above 1 ms under steady load is unexpected. File an issue
with a CPU profile.

---

### `meridian.queue_depth{queue=...}`

| Property | Value |
|----------|-------|
| Type | gauge |
| Unit | requests |
| Labels | `queue` ∈ `{output, think}` |
| Cardinality | 2 series |
| Source | `MeridianScheduler` |
| Why | Monitors queue backlog. Output queue depth growing without draining means the GPU is bottlenecked. Think queue depth growing without `budget_force_triggered` activity means the entropy probe is not converging and long chains are piling up. |

**Operator action**: alert when `queue_depth{queue="think"}` P95 exceeds 4× its
1-hour baseline for 5 consecutive minutes without accompanying forcing activity.

---

### `meridian.block_manager.used_bytes`

| Property | Value |
|----------|-------|
| Type | gauge |
| Unit | bytes |
| Cardinality | 1 series |
| Source | `PhaseAwareBlockManager` |
| Why | Total KV bytes currently allocated across all tiers. Rising towards `kv_memory.capacity_bytes` predicts incoming eviction pressure. |

**Operator action**: alert when `block_manager.used_bytes / capacity_bytes` exceeds
0.90 for 10 minutes — this is the early-warning threshold before
`output_critical_eviction` events begin.

---

### `meridian.block_manager.evictions{tier=...}`

| Property | Value |
|----------|-------|
| Type | counter |
| Unit | events |
| Labels | `tier` ∈ `{think_complete, think_active, output_critical}` |
| Cardinality | 3 series |
| Source | `PhaseAwareBlockManager` |
| Why | Per-tier eviction rate reveals the shape of memory pressure. `think_complete` evictions are routine; `think_active` indicates moderate pressure; `output_critical` is a user-visible degradation event identical to `meridian.output_critical_eviction`. |

**Operator action**: alert on any `tier=output_critical` increment — use this
series or `meridian.output_critical_eviction`, whichever is easier to route in
your alerting stack.

---

### `meridian.scheduler.batch_size{phase=...}`

| Property | Value |
|----------|-------|
| Type | histogram |
| Unit | slots |
| Labels | `phase` ∈ `{output, think}` |
| Cardinality | 2 series |
| Source | `MeridianScheduler` |
| Why | Distribution of actual batch sizes delivered to the vLLM worker per phase. A consistently small `output` batch under load means output requests are draining faster than think-phase completions replenish the pool. |

**Operator action**: compare `scheduler.batch_size{phase=output}` P50 against
`queue_depth{queue=output}` to verify output requests are being served promptly.

---

### `meridian_disagg_blocks_offloaded_total{fabric=...}`

| Property | Value |
|----------|-------|
| Type | counter |
| Unit | blocks |
| Labels | `fabric` ∈ `{nixl, mooncake}` |
| Cardinality | 1 series per active fabric |
| Source | `MeridianSchedulerPlugin` on `ExitThink` (flushed at `offload_threshold_blocks`) |
| Why | Tracks disagg throughput. A counter that never moves when disagg is enabled means offload hooks are not firing. |

---

### `meridian_vocab_fallback_total`

| Property | Value |
|----------|-------|
| Type | counter |
| Unit | events |
| Cardinality | 1 series |
| Source | `MeridianSchedulerPlugin` entropy-probe batch path |
| Why | Counts batches where logit rows had heterogeneous vocab sizes and the probe fell back to per-request compute. A rising counter means mixed-model batching is defeating the batched probe; investigate request routing. |

---

### `meridian.speculation.policy_basis{phase,basis}`

| Property | Value |
|----------|-------|
| Type | counter |
| Unit | decisions |
| Labels | `phase` ∈ `{prefill, think, output, complete}`, `basis` ∈ `{baseline, not_decoding, entropy_ceiling, measured_prior, measured_prior_capped_by_entropy}` |
| Cardinality | ≤ 20 series |
| Source | `dspark_bridge::PhaseConditioningHook` |
| Why | Reports on what authority each draft-depth decision rests. `baseline` means the hook did nothing; `entropy_ceiling` means a proved bound justified a shallower draft; `measured_*` means a calibrated prior is in play. |

**Operator action**: until Phase 1 of the phase-conditioned-speculation work has
run, a healthy deployment shows **zero** samples with a `measured_*` basis. Any
`measured_*` traffic means someone installed a
`[speculation.acceptance_prior]` — check it names a real run. See
[ADR-0009](../adr/0009-phase-conditioned-speculation.md).

---

### `meridian.speculation.proposal_len{phase}`

| Property | Value |
|----------|-------|
| Type | histogram |
| Unit | tokens |
| Labels | `phase` |
| Source | `dspark_bridge::PhaseConditioningHook` |
| Why | The recommended draft depth. Compare against `speculation.baseline_proposal_len`: a distribution pinned at the baseline means the entropy ceiling never binds; one pinned at `min_proposal_len` means it always does. |

**Operator action**: if the histogram sits at `min_proposal_len` across both
phases, the cost model constants are likely wrong for your hardware — profile
and set `draft_token_us` / `verify_fixed_us` / `verify_token_us`. They ship as
placeholders, not measurements.

---

### `meridian.speculation.accepted_length{phase}`

| Property | Value |
|----------|-------|
| Type | histogram |
| Unit | tokens (bonus token included) |
| Labels | `phase` ∈ `{think, output}` |
| Source | `dspark_bridge::AcceptanceLedger` |
| Why | Observed accepted draft length per verification step, segmented by phase. This is the quantity the Phase 1 hypothesis is about. Uses DeepSpec's definition so the numbers are comparable to its published tables. |

---

### `meridian.speculation.straddling_steps`

| Property | Value |
|----------|-------|
| Type | counter |
| Unit | events |
| Cardinality | 1 series |
| Source | `dspark_bridge::AcceptanceLedger` |
| Why | Verification steps whose committed span crossed the `</think>` boundary and so belong cleanly to neither phase. |

**Operator action**: compare against the total recorded steps. Above ~5 %, phase
attribution is too coarse relative to the draft block size and any think-vs-output
comparison drawn from it should be discarded rather than caveated.

---

## OTLP export

Prometheus is the primary metric surface. When `[telemetry] otlp_enabled = true`
(requires the `otel` extra), the plugin additionally exports its counters to an
OTLP/HTTP collector at `[telemetry] otlp_endpoint`, and the Rust core can wire
its `tracing` spans to OTLP via the `otel` crate feature
(`meridian_core::telemetry::install`). Both are off by default.

## Trace spans

Each `MeridianScheduler::schedule_batch` call opens a `meridian.schedule_batch`
OpenTelemetry span. `PhaseEvent`s propagate `meridian.phase_event{kind=...}`
events on the active request's span, allowing per-request phase timelines to be
reconstructed from trace data.

## Alerting summary

| Metric | Alert condition | Severity |
|--------|-----------------|----------|
| `output_critical_eviction` | rate > 0 for 5 min | High — user-visible |
| `block_manager.evictions{tier=output_critical}` | rate > 0 for 1 min | High — user-visible (same event, finer label) |
| `block_manager.used_bytes` | > 90% of capacity for 10 min | Medium — pre-eviction warning |
| `queue_depth{queue=think}` | P95 > 4× baseline for 5 min with no forcing | Medium — starvation risk |
| `budget_force_reason{reason=hard_cap}` | ratio > 0.5 over 1 h | Low — probe investigation |
| `phase_router.tracked_requests` | monotonically growing > 15 min | Low — reap misconfiguration |
| `speculation.policy_basis{basis=measured_prior}` | rate > 0 while no measurement is on file | Medium — unverified policy in production |
| `speculation.straddling_steps` | > 5% of recorded steps | Medium — phase attribution unreliable |
