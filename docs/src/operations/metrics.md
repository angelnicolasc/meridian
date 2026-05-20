# Metrics

Meridian emits both Prometheus metrics and OpenTelemetry traces. All names
are stable contracts — renames trigger a major-version bump.

## Metric catalog

| Name                                       | Type      | Meaning                                                       |
|--------------------------------------------|-----------|---------------------------------------------------------------|
| `meridian.think_tokens_per_request`        | histogram | Think tokens consumed before `ExitThink`.                     |
| `meridian.budget_force_triggered`          | counter   | Times the router emitted `ForceBudget`.                       |
| `meridian.budget_force_reason{reason=...}` | counter   | One of `converged`, `overthinking`, `hard_cap`.               |
| `meridian.output_critical_eviction`        | counter   | Block manager had to evict an `OutputCritical` block.         |
| `meridian.phase_router.tracked_requests`   | gauge     | Requests currently tracked by `PhaseRouter`.                  |
| `meridian.schedule_batch.duration_ns`      | histogram | Wall time spent inside `MeridianScheduler::schedule_batch`.   |
| `meridian.queue_depth{queue=...}`          | gauge     | Depth of `output_queue` / `think_queue`.                      |

## Alerting recommendations

- **`meridian.output_critical_eviction`** — every increment is user-visible
  degradation. Alert at `rate(...) > 0` for 5 minutes.
- **`meridian.queue_depth{queue="think"}`** — sustained growth without
  `meridian.budget_force_triggered` activity means the think queue is
  starving. Alert at p95 depth > 4× baseline for 5 minutes.
- **`meridian.budget_force_reason{reason="hard_cap"}`** — increments mean
  the entropy probe failed to detect convergence in time. Investigate the
  ratio of `hard_cap` vs. `converged`+`overthinking` over a 1h window.

## Trace spans

Each `MeridianScheduler::schedule_batch` opens a `meridian.schedule_batch`
span. Phase events propagate `meridian.phase_event{kind=...}` events on the
active request's span.
