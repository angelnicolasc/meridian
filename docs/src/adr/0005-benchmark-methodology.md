# ADR-0005: Benchmark methodology and metric selection

- **Status**: Accepted
- **Date**: 2026-05-20
- **Authors**: angelnicolasc
- **Reviewers**: sole-maintainer decision record

## Context

Meridian's value proposition is *phase-differentiated scheduling*. The
benchmark harness has to surface that value — measuring overall throughput
or aggregate TPOT will not distinguish Meridian from a stock vLLM scheduler
running the same workload, even if Meridian is substantially better for
the user-visible streaming experience.

We must choose:

1. **What metrics to report.**
2. **What workload to drive the system with.**
3. **How to run the benchmark without a GPU in CI.**

## Decision

### Metrics

The benchmark reports the following primary metrics (all per-request,
aggregated to percentiles in the report):

| Metric | Definition | Why it matters |
|---|---|---|
| **TTFT** | Time-to-first-token (prefill + first decoded token). | Industry-standard latency floor; baseline parity. |
| **TTOT** | Time-to-first-OUTPUT-token. Measured from `</think>` emission to the first user-visible token after it. | **The metric stock vLLM does not even track.** This is where dual-queue scheduling shows its value: in a baseline, output tokens can be preempted by think tokens, inflating TTOT P95. |
| **Output ITL** | Inter-token latency *during the output phase only*. Measured per (token N → token N+1) pair. P50/P95/P99. | Streaming fluidity — the perceptual quality of the user-visible output. |
| **Think tokens** | Tokens emitted in the think phase per request. Avg + P95. | Cost driver; budget-forcing efficacy is measured against this. |
| **Budget force rate** | Percentage of reasoning requests where budget force fired. Broken down by reason (`converged`, `overthinking`, `hard_cap`). | Quality signal: if `hard_cap` dominates we are forcing blindly; if `converged` dominates the entropy probe is doing its job. |
| **OutputCritical eviction events** | Count of eviction events that reached the OutputCritical tier during the run. | User-visible degradation event. Any non-zero is alertable. |

The harness explicitly does NOT report aggregate throughput
(tokens/sec/GPU). That is what every other benchmark already reports, and
it does not distinguish Meridian from the baseline. **Operators who only
want a throughput number can run vLLM's own benchmark harness.**

### Workload

Reference workload: **synthetic mix** of two categories.

- **Chat** — short prompts, 40–240 output tokens, no think phase. Models
  the ShareGPT-style background traffic that should never stutter.
- **Reasoning** — math-style prompts with expected think token counts in
  `[600, 6000]`. Models a MATH-500-equivalent distribution.

Mix ratio is operator-configurable (`--reasoning-ratio`); default 0.4 is
the realistic balance for a 2026 reasoning-model deployment.

Arrivals are **Poisson** (exponential inter-arrival) at a configurable
rate. This matches how production traffic actually arrives and exercises
the dual-queue policy under realistic burst conditions.

### Two execution modes

- **`synthetic-replay`** — uses the native Meridian components
  (`PhaseRouter`, `MeridianScheduler`, `BlockManager`) over a *synthetic*
  decoder loop that does not require a GPU or a real vLLM. The phase
  events, scheduler queue transitions, KV allocations and eviction
  pressure are all real; only the per-token compute is simulated as a
  fixed-cost sleep. **This mode runs in CI** and detects regressions in
  the scheduler / block manager dynamics.

- **`real-vllm`** — drives a real `AsyncLLMEngine` with the
  `MeridianSchedulerPlugin` attached. Requires a CUDA-capable host and
  a model checkpoint. Runs on the GPU CI job and on demand for release
  validation.

Both modes emit the same `BenchmarkReport` JSON+Markdown shape so reports
are directly comparable. CI uploads both as artefacts and the Markdown
form can be posted as a PR comment for visual diff.

## Consequences

### Positive

- **Reproducibility**: synthetic-replay is deterministic given `--seed`
  and runs in seconds. Two PRs can be compared apples-to-apples without
  GPU access.
- **Honest reporting**: metrics call out exactly where Meridian wins
  (TTOT, output ITL variance) and acknowledge what we don't measure
  (raw throughput).
- **Cross-mode parity**: the same `BenchmarkReport` schema for both
  modes means a regression caught in synthetic-replay translates
  directly to expected behaviour under real-vllm.
- **Honest about failures**: the `output_critical_eviction_events`
  counter surfaces user-visible degradation immediately; the budget-force
  reason breakdown surfaces when the entropy probe is doing real work
  vs. just hitting the cap.

### Negative / risks

- **synthetic-replay does not exercise the CUDA kernels.** A regression
  in the kernels will not be caught by CI; the GPU job's
  `kernel_correctness` test is the line of defence there.
- **synthetic per-token latency is calibrated, not measured.** The
  default sleeps (6 µs / 18 µs for think / output) are approximations
  of bf16-Qwen3-on-H100 wall-clock times. Operators tuning a different
  hardware target should override via the `SyntheticDecoder` constructor.
- **Mix is synthetic.** Real ShareGPT / MATH-500 replays are
  available via `--workload sharegpt|math500` and use the offline
  HuggingFace dataset loader; they do not require a GPU.

### Neutral

- Report artefacts are JSON + Markdown only. Operators who want an
  HTML dashboard can render the JSON externally.

## Alternatives considered

### "Just use vLLM's benchmark harness"

Rejected. vLLM's harness reports throughput and TTFT, neither of which
shows Meridian's value. We would have to extend it to track TTOT and
phase-differentiated latencies — at which point we have already built
this harness, but with a tight coupling to vLLM's internal benchmark
abstractions.

### MLPerf-style reference workloads

Considered, deferred. MLPerf targets raw throughput and aggregate API-level
latency, not phase-differentiated metrics. A MLPerf-compatible reporting mode
could be added if there is downstream demand, but it does not fit the primary
signal Meridian optimises for.

### Per-token wall-clock tracing

Considered (would record every decode-step timestamp and reconstruct the
queue depths after the fact). Rejected because it generates GiB of trace
data per run for marginal additional signal — the aggregated percentiles
are enough to identify regressions and the OpenTelemetry spans give the
deep-dive when needed.

## References

- Playbook §5 — target metrics table.
- ADR-0001 — dual-queue scheduling, the design this benchmark validates.
- ADR-0004 — KV tier policy, the design that
  `output_critical_eviction_events` directly measures.
