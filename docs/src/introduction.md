# Introduction

Meridian is an inference-time compute scheduler for reasoning-model serving.
It treats **think-decode** and **output-decode** as separate scheduling
domains with separate SLOs, separate KV eviction priorities, and real-time
entropy-driven budget control.

## Why this exists

Reasoning models (DeepSeek-R1, Qwen3, Qwen2.5, o3) emit two structurally
different token sequences within a single request:

```text
[prompt] → <think> ... N reasoning tokens ... </think> → [output tokens]
```

These two phases have **opposite latency profiles**:

| Phase         | User-visible latency tolerance | Throughput importance | Correct SLO target |
|---------------|-------------------------------|----------------------|-------------------|
| Think-decode  | High — user waits regardless  | Critical (cost driver) | TPOT-relaxed      |
| Output-decode | Zero — streaming experience   | Secondary            | TTOT-strict       |

Standard continuous-batching schedulers (including vLLM's default) process all
decode tokens — thinking and output — from the same priority queue with the
same TPOT target. Meridian is the scheduling layer that knows the difference.

## What Meridian does

1. **Dual-queue scheduling.** Output-phase requests have absolute priority.
   Think-phase requests fill remaining capacity with a larger effective batch
   token budget.
2. **Phase-aware KV block manager.** Three-tier eviction:
   `ThinkComplete` < `ThinkActive` < `OutputCritical`. Blocks from a
   completed reasoning chain are demoted the moment `</think>` is emitted.
3. **Entropy-driven budget forcing.** EAT (`arXiv:2509.26522`) and RPDI
   (`arXiv:2603.14251`) signals inject `</think>` only when the model itself
   is signalling convergence or overthinking — not on a static token counter.
4. **Drop-in vLLM plugin.** No fork required. Wraps the existing scheduler
   via the plugin interface; exposes Prometheus + OpenTelemetry telemetry.
5. **Disagg KV transfer.** `offload_block` / `ingest_block` hooks on the
   block manager support prefill-decode disaggregation fabrics (NIXL,
   Mooncake-compatible). Documented in [ADR-0006](adr/0006-disagg-kv-transfer.md).

## Scope and assumptions

- **In scope**: latency-differentiated scheduling for reasoning models served
  via vLLM on a single node or a disaggregated prefill-decode topology.
- **Out of scope**: model training, quantisation, speculative decoding, or
  serving systems other than vLLM. See [Non-goals](non-goals.md).
- **Assumed**: the serving stack emits per-request token IDs to a hook point
  that Meridian can intercept (the vLLM plugin interface satisfies this).
- **Assumed**: think/output phase boundaries are detectable from the decoded
  token stream via model-specific boundary token IDs (configurable per model
  in `meridian.toml`).

## Threats to validity

- **Synthetic benchmark results are directional, not absolute.** The
  synthetic-replay harness simulates latency with a calibrated decoder; it
  does not model multi-tenant memory pressure or real CUDA kernel variance.
  Results should be reproduced on target hardware before drawing conclusions.
- **Budget forcing can affect accuracy.** Injecting `</think>` early may
  shorten correct reasoning chains. Meridian fires forcing only on entropy
  convergence signals, but the threshold is a tunable heuristic, not a
  guarantee. Accuracy measurement is the operator's responsibility.
- **Phase detection depends on token IDs.** If a model tokenises
  `<think>` / `</think>` differently than the configured boundary IDs,
  Meridian treats the entire request as single-phase. The `models/*.toml`
  files in the repo carry vetted IDs for supported models.

## How to read this book

- **[Architecture](architecture.md)** — component map, per-component contracts,
  failure modes, and observability hooks.
- **[ADRs](adr/README.md)** — every architectural decision with the rejected
  alternatives. Read these alongside the code to understand the *why*.
- **[Configuration](configuration.md)** — every `meridian.toml` field with
  type, default, valid range, and tuning guidance.
- **[API reference](api/rust.md)** — Rust and Python surfaces with lifecycle
  and concurrency notes.
- **[Operations](operations/metrics.md)** — metrics catalogue, alerting
  recommendations, troubleshooting runbooks, and benchmark methodology.
- **[Non-goals](non-goals.md)** — what Meridian explicitly does not do.
- **[Glossary](glossary.md)** — definitions for TTFT, TTOT, ITL, EAT, RPDI,
  KV, disagg, NIXL, and other terms used throughout.
