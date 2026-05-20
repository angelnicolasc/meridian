# Introduction

Meridian is an inference-time compute scheduler for reasoning-model serving.
It treats **think-decode** and **output-decode** as separate scheduling
domains with separate SLOs, separate KV eviction priorities, and real-time
entropy-driven budget control.

## Why does this exist?

Reasoning models (DeepSeek-R1, Qwen3, Claude Opus 4.x, o3) emit two
structurally different token sequences within a single request:

```text
[prompt] → <think> ... N reasoning tokens ... </think> → [output tokens]
```

These two phases have **opposite latency profiles**:

| Phase         | User-visible latency tolerance | Throughput importance | Correct SLO target |
|---------------|-------------------------------|----------------------|-------------------|
| Think-decode  | High — user waits regardless  | Critical (cost driver) | TPOT-relaxed      |
| Output-decode | Zero — streaming experience   | Secondary            | TTOT-strict       |

No serving system today exploits this asymmetry. vLLM's continuous batching
loop processes all decode tokens — thinking and output — from the same
priority queue with the same TPOT target. Meridian is the scheduling layer
that knows the difference.

## What does Meridian do?

1. **Dual-queue scheduling.** Output-phase requests have absolute priority.
   Think-phase requests fill remaining capacity with a larger effective batch
   token budget.
2. **Phase-aware KV block manager.** Three-tier eviction:
   `ThinkComplete` < `ThinkActive` < `OutputCritical`. Blocks that belonged
   to a request's reasoning phase are demoted the moment `</think>` is emitted.
3. **Entropy-driven budget forcing.** EAT (`arXiv:2509.26522`) and RPDI
   (`arXiv:2603.14251`) signals fire `</think>` injection only when the
   model itself is signalling convergence or confusion — not on a static timer.
4. **Drop-in vLLM plugin.** No fork. Wraps the existing scheduler via the
   plugin interface and exposes Prometheus + OpenTelemetry telemetry.

## How to read this book

- The [Architecture](architecture.md) chapter is the component map plus the
  per-component contracts.
- The [ADR](adr/README.md) section is where you'll find *why* a decision was
  made — these are the documents we want auditors to read alongside the code.
- The API reference is generated from rustdoc + Python docstrings.
- Operational concerns (metrics, alerting, troubleshooting) live under
  [Operations](operations/metrics.md).

For the full specification with implementation-level detail, see
[`Research/MERIDIAN-playbook.md`](https://github.com/angelnicolasc/meridian/blob/main/Research/MERIDIAN-playbook.md).
