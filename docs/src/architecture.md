# Architecture

```text
Incoming requests
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    Meridian Daemon                        │
│                                                           │
│  ┌──────────────┐    ┌────────────────────────────────┐  │
│  │   Prefill    │───▶│        Phase Router             │  │
│  │   Executor   │    │  (token stream state machine)   │  │
│  └──────────────┘    └───────────┬─────────────────────┘  │
│                                  │                         │
│                    ┌─────────────┴──────────────┐          │
│                    │                            │          │
│        ┌───────────▼──────────┐  ┌─────────────▼───────┐  │
│        │   Think-Decode       │  │   Output-Decode      │  │
│        │   Scheduler          │  │   Scheduler          │  │
│        │                      │  │                      │  │
│        │  TPOT: relaxed       │  │  TTOT: strict SLO    │  │
│        │  Batch: 2.5× larger  │  │  Batch: standard     │  │
│        │  Entropy probe live  │  │  Stream priority     │  │
│        │  Budget force ready  │  │                      │  │
│        └──────────┬───────────┘  └────────┬─────────────┘  │
│                   │                       │                 │
│                   └──────────┬────────────┘                 │
│                              │                              │
│               ┌──────────────▼─────────────┐               │
│               │   Phase-Aware KV Block Mgr  │               │
│               │                             │               │
│               │  Tier 0: ThinkComplete      │               │
│               │  Tier 1: ThinkActive        │               │
│               │  Tier 2: OutputCritical     │               │
│               └─────────────────────────────┘               │
└────────────────────────────┬────────────────────────────────┘
                             │
                        vLLM worker
                    (decode kernel, KV store)
```

## Components

### Phase Router

The only component that touches every generated token. Maintains
per-request `ThinkPhase` state in a lock-free map; emits `PhaseEvent`s the
scheduler consumes. **Hot path must be O(1) and zero-allocation in the
common case.**

Source: [`crates/meridian-core/src/phase_router.rs`](https://github.com/angelnicolasc/meridian/blob/main/crates/meridian-core/src/phase_router.rs).

### Dual-Queue Scheduler

Two independent queues sharing the same GPU workers. Output drains first,
think fills remaining capacity. See [ADR-0001](adr/0001-dual-queue-rationale.md)
for the design alternative this rejects.

Source: [`crates/meridian-core/src/scheduler.rs`](https://github.com/angelnicolasc/meridian/blob/main/crates/meridian-core/src/scheduler.rs).

### Phase-Aware Block Manager

Three-tier KV eviction priority. `ThinkComplete` blocks — those from a
request whose `</think>` has already been emitted — are evicted first.
`OutputCritical` blocks (the ones backing the user-visible stream) only
under severe pressure, and only after triggering an alert-worthy counter.

Source: [`crates/meridian-core/src/block_manager.rs`](https://github.com/angelnicolasc/meridian/blob/main/crates/meridian-core/src/block_manager.rs).

### Entropy Probe

Runs on a dedicated secondary CUDA stream, reading logits after each
forward pass and computing token entropy + EAT signal without stalling the
generation stream. The Python facade exposes a backend-switchable
interface so tests run against the NumPy reference and the CUDA kernel
must agree with it within `atol=1e-5`.

Sources:
- [`crates/meridian-kernels/`](https://github.com/angelnicolasc/meridian/tree/main/crates/meridian-kernels) (CUDA + FFI).
- [`python/meridian/entropy_probe.py`](https://github.com/angelnicolasc/meridian/blob/main/python/meridian/entropy_probe.py) (facade + EMA state).
- [`python/meridian/_backends/`](https://github.com/angelnicolasc/meridian/tree/main/python/meridian/_backends) (CPU & CUDA backends).

### vLLM Plugin

Drop-in scheduler wrapper. Requires no fork; works against any vLLM
`Scheduler` via attribute delegation. Sprint 0 ships the integration
boundary transparent (no reordering); Sprint 1 enables dispatch.

Source: [`python/meridian/vllm_plugin.py`](https://github.com/angelnicolasc/meridian/blob/main/python/meridian/vllm_plugin.py).
