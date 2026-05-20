# ADR-0006: Disaggregated KV transfer protocol

- **Status**: Accepted
- **Date**: 2026-05-23
- **Authors**: angelnicolasc
- **Reviewers**: angelnicolasc

## Context

The 2026 frontier of LLM serving infrastructure has settled on
*prefill-decode disaggregation*: prompt-processing (compute-bound) and
token-decoding (memory-bandwidth-bound) run on separate worker pools and
exchange KV blocks across a high-bandwidth fabric. NVIDIA NIXL (June
2026) is the CUDA-blessed reference; Mooncake (Moonshot, ASPLOS '25) is
the open-source protocol that the rest of the ecosystem implements.

Meridian's three-tier block manager already produces the exact signal a
disaggregated pool wants — `ThinkComplete` blocks at `ExitThink` are
*known-cold*. A scheduler with phase visibility is the natural producer
of well-batched, well-timed offload events: no other layer in the stack
knows that *right now*, this request just finished reasoning and won't
re-read its think KV.

The remaining question is *how* a phase-aware scheduler talks to the
fabric. We need a wire format and a set of trigger points that work
across NIXL and Mooncake (and any future fabric) without forcing
Meridian to depend on a specific runtime.

Constraints:

- The fabric layer cannot live inside `meridian-core` — that crate is
  `forbid(unsafe_code)` and has no transitive CUDA dependency. It must
  live in `meridian-kernels` behind a cargo feature so non-disagg
  deployments pay nothing.
- The wire format must be readable from Python and C++ NIXL agents,
  not just from Rust. NIXL's reference implementation is C/C++ + Python
  bindings; Mooncake's reference implementation is C++ + Python.
- A `meridian` deployment that wants disagg but doesn't have a libnixl
  runtime available must still be exercisable end-to-end — otherwise
  the integration is impossible to test without specialised hardware.

## Decision

Meridian defines a small versioned wire protocol (`MRDN` v1) for
disaggregated KV transfer, implements it behind the `nixl` cargo
feature of `meridian-kernels`, and exposes the trigger points through
the `BlockManager` trait.

### Wire format

```
+---------------+---------------+---------------+---------------+
| magic (4)     | version (4)   | body_len (4)  | tier (1)+pad(3)|
+---------------+---------------+---------------+---------------+
| checksum (16, Blake3-128)                                     |
+---------------+---------------+---------------+---------------+
| body (body_len bytes — opaque to Meridian; NIXL/Mooncake      |
| interpret as raw KV bytes)                                    |
+---------------+---------------+---------------+---------------+
```

- `magic = b"MRDN"` — fail-fast on misrouted payloads.
- `version = 1` — incremented on any breaking framing change.
  We commit to preserving v1 across all `0.x` releases.
- `tier` — the producer's tier label (`ThinkComplete | ThinkActive |
  OutputCritical`). The consumer may ingest into a different tier;
  the field exists for telemetry and for fabric-side admission policy.
- `checksum` — Blake3 of the body, truncated to 128 bits. Detects bit
  flips on RDMA and silent corruption inside fabric staging buffers.

The header is exactly 32 bytes. Body is opaque to the protocol — NIXL
treats it as a raw KV slab; Mooncake adds its own framing inside.

### Trigger points

- **`ExitThink`** — the producer's natural offload window. The
  scheduler batches the request's `ThinkComplete` blocks and pushes
  them to the fabric in a single shot, amortised by
  `disagg.offload_threshold_blocks`.
- **`OutputCritical` allocation pressure** — if the local pool is
  thrashing `OutputCritical` (a user-visible degradation event), the
  scheduler may pull blocks back from the fabric to satisfy
  allocations. Sprint 3 ships the hook; the pull policy is scheduled
  for Sprint 4 once we have a measured offload latency budget.

### Fabric trait

```rust
pub trait Fabric: Send + Sync + std::fmt::Debug {
    fn push(&self, payload: Vec<u8>) -> Result<u64>;
    fn pull(&self, handle: u64) -> Result<Vec<u8>>;
    fn label(&self) -> &'static str;
}
```

Implementations Sprint 3 ships:

- `SyntheticNixlFabric` — in-process keyed map, wire-format-identical
  to a real NIXL agent on the host side. Used for integration tests
  and for portfolio deployments where libnixl isn't reachable.
- Real libnixl FFI — gated on `nvidia-nixl-sys` becoming available on
  crates.io. The call sites already speak the protocol; switching
  swaps the `Fabric` implementation only.

Mooncake compatibility is achieved by writing a `MooncakeAdapter:
Fabric` that re-frames the v1 wire body inside Mooncake's transport.
The header survives unchanged, so an end-to-end conversation between a
Meridian producer and a Mooncake-only consumer is a one-adapter delta.

## Consequences

### Positive

- A single wire format covers two ecosystems (NIXL + Mooncake) and is
  forward-compatible because `version` is on the wire.
- Checksum-on-body catches silent corruption — the kind of incident
  that takes 48 h to diagnose in a heterogeneous fabric.
- The `BlockManager` trait gains `offload_block` / `ingest_block` /
  `block_location` as **non-breaking** additions: the default impls
  return `DisaggUnavailable` for offload/ingest and `Local` for
  location, so existing implementations keep working.
- Portfolio deployments can demonstrate the full disagg path without
  GPU hardware, via the synthetic fabric.

### Negative / risks

- Adding `Vec<u8>` allocations on every offload contradicts the
  zero-allocation discipline of the router hot path. Mitigation: the
  offload path runs at `ExitThink`, which is at most once per request
  — well off the per-token critical path.
- The 32-byte header is overhead for blocks that may be only a few KiB
  each. At a typical 16 KiB block this is 0.2% — acceptable.
- The synthetic fabric is *not* a substitute for real NIXL benchmarks.
  Anyone reading a synthetic-fabric A/B chart should know what they're
  reading. We label the fabric as `"nixl-synth"` in all telemetry to
  make this unambiguous.

### Neutral

- The `version` field commits us to a backward-compatibility plan once
  v2 ships. ADR-0007 documents the policy.

## Alternatives considered

**gRPC point-to-point**. Considered briefly. Rejected because every
KV transfer would pay HTTP/2 framing overhead on top of the actual
payload, and the standard NIXL/Mooncake clients don't speak gRPC. Our
producer would be the odd one out.

**RDMA-only, no framing**. Considered. Rejected because RDMA without
framing requires every consumer to know the producer's exact tier and
checksum convention out-of-band. The 32-byte header buys us
self-describing payloads at a 0.2% overhead — strictly worth it.

**Per-block streams over Mooncake without our own header**. Considered.
Rejected because we'd lose the `tier` and `version` fields, which makes
mixed-version cluster rollouts dangerous: a producer at v1 talking to
a consumer at v0 must fail fast, not silently mis-tier blocks.

## References

- NIXL technical brief, NVIDIA Developer Blog, March 2026.
- Mooncake: Trading More Storage for Less Computation — A KVCache-centric
  Architecture for Serving LLM Chatbot, ASPLOS '25.
- Blake3 specification, <https://github.com/BLAKE3-team/BLAKE3-specs>.
- Meridian playbook §6 (Disaggregation outlook).
- ADR-0004 (KV tier promotion policy) — describes why
  `ThinkComplete` is a natural offload candidate.
