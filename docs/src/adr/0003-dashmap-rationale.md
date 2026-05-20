# ADR-0003: `DashMap` for per-request phase state

- **Status**: Accepted
- **Date**: 2026-05-20
- **Authors**: angelnicolasc
- **Reviewers**: sole-maintainer decision record

## Context

The `PhaseRouter` mutates per-request state (`ThinkPhase`) on every decoded
token. A single decode worker processes a continuous-batch step that can
touch dozens of requests; in a tensor-parallel deployment, multiple worker
threads may invoke `on_token` for *different* requests in the same wall-clock
window. We need:

- O(1) lookup keyed by `req_id`.
- Concurrent mutation of *different* keys without serialising the whole map.
- Cheap clone / sharing — the router is shared across the scheduler, the
  block manager touch path, and the vLLM plugin.
- No GC, no allocations on the hot path.

Candidates evaluated:

1. `parking_lot::RwLock<HashMap<u64, ThinkPhase>>`
2. `DashMap<u64, ThinkPhase>` (sharded `HashMap` behind per-shard `RwLock`)
3. `papaya::HashMap<u64, ThinkPhase>` (lock-free, June 2025)
4. `scc::HashMap<u64, ThinkPhase>` (lock-free, sharded)

## Decision

**`DashMap` 6.x.** It is the boring, well-audited choice that hits the
performance floor we need without introducing a less-battle-tested
dependency on the hot path.

## Consequences

### Positive

- **Sharded locking.** Operations on different `req_id`s do not contend.
- **Mature API.** `get_mut`, `entry`, `remove` cover every access pattern in
  `PhaseRouter`. No need to invent abstractions on top.
- **No `unsafe`.** Internally backed by `parking_lot`; we keep the
  `#![forbid(unsafe_code)]` invariant in `meridian-core` intact.
- **Crates.io top-100.** Wide deployment, frequent security audits, stable
  semver.

### Negative / risks

- **Sharded lock is not truly lock-free.** Under extreme contention (a
  single shard absorbing many requests because of hash skew), throughput
  degrades. Mitigation: monitor `meridian.phase_router.tracked_requests`;
  if the gauge approaches `n_shards × shard_capacity` (default ~512 per
  shard), revisit with `papaya` or shard-aware partitioning. In our
  workload (~1k concurrent requests max per worker) we are far from this
  ceiling.
- **`AHasher` by default.** Good for our integer keys; not a downside,
  but worth noting that we benefit from non-cryptographic hashing here.

### Neutral

- Memory per entry is slightly higher than a flat `HashMap` because of the
  shard metadata. Negligible (~64 bytes overhead total).

## Alternatives considered

### `RwLock<HashMap<...>>`

Rejected: every `on_token` call needs a write lock to bump
`tokens_so_far`, so the `RwLock` collapses to a `Mutex` in practice.
Single-shard serialisation across all requests is unacceptable.

### `papaya::HashMap`

Watched, not adopted. Genuinely lock-free, with better tail latency under
contention than `DashMap`. Adoption deferred until: (a) it stabilises a
1.0 API, and (b) we can demonstrate a workload where `DashMap` is the
bottleneck. Tracked as future work in the DEVLOG.

### `scc::HashMap`

Comparable to `papaya` but more API surface; same deferral rationale.

## References

- DashMap source: <https://github.com/xacrimon/dashmap>.
- The [phase_router bench](https://github.com/angelnicolasc/meridian/blob/main/crates/meridian-core/benches/phase_router.rs)
  measures the steady-state cost of `on_token` under no contention.
