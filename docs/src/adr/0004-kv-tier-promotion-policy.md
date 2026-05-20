# ADR-0004: KV tier promotion is one-way; demotion is the only direction

- **Status**: Accepted
- **Date**: 2026-05-20
- **Authors**: angelnicolasc
- **Reviewers**: sole-maintainer decision record

## Context

The block manager maintains three tiers (`ThinkActive`, `ThinkComplete`,
`OutputCritical`). Tier transitions are emitted by the scheduler on phase
events. We had to decide whether a block can *promote* (e.g. a
`ThinkComplete` block re-attended during output generation gets restored
to `ThinkActive` or even `OutputCritical`).

Two camps:

1. **Bidirectional**: any block currently accessed gets promoted to the
   highest applicable tier.
2. **One-way demotion**: blocks only move *down* the eviction order
   (`OutputCritical → … → freed`). Promotion is explicitly disallowed.

## Decision

**One-way demotion.** A block's tier is set at allocation time
(via `BlockTier::ThinkActive` or `BlockTier::OutputCritical`) and can only
move *toward* eviction:

- `ThinkActive → ThinkComplete` via `demote_think_blocks`.
- Any tier → freed via `evict_for` or `free`.

There is no API to move a `ThinkComplete` block back to `ThinkActive`, or
a `ThinkActive` block to `OutputCritical`.

## Consequences

### Positive

- **Reasoning about eviction stability becomes trivial.** A block's tier
  monotonically decreases. Cross-attention back-references over a
  reasoning span cannot accidentally "rescue" blocks the scheduler has
  already decided are evictable — operators can reason about KV pressure
  without tracking promotion races.
- **The block manager API is smaller.** No `promote()` method to test, no
  invariant to enforce ("you can only promote within the same request").
- **Aligns with the playbook intent.** `kv_memory.aggressive_think_eviction`
  is a one-way knob: think blocks either survive demotion (default) or
  are freed immediately (`aggressive`). Promotion would make this knob
  semantically incoherent.

### Negative / risks

- **A reasoning model that re-attends over its own think span during
  output generation pays the eviction cost twice.** Cross-attention reads
  on a `ThinkComplete` block bring the block into the GPU's L1/L2 cache
  but do not promote its eviction tier. If that block is then evicted by
  memory pressure, the next cross-attention read forces a recompute.
  Mitigation: keep `kv_memory.aggressive_think_eviction = false` so blocks
  stay resident as `ThinkComplete` until pressure actually demands their
  eviction.
- **No way to mark "this block is hot, please keep it"** beyond keeping
  it in the lowest tier it was admitted to. The LRU within a tier is the
  only signal of recency.

### Neutral

- The `touch()` API exists to update LRU position within a tier, not to
  promote across tiers. This is explicit in the trait documentation.

## Alternatives considered

### Bidirectional with `promote_block(block_id, tier)`

Rejected because:

- Adds a new invariant to police (can a `ThinkComplete` of request A
  promote to `OutputCritical` of request B? Obviously not, but the API
  must enforce that).
- Promotion races with eviction: a block that the eviction iterator has
  selected as next victim might be promoted mid-eviction. Resolving this
  needs either a lock around the whole eviction loop (kills throughput)
  or a generation counter (more complexity).
- The cross-attention rescue use case is real but rare; the simpler
  fallback (operator tunes `think_phase_memory_fraction`) handles it
  without architectural complexity.

### Implicit promotion on `touch()`

Rejected: `touch()` is called on the hot path for every cache hit. Doing
tier promotion there would dramatically slow the common case to optimise
a rare one.

## References

- ADR-0001 — dual-queue scheduling sits alongside this tier policy.
- Playbook §3.4 — original three-tier eviction design.
