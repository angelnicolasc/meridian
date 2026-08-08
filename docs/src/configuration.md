# Configuration

Meridian is configured through a single TOML file consumed by both the Rust
core (`meridian-core::config::MeridianConfig`) and the Python facade
(`meridian.config.MeridianConfig`). Both parsers agree on field names; a
round-trip test in `crates/meridian-core/tests/config_parse.rs` exercises
every field.

The fully-annotated example lives at
[`meridian.toml.example`](https://github.com/angelnicolasc/meridian/blob/main/meridian.toml.example).
Tune by overlay: keep the example as a reference and write a smaller
`meridian.toml` containing only fields that differ from their defaults.

## Validation

Both parsers reject unknown fields and out-of-range values. Cross-field
violations (e.g. `min_think_tokens >= max_think_tokens`) are also caught.
Errors carry the dotted field path and a human-readable message.

---

## `[scheduler]`

Dual-queue scheduling policy. See [ADR-0001](adr/0001-dual-queue-rationale.md).

### `think_tpot_budget_ms`

| Property | Value |
|----------|-------|
| Type | `f64` |
| Default | `80.0` |
| Unit | milliseconds per think-phase token |
| Valid range | `> 0` |

TPOT budget for think-phase tokens. The user does not see inter-token latency
during reasoning, so this can be set much higher than the output budget.
Setting it 4× the output budget gives the batcher room to pack a larger
effective batch during think. Raise if your GPU is underutilised during think;
lower if think-phase requests monopolise capacity at the expense of output.

### `output_tpot_budget_ms`

| Property | Value |
|----------|-------|
| Type | `f64` |
| Default | `20.0` |
| Unit | milliseconds per output-phase token |
| Valid range | `> 0` |

TPOT budget for output-phase tokens. This is the user-visible streaming
latency floor. 20 ms keeps streams fluid on a 30–50 tok/s display target.
Lower values produce tighter streams but reduce think-phase throughput.

### `think_batch_multiplier`

| Property | Value |
|----------|-------|
| Type | `f64` |
| Default | `2.5` |
| Unit | ×output batch token budget |
| Valid range | `>= 1.0` |

The think-phase batch can fill this multiple of the output-phase token budget.
2.5× is conservative — empirically stable across H100-class hardware with
MLA-aware allocation. Values above 3.5× risk output ITL variance spikes if
think requests fail to yield promptly. Monitor `meridian.queue_depth{queue=think}`.

### `max_think_tokens`

| Property | Value |
|----------|-------|
| Type | `u64` |
| Default | `32768` |
| Unit | tokens |
| Valid range | `> min_think_tokens` |

Hard cap on think tokens per request. Budget forcing fires unconditionally at
this limit regardless of entropy signals. 32 768 matches DeepSeek-R1's
documented maximum reasoning length and bounds the KV memory a single request
can monopolise.

### `min_think_tokens`

| Property | Value |
|----------|-------|
| Type | `u64` |
| Default | `512` |
| Unit | tokens |
| Valid range | `< max_think_tokens` |

No budget forcing is allowed before this many think tokens. EAT/RPDI signals
are noisy below 512 tokens; early forcing can prematurely terminate
short-but-correct reasoning chains.

---

## `[entropy]`

Entropy probe and convergence-detection thresholds.

### `enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

When `false`, the entropy probe is disabled and all budget forcing uses
`hard_cap` only (pure token-count limiting). Useful for A/B comparison
or when the CUDA kernel is not available.

### `ema_alpha`

| Property | Value |
|----------|-------|
| Type | `f64` |
| Default | `0.05` |
| Unit | dimensionless (EMA decay) |
| Valid range | `(0.0, 1.0]` |

EMA decay applied to all entropy signals. Smaller values give longer memory.
α = 0.05 → ~95% mass within the last ~60 samples. Long enough to smooth
single-token spikes; short enough to react within a reasoning chain.

### `rpdi_threshold`

| Property | Value |
|----------|-------|
| Type | `f64` |
| Default | `3.0` |
| Unit | ratio (local RPDI / global RPDI) |
| Valid range | `> 1.0` |

Overthinking is declared when `rpdi_local / rpdi_global > threshold`. The
value 3.0 is the empirical threshold from arXiv:2603.14251. Raise to be more
permissive (longer chains); lower to be more aggressive.

### `eat_ema_variance_threshold`

| Property | Value |
|----------|-------|
| Type | `f64` |
| Default | `0.001` |
| Unit | nats² |
| Valid range | `> 0.0` |

Convergence is declared when EAT EMA variance drops below this threshold. 0.001
is approximately the noise floor of EAT in steady state. Lower values defer
forcing; higher values fire earlier.

### `transition_entropy_threshold`

| Property | Value |
|----------|-------|
| Type | `f64` |
| Default | `2.5` |
| Unit | nats |
| Valid range | `> 0.0` |

A token counts as a "transition" for RPDI when its per-token entropy exceeds
this value. 2.5 nats ≈ effective branching factor of 12 — a genuine decision
point rather than low-entropy continuation.

### `eat_probe_interval_tokens`

| Property | Value |
|----------|-------|
| Type | `u32` |
| Default | `32` |
| Unit | tokens |
| Valid range | `>= 1` |

The EAT kernel runs every N think tokens. `1` = every token; higher values
trade signal latency for reduced kernel-launch overhead. 32 is the sweet spot
on H100-class hardware. Halving this on slower GPUs is safe.

---

## `[kv_memory]`

Phase-aware KV block manager policy.

### `aggressive_think_eviction`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `false` |

When `true`, `ThinkComplete` blocks are freed immediately on phase transition.
Leave `false` until cross-attention back-references are audited for your model;
some reasoning-parser pipelines re-attend over the think segment when generating
output and need those blocks resident.

### `think_phase_memory_fraction`

| Property | Value |
|----------|-------|
| Type | `f64` |
| Default | `0.40` |
| Unit | fraction of total KV budget |
| Valid range | `(0.0, 1.0)` |

Fraction of total KV budget reserved for think-phase blocks. 0.40 leaves 60%
for output-phase blocks, which accommodates `think_batch_multiplier = 2.5`
without crowding output. Raise for workloads with very long reasoning chains;
lower for workloads with long output sequences.

### `block_size_bytes`

| Property | Value |
|----------|-------|
| Type | `u64` |
| Default | `16384` (16 KiB) |
| Unit | bytes per KV block |
| Valid range | `> 0` |

Must match the actual vLLM block layout for your model. The canonical vLLM
layout is 16 KiB for bf16/fp16 KV at 16 tokens per block. MLA-aware models
can run smaller blocks.

### `capacity_bytes`

| Property | Value |
|----------|-------|
| Type | `u64` or `"auto"` |
| Default | `"auto"` |
| Unit | bytes |
| Valid range | `> 0` or `"auto"` |

Total KV memory budget. `"auto"` queries the device at startup and uses
85% of `torch.cuda.mem_get_info().total`. Pin an integer for deterministic
or multi-tenant deployments where you want to reserve GPU memory for
other workloads.

---

## `[disagg]`

Disaggregated KV transfer. Disabled by default. See
[ADR-0006](adr/0006-disagg-kv-transfer.md).

### `enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `false` |

Master switch. Leave `false` for single-node deployments.

### `fabric`

| Property | Value |
|----------|-------|
| Type | `"nixl"` \| `"mooncake"` \| `"none"` |
| Default | `"none"` |

Selects the disagg transport. `nixl` uses the NVIDIA NIXL library (requires
`cargo build --features nixl` and libnixl on the deploy host). `mooncake`
uses the Mooncake-compatible protocol adapter. `none` is only valid when
`enabled = false`.

### `offload_threshold_blocks`

| Property | Value |
|----------|-------|
| Type | `u32` |
| Default | `4` |
| Unit | KV blocks |
| Valid range | `>= 1` |

Minimum `ThinkComplete` blocks to accumulate before flushing to the fabric.
Larger values amortise transfer overhead; smaller values reduce latency.

---

## `[speculation]`

Phase-conditioned speculative decoding. See
[ADR-0009](adr/0009-phase-conditioned-speculation.md) and the
[research note](notes/phase-conditioned-speculation.md).

**Read before enabling.** The hypothesis this section exists to act on — that
draft acceptance is lower during the think phase, because DeepSpec's released
Qwen3 drafters were trained only on non-thinking-mode generations — has **not
been measured**. The hook is therefore deliberately half-inert: without
`[speculation.acceptance_prior]` it can only ever *reduce* the baseline draft
depth, and only where the entropy probe proves a deeper draft cannot pay off.
Only a measured prior unlocks upward adjustment and phase conditioning.

### `enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `false` |

Master switch. When `false` the serving stack never consults the hook.

### `baseline_proposal_len`

| Property | Value |
|----------|-------|
| Type | `u32` |
| Default | `7` |
| Unit | tokens |

Draft depth the serving stack would use without the hook, and the ceiling on
what an uncalibrated hook may return. Defaults to 7 because DeepSpec's released
DSpark Qwen3 checkpoints are `block7`.

### `min_proposal_len` / `max_proposal_len`

| Property | Value |
|----------|-------|
| Type | `u32` |
| Default | `1` / `7` |
| Valid range | `min <= baseline <= max` |

Bounds on the recommended draft depth. `max_proposal_len` must not exceed the
drafter's block size — proposing past it is not something the drafter can do.

### `vocab_size`

| Property | Value |
|----------|-------|
| Type | `u32` |
| Default | `151936` (Qwen3) |
| Valid range | `>= 2` |

Target-model vocabulary size. Required for the entropy-derived acceptance
ceiling, which depends on `V` as well as on entropy.

### `use_entropy_ceiling`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

Whether to derive a draft-depth ceiling from the entropy probe. Setting this
`false` makes an uncalibrated hook a strict no-op — the most conservative
posture available.

### `draft_token_us` / `verify_fixed_us` / `verify_token_us`

| Property | Value |
|----------|-------|
| Type | `f32` |
| Default | `40.0` / `900.0` / `25.0` |
| Unit | microseconds |
| Valid range | `>= 0`; `verify_fixed_us > 0` |

Cycle cost model driving the throughput objective
`Θ(γ) = τ(a, γ) / (draft_token_us·γ + verify_fixed_us + verify_token_us·γ)`.

**The defaults are placeholders.** They produce sane relative ordering and are
not a measurement of any hardware. Profile your deployment and replace them; if
`meridian.speculation.proposal_len` sits pinned at `min_proposal_len`, these are
the first thing to check.

### `[speculation.acceptance_prior]`

| Property | Value |
|----------|-------|
| Type | table |
| Default | absent |

Measured per-phase acceptance rates. Absent until a Phase 1 run exists — see the
[protocol](notes/phase-1-protocol.md).

Every field is required: `think`, `output`, `harness`, `draft_checkpoint`,
`target_model`, `thinking_mode`, `recorded_on`. There is deliberately **no way
to express acceptance rates without naming the run that produced them**, so
phase conditioning cannot be switched on by a hunch. Partial or blank provenance
is rejected by both parsers.

```toml
[speculation.acceptance_prior]
think            = 0.42
output           = 0.88
harness          = "DeepSpec@<commit-sha>"
draft_checkpoint = "deepseek-ai/dspark_qwen3_4b_block7"
target_model     = "Qwen/Qwen3-4B"
thinking_mode    = true
recorded_on      = "2026-08-07"
```

---

## `[model.<name>]`

Per-model token-boundary configuration. One `[model.*]` table per model
served. The phase router watches for boundary token IDs in the decoded stream.

See [`models/*.toml`](https://github.com/angelnicolasc/meridian/tree/main/models) for
the vetted token IDs for supported models.

### `think_start_token_ids`

| Property | Value |
|----------|-------|
| Type | `[u32]` |

Token IDs that mark the start of a reasoning chain. Model- and
tokenizer-specific. If empty, the router never enters think-phase for this
model.

### `think_end_token_ids`

| Property | Value |
|----------|-------|
| Type | `[u32]` |

Token IDs that mark the end of a reasoning chain. When the decoded stream
contains any of these IDs, the router emits `ExitThink`.

### `reasoning_parser`

| Property | Value |
|----------|-------|
| Type | `string` |
| Values | `"deepseek_r1"`, `"qwen3"`, `"granite"`, `"anthropic"` |

Selects the reasoning-chain parser. Different models structure their
think-output boundary differently; the parser handles model-specific
normalization.

### `supports_think_disable`

| Property | Value |
|----------|-------|
| Type | `bool` |

Whether the model supports a `/no_think` directive to suppress the reasoning
phase entirely. When `true` and the prompt contains the directive, the router
stays in output-phase for the entire request.
