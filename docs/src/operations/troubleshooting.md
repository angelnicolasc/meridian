# Troubleshooting

## Symptom: output streams stutter under load

Check `meridian.output_critical_eviction`. If it is incrementing, the
block manager is being forced to evict user-visible KV blocks. Either:

1. Lower `kv_memory.think_phase_memory_fraction` to give output more room.
2. Lower `scheduler.think_batch_multiplier`.
3. Lower `scheduler.max_think_tokens` so individual reasoning chains
   release blocks sooner.

## Symptom: budget force never fires; chains hit hard cap

Check `meridian.budget_force_reason`:

- If only `hard_cap` increments, the entropy probe is not converging.
  Inspect a sample: capture EAT EMA + variance and verify the model is
  actually being asked to reason (some prompts produce trivial reasoning).
- If `entropy.enabled = false`, you are running in count-only mode — every
  termination is `hard_cap` by design.

## Symptom: phase router shows runaway tracked-requests gauge

`meridian.phase_router.tracked_requests` grows monotonically when a
component fails to call `PhaseRouter::reap` on request completion.
Confirm the vLLM plugin's `post_step` is receiving the EOS event.

## Symptom: CUDA kernel returns `Unavailable`

The kernel crate built without the `cuda` feature, or the CUDA runtime
shared library is missing on the deploy host. Rebuild with
`cargo build -p meridian-kernels --features cuda` and verify
`libcudart.so` is on the loader path.
