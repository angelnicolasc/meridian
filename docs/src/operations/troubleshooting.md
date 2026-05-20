# Troubleshooting

Each entry follows runbook format: **Symptom → Likely cause → How to verify → Immediate mitigation → Longer-term fix**.

---

## Output streams stutter under load

**Symptom**: users report visible gaps in the output token stream; output ITL P99 spikes.

**Likely cause**: `OutputCritical` KV blocks are being evicted under memory pressure.

**How to verify**:
```promql
rate(meridian.output_critical_eviction[1m]) > 0
```
Any non-zero rate confirms the block manager is evicting user-visible KV.

**Immediate mitigation**: reduce load. Lower the arrival rate or raise `kv_memory.capacity_bytes`
if headroom exists.

**Longer-term fix** (pick one or more):
1. Lower `kv_memory.think_phase_memory_fraction` (e.g. 0.40 → 0.30) to give output more room.
2. Lower `scheduler.think_batch_multiplier` to reduce think-phase KV pressure.
3. Lower `scheduler.max_think_tokens` so individual reasoning chains release blocks sooner.

---

## Budget forcing never fires; chains always hit hard cap

**Symptom**: `meridian.budget_force_reason{reason=hard_cap}` increments constantly;
`converged` and `overthinking` stay at zero.

**Likely cause**: the entropy probe is not detecting convergence. Either the probe is
disabled, the thresholds are too tight, or the model is being asked to reason on prompts
that produce genuinely non-converging chains.

**How to verify**:
1. Check `meridian.toml`: confirm `entropy.enabled = true`.
2. Capture a sample EAT trace: add `logging.level = "debug"` temporarily and inspect
   EAT EMA values in the logs for a known-good reasoning prompt.

**Immediate mitigation**: none — hard-cap forcing is safe, just less adaptive.

**Longer-term fix**:
- If probe is disabled: set `entropy.enabled = true`.
- If thresholds are too tight: relax `eat_ema_variance_threshold` (e.g. 0.001 → 0.005)
  or lower `rpdi_threshold` (e.g. 3.0 → 2.0).
- If prompts are genuinely non-converging: lower `max_think_tokens` to bound KV cost.

---

## Phase router shows runaway tracked-requests gauge

**Symptom**: `meridian.phase_router.tracked_requests` grows monotonically without
a corresponding growth in active requests.

**Likely cause**: completed requests are not being reaped from the router. Either
`post_step` is not receiving EOS events, or `reap_stale_older_than` is not being
called.

**How to verify**:
1. Confirm the vLLM plugin's `post_step` is wired to the engine's step callback.
2. Check the reap interval in the plugin config — default 60 s. If request lifetime
   is shorter than 60 s on average, the reaper may be lagging.

**Immediate mitigation**: call `plugin.reap_stale_now()` manually via the debug CLI
if available.

**Longer-term fix**: reduce the reap period in `vllm_plugin.py` or ensure `post_step`
receives every EOS signal from the vLLM worker.

---

## CUDA kernel returns `Unavailable`

**Symptom**: logs show `KernelError::Unavailable`; entropy probe silently falls back
to CPU; `meridian.budget_force_reason{reason=hard_cap}` is the only firing signal.

**Likely cause**: the `meridian-kernels` crate was built without `--features cuda`,
or `libcudart.so` is not on the dynamic linker path at runtime.

**How to verify**:
```bash
# Check the linked backend.
python -c "from meridian._backends.cuda import cuda_available; print(cuda_available())"
# Should print True on a CUDA-capable host. Prints False if the stub was linked.
```

**Immediate mitigation**: the CPU fallback is correct — entropy values are identical.
The only cost is CPU cycles and higher `schedule_batch.duration_ns` latency.

**Longer-term fix**:
1. Rebuild with `cargo build -p meridian-kernels --features cuda`.
2. Verify `libcudart.so` is on `LD_LIBRARY_PATH` or in `/usr/local/cuda/lib64`.
3. Run `maturin develop --release -m crates/meridian-python/Cargo.toml` to regenerate
   the Python extension.

---

## Disagg offload not firing

**Symptom**: `meridian.disagg.blocks_offloaded` counter never increments after
`[disagg] enabled = true` is set.

**Likely cause**: the fabric is not reachable, the `offload_threshold_blocks` has
not been reached, or the NIXL feature was not compiled in.

**How to verify**:
1. Confirm `cargo build -p meridian-kernels --features nixl` succeeds.
2. Confirm `config.disagg.fabric != "none"`.
3. Check whether `ThinkComplete` block count per step is below `offload_threshold_blocks`.

**Immediate mitigation**: lower `offload_threshold_blocks` to `1` temporarily to
force immediate offload on every `ExitThink`.

**Longer-term fix**: verify fabric connectivity (NIXL service running, network path
open) and restore `offload_threshold_blocks` to the production value.

---

## Plugin does not intercept schedule_batch

**Symptom**: no Meridian metrics appear; vLLM appears to schedule normally without
phase-awareness.

**Likely cause**: `MeridianSchedulerPlugin.attach()` was never called, or the plugin
was attached to a different scheduler instance than the one the engine uses.

**How to verify**:
```python
print(type(engine.scheduler))
# Should print: <class 'meridian.vllm_plugin.MeridianSchedulerPlugin'>
# Not: <class 'vllm.core.scheduler.Scheduler'>
```

**Immediate mitigation**: call `plugin.attach()` explicitly after engine construction.

**Longer-term fix**: verify the plugin initialisation order in the serving entrypoint.
The plugin must be attached after `AsyncLLMEngine` is fully constructed but before
the first request is submitted.
