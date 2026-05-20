# Benchmarks

The Meridian benchmark harness measures phase-differentiated latency under
controlled load. It operates in two modes — one CI-friendly and GPU-free, one
requiring a real NVIDIA GPU — and produces identically-shaped JSON + Markdown
reports so any two runs can be diffed directly.

## Quick start

```bash
# CI-friendly: no GPU, no vLLM. Synthetic decoder loop. Finishes in <30 s.
uv --project python run python -m benchmarks.meridian_bench synthetic-replay \
    --duration-s 30 --arrival-rate 8 --reasoning-ratio 0.4 \
    --out-dir bench-out/

# A/B against stock priority-weight baseline (same synthetic mode, no GPU).
uv --project python run python -m benchmarks.meridian_bench synthetic-replay \
    --baseline stock \
    --duration-s 30 --arrival-rate 8 \
    --out-dir bench-out/

# Real-dataset workload (downloads ShareGPT from HuggingFace, cached locally).
uv --project python run python -m benchmarks.meridian_bench synthetic-replay \
    --workload sharegpt \
    --duration-s 60 --arrival-rate 8 \
    --out-dir bench-out/

# GPU-required: drives a real AsyncLLMEngine via the Meridian plugin.
uv --project python run python -m benchmarks.meridian_bench real-vllm \
    --model Qwen/Qwen2.5-0.5B --duration-s 60 --arrival-rate 4 \
    --baseline stock \
    --out-dir bench-out/
```

## Workloads

| Flag            | Source                                             | GPU required |
|-----------------|----------------------------------------------------|--------------|
| `synthetic`     | Deterministic LCG-driven decoder; no model         | No           |
| `sharegpt`      | `anon8231489123/ShareGPT_Vicuna_unfiltered` subset | No (offline replay) |
| `math500`       | `HuggingFaceH4/MATH-500`; all requests `kind=reasoning` | No (offline replay) |
| `mix`           | 50/50 split of `sharegpt` + `math500`              | No (offline replay) |

Datasets are downloaded once and cached at `~/.cache/meridian/datasets/`.

## Output artefacts

Both modes write to `--out-dir`:

| File          | Contents                                                        |
|---------------|-----------------------------------------------------------------|
| `report.json` | Full structured report — all metric percentiles, metadata, seed |
| `report.md`   | Markdown summary suitable for PR comments and direct reading    |

When `--baseline stock` is set, the report includes an A/B comparison table
showing `Stock | Meridian | Δ (%)` for every metric.

## Metric catalog

| Name                           | Definition                                                           |
|--------------------------------|----------------------------------------------------------------------|
| **TTFT P50/P95**               | Time-to-first-token. Prefill latency + first decoded token.          |
| **TTOT P50/P95**               | Time from `</think>` to first user-visible output token.             |
| **Output ITL P50/P95/P99**     | Inter-token latency during output phase (streaming jitter).          |
| **Think tokens avg/P95**       | Distribution of reasoning-chain length per request.                  |
| **Output tokens avg**          | Mean output token count per request.                                 |
| **Budget forced %**            | Fraction of reasoning requests where the router injected `</think>`. |
| **Force reason breakdown**     | `converged` / `overthinking` / `hard_cap` per-bucket fractions.      |
| **OutputCritical evictions**   | KV pressure events that reached the user-visible tier.               |

See [`benchmarks/metrics.py`](metrics.py) for the exact serialised shape.

## What this harness is, and what it is not

**It is:**
- A phase-differentiated latency regression suite. It catches regressions in
  TTOT and output-ITL, the metrics Meridian was built to protect.
- Reproducible: synthetic-replay mode is fully deterministic given `--seed`.
  Two PRs can be diffed apples-to-apples against identical seeds.
- An honest A/B harness: when `--baseline stock` is given, both paths run the
  same synthetic workload. The stock baseline implements a priority-weight
  single-queue scheduler equivalent to vLLM ≤0.8.

**It is not:**
- A raw-throughput benchmark. vLLM's own harness reports `tokens/sec/GPU`;
  that metric does not differentiate Meridian's phase-aware scheduling.
  Operators who need a throughput number should run vLLM's `benchmark_throughput.py`.
- An accuracy benchmark. Budget forcing can in principle shorten reasoning
  chains on hard problems. Accuracy measurement requires a separate ground-truth
  evaluation suite; this harness does not provide one.
- A production-load replay. The synthetic decoder and offline dataset replay
  do not capture memory pressure, multi-tenant contention, or model-dependent
  decode variance. Treat synthetic results as directional, not absolute.

## Test environment disclosure

Reproducible synthetic results in this repo were collected with:

- `--seed 42` (default), deterministic LCG workload generator
- Arrival rate 8 req/s, duration 30 s, reasoning ratio 0.4

Real-vLLM results are hardware-dependent. The CUDA workflow targets an NVIDIA
GPU runner; results vary with GPU model, driver version, and memory state.
Disclose your hardware when sharing or comparing numbers.

## Running the unit tests

```bash
# No GPU required. Tests the baseline scheduler logic and A/B report maths.
uv --project python run pytest benchmarks/tests -q
```
