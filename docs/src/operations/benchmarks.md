# Benchmarks

The benchmark harness lives at [`benchmarks/`](https://github.com/angelnicolasc/meridian/tree/main/benchmarks).
The methodology behind metric choice is recorded in
[ADR-0005](../adr/0005-benchmark-methodology.md).

## Quick start

```bash
# CI-friendly: no GPU, no vLLM. Drives native Meridian components over a
# synthetic decoder loop. Finishes in seconds.
uv --project python run python -m benchmarks.meridian_bench synthetic-replay \
    --duration-s 30 --arrival-rate 8 --reasoning-ratio 0.4 \
    --out-dir bench-out/

# GPU-required: drives a real AsyncLLMEngine via the Meridian plugin.
uv --project python run python -m benchmarks.meridian_bench real-vllm \
    --model Qwen/Qwen2.5-0.5B --duration-s 30 --arrival-rate 4 \
    --out-dir bench-out/
```

Both modes produce identically-shaped artefacts in `--out-dir`:

- `report.json` — full structured report, diffable.
- `report.md` — Markdown summary suitable for PR comments.

## Metric catalog

| Name                   | Definition                                                   |
|------------------------|--------------------------------------------------------------|
| **TTFT P50/P95**       | Time-to-first-token. Prefill + first decoded token.          |
| **TTOT P50/P95**       | Time from `</think>` emission to the first user-visible token. |
| **Output ITL P50/P95/P99** | Inter-token latency during output phase (streaming jitter). |
| **Think tokens avg/P95** | Distribution of reasoning-chain length per request.        |
| **Output tokens avg**  | Mean output token count per request.                         |
| **Budget forced %**    | Percentage of reasoning requests where the router forced `</think>`. |
| **Force reason**       | Breakdown by `converged` / `overthinking` / `hard_cap`.      |
| **OutputCritical evictions** | KV pressure events that reached the user-visible tier. |

See [`benchmarks/metrics.py`](https://github.com/angelnicolasc/meridian/blob/main/benchmarks/metrics.py)
for the exact serialised shape.

## A/B comparison mode

`--baseline` runs the same workload through one or more baseline schedulers
alongside Meridian and writes `ab-report.{json,md}` to `--out-dir`:

- `stock` — `StockSchedulerBaseline`, a priority-weight single-queue scheduler
  equivalent to vLLM ≤0.8 (no phase awareness, never forces budget).
- `static-budget` — `StaticBudgetBaseline`, a fixed think-token cap equivalent
  to vLLM 0.9's `thinking_token_budget` (forces `</think>` on a counter, with no
  entropy signal). This is the prior art Meridian's EAT/RPDI forcing aims to beat.
- `all` — run every baseline; the report gets one value column per run and a
  `Δ% vs <baseline>` column per baseline with a `WIN/win/FLAT/loss/LOSS` flag.

### Five-minute A/B (no GPU)

```bash
# Stock + static-budget + Meridian over a real prompt-length distribution.
python -m benchmarks.meridian_bench synthetic-replay \
    --workload sharegpt --baseline all \
    --duration-s 30 --arrival-rate 8 --out-dir bench-out/

# Read the comparison. Meridian should win TTOT P95 vs both baselines.
cat bench-out/ab-report.md
```

The synthetic-replay mode requires the native extension
(`maturin develop -m crates/meridian-python/Cargo.toml`); the baseline and
report logic alone are exercised by `benchmarks/tests/test_baselines.py`
without it.

## Dataset loaders

Pass `--workload sharegpt` or `--workload math500` to load real traffic
distributions from HuggingFace. Datasets are downloaded once and cached at
`~/.cache/meridian/datasets/`. Requires no GPU — the offline replay drives the
synthetic decoder with the real prompt/response length distribution.

## Test environment disclosure

When comparing numbers across runs:

| Parameter | Default |
|-----------|---------|
| `--seed` | `42` |
| `--arrival-rate` | `8 req/s` |
| `--duration-s` | `30` |
| `--reasoning-ratio` | `0.4` |

Always report `--seed` and workload flag. Synthetic results are hardware-independent;
real-vLLM results depend on GPU model, driver, and memory state — disclose all three.

## How to compare two runs

```bash
# Run A (baseline config)
python -m benchmarks.meridian_bench synthetic-replay --seed 42 --out-dir bench-out/a/

# Run B (modified config)
python -m benchmarks.meridian_bench synthetic-replay --seed 42 --out-dir bench-out/b/

# Diff the JSON reports
diff <(jq -S . bench-out/a/report.json) <(jq -S . bench-out/b/report.json)
```

## What this harness is, and what it isn't

- It **is** a phase-differentiated latency regression suite. It catches
  changes that move the TTOT or output-ITL distributions, the metrics
  Meridian was built to improve.
- It **is** reproducible: the synthetic-replay mode is deterministic
  given `--seed`. Two PRs can be diffed apples-to-apples.
- It **is not** a raw-throughput benchmark. vLLM's own harness already
  reports `tokens/sec/GPU` and that metric does not differentiate
  Meridian from the baseline. Operators who want a throughput number
  should run vLLM's benchmark.
- It **is not** an accuracy benchmark. Budget forcing can in principle
  hurt reasoning accuracy on hard problems. Accuracy measurement requires a
  separate ground-truth evaluation suite; this harness does not provide one.
