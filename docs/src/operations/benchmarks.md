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
  hurt reasoning accuracy on hard problems; we measure that separately
  in [`benchmarks/phase_accuracy_eval.py`](https://github.com/angelnicolasc/meridian/blob/main/benchmarks/README.md)
  (Sprint 3).
