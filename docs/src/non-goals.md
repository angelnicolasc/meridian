# Non-goals

Meridian has explicit scope boundaries. Documenting what it does **not** do
is as important as documenting what it does — it prevents inflated expectations
and makes the design auditable.

## Not a serving engine

Meridian is a scheduling layer. It requires vLLM (or a compatible serving
backend) underneath it. It does not implement:

- Model loading or weight management.
- Prefill execution (prompt processing).
- Decode kernel scheduling (CUDA streams, tensor parallelism).
- Tokenisation or detokenisation.
- HTTP/gRPC serving interfaces.

## Not a throughput optimiser

Meridian optimises **latency differentiation** — protecting output-phase
inter-token latency at the cost of slightly reduced think-phase throughput.
It does not optimise:

- Raw tokens/sec/GPU throughput. For throughput, run vLLM's own scheduler.
- Batch filling efficiency in non-reasoning workloads.
- Speculative decoding, continuous batching variants, or chunked prefill.

## Not an accuracy guarantee

Budget forcing injects `</think>` based on entropy signals, not correctness.
This can shorten reasoning chains on hard problems. Meridian does not:

- Measure or guarantee reasoning accuracy.
- Validate that forced-short chains produce correct answers.
- Implement any feedback loop between output quality and forcing thresholds.

Operators running accuracy-sensitive workloads should validate that their
`eat_ema_variance_threshold` and `rpdi_threshold` settings do not degrade
task accuracy on representative prompts.

## Not a multi-model router

Meridian schedules within a single `vllm.AsyncLLMEngine` instance. It does not:

- Route requests across multiple models.
- Implement A/B model experiments.
- Manage model replicas or horizontal scaling.

## Not a vLLM fork

Meridian is a plugin, not a fork. It does not modify vLLM internals and does
not ship a patched vLLM binary. Compatibility with a specific vLLM version is
the operator's responsibility. See the [Compatibility Matrix](compatibility.md).

## Not production-certified

v0.1.0 is an early release. It has not been validated under production traffic
at scale. Known gaps:

- Real-vLLM end-to-end results exist only for `Qwen/Qwen2.5-0.5B` in the
  repo's GPU CI workflow.
- The disagg fabric integration uses a synthetic in-process mock when
  `libnixl` is not available; real NIXL interop has not been exercised in CI.
- No back-pressure mechanism exists if the disagg fabric becomes unavailable.
