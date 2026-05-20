# Meridian

**Inference-time compute scheduler for reasoning models.**

[![CI](https://github.com/angelnicolasc/meridian/actions/workflows/ci.yml/badge.svg)](https://github.com/angelnicolasc/meridian/actions/workflows/ci.yml)
[![Docs](https://github.com/angelnicolasc/meridian/actions/workflows/docs.yml/badge.svg)](https://angelnicolasc.github.io/meridian)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85+-orange.svg)](rust-toolchain.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](python/pyproject.toml)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76b900.svg)](.devcontainer/Dockerfile)
[![Release](https://img.shields.io/github/v/release/angelnicolasc/meridian?sort=semver&display_name=tag)](https://github.com/angelnicolasc/meridian/releases)
[![SLSA Level 2](https://img.shields.io/badge/SLSA-Level_2-success.svg)](https://github.com/angelnicolasc/meridian/releases)

---

## What Meridian is

Meridian is a scheduling layer for vLLM that treats the two token-generation
phases of reasoning models as distinct workloads. It intercepts the vLLM
scheduler at runtime — no fork required — and applies separate SLOs,
eviction policies, and entropy-driven budget control to each phase.

## What problem it solves

Reasoning models (DeepSeek-R1, Qwen3, o3, Granite 3.2) emit two structurally
different sequences in a single request:

```
[prompt] → <think> ... N reasoning tokens ... </think> → [output tokens]
```

| Phase         | User-visible? | Latency tolerance | Cost profile |
|---------------|--------------|-------------------|--------------|
| Think-decode  | No           | High              | Throughput-bound |
| Output-decode | Yes          | Zero tolerance    | Latency-bound |

Standard continuous-batching schedulers process both phases from the same
priority queue with the same inter-token latency target. This leaves
output-phase latency unnecessarily constrained by think-phase batch dynamics.
Meridian exploits the asymmetry.

## What is verified today

| Component | Verified in CI | Verified on GPU runner |
|-----------|:--------------:|:----------------------:|
| `PhaseRouter` state machine | ✓ | — |
| `MeridianScheduler` dual-queue dispatch | ✓ | — |
| `PhaseAwareBlockManager` three-tier eviction | ✓ | — |
| `EntropyProbe` CPU backend (NumPy) | ✓ | — |
| `EntropyProbe` CUDA backend — correctness vs CPU | — | ✓ |
| vLLM plugin — attach / detach / reorder / inject | ✓ | ✓ |
| Disagg block manager surface (`offload_block`, `ingest_block`) | ✓ | — |
| NIXL fabric wire protocol (synthetic mock) | ✓ | — |
| Benchmark harness — synthetic A/B | ✓ | — |
| Benchmark harness — real-vLLM with `Qwen2.5-0.5B` | — | ✓ |
| SLSA L2 provenance attestation | ✓ (on every `v*` tag) | — |
| `cargo deny` supply-chain audit | ✓ | — |

---

## What Meridian implements

```
Incoming requests
      │
      ▼
 Phase Router  ──── token stream state machine, O(1) per token
      │
      ├── ExitThink / ForceBudget ─▶  EAT + RPDI entropy signals
      │
      ├── Think-Decode Scheduler  (TPOT-relaxed, 2.5× batch budget)
      │
      └── Output-Decode Scheduler (TTOT-strict, stream priority)
              │
              ▼
    Phase-Aware KV Block Manager
      ThinkComplete ← evicted first
      ThinkActive
      OutputCritical ← never evicted without alerting
```

1. **Dual-queue scheduling** — output-phase requests drain the GPU batch
   before think-phase requests fill remaining capacity. Backed by
   [`crates/meridian-core/src/scheduler.rs`](crates/meridian-core/src/scheduler.rs).

2. **Phase-aware KV eviction** — three tiers with strict priority ordering.
   `OutputCritical` eviction fires an observable counter; the target rate is zero.
   Backed by [`crates/meridian-core/src/block_manager.rs`](crates/meridian-core/src/block_manager.rs).

3. **Entropy-driven budget forcing** — EAT ([arXiv:2509.26522](https://arxiv.org/abs/2509.26522))
   and RPDI ([arXiv:2603.14251](https://arxiv.org/abs/2603.14251)) signals inject
   `</think>` when the model signals convergence, not on a static timer. CUDA
   kernels run on a dedicated secondary stream. Backed by
   [`crates/meridian-kernels/`](crates/meridian-kernels/) and
   [`python/meridian/entropy_probe.py`](python/meridian/entropy_probe.py).

4. **Drop-in vLLM plugin** — wraps the existing scheduler via attribute
   delegation; fully reversible; no vLLM source modification.
   Backed by [`python/meridian/vllm_plugin.py`](python/meridian/vllm_plugin.py).

5. **Disagg KV transfer** — `offload_block` / `ingest_block` hooks support
   prefill-decode disaggregation fabrics (NIXL, Mooncake-compatible).
   Documented in [ADR-0006](docs/src/adr/0006-disagg-kv-transfer.md).

---

## Quickstart

Requires Linux (or WSL2), Rust 1.85+, Python 3.11+. GPU + CUDA 12.6 for the
CUDA backend; not required for synthetic benchmarks or unit tests.

```bash
git clone https://github.com/angelnicolasc/meridian.git
cd meridian

# Install Python deps and build the native extension.
uv sync --project python
maturin develop --release -m crates/meridian-python/Cargo.toml

# Run all tests (no GPU required).
cargo nextest run --workspace
uv run --project python pytest -m "not gpu and not vllm"

# Run the synthetic A/B benchmark (no GPU, no vLLM).
uv --project python run python -m benchmarks.meridian_bench synthetic-replay \
    --baseline stock --duration-s 30 --arrival-rate 8 --out-dir bench-out/
```

For CUDA kernel support, append `--cargo-extra-args="--features cuda"` to the
`maturin develop` call (requires `nvcc` + CUDA 12.6).

---

## Repository layout

```
crates/meridian-core/    Rust scheduler core — no CUDA, no Python dep
crates/meridian-kernels/ CUDA entropy and EAT kernels + C FFI + NIXL wrappers
crates/meridian-python/  pyo3 bindings (built via maturin)
python/meridian/         Python package — EntropyProbe, vLLM plugin, config
docs/                    mdBook documentation + 7 ADRs
models/                  Per-model token boundary configs (DeepSeek-R1, Qwen3, ...)
benchmarks/              Replay harness — synthetic + real-vLLM + A/B mode
```

---

## Non-goals

Meridian is not a throughput optimiser, accuracy guarantor, or full serving
engine. See [Non-goals](docs/src/non-goals.md) for explicit scope boundaries.

---

## Evidence

| Artifact | Link |
|----------|------|
| CI results | [GitHub Actions](https://github.com/angelnicolasc/meridian/actions) |
| Release provenance (SLSA L2) | [Releases](https://github.com/angelnicolasc/meridian/releases) |
| Supply-chain audit | `cargo deny check` in [ci.yml](.github/workflows/ci.yml) |
| Architecture & ADRs | [docs/src/](docs/src/) |
| Benchmark methodology | [ADR-0005](docs/src/adr/0005-benchmark-methodology.md) |

---

## Contributing

Contributions welcome under [DCO sign-off](CONTRIBUTING.md). Read
[CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
before opening a PR. Security disclosures: [SECURITY.md](SECURITY.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
