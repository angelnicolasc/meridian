# Meridian

**Inference-time compute scheduler for reasoning models.**

[![CI](https://github.com/angelnicolasc/meridian/actions/workflows/ci.yml/badge.svg)](https://github.com/angelnicolasc/meridian/actions/workflows/ci.yml)
[![Docs](https://github.com/angelnicolasc/meridian/actions/workflows/docs.yml/badge.svg)](https://angelnicolasc.github.io/meridian)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85+-orange.svg)](rust-toolchain.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](python/pyproject.toml)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76b900.svg)](.devcontainer/Dockerfile)
[![Release](https://img.shields.io/github/v/release/angelnicolasc/meridian?sort=semver&display_name=tag)](https://github.com/angelnicolasc/meridian/releases)
[![SLSA Level 2](https://img.shields.io/badge/SLSA-Level_2-success.svg)](docs/src/adr/0007-release-versioning-policy.md)

> Every LLM serving system in 2026 treats thinking tokens identically to output tokens.
> They are not the same workload. **Meridian is the scheduler that knows the difference.**

---

## The thesis

Reasoning models (DeepSeek-R1, Qwen3, Claude Opus 4.x, o3) produce two structurally
different token sequences within a single request:

```
[prompt] -> <think> ... N reasoning tokens ... </think> -> [output tokens]
```

These phases have opposite latency profiles. **Think-decode** is throughput-bound and
the user does not see inter-token latency. **Output-decode** is the user-visible
streaming experience and must be protected. No serving system today acts on this
asymmetry.

Meridian implements:

1. **Dual-queue scheduling** with phase-differentiated SLOs (TPOT for think, TTOT for output).
2. **Phase-aware KV block manager** with three-tier eviction (`ThinkComplete` evicted before `OutputCritical`).
3. **Entropy-driven budget forcing** using EAT (`arXiv:2509.26522`) and RPDI (`arXiv:2603.14251`) signals computed on a dedicated CUDA stream.
4. **Drop-in vLLM plugin** — no fork required.

See [Research/MERIDIAN-playbook.md](Research/MERIDIAN-playbook.md) for the full spec
and [docs/src/adr/0001-dual-queue-rationale.md](docs/src/adr/0001-dual-queue-rationale.md)
for the architectural decision record.

---

## Status

**v0.1.0 — Sprint 3 (disagg-native)** is the first tagged release.
Meridian now speaks a versioned wire protocol for prefill-decode
disaggregation, ships an A/B benchmark harness against a stock
priority-weight baseline, and loads real ShareGPT / MATH-500 traffic
from HuggingFace. Provenance is attested at SLSA Level 2 on every
tagged release per [ADR-0007](docs/src/adr/0007-release-versioning-policy.md).

| Component                        | Status         |
|----------------------------------|----------------|
| `PhaseRouter` state machine      | functional     |
| `EntropyProbe` (CPU backend)     | functional + batch |
| `EntropyProbe` (CUDA backend)    | functional + correctness-tested |
| `MeridianScheduler` dispatch     | functional     |
| `PhaseAwareBlockManager` evict   | functional + pyo3-exposed |
| `BlockManager` disagg surface    | `offload_block` / `ingest_block` / `block_location` |
| NIXL fabric (`--features nixl`)  | wire protocol + synthetic in-process fabric |
| vLLM plugin                      | active (reorders + injects + Duration reap + disagg hooks) |
| Benchmark harness                | synthetic-replay + real-vllm + A/B against stock baseline |
| Real dataset loaders             | ShareGPT + MATH-500 (cached locally) |
| SLSA Level 2 provenance          | attested on every `v*` tag |

---

## Quickstart

Requires a Linux host (or WSL2) with NVIDIA driver 555+ and CUDA 12.6 toolkit.
Devcontainer support included.

```bash
# Clone and enter devcontainer
git clone https://github.com/angelnicolasc/meridian.git && cd meridian
./scripts/dev-up.sh

# Build the Rust workspace
cargo build --workspace

# Build Python bindings
uv sync --project python
maturin develop --release -m crates/meridian-python/Cargo.toml

# Run the test suite (no GPU required)
cargo nextest run --workspace
uv run --project python pytest -m "not gpu"
```

---

## Repository layout

```
crates/meridian-core/    Rust scheduler core (no CUDA, no Python)
crates/meridian-kernels/ CUDA kernels + Rust FFI
crates/meridian-python/  pyo3 bindings (built via maturin)
python/meridian/         Python package: EntropyProbe + vLLM plugin
docs/                    mdbook + ADRs
models/                  Per-model token boundary configs
benchmarks/              Replay harness (Sprint 1+)
```

---

## Contributing

Contributions welcome under [DCO sign-off](CONTRIBUTING.md). Please read
[CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
before opening a PR. Security disclosures: [SECURITY.md](SECURITY.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
