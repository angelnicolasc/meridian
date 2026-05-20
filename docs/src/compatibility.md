# Compatibility Matrix

## Runtime requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| Linux | Ubuntu 22.04 | Ubuntu 24.04 |
| WSL2 | WSL2 on Windows 10/11 | Windows 10 22H2 |
| Rust toolchain | 1.85.0 | 1.85.0 (pinned) |
| Python | 3.11 | 3.11 |
| vLLM | 0.9.0 | 0.21.0 (resolved in `uv.lock`) |
| NVIDIA driver | 555.x | 555.x |
| CUDA toolkit | 12.6 | 12.6 |
| CUDA Compute Capability | 8.0 (A100) | 8.0+ |

## Build requirements

| Tool | Version |
|------|---------|
| `cargo` + `rustup` | Rust 1.85.0 |
| `maturin` | latest (install via `pip install maturin`) |
| `uv` | 0.4+ |
| `nvcc` | 12.6 (only for `--features cuda`) |
| `mdbook` | latest (only for docs) |

## Model compatibility

Models that have been verified to work with Meridian's phase detection:

| Model family | Boundary detection | Config |
|---|---|---|
| DeepSeek-R1 | Token IDs `[128799, 128800]` | `models/deepseek_r1.toml` |
| Qwen3 / Qwen2.5 | Token IDs `[151648, 151649]` | `models/qwen3.toml` |
| IBM Granite 3.2 | Prose markers (no distinct token IDs) | `models/granite_3_2.toml` |

Models that are **not** verified to work:

- Models with non-standard `<think>` tokenisation not listed above — configure
  `think_start_token_ids` / `think_end_token_ids` manually and validate with
  a sample prompt before production use.
- Models served through streaming APIs (e.g. Claude via Anthropic API) —
  Meridian requires direct access to the logit vector, which API-served models
  do not expose.

## Feature flags

| Feature flag | Requires | Status |
|---|---|---|
| *(default — no flags)* | Linux, Rust | Fully CI-tested |
| `--features prometheus` | `prometheus` crate | CI-tested |
| `--features unstable` | Rust nightly-gated APIs | CI-tested |
| `--features nixl` | `libnixl.so` on deploy host | Compiles; integration tested with synthetic mock |
| `--features cuda` | `nvcc`, CUDA 12.6, GPU at runtime | Build-tested on GPU CI runner |

## CI coverage

| Job | Platform | GPU | Status |
|-----|----------|-----|--------|
| `rust-core` | ubuntu-24.04 | No | [![CI](https://github.com/angelnicolasc/meridian/actions/workflows/ci.yml/badge.svg)](https://github.com/angelnicolasc/meridian/actions/workflows/ci.yml) |
| `rust-kernels` (stub) | ubuntu-24.04 | No | Same badge |
| `python` | ubuntu-24.04 | No | Same badge |
| `docs` | ubuntu-24.04 | No | Same badge |
| `supply-chain` | ubuntu-24.04 | No | Same badge |
| `cuda-build` | self-hosted `gpu` runner | Yes | Runs only on `angelnicolasc` org pushes |

The GPU jobs are gated to prevent arbitrary code execution on the self-hosted
runner from fork PRs. See [GPU CI runner setup](operations/ci-gpu-runner.md).

## Known incompatibilities

- **vLLM below 0.9.0**: the dependency constraint is `vllm>=0.9.0`. Earlier
  versions are not supported and will be rejected at install time.
- **Windows (native)**: the Rust workspace builds on Windows (tested in development),
  but the Python extension and benchmarks require Linux or WSL2 for the CUDA and
  maturin paths.
- **macOS**: not supported. CUDA is not available on macOS.
