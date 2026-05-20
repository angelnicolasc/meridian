# Deployment Model

## Single-node deployment

The primary and most-tested deployment topology. One `AsyncLLMEngine` instance,
one Meridian plugin, all on the same GPU node.

```
┌──────────────────────────────────────────────┐
│   GPU node                                    │
│                                               │
│  vLLM AsyncLLMEngine                         │
│   └── MeridianSchedulerPlugin (attached)     │
│        ├── PhaseRouter                       │
│        ├── MeridianScheduler                 │
│        └── PhaseAwareBlockManager            │
└──────────────────────────────────────────────┘
```

**Prerequisites**: Linux (or WSL2), NVIDIA driver 555+, CUDA 12.6, vLLM 0.4.x.

**Configuration**: standard `meridian.toml` with `[disagg] enabled = false`.

## Disaggregated prefill-decode

Experimental. Requires NIXL-capable infrastructure. The block manager's
`offload_block` / `ingest_block` hooks transfer `ThinkComplete` KV blocks to
a remote decode node after `</think>` is emitted.

```
┌─────────────────┐         NIXL fabric          ┌─────────────────┐
│  Prefill node   │  ──── KV block transfer ────▶ │  Decode node    │
│                 │                               │                 │
│  vLLM prefill   │                               │  vLLM decode    │
│  Meridian plug  │                               │  Meridian plug  │
└─────────────────┘                               └─────────────────┘
```

**Status**: the disagg wire protocol and block manager hooks are implemented
and verified with a synthetic in-process NIXL mock. Real NIXL interop requires
`cargo build --features nixl` and `libnixl.so` on the deploy host.

**Configuration**:
```toml
[disagg]
enabled = true
fabric = "nixl"
offload_threshold_blocks = 4
```

See [ADR-0006](adr/0006-disagg-kv-transfer.md) for the protocol specification.

## Installation

### From source (recommended)

```bash
git clone https://github.com/angelnicolasc/meridian.git
cd meridian

# Build and install the Rust core + Python bindings.
uv sync --project python
maturin develop --release -m crates/meridian-python/Cargo.toml

# Optional: build with CUDA kernel support.
# Requires nvcc + CUDA 12.6 toolkit.
maturin develop --release \
  -m crates/meridian-python/Cargo.toml \
  --cargo-extra-args="--features cuda"
```

### Devcontainer

The repo includes a devcontainer configuration with the full toolchain
pre-installed:

```bash
# Open in VS Code with the Dev Containers extension, or:
./scripts/dev-up.sh
```

## Configuration loading

The plugin looks for `meridian.toml` in the current working directory, then
`~/.config/meridian/meridian.toml`. Override with:

```python
from meridian import load_config
from meridian.vllm_plugin import MeridianSchedulerPlugin

cfg = load_config("/path/to/meridian.toml")
plugin = MeridianSchedulerPlugin(scheduler=engine.scheduler, config=cfg)
```

## Known limits

| Dimension | Limit | Notes |
|-----------|-------|-------|
| Models per instance | 1 | One engine, one config |
| Concurrent requests | Limited by GPU VRAM / block budget | Set `capacity_bytes` |
| vLLM version | 0.4.x verified | 0.5+ compatibility not tested |
| Disagg fabric | NIXL (production) or synthetic mock | Real NIXL requires libnixl |
