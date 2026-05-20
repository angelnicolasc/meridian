# Python API

## Installation

```bash
# From source — requires a Linux host with Rust 1.85+ and maturin.
uv sync --project python
maturin develop --release -m crates/meridian-python/Cargo.toml
```

The package exposes no CUDA dependency at import time. CUDA is lazily loaded
when `backend="cuda"` is requested on `EntropyProbe`.

## Top-level surface

| Symbol | Kind | Purpose |
|--------|------|---------|
| `meridian.EntropyProbe` | class | Stateful per-request entropy probe |
| `meridian.EntropySignal` | dataclass | Per-token signal record |
| `meridian.MeridianConfig` | Pydantic model | Runtime configuration |
| `meridian.load_config(path)` | function | Convenience TOML loader |
| `meridian.vllm_plugin.MeridianSchedulerPlugin` | class | vLLM scheduler wrapper |

## Object lifecycle

### EntropyProbe

One instance per request. **Not thread-safe** — do not share an instance across
concurrent requests. Create, use through the token sequence, then discard.

```python
from meridian import EntropyProbe, load_config
import numpy as np

cfg = load_config("meridian.toml")
probe = EntropyProbe(
    think_end_token_ids=cfg.model["qwen3"].think_end_token_ids,
    backend="cpu",          # "cpu" (NumPy) or "cuda" (CUDA kernel)
    ema_alpha=cfg.entropy.ema_alpha,
)

# Per-token call — call once per decoded token.
logits = np.random.randn(151_936).astype(np.float32)
sig = probe.compute(req_id=42, logits=logits)
print(sig.token_entropy, sig.eat, sig.eat_ema_variance)

# Batch path — more efficient for large batch sizes.
batch_logits = np.random.randn(8, 151_936).astype(np.float32)
signals = probe.compute_batch(req_ids=list(range(8)), logits_batch=batch_logits)
```

### MeridianSchedulerPlugin

Wraps an existing `vllm.core.scheduler.Scheduler` at runtime. Safe to attach
and detach. Holds no GPU resources; all GPU work goes through the underlying
vLLM scheduler.

```python
from meridian import load_config
from meridian.vllm_plugin import MeridianSchedulerPlugin

cfg = load_config("meridian.toml")
plugin = MeridianSchedulerPlugin(scheduler=engine.scheduler, config=cfg)
plugin.attach()  # Replaces engine.scheduler with the wrapped version.

# ... serve requests ...

plugin.detach()  # Restores the original scheduler.
```

## Error model

All configuration errors are raised at construction time as `ValueError` with
a dotted field path. For example:

```
ValueError: entropy.ema_alpha must be in (0, 1]; got 1.5
```

Runtime errors from the CUDA kernel surface as `meridian.KernelError` (a
subclass of `RuntimeError`). When the kernel returns `Unavailable` (built
without the `cuda` feature, or missing runtime library), the probe falls back
to the CPU backend automatically.

## Concurrency notes

- `EntropyProbe` instances are **not** thread-safe. One instance per request.
- `MeridianSchedulerPlugin` is designed to be used from vLLM's single async
  event loop. Do not call `schedule_batch` concurrently.
- The Rust `PhaseRouter` and `BlockManager` bindings are thread-safe; they
  use interior mutability backed by `DashMap`.

## Stability guarantees

Pre-1.0. Signatures may change on minor bumps. Breaking changes are listed
under `BREAKING CHANGE:` in [`CHANGELOG.md`](https://github.com/angelnicolasc/meridian/blob/main/CHANGELOG.md) and announced before merging.

## Backends

`EntropyProbe` accepts `backend="cpu"` (default, pure NumPy) or `backend="cuda"`.
Both backends implement the same mathematical operations; the CUDA backend
offloads the logit reduction to a dedicated secondary CUDA stream. Tests enforce
agreement within `atol=1e-5` on a shared set of reference vectors.
