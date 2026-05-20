# Python API

## Top-level surface

- `meridian.EntropyProbe` — stateful per-request entropy probe.
- `meridian.EntropySignal` — per-token signal record.
- `meridian.MeridianConfig` — Pydantic-backed TOML config.
- `meridian.load_config(path)` — convenience loader.
- `meridian.vllm_plugin.MeridianSchedulerPlugin` — vLLM scheduler wrapper.

## Backends

`EntropyProbe` accepts `backend="cpu"` (default, pure NumPy) or
`backend="cuda"`. Sprint 0 ships CUDA as a delegate to CPU; the public
contract does not change when the real kernel lands in Sprint 1.

## Example

```python
from meridian import EntropyProbe, load_config
import numpy as np

cfg = load_config("meridian.toml")
probe = EntropyProbe(
    think_end_token_ids=cfg.model["qwen3"].think_end_token_ids,
    backend="cpu",
    ema_alpha=cfg.entropy.ema_alpha,
)

logits = np.random.randn(cfg_vocab := 151_936).astype(np.float32)
sig = probe.compute(req_id=42, logits=logits)
print(sig.token_entropy, sig.eat, sig.eat_ema_variance)
```

## vLLM integration

See [`python/meridian/vllm_plugin.py`](https://github.com/angelnicolasc/meridian/blob/main/python/meridian/vllm_plugin.py).
The plugin attaches to an existing `AsyncLLMEngine` without forking vLLM.
