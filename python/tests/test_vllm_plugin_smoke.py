"""Plugin smoke tests.

Two layers:

1. **Pure-Python** (no vLLM, no GPU): construct the plugin against an in-memory
   fake scheduler and validate the transparency contract — calls fall through,
   metrics increment.
2. **vLLM integration** (requires `vllm` + a tiny model + GPU): instantiate
   :class:`vllm.AsyncLLMEngine`, attach the plugin, run a single generation
   request and assert no crash. Marked ``@pytest.mark.gpu`` and ``@pytest.mark.vllm``
   so CI without a GPU skips it.
"""

from __future__ import annotations

import pytest

from meridian.config import MeridianConfig
from meridian.vllm_plugin import MeridianSchedulerPlugin


class _FakeScheduler:
    """Minimal stand-in for vLLM's Scheduler."""

    def __init__(self) -> None:
        self.schedule_calls = 0
        self.arbitrary_attr = "passthrough-works"

    def schedule(self) -> str:
        self.schedule_calls += 1
        return "ok"


def _bootstrap_config() -> MeridianConfig:
    return MeridianConfig.from_str(
        """
[scheduler]
think_tpot_budget_ms   = 80.0
output_tpot_budget_ms  = 20.0
think_batch_multiplier = 2.5
max_think_tokens       = 1024
min_think_tokens       = 8

[entropy]
enabled = false

[kv_memory]
think_phase_memory_fraction = 0.40

[model.qwen3]
think_start_token_ids  = [10, 11]
think_end_token_ids    = [12]
reasoning_parser       = "qwen3"
supports_think_disable = true
""",
    )


def test_plugin_passes_schedule_through_when_no_candidates() -> None:
    """When the base scheduler returns a non-object (string in our fake), the
    plugin should not attempt to reorder it and should not crash."""
    fake = _FakeScheduler()
    cfg = _bootstrap_config()
    plugin = MeridianSchedulerPlugin(fake, cfg, model_key="qwen3")

    assert plugin.schedule() == "ok"
    assert plugin.schedule_count == 1
    assert plugin.last_schedule_latency_ns >= 0

    # Attribute access on unknown names falls through.
    assert plugin.arbitrary_attr == "passthrough-works"  # type: ignore[attr-defined]
    assert fake.schedule_calls == 1


def test_plugin_rejects_unknown_model_key() -> None:
    cfg = _bootstrap_config()
    with pytest.raises(KeyError, match="model key"):
        MeridianSchedulerPlugin(_FakeScheduler(), cfg, model_key="does_not_exist")


def test_post_step_is_no_op_without_router() -> None:
    """When the native extension isn't installed the plugin must still survive
    `post_step` invocations — they degrade to no-ops rather than crashing."""
    cfg = _bootstrap_config()
    plugin = MeridianSchedulerPlugin(_FakeScheduler(), cfg, model_key="qwen3")
    # We don't assume the native module is present in CI; either branch is fine
    # as long as post_step doesn't raise.
    plugin.post_step(seq_groups=[], sampled_tokens=[], logits=None)


# ---------------------------------------------------------------------------
# Real vLLM integration — GPU required
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.vllm
def test_engine_arranca_y_completa_una_request() -> None:  # pragma: no cover — GPU job
    pytest.importorskip("vllm")
    pytest.importorskip("torch")

    from vllm import AsyncLLMEngine  # type: ignore[import-not-found]
    from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore[import-not-found]

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model="Qwen/Qwen2.5-0.5B", max_model_len=512),
    )
    cfg = _bootstrap_config()
    plugin = MeridianSchedulerPlugin.attach(engine, cfg, model_key="qwen3")

    # Drive one trivial completion. We assert non-crash + at least one schedule call.
    import asyncio

    async def _drive() -> None:
        async for _ in engine.generate("Say hello.", request_id="smoke-0"):  # type: ignore[attr-defined]
            pass

    asyncio.run(_drive())
    assert plugin.schedule_count >= 1
