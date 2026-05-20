"""Integration test for the pyo3 bindings.

This test requires the ``meridian._meridian`` native extension built by
``maturin develop``. CI runs it in the dedicated ``cuda`` workflow after the
maturin step; the regular Python CI job skips it via ``importorskip``.
"""

from __future__ import annotations

import pytest

native = pytest.importorskip("meridian._meridian")

THINK_START = 1
THINK_END = 2
EOS = 3


def test_phase_router_state_machine_lifecycle() -> None:
    router = native.PhaseRouter(
        think_start_ids=[THINK_START],
        think_end_ids=[THINK_END],
        eos_ids=[EOS],
        min_think_tokens=2,
        max_think_tokens=64,
    )
    router.register(42)

    assert router.on_token(42, THINK_START, None)["kind"] == "enter_think"
    for _ in range(5):
        ev = router.on_token(42, 99, None)
        assert ev["kind"] == "none"

    ev = router.on_token(42, THINK_END, None)
    assert ev["kind"] == "exit_think"
    assert ev["tokens_used"] == 6

    ev = router.on_token(42, EOS, None)
    assert ev["kind"] == "complete"


def test_phase_router_hard_cap_emits_reason() -> None:
    router = native.PhaseRouter(
        think_start_ids=[THINK_START],
        think_end_ids=[THINK_END],
        eos_ids=[EOS],
        min_think_tokens=1,
        max_think_tokens=8,
    )
    router.register(1)
    router.on_token(1, THINK_START, None)
    for _ in range(7):
        router.on_token(1, 99, None)
    ev = router.on_token(1, 99, None)
    assert ev["kind"] == "force_budget"
    assert ev["reason"] == "hard_cap"
    assert ev["inject_token"] == THINK_END


def test_entropy_signal_round_trip() -> None:
    sig = native.EntropySignal(1.5, 0.2, 0.18, 0.001)
    assert sig.token_entropy == pytest.approx(1.5)
    assert sig.eat == pytest.approx(0.2)
    assert "EntropySignal" in repr(sig)


def test_meridian_scheduler_dual_queue() -> None:
    s = native.MeridianScheduler(
        think_tpot_budget_ms=80.0,
        output_tpot_budget_ms=20.0,
        think_batch_multiplier=2.0,
        max_think_tokens=1024,
        min_think_tokens=8,
    )
    for r in (10, 11, 12):
        s.admit(r)
    s.signal_exit_think(11, 50)
    assert s.output_queue_depth == 1

    batch = s.schedule_batch(1, 100)
    assert batch["output"] == [11]
    assert len(batch["think"]) >= 1


def test_meridian_scheduler_pending_injection() -> None:
    s = native.MeridianScheduler()
    s.queue_injection(7, 128_800)
    pi = s.pop_pending_injection()
    assert pi == {"req_id": 7, "token_id": 128_800}
    assert s.pop_pending_injection() is None
