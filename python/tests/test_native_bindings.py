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


def test_block_manager_enumerate_and_free_by_id() -> None:
    # 16-byte blocks, capacity for 8 — keeps the arithmetic obvious.
    bm = native.BlockManager(16, 8 * 16)
    ids = bm.allocate(1, "think_complete", 3)
    assert len(ids) == 3
    assert sorted(bm.blocks_for_request(1)) == sorted(ids)
    assert bm.blocks_for_request(999) == []

    # Free one slot by id: present -> True, used_bytes drops one block.
    assert bm.free_block_by_id(ids[0]) is True
    assert bm.used_bytes == 2 * 16
    assert sorted(bm.blocks_for_request(1)) == sorted(ids[1:])

    # Freeing an absent id is a no-op returning False.
    assert bm.free_block_by_id(ids[0]) is False
    assert bm.free_block_by_id(4242) is False


# ---------------------------------------------------------------------------
# Phase-conditioned speculative decoding (ADR-0009)
# ---------------------------------------------------------------------------

MEASURED_RUN = {
    "harness": "DeepSpec@0000000",
    "draft_checkpoint": "deepseek-ai/dspark_qwen3_4b_block7",
    "target_model": "Qwen/Qwen3-4B",
    "thinking_mode": True,
    "recorded_on": "2026-08-07",
}


def test_hook_is_uncalibrated_and_inert_by_default() -> None:
    hook = native.PhaseConditioningHook(baseline_proposal_len=7)
    assert hook.is_calibrated is False

    policy = hook.policy_for("think")
    assert policy["proposal_len"] == 7
    assert policy["basis"] == "baseline"
    assert policy["planning_acceptance"] is None


def test_hook_accepts_both_phase_vocabularies() -> None:
    hook = native.PhaseConditioningHook()
    # The router's `phase_of_kind` labels and the short forms must agree.
    assert (
        hook.policy_for("think_decode")["proposal_len"] == hook.policy_for("think")["proposal_len"]
    )
    assert (
        hook.policy_for("output_decode")["proposal_len"]
        == hook.policy_for("output")["proposal_len"]
    )
    with pytest.raises(ValueError, match="unknown speculation phase"):
        hook.policy_for("nonsense")


def test_uncalibrated_hook_never_drafts_deeper_than_baseline() -> None:
    hook = native.PhaseConditioningHook(baseline_proposal_len=4, max_proposal_len=16)
    for entropy in [0.0, 1.0, 4.0, 8.0, 11.9]:
        signal = native.EntropySignal(entropy, 0.1, 0.1, 0.0)
        for phase in ("think", "output"):
            policy = hook.policy_for(phase, signal)
            assert policy["proposal_len"] <= 4
            assert not policy["basis"].startswith("measured")


def test_high_entropy_shrinks_the_draft_budget() -> None:
    hook = native.PhaseConditioningHook(baseline_proposal_len=7)
    near_uniform = native.EntropySignal(11.5, 0.0, 0.0, 0.0)
    policy = hook.policy_for("think", near_uniform)
    assert policy["basis"] == "entropy_ceiling"
    assert policy["proposal_len"] < 7


def test_measured_prior_unlocks_phase_conditioning() -> None:
    hook = native.PhaseConditioningHook(
        baseline_proposal_len=7, max_proposal_len=12
    ).with_measured_prior(think=0.45, output=0.92, **MEASURED_RUN)
    assert hook.is_calibrated is True

    think = hook.policy_for("think")
    output = hook.policy_for("output")
    assert think["basis"] == "measured_prior"
    assert think["proposal_len"] < output["proposal_len"]


def test_ledger_segments_by_phase_and_computes_the_statistic() -> None:
    ledger = native.AcceptanceLedger.measured(**MEASURED_RUN)
    for step in range(400):
        jitter = step % 2
        ledger.record("think", 2 + jitter, 7)
        ledger.record("output", 6 + jitter, 7)

    report = ledger.report()
    assert report["provenance"] == "measured"
    assert report["think"]["steps"] == 400
    assert report["think"]["mean_accepted_length"] < report["output"]["mean_accepted_length"]
    assert report["welch"]["mean_difference"] < 0.0
    assert report["welch"]["ci95_upper"] < 0.0
    assert ledger.verdict(0.5) == "supported"

    claim = ledger.measured_claim()
    assert claim["target_model"] == "Qwen/Qwen3-4B"
    assert claim["thinking_mode"] is True


def test_synthetic_ledger_cannot_produce_a_claim() -> None:
    # Constant values on both arms leave the Welch statistic undefined, so this
    # also pins the error precedence: the unfixable problem (synthetic data) is
    # reported ahead of the fixable one (thin data).
    ledger = native.AcceptanceLedger.synthetic("python-binding-fixture")
    for _ in range(50):
        ledger.record("think", 2, 7)
        ledger.record("output", 5, 7)

    assert ledger.report()["provenance"] == "synthetic"
    assert "NOT A MEASUREMENT" in ledger.render()
    with pytest.raises(ValueError, match="synthetic"):
        ledger.measured_claim()


def test_ledger_straddle_policies_route_the_boundary_step() -> None:
    for policy, think_steps, output_steps in (
        ("exclude", 0, 0),
        ("attribute_to_think", 1, 0),
        ("attribute_to_output", 0, 1),
    ):
        ledger = native.AcceptanceLedger.synthetic("straddle", straddle_policy=policy)
        ledger.record("think", 4, 7, straddles_boundary=True)
        report = ledger.report()
        assert report["think"]["steps"] == think_steps, policy
        assert report["output"]["steps"] == output_steps, policy
        assert report["straddling_steps"] == 1, policy

    with pytest.raises(ValueError, match="unknown straddle policy"):
        native.AcceptanceLedger.synthetic("bad", straddle_policy="nope")


def test_empty_ledger_reports_insufficient_data() -> None:
    ledger = native.AcceptanceLedger.synthetic("empty")
    assert ledger.report()["welch"] is None
    assert ledger.verdict() == "insufficient_data"
