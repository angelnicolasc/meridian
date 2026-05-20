"""Tests for the workload generator and metrics aggregation."""

from __future__ import annotations

import pytest

from benchmarks.metrics import RequestResult, aggregate
from benchmarks.workloads import synthetic_mix


def test_synthetic_mix_is_deterministic() -> None:
    a = synthetic_mix(64, reasoning_ratio=0.4, seed=42)
    b = synthetic_mix(64, reasoning_ratio=0.4, seed=42)
    assert [r.request_id for r in a] == [r.request_id for r in b]
    assert [r.kind for r in a] == [r.kind for r in b]


def test_synthetic_mix_respects_ratio() -> None:
    n = 1000
    mix = synthetic_mix(n, reasoning_ratio=0.3, seed=0)
    reasoning = sum(1 for r in mix if r.kind == "reasoning")
    # Allow a fairly wide ±5% band for the 1000-sample bootstrap.
    assert 0.25 * n <= reasoning <= 0.35 * n


def test_synthetic_mix_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError, match="reasoning_ratio"):
        synthetic_mix(10, reasoning_ratio=1.5)


def test_aggregate_produces_finite_percentiles() -> None:
    results = [
        RequestResult(
            req_id=f"r{i}",
            kind="chat" if i % 2 == 0 else "reasoning",
            ttft_ms=80.0 + i,
            ttot_ms=20.0 + i,
            think_tokens=200 if i % 2 else 0,
            output_tokens=100,
            output_itl_ms=[10.0, 15.0, 12.0, 18.0, 11.0],
            budget_forced=i % 4 == 1,
            force_reason="converged" if i % 4 == 1 else None,
        )
        for i in range(8)
    ]
    report = aggregate(
        results,
        config_name="unit",
        duration_s=10.0,
        arrival_rate_rps=2.0,
        output_critical_eviction_events=0,
    )
    assert report.n_requests == 8
    assert report.n_reasoning == 4
    assert report.n_chat == 4
    assert report.ttft_p50_ms > 0
    assert report.output_itl_p95_ms >= report.output_itl_p50_ms
    assert report.budget_forced_pct > 0
    assert report.budget_forced_by_reason.get("converged", 0) >= 1


def test_report_serialisation_round_trip() -> None:
    results = [
        RequestResult(req_id="r0", kind="chat", ttft_ms=50.0, ttot_ms=10.0,
                      output_tokens=50, output_itl_ms=[20.0]),
    ]
    report = aggregate(
        results,
        config_name="serial",
        duration_s=1.0,
        arrival_rate_rps=1.0,
        output_critical_eviction_events=0,
    )
    j = report.to_json()
    md = report.to_markdown()
    assert "config_name" in j
    assert "Meridian benchmark" in md
