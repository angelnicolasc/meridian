"""Tests for the A/B baseline schedulers and the comparison report."""

from __future__ import annotations

import json
import math
from pathlib import Path

from benchmarks.baselines import (
    StaticBudgetBaseline,
    StockSchedulerBaseline,
    run_static_budget_baseline,
    run_stock_baseline,
)
from benchmarks.metrics import ABComparisonReport, BenchmarkReport, RequestResult, aggregate
from benchmarks.workloads import WorkloadRequest, synthetic_mix


def _tiny_workload() -> list[WorkloadRequest]:
    # Tiny per-token latencies keep the test fast (well under 1 s).
    return synthetic_mix(8, 0.5, seed=42)


def _aggregate(
    name: str, results: list[RequestResult], duration: float,
) -> BenchmarkReport:
    return aggregate(
        results,
        config_name=name,
        duration_s=duration,
        arrival_rate_rps=4.0,
        output_critical_eviction_events=0,
    )


def _scaled(base: BenchmarkReport, name: str, factor: float) -> BenchmarkReport:
    """A copy of ``base`` with latency metrics scaled by ``factor``."""
    return BenchmarkReport(
        config_name=name,
        duration_s=base.duration_s,
        arrival_rate_rps=base.arrival_rate_rps,
        n_requests=base.n_requests,
        n_reasoning=base.n_reasoning,
        n_chat=base.n_chat,
        ttft_p50_ms=base.ttft_p50_ms * factor,
        ttft_p95_ms=base.ttft_p95_ms * factor,
        ttot_p50_ms=base.ttot_p50_ms * factor,
        ttot_p95_ms=base.ttot_p95_ms * factor,
        output_itl_p50_ms=base.output_itl_p50_ms * factor,
        output_itl_p95_ms=base.output_itl_p95_ms * factor,
        output_itl_p99_ms=base.output_itl_p99_ms * factor,
        think_tokens_avg=base.think_tokens_avg * factor,
        think_tokens_p95=base.think_tokens_p95 * factor,
        output_tokens_avg=base.output_tokens_avg,
        budget_forced_pct=42.0,
        budget_forced_by_reason={"converged": 3},
        output_critical_eviction_events=0,
    )


def test_stock_baseline_produces_plausible_metrics() -> None:
    sched = StockSchedulerBaseline(
        think_per_token_us=0.2,
        output_per_token_us=0.5,
        prefill_per_token_us=0.1,
    )
    wl = _tiny_workload()
    results = [sched.run_request(r) for r in wl]
    assert len(results) == len(wl)
    for r, w in zip(results, wl, strict=True):
        assert r.kind == w.kind
        assert r.ttft_ms > 0
        # Output ITL list has output_tokens - 1 entries (first is TTOT).
        expected_itl = max(0, w.expected_output_tokens - 1)
        assert len(r.output_itl_ms) == expected_itl
        # Stock baseline does not force budget under any circumstances.
        assert not r.budget_forced


def test_run_stock_baseline_module_function() -> None:
    wl = _tiny_workload()
    results = run_stock_baseline(
        wl, think_per_token_us=0.1, output_per_token_us=0.3,
    )
    assert len(results) == len(wl)


def test_static_budget_baseline_forces_at_cap() -> None:
    # A budget below the smallest reasoning think target guarantees a force.
    sched = StaticBudgetBaseline(
        think_token_budget=1,
        think_per_token_us=0.1,
        output_per_token_us=0.3,
        prefill_per_token_us=0.1,
    )
    wl = _tiny_workload()
    results = [sched.run_request(r) for r in wl]
    reasoning = [(r, w) for r, w in zip(results, wl, strict=True) if w.kind == "reasoning"]
    assert reasoning, "tiny workload should contain reasoning requests"
    for r, w in reasoning:
        # Think tokens are capped at the budget; the force flag fires.
        assert r.think_tokens <= 1
        assert r.budget_forced
        assert r.force_reason == "static_cap"
        assert w.expected_think_tokens > 1


def test_run_static_budget_baseline_module_function() -> None:
    wl = _tiny_workload()
    results = run_static_budget_baseline(wl, think_token_budget=4096)
    assert len(results) == len(wl)


def test_ab_comparison_report_pair_writes_artifacts(tmp_path: Path) -> None:
    wl = _tiny_workload()
    stock = _aggregate("stock", run_stock_baseline(wl), duration=0.5)
    meridian = _scaled(stock, "meridian", 0.7)

    ab = ABComparisonReport.pair(stock, meridian)
    md = ab.to_markdown()
    assert "under test `meridian`" in md
    assert "stock" in md
    js = json.loads(ab.to_json())
    assert js["under_test"] == "meridian"
    assert set(js["reports"]) == {"stock", "meridian"}
    assert "stock" in js["delta_pct"]
    # ~30% faster -> negative deltas on lower-is-better metrics.
    assert js["delta_pct"]["stock"]["ttot_p95_ms"] < 0

    ab.write_artifacts(tmp_path)
    assert (tmp_path / "ab-report.json").exists()
    assert (tmp_path / "ab-report.md").exists()


def test_ab_comparison_report_n_way() -> None:
    wl = _tiny_workload()
    stock = _aggregate("stock", run_stock_baseline(wl), duration=0.5)
    static = _scaled(stock, "static", 0.9)
    meridian = _scaled(stock, "meridian", 0.6)

    ab = ABComparisonReport(
        reports={"stock": stock, "static": static, "meridian": meridian},
        under_test="meridian",
    )
    js = json.loads(ab.to_json())
    assert set(js["delta_pct"]) == {"stock", "static"}
    md = ab.to_markdown()
    # One delta column per baseline.
    assert "Δ% vs stock" in md
    assert "Δ% vs static" in md


def test_ab_delta_pct_handles_zero_baseline_gracefully() -> None:
    from dataclasses import replace
    wl = _tiny_workload()
    stock = _aggregate("stock", run_stock_baseline(wl), duration=0.1)
    # Force a zero on the baseline side to exercise the divide-by-zero guard.
    stock = replace(stock, ttft_p50_ms=0.0)
    meridian = replace(stock, config_name="meridian", ttft_p50_ms=1.0)
    ab = ABComparisonReport.pair(stock, meridian)
    deltas = json.loads(ab.to_json())["delta_pct"]["stock"]
    assert math.isnan(deltas["ttft_p50_ms"])
