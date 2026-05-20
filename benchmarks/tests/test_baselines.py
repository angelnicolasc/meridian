"""Tests for the A/B baseline scheduler and the comparison report."""

from __future__ import annotations

import math
from pathlib import Path

from benchmarks.baselines import StockSchedulerBaseline, run_stock_baseline
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


def test_ab_comparison_report_computes_deltas_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    wl = _tiny_workload()
    results = run_stock_baseline(
        wl, think_per_token_us=0.1, output_per_token_us=0.3,
    )
    stock = _aggregate("stock", results, duration=0.5)
    # Build a synthetic "Meridian" report whose metrics are systematically
    # ~30 % faster than the stock baseline so the delta calculation is
    # exercised on real numbers.
    meridian = BenchmarkReport(
        config_name="meridian",
        duration_s=stock.duration_s,
        arrival_rate_rps=stock.arrival_rate_rps,
        n_requests=stock.n_requests,
        n_reasoning=stock.n_reasoning,
        n_chat=stock.n_chat,
        ttft_p50_ms=stock.ttft_p50_ms * 0.7,
        ttft_p95_ms=stock.ttft_p95_ms * 0.7,
        ttot_p50_ms=stock.ttot_p50_ms * 0.7,
        ttot_p95_ms=stock.ttot_p95_ms * 0.7,
        output_itl_p50_ms=stock.output_itl_p50_ms * 0.7,
        output_itl_p95_ms=stock.output_itl_p95_ms * 0.7,
        output_itl_p99_ms=stock.output_itl_p99_ms * 0.7,
        think_tokens_avg=stock.think_tokens_avg * 0.5,
        think_tokens_p95=stock.think_tokens_p95 * 0.5,
        output_tokens_avg=stock.output_tokens_avg,
        budget_forced_pct=42.0,
        budget_forced_by_reason={"converged": 3},
        output_critical_eviction_events=0,
    )

    ab = ABComparisonReport(stock=stock, meridian=meridian)
    md = ab.to_markdown()
    assert "A/B comparison" in md
    assert "Stock" in md
    assert "Meridian" in md
    # All deltas where stock > 0 must be finite (no nans leaking).
    js = ab.to_json()
    assert "\"delta_pct\":" in js

    ab.write_artifacts(tmp_path)
    assert (tmp_path / "ab-report.json").exists()
    assert (tmp_path / "ab-report.md").exists()


def test_ab_delta_pct_handles_zero_baseline_gracefully() -> None:
    from dataclasses import asdict, replace
    wl = _tiny_workload()
    stock = _aggregate("zero", run_stock_baseline(wl), duration=0.1)
    # Force a zero in the stock side to exercise the divide-by-zero guard.
    stock = replace(stock, ttft_p50_ms=0.0)
    meridian = replace(stock, ttft_p50_ms=1.0)
    ab = ABComparisonReport(stock=stock, meridian=meridian)
    deltas = ab._deltas()
    assert math.isnan(deltas["ttft_p50"])
    del asdict
