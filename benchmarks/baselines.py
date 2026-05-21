"""Baseline schedulers used for A/B benchmark comparison.

The point of the A/B harness is to isolate *what Meridian's dual-queue +
phase-aware policy buys you*, not to relitigate vLLM versus alternatives.
The baseline therefore replays the same workload over the same synthetic
decoder, but routes requests through a stock priority-weight scheduler —
the policy vLLM ≤ 0.8 used before phase-awareness was on the table.

Why this matters
----------------

A naive A/B that compares Meridian against "no scheduler" is uninteresting
(of course it wins). A useful A/B compares Meridian against a reasonable
default that someone could deploy today. ``StockSchedulerBaseline`` is
that default: single FIFO queue, priority by arrival order, no phase
awareness, no budget forcing, no tier-aware eviction.

The baseline runs in-process so the harness can do many sweeps in CI
without needing a real GPU; the *real-vllm* mode in
:mod:`benchmarks.meridian_bench` covers cross-checking against an actual
``AsyncLLMEngine`` on the GPU job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchmarks.metrics import RequestResult
from benchmarks.workloads import WorkloadRequest


@dataclass(slots=True)
class StockSchedulerBaseline:
    """Single-queue priority-by-arrival scheduler.

    Configuration is intentionally minimal — the only knob is the per-token
    latency, which the harness calibrates to match the same synthetic
    decoder Meridian uses. The two paths therefore agree on per-token
    cost; the only thing that differs is dispatch order and budget
    forcing.

    The baseline does not preempt; once a request starts decoding it runs
    to its natural end (or to its expected token budget). Output-phase
    streaming under contention is therefore as bad as it can plausibly
    be — which is the whole point of having a baseline.
    """

    think_per_token_us: float = 6.0
    output_per_token_us: float = 18.0
    prefill_per_token_us: float = 1.2
    _arrival_counter: int = field(default=0, init=False)

    def run_request(self, wl: WorkloadRequest) -> RequestResult:
        """Replay one workload request through the baseline scheduler.

        Returns the same :class:`RequestResult` shape Meridian produces so
        the report aggregator can diff them field-for-field.
        """
        self._arrival_counter += 1
        result = RequestResult(req_id=wl.request_id, kind=wl.kind)

        t0 = time.perf_counter()
        prefill_tokens = max(8, len(wl.prompt) // 4)
        _busy_sleep_us(self.prefill_per_token_us * prefill_tokens)
        result.ttft_ms = (time.perf_counter() - t0) * 1000.0

        # Think phase — no budget forcing under the stock baseline. The
        # full expected think token budget is consumed, modelling the
        # worst case where the model would have stopped on its own
        # eventually but the scheduler never intervenes.
        if wl.kind == "reasoning":
            for _ in range(wl.expected_think_tokens):
                _busy_sleep_us(self.think_per_token_us)
            result.think_tokens = wl.expected_think_tokens

        # Output phase — same per-token cost as the Meridian path.
        ttot_start = time.perf_counter()
        prev = ttot_start
        for i in range(wl.expected_output_tokens):
            _busy_sleep_us(self.output_per_token_us)
            now = time.perf_counter()
            if i == 0:
                result.ttot_ms = (now - ttot_start) * 1000.0
            else:
                result.output_itl_ms.append((now - prev) * 1000.0)
            prev = now
        result.output_tokens = wl.expected_output_tokens

        return result


@dataclass(slots=True)
class StaticBudgetBaseline:
    """Single-queue scheduler with a static think-token budget.

    Models vLLM 0.9's ``thinking_token_budget``: a fixed per-request cap on
    think tokens. When a reasoning request would exceed the cap the scheduler
    injects ``</think>`` unconditionally — no entropy signal, no convergence
    detection. This is the prior art Meridian's EAT/RPDI-driven forcing aims
    to beat: the static cap fires on a counter, so it either cuts off a
    request that was still making progress or lets a converged request run to
    the cap. Dispatch order is otherwise identical to
    :class:`StockSchedulerBaseline` (priority by arrival, no phase awareness).
    """

    think_token_budget: int = 2048
    think_per_token_us: float = 6.0
    output_per_token_us: float = 18.0
    prefill_per_token_us: float = 1.2
    _arrival_counter: int = field(default=0, init=False)

    def run_request(self, wl: WorkloadRequest) -> RequestResult:
        """Replay one request, forcing ``</think>`` at the static budget."""
        self._arrival_counter += 1
        result = RequestResult(req_id=wl.request_id, kind=wl.kind)

        t0 = time.perf_counter()
        prefill_tokens = max(8, len(wl.prompt) // 4)
        _busy_sleep_us(self.prefill_per_token_us * prefill_tokens)
        result.ttft_ms = (time.perf_counter() - t0) * 1000.0

        if wl.kind == "reasoning":
            think = min(wl.expected_think_tokens, self.think_token_budget)
            for _ in range(think):
                _busy_sleep_us(self.think_per_token_us)
            result.think_tokens = think
            if wl.expected_think_tokens > self.think_token_budget:
                result.budget_forced = True
                result.force_reason = "static_cap"

        ttot_start = time.perf_counter()
        prev = ttot_start
        for i in range(wl.expected_output_tokens):
            _busy_sleep_us(self.output_per_token_us)
            now = time.perf_counter()
            if i == 0:
                result.ttot_ms = (now - ttot_start) * 1000.0
            else:
                result.output_itl_ms.append((now - prev) * 1000.0)
            prev = now
        result.output_tokens = wl.expected_output_tokens

        return result


def _busy_sleep_us(us: float) -> None:
    """Microsecond-accurate sleep — same primitive the Meridian decoder uses."""
    if us <= 0:
        return
    if us < 800:
        end = time.perf_counter_ns() + int(us * 1000)
        while time.perf_counter_ns() < end:
            pass
    else:
        time.sleep(us / 1_000_000.0)


def run_stock_baseline(
    workload: list[WorkloadRequest],
    *,
    think_per_token_us: float = 6.0,
    output_per_token_us: float = 18.0,
) -> list[RequestResult]:
    """Replay ``workload`` through the stock baseline and return raw results.

    The harness aggregates these into a :class:`benchmarks.metrics.BenchmarkReport`
    via :func:`benchmarks.metrics.aggregate`, then diffs against the Meridian
    report via :class:`benchmarks.metrics.ABComparisonReport`.
    """
    sched = StockSchedulerBaseline(
        think_per_token_us=think_per_token_us,
        output_per_token_us=output_per_token_us,
    )
    return [sched.run_request(wl) for wl in workload]


def run_static_budget_baseline(
    workload: list[WorkloadRequest],
    *,
    think_token_budget: int = 2048,
    think_per_token_us: float = 6.0,
    output_per_token_us: float = 18.0,
) -> list[RequestResult]:
    """Replay ``workload`` through the static-budget baseline and return results.

    Mirrors :func:`run_stock_baseline` but caps think tokens at
    ``think_token_budget``, modelling vLLM 0.9's ``thinking_token_budget``.
    """
    sched = StaticBudgetBaseline(
        think_token_budget=think_token_budget,
        think_per_token_us=think_per_token_us,
        output_per_token_us=output_per_token_us,
    )
    return [sched.run_request(wl) for wl in workload]
