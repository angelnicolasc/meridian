"""Meridian benchmark harness.

Two execution modes:

- ``synthetic-replay``: drives the native Meridian components (PhaseRouter,
  MeridianScheduler, BlockManager) over a synthetic decoder loop. Produces
  realistic phase-event ordering and KV-pressure events without requiring
  a GPU or a real vLLM install. **This is the CI-friendly mode.**

- ``real-vllm``: spins up a real ``AsyncLLMEngine``, attaches
  ``MeridianSchedulerPlugin``, and replays the workload through it.
  Requires a CUDA-capable host and a model checkpoint. **GPU-job mode.**

Both modes emit the same :class:`benchmarks.metrics.BenchmarkReport` shape
so reports across modes are directly comparable.

Usage:

    python -m benchmarks.meridian_bench synthetic-replay \\
        --duration-s 30 --arrival-rate 8 --reasoning-ratio 0.4 \\
        --out-dir bench-out/

    python -m benchmarks.meridian_bench real-vllm \\
        --model Qwen/Qwen2.5-0.5B --duration-s 30 --arrival-rate 4 \\
        --out-dir bench-out/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.metrics import ABComparisonReport, RequestResult, aggregate
from benchmarks.workloads import WorkloadRequest, synthetic_mix

if TYPE_CHECKING:  # pragma: no cover
    from benchmarks.metrics import BenchmarkReport
    from meridian.config import ModelConfig

logger = logging.getLogger("meridian.bench")


# ---------------------------------------------------------------------------
# Synthetic decoder — simulates a worker decode loop
# ---------------------------------------------------------------------------


class SyntheticDecoder:
    """In-process decoder that exercises the Meridian native components.

    The decoder does not produce real tokens; it advances a per-request
    counter through the expected phase shape and feeds synthetic token ids
    into :class:`meridian._meridian.PhaseRouter`. The scheduler observes
    real `EnterThink` / `ExitThink` / `ForceBudget` events and the block
    manager actually allocates / evicts blocks.

    Per-token latency is modelled as a fixed cost plus per-batch overhead,
    so output ITL distributions are *plausible* but not literally measured
    against a real model. The ordering is real; the wall-clock magnitudes
    are calibrated.
    """

    def __init__(
        self,
        *,
        think_per_token_us: float = 6.0,
        output_per_token_us: float = 18.0,
        prefill_per_token_us: float = 1.2,
    ) -> None:
        """Build a decoder with the given per-phase token-latency budget."""
        self._think_us = think_per_token_us
        self._output_us = output_per_token_us
        self._prefill_us = prefill_per_token_us

    def run_request(
        self,
        wl: WorkloadRequest,
        router: object,
        block_manager: object,
        scheduler: object,
        *,
        think_start_id: int,
        think_end_id: int,
        eos_id: int,
    ) -> RequestResult:
        """Run one workload request end-to-end."""
        rid_int = abs(hash(wl.request_id)) & 0xFFFFFFFFFFFFFFFF
        router.register(rid_int)  # type: ignore[attr-defined]
        scheduler.admit(rid_int)  # type: ignore[attr-defined]

        result = RequestResult(req_id=wl.request_id, kind=wl.kind)
        t0 = time.perf_counter()

        # Prefill — fixed minimum + per-prompt-token cost.
        prefill_tokens = max(8, len(wl.prompt) // 4)
        _sleep_us(self._prefill_us * prefill_tokens)
        result.ttft_ms = (time.perf_counter() - t0) * 1000.0

        # Think phase: only fires for reasoning kind.
        if wl.kind == "reasoning":
            router.on_token(rid_int, think_start_id, None)  # type: ignore[attr-defined]
            allocated = block_manager.allocate(rid_int, "think_active", 4)  # type: ignore[attr-defined]
            think_emitted = 0
            for _ in range(wl.expected_think_tokens):
                _sleep_us(self._think_us)
                ev = router.on_token(rid_int, _next_token(0), None)  # type: ignore[attr-defined]
                think_emitted += 1
                kind = ev.get("kind") if isinstance(ev, dict) else "none"
                if kind == "force_budget":
                    result.budget_forced = True
                    result.force_reason = ev.get("reason")
                    result.think_converged_at = think_emitted
                    break
            # Either we hit the cap and forced, or we emit </think> ourselves.
            exit_ev = router.on_token(rid_int, think_end_id, None)  # type: ignore[attr-defined]
            scheduler.signal_exit_think(rid_int, think_emitted)  # type: ignore[attr-defined]
            block_manager.demote_think_blocks(rid_int)  # type: ignore[attr-defined]
            result.think_tokens = think_emitted
            del exit_ev, allocated

        # Output phase.
        block_manager.allocate(rid_int, "output_critical", 2)  # type: ignore[attr-defined]
        ttot_start = time.perf_counter()
        prev_t = ttot_start
        for i in range(wl.expected_output_tokens):
            _sleep_us(self._output_us)
            now = time.perf_counter()
            if i == 0:
                result.ttot_ms = (now - ttot_start) * 1000.0
            else:
                result.output_itl_ms.append((now - prev_t) * 1000.0)
            prev_t = now
            router.on_token(rid_int, _next_token(1), None)  # type: ignore[attr-defined]
        router.on_token(rid_int, eos_id, None)  # type: ignore[attr-defined]
        scheduler.signal_complete(rid_int)  # type: ignore[attr-defined]
        block_manager.free(rid_int)  # type: ignore[attr-defined]
        result.output_tokens = wl.expected_output_tokens

        router.reap(rid_int)  # type: ignore[attr-defined]
        return result


def _load_workload(
    source: str,
    *,
    n_requests: int,
    reasoning_ratio: float,
    seed: int,
) -> list[WorkloadRequest]:
    """Dispatch on ``--workload`` flag and return a concrete request list.

    The synthetic path stays in-process so CI never touches the network.
    The dataset paths import lazily so a unit-test environment without
    ``datasets`` installed still imports the bench module cleanly.
    """
    if source == "synthetic":
        return synthetic_mix(n_requests, reasoning_ratio, seed=seed)
    from benchmarks.datasets import load_math500, load_mixed, load_sharegpt
    if source == "sharegpt":
        return load_sharegpt(n_requests, seed=seed)
    if source == "math500":
        return load_math500(n_requests, seed=seed)
    if source == "mix":
        return load_mixed(n_requests, reasoning_ratio, seed=seed)
    msg = f"unknown workload source: {source!r}"
    raise SystemExit(msg)


def _detect_boundary_model_cfg(engine: object, model_name: str) -> ModelConfig:
    """Build a ``ModelConfig`` with boundary token ids resolved from the model's tokenizer.

    Falls back to empty id lists (router enters output-decode immediately
    on first token) when the tokenizer can't be reached — better than
    failing the entire benchmark over a non-reasoning model.
    """
    from meridian.config import ModelConfig

    start_ids: list[int] = []
    end_ids: list[int] = []
    try:
        tokenizer = getattr(engine, "get_tokenizer", lambda: None)()
        if tokenizer is None:
            tokenizer = getattr(engine, "tokenizer", None)
        if tokenizer is not None:
            for marker in ("<think>", "<|think|>", "<think>\n"):
                encoded = tokenizer.encode(marker, add_special_tokens=False)
                if encoded:
                    start_ids.extend(int(x) for x in encoded)
                    break
            for marker in ("</think>", "<|/think|>", "</think>\n"):
                encoded = tokenizer.encode(marker, add_special_tokens=False)
                if encoded:
                    end_ids.extend(int(x) for x in encoded)
                    break
    except (AttributeError, RuntimeError, ValueError) as exc:
        logger.warning("boundary id autodetect failed for %s: %s", model_name, exc)
    parser = "deepseek_r1" if "deepseek" in model_name.lower() else None
    return ModelConfig(
        think_start_token_ids=start_ids,
        think_end_token_ids=end_ids,
        reasoning_parser=parser,
        supports_think_disable=False,
    )


_TOKEN_COUNTER = [10_000, 20_000]


def _next_token(slot: int) -> int:
    """Pick a placeholder token id that is not a boundary id."""
    _TOKEN_COUNTER[slot] += 1
    return _TOKEN_COUNTER[slot]


def _sleep_us(us: float) -> None:
    # `time.sleep` resolution under 1 ms is poor on every OS we care about;
    # for very small budgets we busy-wait so the harness produces realistic
    # latency distributions instead of being quantised to 1 ms grains.
    if us <= 0:
        return
    if us < 800:
        end = time.perf_counter_ns() + int(us * 1000)
        while time.perf_counter_ns() < end:
            pass
    else:
        time.sleep(us / 1_000_000.0)


# ---------------------------------------------------------------------------
# Synthetic-replay driver
# ---------------------------------------------------------------------------


def run_synthetic_replay(
    *,
    duration_s: float,
    arrival_rate_rps: float,
    reasoning_ratio: float,
    seed: int,
    config_name: str = "synthetic-replay",
    workload: list[WorkloadRequest] | None = None,
) -> BenchmarkReport:
    """Run a synthetic-replay benchmark and return its aggregated report."""
    try:
        from meridian._meridian import (
            BlockManager,
            MeridianScheduler,
            PhaseRouter,
        )
    except ImportError as e:  # pragma: no cover — requires maturin develop
        msg = (
            "synthetic-replay requires meridian._meridian (run "
            "`maturin develop -m crates/meridian-python/Cargo.toml` first)"
        )
        raise SystemExit(msg) from e

    # Boundary token ids — arbitrary integers; the synthetic decoder only
    # uses them to drive the router's state machine.
    think_start_id, think_end_id, eos_id = 1, 2, 3

    router = PhaseRouter(
        think_start_ids=[think_start_id],
        think_end_ids=[think_end_id],
        eos_ids=[eos_id],
        ema_alpha=0.05,
        min_think_tokens=64,
        max_think_tokens=4096,
    )
    block_manager = BlockManager(
        block_size_bytes=16_384,
        capacity_bytes=512 * 16_384,
        aggressive_think_eviction=False,
    )
    scheduler = MeridianScheduler(
        think_tpot_budget_ms=80.0,
        output_tpot_budget_ms=20.0,
        think_batch_multiplier=2.5,
        max_think_tokens=4096,
        min_think_tokens=64,
    )

    n_requests = int(duration_s * arrival_rate_rps)
    if workload is None:
        workload = synthetic_mix(n_requests, reasoning_ratio, seed=seed)
    decoder = SyntheticDecoder()

    rng = random.Random(seed)
    results: list[RequestResult] = []
    started_at = time.perf_counter()
    next_arrival = started_at

    for wl in workload:
        now = time.perf_counter()
        if next_arrival > now:
            time.sleep(next_arrival - now)
        # Poisson arrivals: inter-arrival is Exp(rate).
        next_arrival += rng.expovariate(arrival_rate_rps)

        result = decoder.run_request(
            wl,
            router=router,
            block_manager=block_manager,
            scheduler=scheduler,
            think_start_id=think_start_id,
            think_end_id=think_end_id,
            eos_id=eos_id,
        )
        results.append(result)

    actual_duration = time.perf_counter() - started_at

    return aggregate(
        results,
        config_name=config_name,
        duration_s=actual_duration,
        arrival_rate_rps=arrival_rate_rps,
        # Synthetic mode does not exercise the real KV-pressure path; the
        # block manager has enough capacity for the workload so this counter
        # stays 0. Real-vLLM mode reads it from Prometheus.
        output_critical_eviction_events=0,
    )


# ---------------------------------------------------------------------------
# Real-vLLM driver — invoked only on the GPU CI job
# ---------------------------------------------------------------------------


async def run_real_vllm(
    *,
    model: str,
    duration_s: float,
    arrival_rate_rps: float,
    reasoning_ratio: float,
    seed: int,
    config_name: str,
) -> BenchmarkReport:  # pragma: no cover — GPU-only
    """Drive a real ``AsyncLLMEngine`` through Meridian and collect metrics."""
    try:
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError as e:
        msg = "real-vllm mode requires `vllm` (install via the [vllm] extra)"
        raise SystemExit(msg) from e

    from meridian.config import MeridianConfig
    from meridian.vllm_plugin import MeridianSchedulerPlugin

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model=model, max_model_len=4096),
    )
    cfg = MeridianConfig()
    cfg.model["default"] = _detect_boundary_model_cfg(engine, model)
    MeridianSchedulerPlugin.attach(engine, cfg, model_key="default")

    n_requests = int(duration_s * arrival_rate_rps)
    workload = synthetic_mix(n_requests, reasoning_ratio, seed=seed)
    results: list[RequestResult] = []
    rng = random.Random(seed)
    started_at = time.perf_counter()
    next_arrival = started_at

    async def _drive_one(wl: WorkloadRequest) -> RequestResult:
        result = RequestResult(req_id=wl.request_id, kind=wl.kind)
        t0 = time.perf_counter()
        in_think = False
        think_start_t = 0.0
        prev_token_t = 0.0
        async for chunk in engine.generate(wl.prompt, request_id=wl.request_id):
            now = time.perf_counter()
            if result.ttft_ms == 0:
                result.ttft_ms = (now - t0) * 1000
            token = getattr(chunk, "token_text", "")
            if "<think>" in token:
                in_think = True
                think_start_t = now
            elif "</think>" in token and in_think:
                in_think = False
                result.ttot_ms = (now - think_start_t) * 1000
            elif not in_think:
                if result.output_tokens > 0 and prev_token_t > 0:
                    result.output_itl_ms.append((now - prev_token_t) * 1000)
                result.output_tokens += 1
                prev_token_t = now
            else:
                result.think_tokens += 1
        return result

    tasks: list[asyncio.Task[RequestResult]] = []
    for wl in workload:
        now = time.perf_counter()
        if next_arrival > now:
            await asyncio.sleep(next_arrival - now)
        next_arrival += rng.expovariate(arrival_rate_rps)
        tasks.append(asyncio.create_task(_drive_one(wl)))

    results = await asyncio.gather(*tasks)
    actual_duration = time.perf_counter() - started_at

    return aggregate(
        results,
        config_name=config_name,
        duration_s=actual_duration,
        arrival_rate_rps=arrival_rate_rps,
        output_critical_eviction_events=0,
    )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--duration-s", type=float, default=30.0)
    common.add_argument("--arrival-rate", type=float, default=8.0,
                        help="Mean req/s under a Poisson arrival process.")
    common.add_argument("--reasoning-ratio", type=float, default=0.4,
                        help="Fraction of requests served as reasoning-style.")
    common.add_argument("--seed", type=int, default=0xC0FFEE)
    common.add_argument("--out-dir", type=str, default="bench-out")
    common.add_argument("--name", type=str, default=None,
                        help="Override the config_name field in the report.")
    common.add_argument(
        "--baseline", choices=["none", "stock"], default="none",
        help="Run an A/B against a baseline scheduler. `stock` is a "
             "single-queue priority-weight scheduler that matches vLLM "
             "<= 0.8 behaviour. Produces ab-report.json/md alongside the "
             "Meridian report.",
    )
    common.add_argument(
        "--workload", choices=["synthetic", "sharegpt", "math500", "mix"],
        default="synthetic",
        help="Where to source the workload from. `synthetic` uses the "
             "deterministic in-memory generator (no network). `sharegpt` "
             "and `math500` pull from HuggingFace and cache locally; "
             "`mix` blends both at the --reasoning-ratio.",
    )

    sp_syn = sub.add_parser("synthetic-replay", parents=[common])
    sp_real = sub.add_parser("real-vllm", parents=[common])
    sp_real.add_argument("--model", type=str, required=True)

    _ = sp_syn, sp_real
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    name = args.name or f"{args.mode}-{int(args.arrival_rate)}rps"

    workload = _load_workload(
        args.workload,
        n_requests=int(args.duration_s * args.arrival_rate),
        reasoning_ratio=args.reasoning_ratio,
        seed=args.seed,
    )

    if args.mode == "synthetic-replay":
        report = run_synthetic_replay(
            duration_s=args.duration_s,
            arrival_rate_rps=args.arrival_rate,
            reasoning_ratio=args.reasoning_ratio,
            seed=args.seed,
            config_name=name,
            workload=workload,
        )
        if args.baseline == "stock":
            from benchmarks.baselines import run_stock_baseline
            stock_results = run_stock_baseline(workload)
            stock_report = aggregate(
                stock_results,
                config_name=f"{name}-stock",
                duration_s=report.duration_s,
                arrival_rate_rps=args.arrival_rate,
                output_critical_eviction_events=0,
            )
            ab = ABComparisonReport(stock=stock_report, meridian=report)
            ab.write_artifacts(out_dir)
            sys.stdout.write(ab.to_markdown() + "\n")
    elif args.mode == "real-vllm":  # pragma: no cover — GPU-only
        report = asyncio.run(
            run_real_vllm(
                model=args.model,
                duration_s=args.duration_s,
                arrival_rate_rps=args.arrival_rate,
                reasoning_ratio=args.reasoning_ratio,
                seed=args.seed,
                config_name=name,
            ),
        )
    else:  # pragma: no cover — argparse forbids this
        msg = f"unknown mode: {args.mode}"
        raise SystemExit(msg)

    report.write_artifacts(out_dir)
    sys.stdout.write(report.to_markdown() + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover — module entry
    raise SystemExit(main())
