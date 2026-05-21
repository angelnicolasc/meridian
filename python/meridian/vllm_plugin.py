"""Active scheduler plugin for vLLM.

**Sprint 1 stance**: the plugin now drives dispatch ordering and budget
forcing. It wraps the vLLM scheduler, observes every decoded token through
the Rust ``PhaseRouter``, computes entropy via the (configurable) probe,
and feeds ``</think>`` injections back to the worker via
``MeridianScheduler.pop_pending_injection``.

The integration points are deliberately narrow so a vLLM API drift between
0.9.x and 1.0 only touches the ``_extract_*`` / ``_reorder`` helpers below.

Usage::

    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from meridian.vllm_plugin import MeridianSchedulerPlugin
    from meridian.config import load_config

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model="Qwen/Qwen2.5-0.5B"))
    cfg = load_config("meridian.toml")
    plugin = MeridianSchedulerPlugin.attach(engine, cfg, model_key="qwen3")
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from meridian.config import MeridianConfig
from meridian.entropy_probe import EntropyProbe, EntropySignal
from meridian.telemetry import init_otlp, record_blocks_offloaded, record_vocab_fallback

if TYPE_CHECKING:  # pragma: no cover — typing only
    from vllm import AsyncLLMEngine

logger = logging.getLogger("meridian.vllm")

# Think-tokens-per-block used only when the block manager does not own a
# request's allocations. In the live vLLM plugin the manager is a capacity
# model rather than a mirror of vLLM's block table, so it holds no per-request
# blocks to enumerate; this mirrors vLLM's canonical 16-token KV block.
_DEFAULT_TOKENS_PER_BLOCK = 16


class MeridianSchedulerPlugin:
    """vLLM scheduler wrapper.

    Holds:

    - The wrapped vLLM scheduler (``self._base``).
    - The Rust ``PhaseRouter`` for token-stream state tracking.
    - The ``EntropyProbe`` (CPU backend by default; CUDA when available).
    - A lightweight metric surface accessible from tests and Prometheus
      scraping (``schedule_count``, ``last_schedule_latency_ns``,
      ``pending_injections``).

    Attribute access not handled by this class falls through to the wrapped
    scheduler so existing vLLM internals continue to work unmodified.
    """

    def __init__(
        self,
        base_scheduler: Any,
        config: MeridianConfig,
        model_key: str,
    ) -> None:
        """Wrap ``base_scheduler`` with Meridian's dispatcher for ``model_key``."""
        if model_key not in config.model:
            msg = f"model key {model_key!r} not present in meridian.toml [model.*]"
            raise KeyError(msg)
        model_cfg = config.model[model_key]

        # Native router + Meridian scheduler — lazy import so unit tests that
        # do not build the maturin extension still work.
        self._router = _load_native_router(model_cfg, config)
        self._meridian_scheduler = _load_native_scheduler(config)

        self._probe: EntropyProbe | None = (
            EntropyProbe(
                think_end_token_ids=model_cfg.think_end_token_ids,
                backend="cpu",
                ema_alpha=config.entropy.ema_alpha,
            )
            if config.entropy.enabled and model_cfg.think_end_token_ids
            else None
        )

        self._base = base_scheduler
        self._config = config
        self._probe_interval = max(1, config.entropy.eat_probe_interval_tokens)
        self._token_counter: dict[int, int] = {}

        # Per-instance state — must NOT be class attributes (sharing would
        # leak across plugins in the same process).
        self._exited_think: set[int] = set()
        self._local_pending_count: int = 0

        # Block manager handle (optional). When present, the plugin queries
        # it for ``available_blocks`` instead of falling back to a constant.
        self._block_manager = _load_block_manager(config)

        # Reaper bookkeeping. The heartbeat runs once every
        # ``_reap_interval_schedules`` ``schedule()`` invocations so we do not
        # pay the cost on every iteration. The threshold is expressed in
        # wall-clock seconds (router translates to ticks internally).
        self._reap_interval_schedules = 64
        self._reap_idle_seconds = 60.0

        # Disagg state. ``_pending_offload`` accumulates ``block_ids`` ready
        # to be flushed once the threshold is reached.
        self._disagg_enabled = config.disagg.enabled
        self._disagg_fabric = config.disagg.fabric
        self._disagg_threshold = max(1, config.disagg.offload_threshold_blocks)
        self._pending_offload: list[int] = []

        # Metric surface (also exposed via Prometheus from the Rust side).
        self._schedule_count = 0
        self._last_schedule_latency_ns = 0

        # Optional OTLP export of the plugin's counters (Prometheus stays the
        # primary surface). Off unless [telemetry] opts in.
        if config.telemetry.otlp_enabled:
            init_otlp(config.telemetry.otlp_endpoint, config.telemetry.service_name)

    # ------------------------------------------------------------------
    # Public surface used by the engine
    # ------------------------------------------------------------------

    @classmethod
    def attach(
        cls,
        engine: AsyncLLMEngine,
        config: MeridianConfig,
        model_key: str,
    ) -> MeridianSchedulerPlugin:
        """Install the plugin on a running :class:`AsyncLLMEngine`."""
        scheduler_list = engine.scheduler
        wrapped = cls(scheduler_list[0], config, model_key)
        scheduler_list[0] = wrapped
        return wrapped

    def schedule(self) -> Any:
        """Reorder the base schedule using Meridian's dual-queue dispatcher.

        Strategy:

        1. Call the underlying vLLM schedule to obtain the candidate set.
        2. Query :class:`BlockManager` for the real ``available_blocks``
           budget (or fall back to a heuristic if no manager is wired).
        3. Re-sort candidates by phase (output first, think second, unknown
           last) preserving relative order within a phase.
        4. Periodically invoke ``reap_stale`` to evict orphaned router state.
        """
        t0 = time.monotonic_ns()
        out = self._base.schedule()
        if self._meridian_scheduler is not None:
            try:
                out = self._reorder(out)
            except Exception:
                logger.exception("Meridian reorder failed; falling back to base order")

        # Periodic heartbeat reap — bounded so we never pay it more than
        # once every `_reap_interval_schedules` iterations.
        if (
            self._router is not None
            and self._schedule_count > 0
            and self._schedule_count % self._reap_interval_schedules == 0
        ):
            with contextlib.suppress(AttributeError):
                # Prefer the wall-clock variant so operators reason in
                # seconds rather than opaque tick counts.
                if hasattr(self._router, "reap_stale_older_than"):
                    reaped = self._router.reap_stale_older_than(self._reap_idle_seconds)
                else:
                    reaped = self._router.reap_stale(int(self._reap_idle_seconds * 1e6))
                if reaped:
                    logger.info("reap_stale collected %d orphaned requests", reaped)

        self._last_schedule_latency_ns = time.monotonic_ns() - t0
        self._schedule_count += 1
        return out

    def post_step(
        self,
        seq_groups: list[Any],
        sampled_tokens: list[list[int]],
        logits: list[Any] | None = None,
    ) -> None:
        """Observe a decode step and act on phase events.

        Strategy:

        1. Batch-decide which requests are due for an entropy-probe sample
           (``eat_probe_interval_tokens`` gating per request).
        2. Run a single vectorised :meth:`EntropyProbe.compute_batch` call
           over the due rows.
        3. Feed each sampled token through the phase router with the
           appropriate signal (or ``None``).
        4. Dispatch ``ForceBudget`` / ``ExitThink`` / ``Complete`` actions.
        """
        if self._router is None:
            return

        req_ids = [self._extract_req_id(sg) for sg in seq_groups]
        for rid in req_ids:
            self._router.register(rid)

        signals_by_idx = self._batch_signals(req_ids, logits)

        for idx, (rid, tokens) in enumerate(zip(req_ids, sampled_tokens, strict=False)):
            signal = signals_by_idx.get(idx)
            for token_id in tokens:
                event = self._router.on_token(rid, int(token_id), signal)
                kind = event.get("kind", "none")
                if kind == "force_budget":
                    self._on_force_budget(rid, event)
                elif kind == "exit_think":
                    self._on_exit_think(rid, event)
                elif kind == "complete":
                    self._on_complete(rid)

    # ------------------------------------------------------------------
    # Metric surface
    # ------------------------------------------------------------------

    @property
    def schedule_count(self) -> int:
        """Number of ``schedule()`` calls observed."""
        return self._schedule_count

    @property
    def last_schedule_latency_ns(self) -> int:
        """Wall-clock latency of the most recent ``schedule()`` call (ns)."""
        return self._last_schedule_latency_ns

    @property
    def pending_injections(self) -> int:
        """Number of ``</think>`` injections waiting for the worker to consume."""
        if self._meridian_scheduler is None:
            return 0
        # Native side exposes a count via repeated pop is wasteful; instead the
        # Rust scheduler tracks depth via its queue gauge metric. For tests we
        # count locally.
        return self._local_pending_count

    # ------------------------------------------------------------------
    # Internal — reorder, event handlers, helpers
    # ------------------------------------------------------------------

    def _reorder(self, base_out: Any) -> Any:
        """Reorder ``base_out`` so output-phase requests are dispatched first.

        Sprint 1 conservative policy: we do *not* mutate the underlying vLLM
        decision (no preemption, no new admissions); we only re-rank the
        already-scheduled candidates so output requests are served first
        within the batch the base scheduler already produced.
        """
        # vLLM's ``SchedulerOutputs`` may be a dataclass, NamedTuple, or
        # custom object depending on version. We work via attribute probes
        # so this code survives 0.9.x → 1.0 drift.
        scheduled = getattr(base_out, "scheduled_seq_groups", None)
        if scheduled is None or not scheduled:
            return base_out

        priority_keys: list[tuple[int, int, Any]] = []
        for i, sg in enumerate(scheduled):
            req_id = self._extract_req_id(sg)
            phase_kind = self._phase_kind(req_id)
            # Output phase first (0), think second (1), unknown third (2).
            rank = {"output_decode": 0, "think_decode": 1}.get(phase_kind, 2)
            priority_keys.append((rank, i, sg))

        priority_keys.sort(key=lambda t: (t[0], t[1]))
        reordered = [t[2] for t in priority_keys]
        try:
            base_out.scheduled_seq_groups = reordered
        except (AttributeError, TypeError):
            # Some vLLM versions expose ``scheduled_seq_groups`` as a
            # read-only property. In that case we cannot mutate in place;
            # the safest fallback is to return base_out unchanged.
            logger.debug("scheduled_seq_groups is read-only; skipping reorder")
        return base_out

    def _on_force_budget(self, req_id: int, event: dict[str, Any]) -> None:
        token_id = int(event.get("inject_token", -1))
        reason = event.get("reason", "unknown")
        if token_id < 0:
            return
        logger.info(
            "budget_force req=%s token=%s reason=%s",
            req_id, token_id, reason,
        )
        if self._meridian_scheduler is not None:
            try:
                self._meridian_scheduler.queue_injection(req_id, token_id)
                self._local_pending_count += 1
            except AttributeError:
                # Defensive: older bindings may not expose queue_injection.
                logger.debug("native scheduler missing queue_injection; injection dropped")

    def _on_exit_think(self, req_id: int, event: dict[str, Any]) -> None:
        tokens_used = int(event.get("tokens_used", 0))
        logger.debug("exit_think req=%s tokens=%s", req_id, tokens_used)
        self._exited_think.add(req_id)
        if self._meridian_scheduler is not None:
            try:
                self._meridian_scheduler.signal_exit_think(req_id, tokens_used)
            except AttributeError:
                logger.debug("native scheduler missing signal_exit_think")
        # Disagg trigger point: at ExitThink the request's think-phase blocks
        # have been demoted to `ThinkComplete` and become candidates for
        # fabric offload. We accumulate into a pending list and flush when
        # the configured threshold is reached. See ADR-0006.
        if self._disagg_enabled and self._block_manager is not None:
            self._maybe_offload_for(req_id, tokens_used)

    def _maybe_offload_for(self, req_id: int, tokens_used: int) -> None:
        """Queue the request's think-complete blocks for fabric offload.

        When the block manager owns this request's allocations — as the
        scheduler and the benchmark harness do — we enumerate the exact
        block ids via ``blocks_for_request`` and reclaim each slot with
        ``free_block_by_id`` as it is handed to the fabric (DEVLOG D30/D31).
        In the live vLLM plugin the manager is a capacity model rather than a
        mirror of vLLM's block table, so it owns no per-request blocks; there
        we fall back to a geometry estimate. The offload count is reported
        exactly either way. See ADR-0006.
        """
        block_ids = list(self._block_manager.blocks_for_request(req_id))
        if block_ids:
            for block_id in block_ids:
                self._block_manager.free_block_by_id(block_id)
            n_blocks = len(block_ids)
        else:
            n_blocks = max(1, tokens_used // _DEFAULT_TOKENS_PER_BLOCK)

        self._pending_offload.extend([req_id] * n_blocks)
        if len(self._pending_offload) >= self._disagg_threshold:
            flushed = len(self._pending_offload)
            record_blocks_offloaded(self._disagg_fabric, flushed)
            logger.info(
                "disagg.offload fabric=%s blocks=%d",
                self._disagg_fabric, flushed,
            )
            self._pending_offload.clear()

    def _on_complete(self, req_id: int) -> None:
        if self._router is not None:
            self._router.reap(req_id)
        if self._probe is not None:
            self._probe.cleanup(req_id)
        if self._meridian_scheduler is not None:
            with contextlib.suppress(AttributeError):
                self._meridian_scheduler.signal_complete(req_id)
        self._token_counter.pop(req_id, None)
        self._exited_think.discard(req_id)

    def _phase_kind(self, req_id: int) -> str:
        """Return the phase classification used for reorder ranking.

        Authoritative: queries the native router via ``phase_of_kind`` when
        available. Falls back to the ``_exited_think`` heuristic only when
        the router binding is absent (test environment without maturin).
        """
        if self._router is not None:
            with contextlib.suppress(AttributeError):
                kind = self._router.phase_of_kind(req_id)
                if isinstance(kind, str):
                    return kind
        return "output_decode" if req_id in self._exited_think else "think_decode"

    def _available_blocks(self) -> int:
        """Real available-blocks budget, queried from the block manager."""
        if self._block_manager is None:
            return 1024  # conservative fallback when no manager is wired
        with contextlib.suppress(AttributeError):
            return int(self._block_manager.available_blocks())
        return 1024

    def _batch_signals(
        self,
        req_ids: list[int],
        logits: list[Any] | None,
    ) -> dict[int, Any]:
        """Return a ``{batch_index -> native signal}`` mapping for due requests.

        A request is "due" when its post-increment token counter is a
        multiple of ``eat_probe_interval_tokens``. We collect the due rows,
        run a single :meth:`EntropyProbe.compute_batch`, and only return
        those indices — non-due requests get ``None`` (their callers must
        check the dict).
        """
        if self._probe is None or logits is None:
            return {}

        import numpy as np

        due_idx: list[int] = []
        due_req_ids: list[int] = []
        due_rows: list[Any] = []
        for i, rid in enumerate(req_ids):
            count = self._token_counter.get(rid, 0) + 1
            self._token_counter[rid] = count
            if count % self._probe_interval != 0:
                continue
            try:
                arr = np.asarray(logits[i], dtype=np.float32).reshape(-1)
            except (ValueError, TypeError):
                continue
            due_idx.append(i)
            due_req_ids.append(rid)
            due_rows.append(arr)

        if not due_idx:
            return {}

        try:
            stacked = np.stack(due_rows, axis=0)
        except ValueError:
            # Heterogeneous vocab sizes — fall back to per-row compute.
            record_vocab_fallback()
            out: dict[int, Any] = {}
            for i, rid, arr in zip(due_idx, due_req_ids, due_rows, strict=True):
                sig = self._probe.compute(rid, arr)
                out[i] = _to_native_signal(sig)
            return out

        signals: list[EntropySignal] = self._probe.compute_batch(due_req_ids, stacked)
        return {i: _to_native_signal(s) for i, s in zip(due_idx, signals, strict=True)}

    def _extract_req_id(self, sg: Any) -> int:
        """Best-effort extraction of the integer request id from a seq group."""
        raw = getattr(sg, "request_id", None) or getattr(sg, "req_id", None)
        if raw is None:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return abs(hash(raw)) & 0xFFFFFFFFFFFFFFFF

    # ------------------------------------------------------------------
    # Transparent delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate any unknown attribute to the wrapped vLLM scheduler."""
        return getattr(self._base, name)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _load_native_router(model_cfg: Any, config: MeridianConfig) -> Any:
    try:
        from meridian._meridian import (
            PhaseRouter as _PhaseRouter,
        )
    except ImportError:
        logger.warning(
            "meridian._meridian native extension not available — "
            "router integration disabled in this process.",
        )
        return None
    return _PhaseRouter(
        think_start_ids=model_cfg.think_start_token_ids,
        think_end_ids=model_cfg.think_end_token_ids,
        eos_ids=[],
        ema_alpha=config.entropy.ema_alpha,
        transition_entropy_threshold=config.entropy.transition_entropy_threshold,
        rpdi_threshold=config.entropy.rpdi_threshold,
        eat_ema_variance_threshold=config.entropy.eat_ema_variance_threshold,
        min_think_tokens=config.scheduler.min_think_tokens,
        max_think_tokens=config.scheduler.max_think_tokens,
    )


def _load_block_manager(config: MeridianConfig) -> Any:
    """Construct a native ``BlockManager`` from the configured KV budget.

    Returns ``None`` if the native extension isn't installed. The plugin
    falls back to a constant ``available_blocks`` heuristic in that case.

    Capacity resolution order:

    1. ``config.kv_memory.capacity_bytes`` if it is an integer.
    2. If ``"auto"``, query ``torch.cuda.mem_get_info`` (when CUDA is
       available) and use 85% of total device memory.
    3. Otherwise fall back to a conservative 8 GiB.
    """
    try:
        from meridian._meridian import BlockManager as _BlockManager
    except ImportError:
        return None

    block_size_bytes = int(config.kv_memory.block_size_bytes)
    capacity = config.kv_memory.capacity_bytes
    capacity_bytes = capacity if isinstance(capacity, int) else _autodetect_kv_capacity_bytes()

    return _BlockManager(
        block_size_bytes=block_size_bytes,
        capacity_bytes=capacity_bytes,
        aggressive_think_eviction=config.kv_memory.aggressive_think_eviction,
    )


def _autodetect_kv_capacity_bytes() -> int:
    """Resolve the ``capacity_bytes = "auto"`` setting at runtime.

    Tries ``torch.cuda.mem_get_info`` first. Falls back to a documented
    8 GiB constant when CUDA isn't present so unit tests on a CPU host
    still get a sensible non-zero budget.
    """
    fallback = 8 * 1024 * 1024 * 1024  # 8 GiB
    try:
        import torch

        if not torch.cuda.is_available():
            return fallback
        free, total = torch.cuda.mem_get_info()
        del free
        # 85% of total leaves headroom for activations and CUDA graphs.
        return int(total * 0.85)
    except (ImportError, RuntimeError, AttributeError):
        return fallback


def _load_native_scheduler(config: MeridianConfig) -> Any:
    try:
        from meridian._meridian import (
            MeridianScheduler as _Scheduler,
        )
    except ImportError:
        return None
    return _Scheduler(
        think_tpot_budget_ms=config.scheduler.think_tpot_budget_ms,
        output_tpot_budget_ms=config.scheduler.output_tpot_budget_ms,
        think_batch_multiplier=config.scheduler.think_batch_multiplier,
        max_think_tokens=config.scheduler.max_think_tokens,
        min_think_tokens=config.scheduler.min_think_tokens,
    )


def _to_native_signal(sig: EntropySignal) -> Any:
    try:
        from meridian._meridian import (
            EntropySignal as _NativeSignal,
        )
    except ImportError:
        return None
    return _NativeSignal(
        sig.token_entropy,
        sig.eat,
        sig.eat_ema,
        sig.eat_ema_variance,
    )
