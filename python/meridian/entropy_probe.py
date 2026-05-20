"""Public ``EntropyProbe`` facade.

Owns:

- Backend selection (``"cpu"`` or ``"cuda"``).
- Per-request EMA state for ``eat`` so the variance threshold is computed
  consistently regardless of backend.
- The :class:`EntropySignal` record consumed by the Rust ``PhaseRouter`` and
  the vLLM plugin.

The backend itself is stateless — see :mod:`meridian._backends` for the
contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from meridian._backends import CpuEntropyBackend, CudaEntropyBackend, EntropyBackend

BackendName = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class EntropySignal:
    """Per-token entropy signal returned by :meth:`EntropyProbe.compute`."""

    token_entropy: float
    """Shannon entropy of the next-token distribution, in nats."""

    eat: float
    """Raw EAT signal at this decode step (probability mass on `</think>`)."""

    eat_ema: float
    """Exponential moving average of ``eat``."""

    eat_ema_variance: float
    """Running variance of ``eat`` via ``E[X^2] - E[X]^2`` (non-negative)."""


class EntropyProbe:
    """Stateful probe wrapping a stateless backend.

    Lifecycle:

    1. Construct once per serving worker with the resolved ``think_end_token_ids``.
    2. Call :meth:`compute` on each decode step (or every Nth, per
       ``entropy.eat_probe_interval_tokens``).
    3. Call :meth:`cleanup` when a request completes to release EMA state.
    """

    _DEFAULT_ALPHA: Final[float] = 0.05

    def __init__(
        self,
        think_end_token_ids: list[int],
        *,
        backend: BackendName = "cpu",
        ema_alpha: float = _DEFAULT_ALPHA,
    ) -> None:
        """Build a probe bound to a specific set of ``</think>`` token ids."""
        if not 0.0 <= ema_alpha <= 1.0:
            msg = "ema_alpha must be in [0, 1]"
            raise ValueError(msg)
        if not think_end_token_ids:
            msg = "think_end_token_ids must not be empty"
            raise ValueError(msg)

        self._think_end_ids: NDArray[np.int32] = np.asarray(
            think_end_token_ids, dtype=np.int32,
        )
        self._alpha = float(ema_alpha)
        # Both concrete backends satisfy the `EntropyBackend` Protocol at
        # runtime but mypy can't infer Protocol conformance from a union, so
        # we cast explicitly. `@runtime_checkable` keeps the contract real.
        self._backend: EntropyBackend = cast(
            "EntropyBackend",
            CudaEntropyBackend() if backend == "cuda" else CpuEntropyBackend(),
        )

        self._eat_ema: dict[int, float] = {}
        self._eat_ema_sq: dict[int, float] = {}

    @property
    def backend_name(self) -> str:
        """Backend currently in use (`"cpu"` or `"cuda"`)."""
        return self._backend.name

    def compute(self, req_id: int, logits: NDArray[np.float32]) -> EntropySignal:
        """Compute the next-token entropy signal for ``req_id``.

        Updates the per-request EMA state. The logits array must be 1-D
        (vocab-size length) — the caller is responsible for selecting the
        last row when working with multi-step tensors.
        """
        h = self._backend.entropy(logits)
        eat = self._backend.eat(logits, self._think_end_ids)

        prev_ema = self._eat_ema.get(req_id, eat)
        prev_sq = self._eat_ema_sq.get(req_id, eat * eat)

        ema = self._alpha * eat + (1.0 - self._alpha) * prev_ema
        ema_sq = self._alpha * eat * eat + (1.0 - self._alpha) * prev_sq
        variance = max(0.0, ema_sq - ema * ema)

        self._eat_ema[req_id] = ema
        self._eat_ema_sq[req_id] = ema_sq

        return EntropySignal(
            token_entropy=h,
            eat=eat,
            eat_ema=ema,
            eat_ema_variance=variance,
        )

    def compute_batch(
        self,
        req_ids: list[int],
        logits_2d: NDArray[np.float32],
    ) -> list[EntropySignal]:
        """Vectorised :meth:`compute` over a ``(B, V)`` logits tensor.

        ``req_ids`` and ``logits_2d.shape[0]`` must agree. Per-request EMA
        state is updated independently for each row; the returned signals
        are in input order.

        The whole batch shares a single backend pass, so this is
        substantially cheaper than calling :meth:`compute` in a Python loop
        once the batch reaches a handful of requests.
        """
        if logits_2d.ndim != 2:
            msg = f"expected 2-D logits_2d, got shape {logits_2d.shape}"
            raise ValueError(msg)
        if len(req_ids) != logits_2d.shape[0]:
            msg = (
                f"req_ids length {len(req_ids)} mismatched against "
                f"logits_2d.shape[0]={logits_2d.shape[0]}"
            )
            raise ValueError(msg)

        h = self._backend.entropy_batch(logits_2d)
        eat = self._backend.eat_batch(logits_2d, self._think_end_ids)

        out: list[EntropySignal] = []
        a = self._alpha
        for i, rid in enumerate(req_ids):
            eat_i = float(eat[i])
            prev_ema = self._eat_ema.get(rid, eat_i)
            prev_sq = self._eat_ema_sq.get(rid, eat_i * eat_i)
            ema = a * eat_i + (1.0 - a) * prev_ema
            ema_sq = a * eat_i * eat_i + (1.0 - a) * prev_sq
            variance = max(0.0, ema_sq - ema * ema)
            self._eat_ema[rid] = ema
            self._eat_ema_sq[rid] = ema_sq
            out.append(
                EntropySignal(
                    token_entropy=float(h[i]),
                    eat=eat_i,
                    eat_ema=ema,
                    eat_ema_variance=variance,
                ),
            )
        return out

    def cleanup(self, req_id: int) -> None:
        """Release per-request EMA state. Idempotent."""
        self._eat_ema.pop(req_id, None)
        self._eat_ema_sq.pop(req_id, None)

    def tracked_requests(self) -> int:
        """Number of requests with live EMA state."""
        return len(self._eat_ema)
