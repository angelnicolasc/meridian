"""CUDA backend for the entropy probe.

Sprint 0: delegates to :class:`CpuEntropyBackend` so the public surface and
its tests stay backend-agnostic. Sprint 1 will replace the body of these
methods with calls into ``meridian._meridian`` (Rust → CUDA kernels) — the
:class:`EntropyBackend` protocol does not change.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from meridian._backends.cpu import CpuEntropyBackend


class CudaEntropyBackend:
    """CUDA-backed implementation. Currently delegates to CPU (Sprint 0)."""

    name: Final[str] = "cuda"

    def __init__(self) -> None:
        # When the real kernels land, this constructor will probe device
        # availability and raise if CUDA is not present. For now it always
        # succeeds because the underlying impl is pure NumPy.
        self._fallback = CpuEntropyBackend()

    def entropy(self, logits: NDArray[np.float32]) -> float:
        """See :meth:`CpuEntropyBackend.entropy`."""
        return self._fallback.entropy(logits)

    def eat(
        self,
        logits: NDArray[np.float32],
        think_end_ids: NDArray[np.int32],
    ) -> float:
        """See :meth:`CpuEntropyBackend.eat`."""
        return self._fallback.eat(logits, think_end_ids)

    def entropy_batch(
        self,
        logits_2d: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """See :meth:`CpuEntropyBackend.entropy_batch`."""
        return self._fallback.entropy_batch(logits_2d)

    def eat_batch(
        self,
        logits_2d: NDArray[np.float32],
        think_end_ids: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """See :meth:`CpuEntropyBackend.eat_batch`."""
        return self._fallback.eat_batch(logits_2d, think_end_ids)
