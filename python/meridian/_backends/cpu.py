"""NumPy reference backend for the entropy probe.

This is the correctness oracle for the CUDA kernel. The CUDA kernel must
match this implementation within ``atol=1e-5`` on randomized inputs.

Both routines operate on the *last* row of the logits tensor — i.e. the
distribution over the next token at the current decode step.
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
from numpy.typing import NDArray


class CpuEntropyBackend:
    """NumPy implementation of the entropy probe."""

    name: Final[str] = "cpu"

    def entropy(self, logits: NDArray[np.float32]) -> float:
        """Shannon entropy of softmax(logits) in nats, via log-sum-exp.

        The computation is:

            H(p) = log Z - (1/Z) * sum_i exp(x_i - m) * (x_i - m)

        where m = max_i x_i and Z = sum_i exp(x_i - m). This is the same
        identity the CUDA kernel uses (see playbook §3.5).
        """
        if logits.ndim != 1:
            msg = f"expected 1-D logits, got shape {logits.shape}"
            raise ValueError(msg)

        x = logits.astype(np.float32, copy=False)
        m = float(x.max())
        shifted = x - m
        ex = np.exp(shifted, dtype=np.float32)
        z = float(ex.sum())
        # H = log(Z) - sum(ex * shifted) / Z
        weighted = float((ex * shifted).sum())
        return float(np.log(z) - weighted / z)

    def eat(
        self,
        logits: NDArray[np.float32],
        think_end_ids: NDArray[np.int32],
    ) -> float:
        """Sum of softmax probability over the configured think-end ids.

        Uses log-sum-exp for the normaliser. Empty ``think_end_ids`` returns 0.
        """
        if logits.ndim != 1:
            msg = f"expected 1-D logits, got shape {logits.shape}"
            raise ValueError(msg)
        if think_end_ids.size == 0:
            return 0.0

        x = logits.astype(np.float32, copy=False)
        m = float(x.max())
        ex = np.exp(x - m, dtype=np.float32)
        z = float(ex.sum())
        ids = think_end_ids.astype(np.int64, copy=False)
        return float(ex[ids].sum() / z)

    # ------------------------------------------------------------------
    # Batch entry points
    # ------------------------------------------------------------------

    def entropy_batch(
        self,
        logits_2d: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Vectorised entropy over a ``(B, V)`` logits batch."""
        if logits_2d.ndim != 2:
            msg = f"expected 2-D logits_2d, got shape {logits_2d.shape}"
            raise ValueError(msg)

        x = logits_2d.astype(np.float32, copy=False)
        # Per-row max for numerical stability; shape (B, 1).
        m = x.max(axis=1, keepdims=True)
        shifted = x - m
        ex = np.exp(shifted, dtype=np.float32)
        z = ex.sum(axis=1)  # (B,)
        weighted = (ex * shifted).sum(axis=1)  # (B,)
        # H = log Z - weighted / Z; broadcast-safe.
        return cast("NDArray[np.float32]", np.log(z) - weighted / z)

    def eat_batch(
        self,
        logits_2d: NDArray[np.float32],
        think_end_ids: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """Vectorised EAT over a ``(B, V)`` logits batch."""
        if logits_2d.ndim != 2:
            msg = f"expected 2-D logits_2d, got shape {logits_2d.shape}"
            raise ValueError(msg)
        b = logits_2d.shape[0]
        if think_end_ids.size == 0:
            return np.zeros(b, dtype=np.float32)

        x = logits_2d.astype(np.float32, copy=False)
        m = x.max(axis=1, keepdims=True)
        ex = np.exp(x - m, dtype=np.float32)
        z = ex.sum(axis=1)
        ids = think_end_ids.astype(np.int64, copy=False)
        # Sum over the gathered columns; shape (B,).
        gathered = ex[:, ids].sum(axis=1)
        return cast("NDArray[np.float32]", gathered / z)
