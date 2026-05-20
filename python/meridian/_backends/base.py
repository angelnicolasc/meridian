"""Backend contract for the entropy probe.

The probe runs after every (or every Nth) decode step and produces:

- ``token_entropy``: Shannon entropy of the sampled token distribution (nats).
- ``eat``: total softmax probability over the configured ``</think>`` token ids.

The backend protocol is intentionally numerical-only — it does not own EMA
state. EMA tracking lives in :class:`meridian.entropy_probe.EntropyProbe` so it
is identical regardless of which backend computes the raw signals.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EntropyBackend(Protocol):
    """Stateless raw-signal computation surface."""

    name: str
    """Human-readable backend identifier (`"cpu"`, `"cuda"`)."""

    def entropy(self, logits: NDArray[np.float32]) -> float:
        """Shannon entropy of the softmax of ``logits``, in nats.

        Implementations must use log-sum-exp for numerical stability.
        """
        ...

    def eat(
        self,
        logits: NDArray[np.float32],
        think_end_ids: NDArray[np.int32],
    ) -> float:
        """Sum of softmax probability over ``think_end_ids``.

        Returns a value in ``[0, 1]``.
        """
        ...

    def entropy_batch(
        self,
        logits_2d: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Shannon entropy of every row in ``logits_2d`` (shape ``(B, V)``).

        Returns a 1-D array of length ``B``.
        """
        ...

    def eat_batch(
        self,
        logits_2d: NDArray[np.float32],
        think_end_ids: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """EAT for every row of ``logits_2d``. Returns a 1-D array of length ``B``."""
        ...
