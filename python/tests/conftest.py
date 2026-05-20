"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG seeded for reproducibility."""
    return np.random.default_rng(seed=0xC0FFEE)


@pytest.fixture(scope="session")
def vocab_size() -> int:
    """Vocab size used in entropy correctness tests. Mirrors a small reasoning model."""
    return 32_000
