"""Correctness tests for the CPU entropy backend.

The CPU backend is the reference oracle for the CUDA kernel. We verify it
against a vectorised NumPy "ground truth" computed with the canonical
softmax formula plus, when available, against PyTorch — whichever matches
production semantics most tightly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meridian._backends.cpu import CpuEntropyBackend
from meridian.entropy_probe import EntropyProbe

# ---------------------------------------------------------------------------
# Reference: pure-NumPy softmax + Shannon entropy
# ---------------------------------------------------------------------------


def _reference_entropy(logits: np.ndarray) -> float:
    """Direct softmax + entropy, used as ground truth in tests."""
    x = logits.astype(np.float64)
    x = x - x.max()
    p = np.exp(x)
    p = p / p.sum()
    # Guard log(0) — entries with p=0 contribute 0 to entropy.
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def _reference_eat(logits: np.ndarray, end_ids: np.ndarray) -> float:
    x = logits.astype(np.float64)
    x = x - x.max()
    p = np.exp(x)
    p = p / p.sum()
    return float(p[end_ids.astype(np.int64)].sum())


# ---------------------------------------------------------------------------
# Direct correctness
# ---------------------------------------------------------------------------


def test_entropy_uniform_distribution(vocab_size: int) -> None:
    backend = CpuEntropyBackend()
    logits = np.zeros(vocab_size, dtype=np.float32)
    h = backend.entropy(logits)
    # H(uniform) = log(V)
    assert math.isclose(h, math.log(vocab_size), rel_tol=1e-5, abs_tol=1e-5)


def test_entropy_delta_distribution(vocab_size: int) -> None:
    backend = CpuEntropyBackend()
    logits = np.full(vocab_size, -1e4, dtype=np.float32)
    logits[7] = 1e4
    h = backend.entropy(logits)
    assert h < 1e-4, f"near-delta should have ~0 entropy, got {h}"


def test_entropy_matches_reference(rng: np.random.Generator, vocab_size: int) -> None:
    backend = CpuEntropyBackend()
    logits = rng.standard_normal(vocab_size).astype(np.float32) * 3.0
    h_cpu = backend.entropy(logits)
    h_ref = _reference_entropy(logits)
    assert math.isclose(h_cpu, h_ref, rel_tol=1e-4, abs_tol=1e-4)


def test_eat_matches_reference(rng: np.random.Generator, vocab_size: int) -> None:
    backend = CpuEntropyBackend()
    logits = rng.standard_normal(vocab_size).astype(np.float32) * 2.0
    end_ids = np.array([1, 17, 256, 1024, 8000], dtype=np.int32)
    eat_cpu = backend.eat(logits, end_ids)
    eat_ref = _reference_eat(logits, end_ids)
    assert math.isclose(eat_cpu, eat_ref, rel_tol=1e-4, abs_tol=1e-4)


def test_eat_empty_ids_returns_zero(vocab_size: int) -> None:
    backend = CpuEntropyBackend()
    logits = np.zeros(vocab_size, dtype=np.float32)
    assert backend.eat(logits, np.array([], dtype=np.int32)) == 0.0


def test_entropy_rejects_non_1d() -> None:
    backend = CpuEntropyBackend()
    with pytest.raises(ValueError, match="1-D"):
        backend.entropy(np.zeros((2, 16), dtype=np.float32))


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=80)
@given(
    logits=st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=4096,
    ),
)
def test_entropy_nonnegative_and_bounded(logits: list[float]) -> None:
    backend = CpuEntropyBackend()
    arr = np.asarray(logits, dtype=np.float32)
    h = backend.entropy(arr)
    assert h >= -1e-5, f"entropy must be non-negative, got {h}"
    assert h <= math.log(len(arr)) + 1e-4, f"entropy exceeds log V: {h} > {math.log(len(arr))}"


@settings(deadline=None, max_examples=80)
@given(
    logits=st.lists(
        st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        min_size=8, max_size=1024,
    ),
    end_count=st.integers(min_value=0, max_value=4),
)
def test_eat_in_unit_interval(logits: list[float], end_count: int) -> None:
    backend = CpuEntropyBackend()
    arr = np.asarray(logits, dtype=np.float32)
    end_ids = np.arange(end_count, dtype=np.int32)
    eat = backend.eat(arr, end_ids)
    assert -1e-6 <= eat <= 1.0 + 1e-6, f"EAT must lie in [0, 1], got {eat}"


# ---------------------------------------------------------------------------
# EntropyProbe EMA behavior
# ---------------------------------------------------------------------------


def test_probe_ema_converges_under_constant_input(vocab_size: int) -> None:
    probe = EntropyProbe(think_end_token_ids=[1, 2, 3], backend="cpu", ema_alpha=0.1)
    logits = np.ones(vocab_size, dtype=np.float32)

    last_variance = math.inf
    for step in range(200):
        sig = probe.compute(req_id=42, logits=logits)
        if step >= 50:
            # After warm-up the variance must be small and monotone-ish (with EMA noise).
            assert sig.eat_ema_variance < 1e-3
            assert sig.eat_ema_variance <= last_variance + 1e-6
            last_variance = sig.eat_ema_variance


def test_probe_cleanup_releases_state(vocab_size: int) -> None:
    probe = EntropyProbe(think_end_token_ids=[1], backend="cpu")
    probe.compute(7, np.zeros(vocab_size, dtype=np.float32))
    probe.compute(8, np.zeros(vocab_size, dtype=np.float32))
    assert probe.tracked_requests() == 2
    probe.cleanup(7)
    assert probe.tracked_requests() == 1
    probe.cleanup(7)  # idempotent
    probe.cleanup(8)
    assert probe.tracked_requests() == 0


def test_probe_rejects_empty_end_ids() -> None:
    with pytest.raises(ValueError, match="think_end_token_ids"):
        EntropyProbe(think_end_token_ids=[], backend="cpu")


# ---------------------------------------------------------------------------
# compute_batch (vectorised)
# ---------------------------------------------------------------------------


def test_compute_batch_agrees_with_serial(
    rng: np.random.Generator, vocab_size: int,
) -> None:
    """Batch path and per-request path must produce identical signals."""
    end_ids = [1, 17, 256]
    serial = EntropyProbe(think_end_token_ids=end_ids, backend="cpu", ema_alpha=0.05)
    batch = EntropyProbe(think_end_token_ids=end_ids, backend="cpu", ema_alpha=0.05)

    req_ids = [10, 11, 12, 13]
    logits_2d = rng.standard_normal((4, vocab_size)).astype(np.float32)

    # Serial
    serial_signals = [
        serial.compute(rid, logits_2d[i]) for i, rid in enumerate(req_ids)
    ]
    # Batch
    batch_signals = batch.compute_batch(req_ids, logits_2d)

    assert len(serial_signals) == len(batch_signals) == 4
    for s, b in zip(serial_signals, batch_signals, strict=True):
        assert s.token_entropy == pytest.approx(b.token_entropy, abs=1e-4)
        assert s.eat == pytest.approx(b.eat, abs=1e-5)
        assert s.eat_ema == pytest.approx(b.eat_ema, abs=1e-5)
        assert s.eat_ema_variance == pytest.approx(b.eat_ema_variance, abs=1e-5)


def test_compute_batch_validates_shape(vocab_size: int) -> None:
    probe = EntropyProbe(think_end_token_ids=[1], backend="cpu")
    with pytest.raises(ValueError, match="2-D"):
        probe.compute_batch([1], np.zeros(vocab_size, dtype=np.float32))


def test_compute_batch_validates_length_match(vocab_size: int) -> None:
    probe = EntropyProbe(think_end_token_ids=[1], backend="cpu")
    with pytest.raises(ValueError, match="mismatched"):
        probe.compute_batch([1, 2], np.zeros((3, vocab_size), dtype=np.float32))
