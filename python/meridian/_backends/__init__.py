"""EntropyProbe backends.

This package implements the ``EntropyBackend`` Protocol with two concrete
backends: ``cpu`` (NumPy reference, always available) and ``cuda`` (delegates
to CPU in Sprint 0; swaps to the Rust kernels in Sprint 1).

The reference is the CPU backend — the CUDA backend is required to match its
output within ``atol=1e-5`` over all properties exercised by the test suite.
"""

from __future__ import annotations

from meridian._backends.base import EntropyBackend
from meridian._backends.cpu import CpuEntropyBackend
from meridian._backends.cuda import CudaEntropyBackend

__all__ = ["CpuEntropyBackend", "CudaEntropyBackend", "EntropyBackend"]
