"""Meridian — inference-time compute scheduler for reasoning models.

This package is the Python facade over the Rust core (`meridian-core`,
exposed via the `meridian._meridian` pyo3 extension) plus pure-Python
components: the entropy probe (CPU reference backend) and the vLLM plugin.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from meridian.config import MeridianConfig, load_config
from meridian.entropy_probe import EntropyProbe, EntropySignal

try:
    __version__ = version("meridian")
except PackageNotFoundError:  # pragma: no cover — editable install before metadata
    __version__ = "0.1.0"

__all__ = [
    "EntropyProbe",
    "EntropySignal",
    "MeridianConfig",
    "__version__",
    "load_config",
]
