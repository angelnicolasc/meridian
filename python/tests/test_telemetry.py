"""Tests for the telemetry counters and OTLP wiring."""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from meridian import telemetry


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    val = REGISTRY.get_sample_value(name, labels or {})
    return 0.0 if val is None else val


def test_record_vocab_fallback_increments_prometheus() -> None:
    before = _sample("meridian_vocab_fallback_total")
    telemetry.record_vocab_fallback()
    telemetry.record_vocab_fallback()
    after = _sample("meridian_vocab_fallback_total")
    assert after - before == 2.0


def test_record_blocks_offloaded_increments_by_label() -> None:
    name = "meridian_disagg_blocks_offloaded_total"
    before = _sample(name, {"fabric": "nixl"})
    telemetry.record_blocks_offloaded("nixl", 5)
    after = _sample(name, {"fabric": "nixl"})
    assert after - before == 5.0


def test_init_otlp_guard_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    # When already installed, init_otlp returns immediately without building a
    # provider, so no collector connection is attempted during the test run.
    monkeypatch.setattr(telemetry, "_otlp_installed", True)
    telemetry.init_otlp("http://localhost:4318")  # no-op, must not raise
