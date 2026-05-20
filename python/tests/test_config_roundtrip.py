"""Round-trip tests: Python ``MeridianConfig`` parses the canonical example."""

from __future__ import annotations

from pathlib import Path

import pytest

from meridian.config import MeridianConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "meridian.toml.example"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="meridian.toml.example not present")
def test_example_config_parses() -> None:
    cfg = MeridianConfig.from_path(EXAMPLE)
    assert cfg.scheduler.max_think_tokens > cfg.scheduler.min_think_tokens
    assert 0.0 <= cfg.entropy.ema_alpha <= 1.0
    assert "deepseek_r1" in cfg.model


def test_defaults_validate() -> None:
    cfg = MeridianConfig()
    assert cfg.scheduler.think_tpot_budget_ms > 0
    assert cfg.entropy.enabled is True


def test_rejects_unknown_fields() -> None:
    bad = """
[scheduler]
unknown_field = 42
"""
    with pytest.raises(ValueError, match=r"unknown_field|Extra inputs"):
        MeridianConfig.from_str(bad)


def test_rejects_min_above_max() -> None:
    bad = """
[scheduler]
min_think_tokens = 1000
max_think_tokens = 500
"""
    with pytest.raises(ValueError, match="min_think_tokens"):
        MeridianConfig.from_str(bad)
