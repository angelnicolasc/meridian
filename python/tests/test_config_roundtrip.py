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


@pytest.mark.skipif(not EXAMPLE.exists(), reason="meridian.toml.example not present")
def test_qwen35_preset_present() -> None:
    cfg = MeridianConfig.from_path(EXAMPLE)
    assert "qwen35" in cfg.model
    qwen35 = cfg.model["qwen35"]
    assert qwen35.think_start_token_ids
    assert qwen35.think_end_token_ids
    assert qwen35.reasoning_parser == "qwen3"


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


# ---------------------------------------------------------------------------
# [speculation] — ADR-0009
# ---------------------------------------------------------------------------


def test_speculation_defaults_to_off_and_uncalibrated() -> None:
    cfg = MeridianConfig()
    assert cfg.speculation.enabled is False
    assert cfg.speculation.acceptance_prior is None
    assert cfg.speculation.baseline_proposal_len == 7


@pytest.mark.skipif(not EXAMPLE.exists(), reason="meridian.toml.example not present")
def test_example_ships_speculation_disabled_and_unmeasured() -> None:
    cfg = MeridianConfig.from_path(EXAMPLE)
    assert cfg.speculation.enabled is False
    assert cfg.speculation.acceptance_prior is None, (
        "the shipped example must not carry acceptance rates: no phase-segmented "
        "measurement exists yet"
    )


def test_speculation_accepts_a_fully_attributed_prior() -> None:
    good = """
[speculation]
enabled = true

[speculation.acceptance_prior]
think            = 0.42
output           = 0.88
harness          = "DeepSpec@deadbeef"
draft_checkpoint = "deepseek-ai/dspark_qwen3_4b_block7"
target_model     = "Qwen/Qwen3-4B"
thinking_mode    = true
recorded_on      = "2026-08-07"
"""
    cfg = MeridianConfig.from_str(good)
    prior = cfg.speculation.acceptance_prior
    assert prior is not None
    assert prior.think < prior.output
    assert prior.target_model == "Qwen/Qwen3-4B"


def test_speculation_rejects_rates_without_provenance() -> None:
    bad = """
[speculation.acceptance_prior]
think  = 0.42
output = 0.88
"""
    with pytest.raises(ValueError, match=r"harness|Field required"):
        MeridianConfig.from_str(bad)


def test_speculation_rejects_inverted_proposal_bounds() -> None:
    bad = """
[speculation]
min_proposal_len = 9
max_proposal_len = 4
"""
    with pytest.raises(ValueError, match="proposal_len"):
        MeridianConfig.from_str(bad)
