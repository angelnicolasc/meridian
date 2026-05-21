"""TOML configuration loader.

Parses `meridian.toml` into Pydantic models so downstream consumers get
typed access plus structured validation errors. The same TOML is consumed
by the Rust core (`meridian-core::config::MeridianConfig`); the two parsers
agree on field names by convention and are exercised by a round-trip test.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SchedulerConfig(BaseModel):
    """Dual-queue scheduler SLO and batching policy."""

    model_config = ConfigDict(extra="forbid")

    think_tpot_budget_ms: Annotated[float, Field(gt=0.0)] = 80.0
    output_tpot_budget_ms: Annotated[float, Field(gt=0.0)] = 20.0
    think_batch_multiplier: Annotated[float, Field(ge=1.0)] = 2.5
    max_think_tokens: Annotated[int, Field(gt=0)] = 32_768
    min_think_tokens: Annotated[int, Field(ge=0)] = 512

    @model_validator(mode="after")
    def _check_token_bounds(self) -> SchedulerConfig:
        if self.min_think_tokens >= self.max_think_tokens:
            msg = "min_think_tokens must be < max_think_tokens"
            raise ValueError(msg)
        return self


class EntropyConfig(BaseModel):
    """Entropy probe and convergence-detection thresholds."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ema_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    rpdi_threshold: Annotated[float, Field(gt=1.0)] = 3.0
    eat_ema_variance_threshold: Annotated[float, Field(ge=0.0)] = 0.001
    transition_entropy_threshold: Annotated[float, Field(ge=0.0)] = 2.5
    eat_probe_interval_tokens: Annotated[int, Field(ge=1)] = 32


class KvConfig(BaseModel):
    """KV memory tiering and eviction policy."""

    model_config = ConfigDict(extra="forbid")

    aggressive_think_eviction: bool = False
    think_phase_memory_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.40
    block_size_bytes: Annotated[int, Field(gt=0)] = 16_384
    capacity_bytes: Annotated[int, Field(gt=0)] | Literal["auto"] = "auto"


class DisaggConfig(BaseModel):
    """Disaggregated KV transfer (ADR-0006).

    Off by default. When enabled, the plugin offloads ``ThinkComplete``
    blocks via the configured fabric on every ``ExitThink`` event, batched
    by ``offload_threshold_blocks``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fabric: Literal["none", "nixl", "mooncake"] = "none"
    offload_threshold_blocks: Annotated[int, Field(ge=1)] = 4

    @model_validator(mode="after")
    def _check_fabric(self) -> DisaggConfig:
        if self.enabled and self.fabric == "none":
            msg = "disagg.fabric must be 'nixl' or 'mooncake' when disagg.enabled is true"
            raise ValueError(msg)
        return self


class ModelConfig(BaseModel):
    """Per-model token-boundary and parser configuration."""

    model_config = ConfigDict(extra="forbid")

    think_start_token_ids: list[int] = Field(default_factory=list)
    think_end_token_ids: list[int] = Field(default_factory=list)
    reasoning_parser: str | None = None
    supports_think_disable: bool = False


class TelemetryConfig(BaseModel):
    """OpenTelemetry OTLP export toggle.

    Off by default. Prometheus remains the primary metric surface; when
    ``otlp_enabled`` the plugin also exports its counters to an OTLP collector
    at ``otlp_endpoint``. Requires the ``otel`` extra.
    """

    model_config = ConfigDict(extra="forbid")

    otlp_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4318"
    service_name: str = "meridian"


class MeridianConfig(BaseModel):
    """Root configuration parsed from `meridian.toml`."""

    model_config = ConfigDict(extra="forbid")

    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    entropy: EntropyConfig = Field(default_factory=EntropyConfig)
    kv_memory: KvConfig = Field(default_factory=KvConfig)
    disagg: DisaggConfig = Field(default_factory=DisaggConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    model: dict[str, ModelConfig] = Field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> MeridianConfig:
        """Load and validate a config from a TOML file on disk."""
        raw = Path(path).read_bytes()
        return cls.model_validate(tomllib.loads(raw.decode("utf-8")))

    @classmethod
    def from_str(cls, s: str) -> MeridianConfig:
        """Parse a config from an in-memory TOML string."""
        return cls.model_validate(tomllib.loads(s))


def load_config(path: str | Path) -> MeridianConfig:
    """Convenience function mirroring the Rust `MeridianConfig::from_path` API."""
    return MeridianConfig.from_path(path)
