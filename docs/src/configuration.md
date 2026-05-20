# Configuration

Meridian is configured through a single TOML file consumed by both the Rust
core (`meridian-core::config::MeridianConfig`) and the Python facade
(`meridian.config.MeridianConfig`). Both parsers agree on field names by
convention; a round-trip test exercises every field.

The canonical, fully-commented example lives at
[`meridian.toml.example`](https://github.com/angelnicolasc/meridian/blob/main/meridian.toml.example).
Each default value carries an inline `# Rationale:` comment explaining the
reason it was chosen.

## Sections

| Section          | Purpose                                                       |
|------------------|---------------------------------------------------------------|
| `[scheduler]`    | Dual-queue SLO budgets and batching multiplier                |
| `[entropy]`      | Probe activation, EMA decay, RPDI/EAT thresholds              |
| `[kv_memory]`    | Phase-aware block manager policy                              |
| `[model.<name>]` | Per-model token boundaries and reasoning parser key           |

## Validation

Both parsers reject:

- Unknown fields (`extra = "forbid"` in Pydantic; `deny_unknown_fields` in serde).
- Out-of-range values (e.g. `ema_alpha` outside `[0, 1]`).
- Cross-field violations (e.g. `min_think_tokens >= max_think_tokens`).

Validation errors carry the dotted field path and a human-readable reason.
See [`meridian-core::error::Error::ConfigValidation`](https://github.com/angelnicolasc/meridian/blob/main/crates/meridian-core/src/error.rs).
