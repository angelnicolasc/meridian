//! Config TOML round-trip and validation tests.

use meridian_core::config::MeridianConfig;
use pretty_assertions::assert_eq;

#[test]
fn default_config_is_valid() {
    let cfg = MeridianConfig::default();
    cfg.validate().expect("defaults must validate");
}

#[test]
fn full_config_round_trip() {
    let input = r#"
[scheduler]
think_tpot_budget_ms   = 80.0
output_tpot_budget_ms  = 20.0
think_batch_multiplier = 2.5
max_think_tokens       = 32768
min_think_tokens       = 512

[entropy]
enabled                        = true
ema_alpha                      = 0.05
rpdi_threshold                 = 3.0
eat_ema_variance_threshold     = 0.001
transition_entropy_threshold   = 2.5
eat_probe_interval_tokens      = 32

[kv_memory]
aggressive_think_eviction      = false
think_phase_memory_fraction    = 0.40

[model.deepseek_r1]
think_start_token_ids = [128799]
think_end_token_ids   = [128800]
reasoning_parser      = "deepseek_r1"
supports_think_disable = false
"#;

    let cfg = MeridianConfig::from_toml_str(input).expect("parse");
    assert_eq!(cfg.scheduler.max_think_tokens, 32_768);
    assert_eq!(cfg.entropy.ema_alpha, 0.05);
    assert_eq!(cfg.kv_memory.think_phase_memory_fraction, 0.40);

    let model = cfg
        .model
        .get("deepseek_r1")
        .expect("deepseek model present");
    assert_eq!(model.think_start_token_ids, vec![128_799]);
    assert_eq!(model.think_end_token_ids, vec![128_800]);
    assert_eq!(model.reasoning_parser.as_deref(), Some("deepseek_r1"));
}

#[test]
fn rejects_min_above_max_think_tokens() {
    let input = r#"
[scheduler]
think_tpot_budget_ms   = 80.0
output_tpot_budget_ms  = 20.0
think_batch_multiplier = 2.5
max_think_tokens       = 100
min_think_tokens       = 500
"#;
    let err = MeridianConfig::from_toml_str(input).expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.contains("min_think_tokens"), "unexpected error: {msg}");
}

#[test]
fn rejects_unknown_fields() {
    let input = r#"
[scheduler]
think_tpot_budget_ms = 80.0
output_tpot_budget_ms = 20.0
think_batch_multiplier = 2.5
max_think_tokens = 1000
min_think_tokens = 100
this_field_does_not_exist = 42
"#;
    let err = MeridianConfig::from_toml_str(input).expect_err("must reject unknown fields");
    let msg = format!("{err}");
    assert!(msg.contains("this_field_does_not_exist"), "got: {msg}");
}
