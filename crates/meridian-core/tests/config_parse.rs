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

// ---------------------------------------------------------------------------
// [speculation] — ADR-0009
// ---------------------------------------------------------------------------

#[test]
fn speculation_section_defaults_to_off_and_uncalibrated() {
    let cfg = MeridianConfig::default();
    assert!(!cfg.speculation.enabled);
    assert!(cfg.speculation.acceptance_prior.is_none());

    let hook = cfg
        .speculation
        .to_hook_config()
        .expect("defaults are valid");
    assert!(!hook.prior.is_calibrated());
    assert_eq!(hook.baseline_proposal_len, 7);
}

#[test]
fn speculation_section_parses() {
    let input = r#"
[speculation]
enabled               = true
baseline_proposal_len = 5
min_proposal_len      = 2
max_proposal_len      = 8
vocab_size            = 151936
use_entropy_ceiling   = true
draft_token_us        = 35.0
verify_fixed_us       = 850.0
verify_token_us       = 20.0
"#;
    let cfg = MeridianConfig::from_toml_str(input).expect("parse");
    assert!(cfg.speculation.enabled);
    assert_eq!(cfg.speculation.baseline_proposal_len, 5);
    assert_eq!(cfg.speculation.max_proposal_len, 8);
    assert!(cfg.speculation.acceptance_prior.is_none());
}

#[test]
fn speculation_accepts_a_fully_attributed_measured_prior() {
    let input = r#"
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
"#;
    let cfg = MeridianConfig::from_toml_str(input).expect("parse");
    let prior = cfg
        .speculation
        .acceptance_prior
        .as_ref()
        .expect("prior present");
    assert_eq!(prior.target_model, "Qwen/Qwen3-4B");

    let hook = cfg.speculation.to_hook_config().expect("valid");
    assert!(hook.prior.is_calibrated());
    assert!(hook.prior.phase_gap().expect("gap") < 0.0);
}

/// A measured prior without the run that produced it is not admissible input.
/// This is the configuration-level half of the provenance discipline described
/// in `dspark_bridge::provenance`.
#[test]
fn speculation_rejects_acceptance_rates_without_provenance() {
    let input = r#"
[speculation.acceptance_prior]
think  = 0.42
output = 0.88
"#;
    let err = MeridianConfig::from_toml_str(input).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("harness") || msg.contains("missing field"),
        "unexpected error: {msg}",
    );
}

#[test]
fn speculation_rejects_blank_provenance_fields() {
    let input = r#"
[speculation.acceptance_prior]
think            = 0.42
output           = 0.88
harness          = "   "
draft_checkpoint = "deepseek-ai/dspark_qwen3_4b_block7"
target_model     = "Qwen/Qwen3-4B"
thinking_mode    = true
recorded_on      = "2026-08-07"
"#;
    let err = MeridianConfig::from_toml_str(input).expect_err("must reject");
    assert!(
        format!("{err}").contains("acceptance_prior"),
        "unexpected error: {err}",
    );
}

#[test]
fn speculation_rejects_out_of_range_proposal_lengths() {
    let input = r#"
[speculation]
min_proposal_len      = 9
max_proposal_len      = 4
baseline_proposal_len = 4
"#;
    let err = MeridianConfig::from_toml_str(input).expect_err("must reject");
    assert!(
        format!("{err}").contains("proposal_len"),
        "unexpected error: {err}",
    );
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
