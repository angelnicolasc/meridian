//! Integration tests for `PhaseRouter`. Cover the eight scenarios listed in
//! the Sprint 0 plan plus property-based invariants over arbitrary token
//! streams.

use meridian_core::phase_router::{PhaseRouter, PhaseRouterConfig};
use meridian_core::types::{BudgetForceReason, EntropySignal, PhaseEvent, ThinkPhase};
use pretty_assertions::assert_eq;
use proptest::prelude::*;

// Canonical token ids used throughout these tests.
const THINK_START: u32 = 1;
const THINK_END: u32 = 2;
const EOS: u32 = 3;
const NORMAL: u32 = 100;

fn cfg() -> PhaseRouterConfig {
    PhaseRouterConfig {
        ema_alpha: 0.05,
        transition_entropy_threshold: 2.5,
        rpdi_threshold: 3.0,
        eat_ema_variance_threshold: 0.001,
        min_think_tokens: 8,
        max_think_tokens: 64,
        think_start_ids: vec![THINK_START],
        think_end_ids: vec![THINK_END],
        eos_ids: vec![EOS],
    }
}

fn router() -> PhaseRouter {
    PhaseRouter::new(cfg())
}

/// Config tuned for entropy-driven tests: larger `ema_alpha` so the EMA
/// converges quickly within the iteration budget, and a larger
/// `max_think_tokens` so the hard cap does not preempt the EAT/RPDI path.
fn entropy_cfg() -> PhaseRouterConfig {
    PhaseRouterConfig {
        ema_alpha: 0.4,
        eat_ema_variance_threshold: 0.001,
        min_think_tokens: 8,
        max_think_tokens: 1_000_000,
        ..cfg()
    }
}

fn entropy_router() -> PhaseRouter {
    PhaseRouter::new(entropy_cfg())
}

/// Config for the RPDI test specifically — convergence threshold set to a
/// negative value so the EAT path never fires; only RPDI overthinking can
/// trigger force_budget.
fn rpdi_router() -> PhaseRouter {
    PhaseRouter::new(PhaseRouterConfig {
        eat_ema_variance_threshold: -1.0, // unreachable; disables EAT path
        ..entropy_cfg()
    })
}

// ---------------------------------------------------------------------------
// Scenario 1: normal happy path Prefill -> ThinkDecode -> OutputDecode -> Complete
// ---------------------------------------------------------------------------

#[test]
fn happy_path_full_lifecycle() {
    let r = router();
    r.register(1);

    assert_eq!(r.on_token(1, THINK_START, None), PhaseEvent::EnterThink);
    for _ in 0..5 {
        assert_eq!(r.on_token(1, NORMAL, None), PhaseEvent::None);
    }
    assert_eq!(
        r.on_token(1, THINK_END, None),
        PhaseEvent::ExitThink { tokens_used: 6 },
    );
    for _ in 0..3 {
        assert_eq!(r.on_token(1, NORMAL, None), PhaseEvent::None);
    }
    assert_eq!(r.on_token(1, EOS, None), PhaseEvent::Complete);

    assert!(matches!(r.phase_of(1), Some(ThinkPhase::Complete)));
}

// ---------------------------------------------------------------------------
// Scenario 2: non-reasoning model — never emits <think>
// ---------------------------------------------------------------------------

#[test]
fn non_reasoning_model_skips_think_phase() {
    let r = router();
    r.register(7);

    assert_eq!(r.on_token(7, NORMAL, None), PhaseEvent::None);
    assert!(matches!(
        r.phase_of(7),
        Some(ThinkPhase::OutputDecode { think_tokens_used: 0 }),
    ));

    for _ in 0..4 {
        assert_eq!(r.on_token(7, NORMAL, None), PhaseEvent::None);
    }
    assert_eq!(r.on_token(7, EOS, None), PhaseEvent::Complete);
}

// ---------------------------------------------------------------------------
// Scenario 3: nested <think> inside an active think phase is ignored
// ---------------------------------------------------------------------------

#[test]
fn nested_think_start_is_ignored() {
    let r = router();
    r.register(2);

    assert_eq!(r.on_token(2, THINK_START, None), PhaseEvent::EnterThink);
    // Nested <think> — should be treated as noise, not re-enter the phase
    // (which would reset the token counter and the entropy accumulators).
    assert_eq!(r.on_token(2, THINK_START, None), PhaseEvent::None);

    if let Some(ThinkPhase::ThinkDecode { tokens_so_far, .. }) = r.phase_of(2) {
        assert_eq!(tokens_so_far, 1, "counter must advance through the nested token");
    } else {
        panic!("expected ThinkDecode, got {:?}", r.phase_of(2));
    }
}

// ---------------------------------------------------------------------------
// Scenario 4: hard cap fires ForceBudget even without any entropy signal
// ---------------------------------------------------------------------------

#[test]
fn hard_cap_forces_budget_without_entropy_signal() {
    let r = router();
    r.register(3);
    assert_eq!(r.on_token(3, THINK_START, None), PhaseEvent::EnterThink);

    // max_think_tokens = 64 in our config.
    for i in 1..64 {
        let ev = r.on_token(3, NORMAL, None);
        assert_eq!(ev, PhaseEvent::None, "no force expected at token {i}");
    }
    let ev = r.on_token(3, NORMAL, None);
    assert!(
        matches!(
            ev,
            PhaseEvent::ForceBudget {
                inject_token,
                reason: BudgetForceReason::HardCap,
            } if inject_token == THINK_END,
        ),
        "expected ForceBudget(HardCap) at cap, got {ev:?}",
    );
}

// ---------------------------------------------------------------------------
// Scenario 5: EAT convergence triggers ForceBudget (only after min_think_tokens)
// ---------------------------------------------------------------------------

#[test]
fn eat_convergence_triggers_budget_force() {
    let r = entropy_router();
    r.register(4);
    r.on_token(4, THINK_START, None);

    // Feed a stable EAT signal for many tokens — variance will collapse.
    let stable = EntropySignal {
        token_entropy: 0.1,
        eat: 0.95,
        eat_ema: 0.95,
        eat_ema_variance: 0.0,
    };

    let mut forced_at: Option<usize> = None;
    for i in 0..200 {
        if let PhaseEvent::ForceBudget { reason, .. } = r.on_token(4, NORMAL, Some(&stable)) {
            assert_eq!(reason, BudgetForceReason::Converged, "expected Converged reason");
            forced_at = Some(i);
            break;
        }
    }
    let forced_at = forced_at.expect("EAT stabilisation should have forced budget");
    assert!(
        forced_at >= entropy_cfg().min_think_tokens as usize,
        "force fired at {forced_at}, must be >= min_think_tokens ({})",
        entropy_cfg().min_think_tokens,
    );
}

// ---------------------------------------------------------------------------
// Scenario 6: RPDI overthinking triggers ForceBudget
// ---------------------------------------------------------------------------

#[test]
fn rpdi_overthinking_triggers_budget_force() {
    let r = rpdi_router();
    r.register(5);
    r.on_token(5, THINK_START, None);

    // Phase A: warm up rpdi_global to a low baseline with mostly non-transition
    // tokens. We feed enough tokens to clear min_think_tokens and establish a
    // small global rate.
    let calm = EntropySignal {
        token_entropy: 0.2,
        eat: 0.05,
        eat_ema: 0.05,
        eat_ema_variance: 1.0, // keep variance high so EAT path doesn't fire
    };
    for _ in 0..40 {
        let _ = r.on_token(5, NORMAL, Some(&calm));
    }

    // Phase B: burst of transition tokens — drives rpdi_local far above
    // rpdi_global. EAT signal kept unstable so it isn't the trigger.
    let spike = EntropySignal {
        token_entropy: 5.0, // well above transition_entropy_threshold = 2.5
        eat: 0.05,
        eat_ema: 0.05,
        eat_ema_variance: 1.0,
    };
    let mut forced = false;
    for _ in 0..50 {
        if let PhaseEvent::ForceBudget { reason, .. } = r.on_token(5, NORMAL, Some(&spike)) {
            assert_eq!(reason, BudgetForceReason::Overthinking);
            forced = true;
            break;
        }
    }
    assert!(forced, "RPDI overthinking should have forced the budget");
}

// ---------------------------------------------------------------------------
// Scenario 7: force_in_progress prevents double injection
// ---------------------------------------------------------------------------

#[test]
fn force_in_progress_is_idempotent() {
    let r = router();
    r.register(6);
    r.on_token(6, THINK_START, None);

    // Hit the hard cap with a final NORMAL token — produces a single ForceBudget.
    for _ in 0..63 {
        let _ = r.on_token(6, NORMAL, None);
    }
    let first = r.on_token(6, NORMAL, None);
    assert!(matches!(first, PhaseEvent::ForceBudget { .. }));

    // Subsequent tokens in the same phase must not emit a second ForceBudget
    // until the model actually consumes the injection (which is signalled by
    // ExitThink, not by another normal token).
    for _ in 0..5 {
        let ev = r.on_token(6, NORMAL, None);
        assert_eq!(ev, PhaseEvent::None, "duplicate ForceBudget emitted: {ev:?}");
    }
}

// ---------------------------------------------------------------------------
// Unknown / reaped request degrades gracefully
// ---------------------------------------------------------------------------

#[test]
fn unknown_request_is_safe() {
    let r = router();
    assert_eq!(r.on_token(999, NORMAL, None), PhaseEvent::None);
}

#[test]
fn reap_then_token_is_safe() {
    let r = router();
    r.register(11);
    r.reap(11);
    assert_eq!(r.on_token(11, NORMAL, None), PhaseEvent::None);
    assert_eq!(r.phase_of(11), None);
}

// ---------------------------------------------------------------------------
// phase_of_kind contract for plugin integration
// ---------------------------------------------------------------------------

#[test]
fn phase_of_kind_tracks_state_machine() {
    let r = router();
    assert_eq!(r.phase_of_kind(1), None);
    r.register(1);
    assert_eq!(r.phase_of_kind(1), Some("prefill"));
    r.on_token(1, THINK_START, None);
    assert_eq!(r.phase_of_kind(1), Some("think_decode"));
    r.on_token(1, NORMAL, None);
    r.on_token(1, THINK_END, None);
    assert_eq!(r.phase_of_kind(1), Some("output_decode"));
    r.on_token(1, EOS, None);
    assert_eq!(r.phase_of_kind(1), Some("complete"));
}

// ---------------------------------------------------------------------------
// tracked_requests counter is exact (atomic)
// ---------------------------------------------------------------------------

#[test]
fn tracked_requests_counter_is_exact() {
    let r = router();
    assert_eq!(r.tracked_requests(), 0);
    r.register(10);
    r.register(11);
    r.register(12);
    assert_eq!(r.tracked_requests(), 3);
    r.register(10); // idempotent
    assert_eq!(r.tracked_requests(), 3);
    r.reap(11);
    assert_eq!(r.tracked_requests(), 2);
    r.reap(11); // idempotent reap
    assert_eq!(r.tracked_requests(), 2);
}

// ---------------------------------------------------------------------------
// reap_stale heartbeat
// ---------------------------------------------------------------------------

#[test]
fn reap_stale_removes_idle_requests() {
    let r = router();
    r.register(1);
    r.register(2);
    r.register(3);
    // Activity on req 2 — its last_touch advances every tick.
    for _ in 0..50 {
        r.on_token(2, NORMAL, None);
    }
    // Massive cutoff: anything older than 10 ticks ago. Reqs 1 and 3 should
    // go; req 2 was just touched.
    let reaped = r.reap_stale(10);
    assert_eq!(reaped, 2);
    assert_eq!(r.tracked_requests(), 1);
    assert_eq!(r.phase_of_kind(2), Some("output_decode"));
    assert_eq!(r.phase_of_kind(1), None);
    assert_eq!(r.phase_of_kind(3), None);
}

#[test]
fn reap_stale_with_huge_threshold_keeps_everyone() {
    let r = router();
    r.register(1);
    r.register(2);
    let reaped = r.reap_stale(u64::MAX);
    assert_eq!(reaped, 0);
    assert_eq!(r.tracked_requests(), 2);
}

// ---------------------------------------------------------------------------
// Property test: Complete is absorbing; phase never regresses
// ---------------------------------------------------------------------------

fn phase_rank(p: &ThinkPhase) -> u8 {
    match p {
        ThinkPhase::Prefill => 0,
        ThinkPhase::ThinkDecode { .. } => 1,
        ThinkPhase::OutputDecode { .. } => 2,
        ThinkPhase::Complete => 3,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn phase_is_monotonic(tokens in proptest::collection::vec(0u32..=110u32, 1..200)) {
        let r = router();
        r.register(0);

        let mut last_rank = 0u8;
        for tok in tokens {
            let _ = r.on_token(0, tok, None);
            let Some(phase) = r.phase_of(0) else { continue };
            let rank = phase_rank(&phase);
            prop_assert!(
                rank >= last_rank,
                "phase regressed: rank {last_rank} -> {rank} on token",
            );
            last_rank = rank;

            // Once Complete, the router must stay Complete.
            if last_rank == 3 {
                prop_assert!(matches!(phase, ThinkPhase::Complete));
            }
        }
    }
}
