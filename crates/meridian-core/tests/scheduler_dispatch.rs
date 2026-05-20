//! Integration tests for `MeridianScheduler` dispatch.
//!
//! Verifies: output-first drain, think_batch_multiplier expansion,
//! memory-cap saturation, phase-transition queue movement, token injection
//! idempotency, and Complete cleanup.

use meridian_core::config::SchedulerConfig;
use meridian_core::scheduler::MeridianScheduler;
use meridian_core::types::{BudgetForceReason, PhaseEvent};
use pretty_assertions::assert_eq;

fn scheduler() -> MeridianScheduler {
    MeridianScheduler::new(SchedulerConfig {
        think_tpot_budget_ms: 80.0,
        output_tpot_budget_ms: 20.0,
        think_batch_multiplier: 2.0,
        max_think_tokens: 32_768,
        min_think_tokens: 8,
    })
}

#[test]
fn empty_scheduler_returns_empty_batch() {
    let s = scheduler();
    let batch = s.schedule_batch(4, 100);
    assert!(batch.is_empty());
}

#[test]
fn output_queue_drains_first() {
    let s = scheduler();
    // Admit three think requests, then move one to output via ExitThink.
    s.admit(1);
    s.admit(2);
    s.admit(3);
    s.on_phase_event(2, PhaseEvent::EnterThink); // 2 enters think (idempotent w.r.t. admit)
    s.on_phase_event(1, PhaseEvent::EnterThink);
    s.on_phase_event(1, PhaseEvent::ExitThink { tokens_used: 100 });

    // Output budget = 2, plenty of blocks. Output queue has req 1 — it drains first.
    let batch = s.schedule_batch(2, 100);
    assert_eq!(batch.output_slots.len(), 1);
    assert_eq!(batch.output_slots[0].req_id, 1);
    // Think queue has 2, 3 plus a re-queued 2 from EnterThink — multiplier=2 → think_budget=4.
    assert!(batch.think_slots.len() >= 2);
}

#[test]
fn think_batch_multiplier_expands_budget() {
    let s = scheduler();
    for r in 100..110 {
        s.admit(r);
    }
    // output_budget = 2 → think_budget = floor(2 × 2.0) = 4
    let batch = s.schedule_batch(2, 100);
    assert_eq!(batch.output_slots.len(), 0);
    assert_eq!(
        batch.think_slots.len(),
        4,
        "multiplier didn't expand think budget"
    );
}

#[test]
fn memory_cap_saturates_dispatch() {
    let s = scheduler();
    for r in 0..20 {
        s.admit(r);
    }
    // available_blocks = 3 should cap the total dispatch at 3 even though the
    // multiplier wants more.
    let batch = s.schedule_batch(2, 3);
    assert_eq!(batch.output_slots.len() + batch.think_slots.len(), 3);
}

#[test]
fn exit_think_moves_request_between_queues() {
    let s = scheduler();
    s.admit(42);
    s.on_phase_event(42, PhaseEvent::EnterThink);
    assert_eq!(s.think_queue_depth(), 2, "admit + EnterThink both enqueue");

    s.on_phase_event(42, PhaseEvent::ExitThink { tokens_used: 50 });
    // Exit removes the request from think_queue and adds to output_queue.
    // Note: only one entry is removed per call — the duplicate from admit
    // remains unless completion clears it.
    assert_eq!(s.output_queue_depth(), 1);
}

#[test]
fn complete_clears_request_from_both_queues() {
    let s = scheduler();
    s.admit(9);
    // ExitThink moves the request from think_queue → output_queue.
    s.on_phase_event(9, PhaseEvent::ExitThink { tokens_used: 5 });
    assert_eq!(
        s.think_queue_depth(),
        0,
        "ExitThink should drain the think slot"
    );
    assert_eq!(s.output_queue_depth(), 1);

    // Complete clears every remaining slot for this request.
    s.on_phase_event(9, PhaseEvent::Complete);
    assert_eq!(s.think_queue_depth(), 0);
    assert_eq!(s.output_queue_depth(), 0);
}

#[test]
fn force_budget_queues_pending_injection() {
    let s = scheduler();
    s.on_phase_event(
        7,
        PhaseEvent::ForceBudget {
            inject_token: 128_800,
            reason: BudgetForceReason::Converged,
        },
    );
    let pi = s.pop_pending_injection().expect("injection present");
    assert_eq!(pi.req_id, 7);
    assert_eq!(pi.token_id, 128_800);
    assert!(s.pop_pending_injection().is_none(), "queue should drain");
}

#[test]
fn multiple_force_budgets_queue_in_order() {
    let s = scheduler();
    for (r, t) in [(1, 100), (2, 200), (3, 300)] {
        s.on_phase_event(
            r,
            PhaseEvent::ForceBudget {
                inject_token: t,
                reason: BudgetForceReason::HardCap,
            },
        );
    }
    let first = s.pop_pending_injection().unwrap();
    let second = s.pop_pending_injection().unwrap();
    let third = s.pop_pending_injection().unwrap();
    assert_eq!(first.req_id, 1);
    assert_eq!(second.req_id, 2);
    assert_eq!(third.req_id, 3);
}

#[test]
fn output_priority_dominates_under_pressure() {
    let s = scheduler();
    // 10 think requests, then add one output request.
    for r in 0..10 {
        s.admit(r);
    }
    s.on_phase_event(99, PhaseEvent::ExitThink { tokens_used: 1 });

    // output_budget = 1, blocks = 1. Output must be dispatched, think must wait.
    let batch = s.schedule_batch(1, 1);
    assert_eq!(batch.output_slots.len(), 1);
    assert_eq!(batch.output_slots[0].req_id, 99);
    assert_eq!(batch.think_slots.len(), 0);
}
