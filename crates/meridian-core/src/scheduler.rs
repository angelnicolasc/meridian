//! Dual-queue dispatcher.
//!
//! Output-phase requests have absolute priority. Think-phase requests fill the
//! remaining capacity with a larger effective batch budget governed by
//! `scheduler.think_batch_multiplier`. See [ADR-0001](https://github.com/angelnicolasc/meridian/blob/main/docs/src/adr/0001-dual-queue-rationale.md)
//! for the rationale.
//!
//! ## Cross-iteration phase transitions
//!
//! A request that emits `</think>` mid-iteration is detected by the
//! [`crate::phase_router::PhaseRouter`] *after* its current batch slot has
//! already been consumed. [`MeridianScheduler::on_phase_event`] moves the
//! request from `think_queue` to `output_queue` **for the next iteration**
//! and demotes its KV blocks to `ThinkComplete` immediately. This avoids
//! double-dispatch within the same `schedule_batch` call.
//!
//! ## Token injection on `ForceBudget`
//!
//! `ForceBudget` events arrive with the canonical `</think>` token id and a
//! [`crate::types::BudgetForceReason`]. The scheduler records the injection
//! in [`MeridianScheduler::pop_pending_injection`] for the worker to consume
//! before the next decode step. Idempotency is guaranteed by the router's
//! `force_in_progress` flag — a second `ForceBudget` for the same request
//! does not re-queue the injection.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use tracing::{debug, instrument};

use crate::config::SchedulerConfig;
use crate::metrics::names;
use crate::types::{PhaseEvent, Priority, RequestSlot, ScheduledBatch};

/// Pending `</think>` injection scheduled by the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingInjection {
    /// Target request.
    pub req_id: u64,
    /// Token id to inject as the next sampled token.
    pub token_id: u32,
}

/// Phase-aware dual-queue scheduler.
///
/// Internally backed by two FIFO queues (one per phase) protected by a
/// `parking_lot::Mutex`. The queues are mutated only at well-known points
/// (admission, phase transition, completion) so the lock is held for short
/// O(1) bursts.
#[derive(Debug)]
pub struct MeridianScheduler {
    inner: Mutex<Inner>,
    config: SchedulerConfig,
    /// Monotonic count of completed schedule iterations.
    iteration: AtomicU64,
}

#[derive(Debug, Default)]
struct Inner {
    output_queue: VecDeque<RequestSlot>,
    think_queue: VecDeque<RequestSlot>,
    pending_injections: VecDeque<PendingInjection>,
}

impl MeridianScheduler {
    /// Construct a scheduler from a validated [`SchedulerConfig`].
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            config,
            iteration: AtomicU64::new(0),
        }
    }

    /// Read-only view of the configuration in effect.
    #[must_use]
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Admit a fresh request as a think-phase slot at Normal priority.
    pub fn admit(&self, req_id: u64) {
        let mut inner = self.inner.lock();
        inner.think_queue.push_back(RequestSlot {
            req_id,
            priority: Priority::Normal,
        });
        self.publish_queue_depths(&inner);
    }

    /// Per-iteration scheduling decision.
    ///
    /// Drains the output queue first up to `output_budget_slots`, then fills
    /// remaining capacity from the think queue up to
    /// `think_batch_multiplier × output_budget_slots`, capped by `available_blocks`.
    ///
    /// Both budgets are expressed in *slots* (i.e. requests dispatched this
    /// iteration), not bytes — the caller (the worker) is responsible for
    /// translating slots into the model's batch-token budget.
    #[instrument(skip(self), level = "trace", fields(iter = self.iteration.load(Ordering::Relaxed)))]
    pub fn schedule_batch(
        &self,
        output_budget_slots: usize,
        available_blocks: usize,
    ) -> ScheduledBatch {
        let started_at = std::time::Instant::now();
        let mut inner = self.inner.lock();

        // Output queue drains first, up to its slot budget and the available
        // block budget (each slot in flight consumes at least one block).
        let output_take = output_budget_slots.min(available_blocks);
        let mut output_slots = Vec::with_capacity(output_take);
        for _ in 0..output_take {
            let Some(slot) = inner.output_queue.pop_front() else {
                break;
            };
            output_slots.push(slot);
        }

        let blocks_left = available_blocks.saturating_sub(output_slots.len());

        // Think queue fills the remaining slot budget. `think_batch_multiplier`
        // > 1 means we are willing to schedule more think slots per iteration
        // than output, *if* the block budget allows.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let think_budget_slots =
            ((output_budget_slots as f32) * self.config.think_batch_multiplier).floor() as usize;
        let think_take = think_budget_slots.min(blocks_left);
        let mut think_slots = Vec::with_capacity(think_take);
        for _ in 0..think_take {
            let Some(slot) = inner.think_queue.pop_front() else {
                break;
            };
            think_slots.push(slot);
        }

        let batch = ScheduledBatch {
            output_slots,
            think_slots,
        };

        // Metric surface.
        let elapsed_ns = started_at.elapsed().as_nanos() as f64;
        metrics::histogram!(names::SCHEDULE_BATCH_DURATION_NS).record(elapsed_ns);
        metrics::histogram!(names::SCHEDULER_BATCH_SIZE, "phase" => "output")
            .record(batch.output_slots.len() as f64);
        metrics::histogram!(names::SCHEDULER_BATCH_SIZE, "phase" => "think")
            .record(batch.think_slots.len() as f64);
        self.publish_queue_depths(&inner);

        self.iteration.fetch_add(1, Ordering::Relaxed);
        debug!(
            output = batch.output_slots.len(),
            think = batch.think_slots.len(),
            "schedule_batch",
        );
        batch
    }

    /// React to a phase event from the [`crate::phase_router::PhaseRouter`].
    ///
    /// The block manager is *not* owned by the scheduler — the caller (the
    /// worker) is responsible for invoking
    /// [`crate::block_manager::BlockManager::demote_think_blocks`] when this
    /// method returns. Decoupling keeps the scheduler testable without a
    /// concrete block manager.
    #[instrument(skip(self), level = "trace")]
    pub fn on_phase_event(&self, req_id: u64, event: PhaseEvent) {
        let mut inner = self.inner.lock();
        match event {
            PhaseEvent::EnterThink => {
                inner.think_queue.push_back(RequestSlot {
                    req_id,
                    priority: Priority::Normal,
                });
            }
            PhaseEvent::ExitThink { tokens_used: _ } => {
                // Remove from think_queue if still present (would happen if
                // the same request was queued twice — defensive). Then push
                // into output_queue at High priority.
                Self::remove_from_queue(&mut inner.think_queue, req_id);
                inner.output_queue.push_back(RequestSlot {
                    req_id,
                    priority: Priority::High,
                });
            }
            PhaseEvent::ForceBudget {
                inject_token,
                reason: _,
            } => {
                inner.pending_injections.push_back(PendingInjection {
                    req_id,
                    token_id: inject_token,
                });
            }
            PhaseEvent::Complete => {
                Self::remove_from_queue(&mut inner.think_queue, req_id);
                Self::remove_from_queue(&mut inner.output_queue, req_id);
            }
            PhaseEvent::None => {}
        }
        self.publish_queue_depths(&inner);
    }

    /// Pop the next pending `</think>` injection, if any. The worker is
    /// expected to call this once per request that has tokens in flight and
    /// inject the returned `token_id` into its sampler input.
    pub fn pop_pending_injection(&self) -> Option<PendingInjection> {
        self.inner.lock().pending_injections.pop_front()
    }

    /// Current output-queue depth.
    #[must_use]
    pub fn output_queue_depth(&self) -> usize {
        self.inner.lock().output_queue.len()
    }

    /// Current think-queue depth.
    #[must_use]
    pub fn think_queue_depth(&self) -> usize {
        self.inner.lock().think_queue.len()
    }

    fn remove_from_queue(q: &mut VecDeque<RequestSlot>, req_id: u64) {
        if let Some(pos) = q.iter().position(|s| s.req_id == req_id) {
            q.remove(pos);
        }
    }

    fn publish_queue_depths(&self, inner: &Inner) {
        metrics::gauge!(names::QUEUE_DEPTH, "queue" => "output")
            .set(inner.output_queue.len() as f64);
        metrics::gauge!(names::QUEUE_DEPTH, "queue" => "think").set(inner.think_queue.len() as f64);
    }
}
