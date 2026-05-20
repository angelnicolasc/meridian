//! Per-request token-stream state machine.
//!
//! The `PhaseRouter` is the only component that touches every generated token.
//! Its hot path must be `O(1)` and zero-allocation in the common case. State
//! is held in a lock-free `DashMap` keyed by request id so that concurrent
//! decode workers can update independently without contention.
//!
//! See [`ThinkPhase`] for the underlying state machine and [`PhaseEvent`] for
//! the event surface the scheduler consumes.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::metrics::names;
use crate::types::{BudgetForceReason, EntropySignal, PhaseEvent, ThinkPhase};

/// Internal entry held per request — wraps the phase with a last-touch
/// timestamp so the heartbeat reaper can prune orphans whose `Complete`
/// event was never observed.
#[derive(Debug, Clone)]
struct Entry {
    phase: ThinkPhase,
    last_touch_ns: u64,
}

/// Monotonic process-local tick used by the router. The reaper only needs
/// strict ordering across calls; `AtomicU64::fetch_add(1)` is monotonic
/// and cheaper than any syscall-based clock.
///
/// A parallel wall-clock anchor lives in [`process_start`] and is used by
/// [`PhaseRouter::reap_stale_older_than`] to convert wall-clock durations
/// to tick counts via the measured tick rate.
fn now_ns() -> u64 {
    static CLOCK: AtomicU64 = AtomicU64::new(1);
    CLOCK.fetch_add(1, Ordering::Relaxed)
}

/// Process-wide wall-clock anchor used to translate `Duration` thresholds
/// into tick counts. Initialised lazily on first access so test binaries
/// that never touch the router pay no cost.
fn process_start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Snapshot the (tick, elapsed) pair atomically enough for tick-rate
/// estimation. The two reads cannot truly be atomic, but the only
/// consequence of skew is a fractional-tick error on the wall-clock
/// conversion, which is harmless at the scales the reaper operates on.
fn tick_rate_per_sec() -> f64 {
    let ticks = now_ns() as f64;
    let elapsed = process_start().elapsed().as_secs_f64().max(1e-9);
    (ticks / elapsed).max(1.0)
}

/// Configuration for the [`PhaseRouter`].
///
/// These fields mirror the operationally-meaningful subset of
/// [`crate::config::SchedulerConfig`] and [`crate::config::EntropyConfig`]
/// — the router does not own the full config because it has no need for
/// queue or memory parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseRouterConfig {
    /// EMA decay applied to EAT and RPDI signals.
    pub ema_alpha: f32,
    /// Token entropy above which a token is classified as a "transition" for
    /// RPDI accounting.
    pub transition_entropy_threshold: f32,
    /// Ratio (`rpdi_local / rpdi_global`) above which overthinking is declared.
    pub rpdi_threshold: f32,
    /// EAT EMA variance below which convergence is declared.
    pub eat_ema_variance_threshold: f32,
    /// Minimum think tokens before any budget forcing is allowed.
    pub min_think_tokens: u32,
    /// Absolute hard cap (safety valve).
    pub max_think_tokens: u32,
    /// Token id(s) that signal `<think>` start. Multiple are supported because
    /// some tokenizers (DeepSeek-R1) encode `"\n<think>\n"` as a distinct id
    /// from `"<think>"`.
    pub think_start_ids: Vec<u32>,
    /// Token id(s) that signal `</think>` end.
    pub think_end_ids: Vec<u32>,
    /// EOS token ids.
    pub eos_ids: Vec<u32>,
}

impl Default for PhaseRouterConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.05,
            transition_entropy_threshold: 2.5,
            rpdi_threshold: 3.0,
            eat_ema_variance_threshold: 0.001,
            min_think_tokens: 512,
            max_think_tokens: 32_768,
            think_start_ids: Vec::new(),
            think_end_ids: Vec::new(),
            eos_ids: Vec::new(),
        }
    }
}

impl PhaseRouterConfig {
    /// Construct a config with the given think-boundary ids and otherwise
    /// default thresholds. Convenience for tests.
    #[must_use]
    pub fn with_boundary_ids(start: Vec<u32>, end: Vec<u32>, eos: Vec<u32>) -> Self {
        Self {
            think_start_ids: start,
            think_end_ids: end,
            eos_ids: eos,
            ..Self::default()
        }
    }
}

/// The phase router itself.
///
/// Designed to be wrapped in `Arc<PhaseRouter>` and shared across decode
/// workers. Internal state is a sharded `DashMap`; the `tracked_requests`
/// counter is an `AtomicUsize` updated in lock-step with insert/remove so
/// readers never see a stale length.
#[derive(Debug)]
pub struct PhaseRouter {
    /// Per-request entry (phase + last-touch timestamp).
    state: DashMap<u64, Entry>,
    /// Exact count of tracked requests. Maintained in lock-step with `state`.
    tracked: AtomicUsize,
    config: PhaseRouterConfig,
}

impl PhaseRouter {
    /// Construct a router with the given configuration.
    #[must_use]
    pub fn new(config: PhaseRouterConfig) -> Self {
        Self {
            state: DashMap::with_capacity(1024),
            tracked: AtomicUsize::new(0),
            config,
        }
    }

    /// Number of requests currently tracked. Exact; not eventually-consistent.
    #[must_use]
    pub fn tracked_requests(&self) -> usize {
        self.tracked.load(Ordering::Relaxed)
    }

    /// Register a new request. Requests start in [`ThinkPhase::Prefill`].
    ///
    /// Idempotent: re-registering the same id is a no-op (the existing
    /// `last_touch_ns` is NOT refreshed — use [`Self::touch`] if you want to
    /// extend a request's life under the reaper).
    pub fn register(&self, req_id: u64) {
        let inserted = self
            .state
            .entry(req_id)
            .or_insert_with(|| {
                self.tracked.fetch_add(1, Ordering::Relaxed);
                Entry {
                    phase: ThinkPhase::Prefill,
                    last_touch_ns: now_ns(),
                }
            });
        drop(inserted);
        self.publish_tracked_gauge();
    }

    /// Drop tracked state for a request. Called by the scheduler once a request
    /// completes (or is cancelled) to bound memory growth.
    pub fn reap(&self, req_id: u64) {
        if self.state.remove(&req_id).is_some() {
            self.tracked.fetch_sub(1, Ordering::Relaxed);
        }
        self.publish_tracked_gauge();
    }

    /// Mark `req_id` as recently active. Resets its idle clock so the next
    /// [`Self::reap_stale`] sweep does not collect it.
    ///
    /// `on_token` automatically touches the request; call this explicitly only
    /// when the worker observes activity outside the token-stream (e.g. an
    /// admin keepalive).
    pub fn touch(&self, req_id: u64) {
        if let Some(mut entry) = self.state.get_mut(&req_id) {
            entry.last_touch_ns = now_ns();
        }
    }

    /// Reap every tracked request whose `last_touch_ns` is older than
    /// `(now - older_than_ticks)`. Returns the number of requests reaped.
    ///
    /// "Ticks" are the router's monotonic clock units, not wall-clock
    /// nanoseconds — they increment once per relevant state mutation. A
    /// caller running a wall-clock heartbeat can choose a threshold based
    /// on the empirical tick rate of the workload.
    pub fn reap_stale(&self, older_than_ticks: u64) -> usize {
        let now = now_ns();
        let cutoff = now.saturating_sub(older_than_ticks);
        let stale: Vec<u64> = self
            .state
            .iter()
            .filter_map(|item| {
                if item.value().last_touch_ns < cutoff {
                    Some(*item.key())
                } else {
                    None
                }
            })
            .collect();
        let count = stale.len();
        for rid in stale {
            if self.state.remove(&rid).is_some() {
                self.tracked.fetch_sub(1, Ordering::Relaxed);
            }
        }
        if count > 0 {
            self.publish_tracked_gauge();
        }
        count
    }

    /// Wall-clock variant of [`Self::reap_stale`]. Converts `duration` to
    /// router-tick units using the measured tick rate of this process and
    /// delegates to the underlying tick-based reaper.
    ///
    /// Operators express idle timeouts in seconds (`Duration::from_secs(60)`)
    /// without needing to know the router's internal tick scale.
    pub fn reap_stale_older_than(&self, duration: Duration) -> usize {
        // Touch `process_start` once so the wall-clock anchor is initialised
        // before any caller observes the rate.
        let _ = process_start();
        let rate = tick_rate_per_sec();
        let ticks = (duration.as_secs_f64() * rate).round();
        // Saturate at u64::MAX rather than wrap; absurd thresholds simply
        // mean "never reap", which is the intended behaviour.
        let ticks_u64 = if ticks.is_finite() && ticks >= 0.0 && ticks <= u64::MAX as f64 {
            ticks as u64
        } else {
            u64::MAX
        };
        self.reap_stale(ticks_u64)
    }

    /// Inspect the current phase of a request without mutating it. Returns
    /// `None` if the request is unknown (either never registered or already
    /// reaped).
    #[must_use]
    pub fn phase_of(&self, req_id: u64) -> Option<ThinkPhase> {
        self.state.get(&req_id).map(|e| e.phase.clone())
    }

    /// String-typed phase classification. Stable contract for cross-language
    /// callers (the vLLM plugin reads this to rank requests for reorder).
    ///
    /// Returns one of `"prefill"`, `"think_decode"`, `"output_decode"`,
    /// `"complete"`, or `None` if the request is unknown.
    #[must_use]
    pub fn phase_of_kind(&self, req_id: u64) -> Option<&'static str> {
        self.state.get(&req_id).map(|e| match e.phase {
            ThinkPhase::Prefill => "prefill",
            ThinkPhase::ThinkDecode { .. } => "think_decode",
            ThinkPhase::OutputDecode { .. } => "output_decode",
            ThinkPhase::Complete => "complete",
        })
    }

    fn publish_tracked_gauge(&self) {
        metrics::gauge!(names::PHASE_ROUTER_TRACKED_REQUESTS)
            .set(self.tracked.load(Ordering::Relaxed) as f64);
    }

    /// Process a single decoded token for the given request and return the
    /// scheduling-visible event.
    ///
    /// `entropy_signal` is optional because the probe runs on its own stream
    /// and may not have a fresh sample for every token (see
    /// `entropy.eat_probe_interval_tokens`).
    ///
    /// # Examples
    ///
    /// Driving a complete reasoning request through the state machine:
    ///
    /// ```
    /// use meridian_core::phase_router::{PhaseRouter, PhaseRouterConfig};
    /// use meridian_core::types::PhaseEvent;
    ///
    /// const THINK_START: u32 = 1;
    /// const THINK_END:   u32 = 2;
    /// const EOS:         u32 = 3;
    ///
    /// let router = PhaseRouter::new(PhaseRouterConfig::with_boundary_ids(
    ///     vec![THINK_START], vec![THINK_END], vec![EOS],
    /// ));
    /// router.register(42);
    ///
    /// assert_eq!(router.on_token(42, THINK_START, None), PhaseEvent::EnterThink);
    /// for _ in 0..5 { let _ = router.on_token(42, 99, None); }
    /// assert!(matches!(
    ///     router.on_token(42, THINK_END, None),
    ///     PhaseEvent::ExitThink { .. },
    /// ));
    /// for _ in 0..3 { let _ = router.on_token(42, 100, None); }
    /// assert_eq!(router.on_token(42, EOS, None), PhaseEvent::Complete);
    /// ```
    pub fn on_token(
        &self,
        req_id: u64,
        token_id: u32,
        entropy_signal: Option<&EntropySignal>,
    ) -> PhaseEvent {
        let Some(mut entry_ref) = self.state.get_mut(&req_id) else {
            // Unknown request — degrade gracefully. Production code should
            // always call `register` first; this branch exists so a missing
            // call does not corrupt scheduler state.
            return PhaseEvent::None;
        };

        // Bump the per-request idle clock so the heartbeat reaper does not
        // collect a request that is actively producing tokens.
        entry_ref.last_touch_ns = now_ns();
        let entry: &mut ThinkPhase = &mut entry_ref.phase;

        match entry {
            ThinkPhase::Prefill => {
                if self.config.think_start_ids.contains(&token_id) {
                    *entry = ThinkPhase::ThinkDecode {
                        tokens_so_far: 0,
                        eat_ema: 0.5,
                        eat_ema_sq: 0.25,
                        rpdi_local: 0.0,
                        rpdi_global: 0.0,
                        force_in_progress: false,
                    };
                    PhaseEvent::EnterThink
                } else if self.config.eos_ids.contains(&token_id) {
                    // Some non-reasoning models can EOS while still nominally
                    // in prefill (e.g. trivial single-token answers).
                    *entry = ThinkPhase::Complete;
                    PhaseEvent::Complete
                } else {
                    // Non-reasoning model path: any non-think token in prefill
                    // means the request is already producing user-visible
                    // output. Skip ThinkDecode entirely.
                    *entry = ThinkPhase::OutputDecode { think_tokens_used: 0 };
                    PhaseEvent::None
                }
            }

            ThinkPhase::ThinkDecode {
                tokens_so_far,
                eat_ema,
                eat_ema_sq,
                rpdi_local,
                rpdi_global,
                force_in_progress,
            } => {
                *tokens_so_far += 1;

                // Update entropy signals if a fresh probe sample is available.
                let mut force_reason: Option<BudgetForceReason> = None;
                if let Some(sig) = entropy_signal {
                    let a = self.config.ema_alpha;

                    *eat_ema = a * sig.eat + (1.0 - a) * *eat_ema;
                    *eat_ema_sq = a * sig.eat * sig.eat + (1.0 - a) * *eat_ema_sq;
                    let eat_variance = (*eat_ema_sq - (*eat_ema * *eat_ema)).max(0.0);

                    let is_transition =
                        sig.token_entropy > self.config.transition_entropy_threshold;
                    let n = *tokens_so_far as f32;
                    let transition_f = f32::from(is_transition);
                    *rpdi_local = a * transition_f + (1.0 - a) * *rpdi_local;
                    *rpdi_global = (*rpdi_global * (n - 1.0) + transition_f) / n;

                    let past_floor = *tokens_so_far > self.config.min_think_tokens;
                    let converged =
                        past_floor && eat_variance < self.config.eat_ema_variance_threshold;
                    let overthinking = past_floor
                        && *rpdi_global > 0.001
                        && (*rpdi_local / *rpdi_global) > self.config.rpdi_threshold;

                    if !*force_in_progress {
                        // Convergence wins ties with overthinking — converged
                        // means the model itself is willing to stop, which is
                        // a stronger signal than "looks confused".
                        if converged {
                            force_reason = Some(BudgetForceReason::Converged);
                        } else if overthinking {
                            force_reason = Some(BudgetForceReason::Overthinking);
                        }
                    }
                }

                // Hard cap: safety valve regardless of entropy signals.
                if *tokens_so_far >= self.config.max_think_tokens
                    && !*force_in_progress
                    && force_reason.is_none()
                {
                    force_reason = Some(BudgetForceReason::HardCap);
                }

                // Phase transition on </think> token. This check runs *after*
                // the token-count update so the emitted `tokens_used` is
                // inclusive of the boundary token itself.
                if self.config.think_end_ids.contains(&token_id) {
                    let used = *tokens_so_far;
                    *entry = ThinkPhase::OutputDecode {
                        think_tokens_used: used,
                    };
                    metrics::histogram!(names::THINK_TOKENS_PER_REQUEST).record(f64::from(used));
                    return PhaseEvent::ExitThink { tokens_used: used };
                }

                // Nested <think> inside an active think phase is treated as
                // noise — the spec says reasoning is a single contiguous span.

                if let Some(reason) = force_reason {
                    *force_in_progress = true;
                    metrics::counter!(names::BUDGET_FORCE_TRIGGERED).increment(1);
                    metrics::counter!(
                        names::BUDGET_FORCE_REASON,
                        "reason" => reason.as_label(),
                    )
                    .increment(1);
                    PhaseEvent::ForceBudget {
                        inject_token: self.config.think_end_ids[0],
                        reason,
                    }
                } else {
                    PhaseEvent::None
                }
            }

            ThinkPhase::OutputDecode { .. } => {
                if self.config.eos_ids.contains(&token_id) {
                    *entry = ThinkPhase::Complete;
                    PhaseEvent::Complete
                } else {
                    PhaseEvent::None
                }
            }

            ThinkPhase::Complete => PhaseEvent::None,
        }
    }
}
