//! # meridian-core
//!
//! Phase-aware inference-time compute scheduler for reasoning-model serving.
//!
//! ## Why this crate exists
//!
//! Reasoning models (DeepSeek-R1, Qwen3, Claude Opus 4.x, o3) emit two structurally
//! different token sequences within a single request:
//!
//! ```text
//! [prompt] -> <think> ... N reasoning tokens ... </think> -> [output tokens]
//! ```
//!
//! These phases have opposite latency profiles. Think-decode is throughput-bound and
//! the user does not see inter-token latency. Output-decode is the user-visible
//! streaming experience and must be protected. `meridian-core` is the scheduling
//! logic that treats them as separate domains.
//!
//! ## Crate layout
//!
//! - [`phase_router`] — per-request token-stream state machine. The only component
//!   that touches every generated token. Hot path is `O(1)` with zero allocation in
//!   the common case. **Fully implemented in Sprint 0.**
//! - [`scheduler`] — dual-queue dispatcher with phase-differentiated SLOs.
//!   Contracts in place; dispatch logic lands in Sprint 1.
//! - [`block_manager`] — three-tier KV block manager. Contracts in place; tier-aware
//!   eviction logic lands in Sprint 1.
//! - [`config`] — TOML-driven configuration with validated defaults.
//! - [`types`] — shared domain types: `ThinkPhase`, `PhaseEvent`, `BlockTier`,
//!   `EntropySignal`, `RequestSlot`, `ScheduledBatch`.
//! - [`metrics`] — Prometheus / OpenTelemetry signals emitted by the scheduler.
//! - [`error`] — typed error surface (no panicking public API).
//!
//! ## Stability
//!
//! Pre-1.0. The public API is stabilising sprint-by-sprint; breaking changes are
//! recorded in [`CHANGELOG.md`](https://github.com/angelnicolasc/meridian/blob/main/CHANGELOG.md)
//! under a `BREAKING CHANGE` footer.

#![doc(html_root_url = "https://docs.rs/meridian-core/0.1.0")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![warn(
    missing_docs,
    missing_debug_implementations,
    unreachable_pub,
    rust_2018_idioms
)]

pub mod block_manager;
pub mod config;
pub mod error;
pub mod metrics;
pub mod phase_router;
pub mod scheduler;
#[cfg(feature = "otel")]
pub mod telemetry;
pub mod types;

pub use crate::block_manager::{BlockManager, PhaseAwareBlockManager};
pub use crate::config::{
    CapacitySpec, DisaggConfig, DisaggFabric, EntropyConfig, KvConfig, MeridianConfig, ModelConfig,
    SchedulerConfig,
};
pub use crate::error::{Error, Result};
pub use crate::phase_router::{PhaseRouter, PhaseRouterConfig};
pub use crate::scheduler::MeridianScheduler;
pub use crate::types::{
    BlockLocation, BlockTier, BudgetForceReason, EntropySignal, KvBlock, MlaBlockConfig,
    PhaseEvent, Priority, RequestSlot, ScheduledBatch, ThinkPhase,
};

/// Semantic version of `meridian-core`, sourced from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
