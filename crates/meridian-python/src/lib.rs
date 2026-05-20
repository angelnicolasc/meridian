//! Python bindings for `meridian-core`. Compiled into `_meridian.{abi3,so,pyd}`
//! and re-exported from `python/meridian/__init__.py`.
//!
//! Only the contracts the Python plugin actually needs are exposed here —
//! `PhaseRouter`, `PhaseRouterConfig`, `MeridianConfig`, and the event /
//! signal types. Internal Rust-only types remain Rust-only.

#![doc(html_root_url = "https://docs.rs/meridian-python/0.1.0")]
#![warn(missing_docs, missing_debug_implementations, unreachable_pub)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::sync::Mutex;

use meridian_core::block_manager::{BlockManager, PhaseAwareBlockManager};
use meridian_core::config::{MeridianConfig as CoreConfig, SchedulerConfig as CoreSchedConfig};
use meridian_core::phase_router::{
    PhaseRouter as CoreRouter, PhaseRouterConfig as CoreRouterConfig,
};
use meridian_core::scheduler::MeridianScheduler as CoreScheduler;
use meridian_core::types::{
    BlockTier, BudgetForceReason, EntropySignal as CoreSignal, MlaBlockConfig, PhaseEvent,
};

// ---------------------------------------------------------------------------
// PhaseEvent → Python tuple representation
// ---------------------------------------------------------------------------

fn event_to_py(py: Python<'_>, ev: PhaseEvent) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    match ev {
        PhaseEvent::None => d.set_item("kind", "none")?,
        PhaseEvent::EnterThink => d.set_item("kind", "enter_think")?,
        PhaseEvent::ExitThink { tokens_used } => {
            d.set_item("kind", "exit_think")?;
            d.set_item("tokens_used", tokens_used)?;
        }
        PhaseEvent::ForceBudget {
            inject_token,
            reason,
        } => {
            d.set_item("kind", "force_budget")?;
            d.set_item("inject_token", inject_token)?;
            d.set_item("reason", reason.as_label())?;
        }
        PhaseEvent::Complete => d.set_item("kind", "complete")?,
    }
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// EntropySignal
// ---------------------------------------------------------------------------

/// Per-token entropy signal computed by the probe (CPU or CUDA).
#[pyclass(name = "EntropySignal", module = "meridian._meridian", frozen)]
#[derive(Debug, Clone, Copy)]
pub struct PyEntropySignal {
    inner: CoreSignal,
}

#[pymethods]
impl PyEntropySignal {
    #[new]
    fn new(token_entropy: f32, eat: f32, eat_ema: f32, eat_ema_variance: f32) -> Self {
        Self {
            inner: CoreSignal {
                token_entropy,
                eat,
                eat_ema,
                eat_ema_variance,
            },
        }
    }

    #[getter]
    fn token_entropy(&self) -> f32 {
        self.inner.token_entropy
    }
    #[getter]
    fn eat(&self) -> f32 {
        self.inner.eat
    }
    #[getter]
    fn eat_ema(&self) -> f32 {
        self.inner.eat_ema
    }
    #[getter]
    fn eat_ema_variance(&self) -> f32 {
        self.inner.eat_ema_variance
    }

    fn __repr__(&self) -> String {
        format!(
            "EntropySignal(token_entropy={:.4}, eat={:.4}, eat_ema={:.4}, eat_ema_variance={:.6})",
            self.inner.token_entropy,
            self.inner.eat,
            self.inner.eat_ema,
            self.inner.eat_ema_variance,
        )
    }
}

// ---------------------------------------------------------------------------
// PhaseRouter
// ---------------------------------------------------------------------------

/// Phase-aware token-stream state machine.
#[pyclass(name = "PhaseRouter", module = "meridian._meridian")]
#[derive(Debug)]
pub struct PyPhaseRouter {
    inner: CoreRouter,
}

#[pymethods]
impl PyPhaseRouter {
    #[new]
    #[pyo3(signature = (
        think_start_ids,
        think_end_ids,
        eos_ids,
        ema_alpha = 0.05,
        transition_entropy_threshold = 2.5,
        rpdi_threshold = 3.0,
        eat_ema_variance_threshold = 0.001,
        min_think_tokens = 512,
        max_think_tokens = 32_768,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        think_start_ids: Vec<u32>,
        think_end_ids: Vec<u32>,
        eos_ids: Vec<u32>,
        ema_alpha: f32,
        transition_entropy_threshold: f32,
        rpdi_threshold: f32,
        eat_ema_variance_threshold: f32,
        min_think_tokens: u32,
        max_think_tokens: u32,
    ) -> PyResult<Self> {
        if think_end_ids.is_empty() {
            return Err(PyValueError::new_err(
                "think_end_ids must contain at least one id",
            ));
        }
        Ok(Self {
            inner: CoreRouter::new(CoreRouterConfig {
                think_start_ids,
                think_end_ids,
                eos_ids,
                ema_alpha,
                transition_entropy_threshold,
                rpdi_threshold,
                eat_ema_variance_threshold,
                min_think_tokens,
                max_think_tokens,
            }),
        })
    }

    fn register(&self, req_id: u64) {
        self.inner.register(req_id);
    }

    fn reap(&self, req_id: u64) {
        self.inner.reap(req_id);
    }

    fn tracked_requests(&self) -> usize {
        self.inner.tracked_requests()
    }

    fn phase_of_kind(&self, req_id: u64) -> Option<&'static str> {
        self.inner.phase_of_kind(req_id)
    }

    fn touch(&self, req_id: u64) {
        self.inner.touch(req_id);
    }

    fn reap_stale(&self, older_than_ticks: u64) -> usize {
        self.inner.reap_stale(older_than_ticks)
    }

    /// Wall-clock variant. Caller expresses idle threshold in seconds; the
    /// router translates to ticks using its measured rate.
    fn reap_stale_older_than(&self, seconds: f64) -> PyResult<usize> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(PyValueError::new_err(
                "seconds must be a non-negative finite number",
            ));
        }
        Ok(self
            .inner
            .reap_stale_older_than(std::time::Duration::from_secs_f64(seconds)))
    }

    #[pyo3(signature = (req_id, token_id, entropy_signal = None))]
    fn on_token(
        &self,
        py: Python<'_>,
        req_id: u64,
        token_id: u32,
        entropy_signal: Option<PyEntropySignal>,
    ) -> PyResult<PyObject> {
        let ev = match entropy_signal {
            Some(s) => self.inner.on_token(req_id, token_id, Some(&s.inner)),
            None => self.inner.on_token(req_id, token_id, None),
        };
        event_to_py(py, ev)
    }
}

// ---------------------------------------------------------------------------
// MeridianScheduler
// ---------------------------------------------------------------------------

/// Dual-queue scheduler binding.
#[pyclass(name = "MeridianScheduler", module = "meridian._meridian")]
#[derive(Debug)]
pub struct PyMeridianScheduler {
    inner: CoreScheduler,
}

#[pymethods]
impl PyMeridianScheduler {
    #[new]
    #[pyo3(signature = (
        think_tpot_budget_ms = 80.0,
        output_tpot_budget_ms = 20.0,
        think_batch_multiplier = 2.5,
        max_think_tokens = 32_768,
        min_think_tokens = 512,
    ))]
    fn new(
        think_tpot_budget_ms: f32,
        output_tpot_budget_ms: f32,
        think_batch_multiplier: f32,
        max_think_tokens: u32,
        min_think_tokens: u32,
    ) -> PyResult<Self> {
        let cfg = CoreSchedConfig {
            think_tpot_budget_ms,
            output_tpot_budget_ms,
            think_batch_multiplier,
            max_think_tokens,
            min_think_tokens,
        };
        Ok(Self {
            inner: CoreScheduler::new(cfg),
        })
    }

    fn admit(&self, req_id: u64) {
        self.inner.admit(req_id);
    }

    fn schedule_batch(
        &self,
        py: Python<'_>,
        output_budget_slots: usize,
        available_blocks: usize,
    ) -> PyResult<PyObject> {
        let batch = self
            .inner
            .schedule_batch(output_budget_slots, available_blocks);
        let d = PyDict::new(py);
        let outputs: Vec<u64> = batch.output_slots.iter().map(|s| s.req_id).collect();
        let thinks: Vec<u64> = batch.think_slots.iter().map(|s| s.req_id).collect();
        d.set_item("output", outputs)?;
        d.set_item("think", thinks)?;
        Ok(d.into())
    }

    fn queue_injection(&self, req_id: u64, token_id: u32) {
        self.inner.on_phase_event(
            req_id,
            PhaseEvent::ForceBudget {
                inject_token: token_id,
                reason: BudgetForceReason::HardCap,
            },
        );
    }

    fn signal_exit_think(&self, req_id: u64, tokens_used: u32) {
        self.inner
            .on_phase_event(req_id, PhaseEvent::ExitThink { tokens_used });
    }

    fn signal_complete(&self, req_id: u64) {
        self.inner.on_phase_event(req_id, PhaseEvent::Complete);
    }

    fn pop_pending_injection(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner.pop_pending_injection() {
            Some(pi) => {
                let d = PyDict::new(py);
                d.set_item("req_id", pi.req_id)?;
                d.set_item("token_id", pi.token_id)?;
                Ok(d.into())
            }
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn output_queue_depth(&self) -> usize {
        self.inner.output_queue_depth()
    }
    #[getter]
    fn think_queue_depth(&self) -> usize {
        self.inner.think_queue_depth()
    }
}

// ---------------------------------------------------------------------------
// PhaseAwareBlockManager
// ---------------------------------------------------------------------------

/// Tier label accepted by [`PyBlockManager`] methods.
fn tier_from_str(s: &str) -> PyResult<BlockTier> {
    match s {
        "think_complete" => Ok(BlockTier::ThinkComplete),
        "think_active" => Ok(BlockTier::ThinkActive),
        "output_critical" => Ok(BlockTier::OutputCritical),
        other => Err(PyValueError::new_err(format!(
            "unknown block tier: {other}"
        ))),
    }
}

/// Phase-aware KV block manager (Python view).
#[pyclass(name = "BlockManager", module = "meridian._meridian")]
#[derive(Debug)]
pub struct PyBlockManager {
    inner: Mutex<PhaseAwareBlockManager>,
}

#[pymethods]
impl PyBlockManager {
    /// Construct from explicit per-block byte cost. Use for non-MLA models.
    #[new]
    #[pyo3(signature = (block_size_bytes, capacity_bytes, aggressive_think_eviction = false))]
    fn new(block_size_bytes: u32, capacity_bytes: u64, aggressive_think_eviction: bool) -> Self {
        Self {
            inner: Mutex::new(PhaseAwareBlockManager::new_with_block_size(
                block_size_bytes,
                capacity_bytes,
                aggressive_think_eviction,
            )),
        }
    }

    /// Construct from MLA layout.
    #[staticmethod]
    #[pyo3(signature = (
        latent_dim,
        bytes_per_element,
        tokens_per_block,
        capacity_bytes,
        aggressive_think_eviction = false,
    ))]
    fn mla(
        latent_dim: u32,
        bytes_per_element: u8,
        tokens_per_block: u32,
        capacity_bytes: u64,
        aggressive_think_eviction: bool,
    ) -> Self {
        let mla = MlaBlockConfig {
            latent_dim,
            bytes_per_element,
            tokens_per_block,
        };
        Self {
            inner: Mutex::new(PhaseAwareBlockManager::new_mla(
                mla,
                capacity_bytes,
                aggressive_think_eviction,
            )),
        }
    }

    fn allocate(&self, req_id: u64, tier: &str, n_blocks: u32) -> PyResult<Vec<u32>> {
        let tier = tier_from_str(tier)?;
        let mut mgr = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        mgr.allocate(req_id, tier, n_blocks)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    fn free(&self, req_id: u64) -> PyResult<()> {
        let mut mgr = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        mgr.free(req_id);
        Ok(())
    }

    fn demote_think_blocks(&self, req_id: u64) -> PyResult<()> {
        let mut mgr = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        mgr.demote_think_blocks(req_id);
        Ok(())
    }

    fn evict_for(&self, needed_bytes: u64) -> PyResult<u64> {
        let mut mgr = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(mgr.evict_for(needed_bytes))
    }

    fn touch(&self, block_id: u32) -> PyResult<()> {
        let mut mgr = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        mgr.touch(block_id);
        Ok(())
    }

    #[getter]
    fn used_bytes(&self) -> PyResult<u64> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .used_bytes())
    }

    #[getter]
    fn capacity_bytes(&self) -> PyResult<u64> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .capacity_bytes())
    }

    #[getter]
    fn available_bytes(&self) -> PyResult<u64> {
        let mgr = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(mgr.capacity_bytes().saturating_sub(mgr.used_bytes()))
    }

    /// How many *whole blocks* of the configured layout can still be
    /// allocated without eviction. The plugin uses this as the
    /// `available_blocks` budget when calling `MeridianScheduler.schedule_batch`.
    fn available_blocks(&self) -> PyResult<usize> {
        let mgr = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let block_bytes = u64::from(mgr.block_size_bytes());
        if block_bytes == 0 {
            return Ok(0);
        }
        let free = mgr.capacity_bytes().saturating_sub(mgr.used_bytes());
        Ok((free / block_bytes) as usize)
    }
}

// ---------------------------------------------------------------------------
// MeridianConfig loader
// ---------------------------------------------------------------------------

/// Load a `meridian.toml` from a path and return its contents as a dict.
#[pyfunction]
fn load_config(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let cfg = CoreConfig::from_path(path).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let json = serde_json::to_string(&serialize_config(&cfg))
        .map_err(|e| PyValueError::new_err(format!("serialize: {e}")))?;
    let module = py.import("json")?;
    let parsed = module.getattr("loads")?.call1((json,))?;
    Ok(parsed.into())
}

#[derive(serde::Serialize)]
struct SerializableConfig<'a> {
    scheduler: &'a meridian_core::config::SchedulerConfig,
    entropy: &'a meridian_core::config::EntropyConfig,
    kv_memory: &'a meridian_core::config::KvConfig,
    disagg: &'a meridian_core::config::DisaggConfig,
    model: &'a std::collections::BTreeMap<String, meridian_core::config::ModelConfig>,
}

fn serialize_config(cfg: &CoreConfig) -> SerializableConfig<'_> {
    SerializableConfig {
        scheduler: &cfg.scheduler,
        entropy: &cfg.entropy,
        kv_memory: &cfg.kv_memory,
        disagg: &cfg.disagg,
        model: &cfg.model,
    }
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

/// Top-level `_meridian` extension module.
#[pymodule]
fn _meridian(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPhaseRouter>()?;
    m.add_class::<PyEntropySignal>()?;
    m.add_class::<PyMeridianScheduler>()?;
    m.add_class::<PyBlockManager>()?;
    m.add_function(wrap_pyfunction!(load_config, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
