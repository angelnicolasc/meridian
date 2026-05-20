//! Three-tier phase-aware KV block manager.
//!
//! ## Eviction policy
//!
//! Blocks live in one of three tiers. Eviction iterates them in order:
//!
//! 1. [`BlockTier::ThinkComplete`] — think-phase blocks belonging to a
//!    request that has already transitioned to `OutputDecode`. **Always
//!    first to evict.**
//! 2. [`BlockTier::ThinkActive`] — think-phase blocks of a request still
//!    reasoning. Evicting these pauses reasoning until the block is
//!    re-allocated, which is acceptable degradation.
//! 3. [`BlockTier::OutputCritical`] — output-phase blocks. Eviction here is
//!    user-visible (a stream stutter) and emits a `tracing::warn!` plus an
//!    increment on `meridian.output_critical_eviction`. Reserved for last
//!    resort.
//!
//! Within a tier, eviction is LRU on `last_access_ns`. A `BTreeSet<(ns, id)>`
//! gives `O(log n)` insert and `O(1)` "least recently accessed" lookup via
//! `.iter().next()`.
//!
//! ## MLA accounting
//!
//! When constructed with an [`MlaBlockConfig`], block sizing reflects the
//! latent-dim representation rather than the full per-head KV. This matches
//! what an MLA-aware model (DeepSeek-V3, Qwen3) actually allocates on device.
//! See [`crate::types::MlaBlockConfig::block_size_bytes`].

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use tracing::warn;

use crate::error::{Error, Result};
use crate::metrics::names;
use crate::types::{BlockLocation, BlockTier, KvBlock, MlaBlockConfig};

/// Monotonic counter used to time-stamp `last_access_ns` deterministically in
/// tests. In production this would be wall-clock nanoseconds via
/// `std::time::Instant`; for the scheduler the only requirement is
/// monotonicity inside a process.
fn next_access_ts() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Operations every KV block manager must support.
pub trait BlockManager: Send + Sync {
    /// Allocate `n_blocks` blocks for `req_id` in `tier`. Returns the assigned
    /// block ids in allocation order. Returns
    /// [`Error::KvMemoryExhausted`] if memory pressure cannot be relieved.
    fn allocate(&mut self, req_id: u64, tier: BlockTier, n_blocks: u32) -> Result<Vec<u32>>;

    /// Release every block belonging to `req_id`. Idempotent.
    fn free(&mut self, req_id: u64);

    /// Demote every `ThinkActive` block belonging to `req_id` to
    /// `ThinkComplete`. Called from the scheduler immediately on
    /// [`crate::types::PhaseEvent::ExitThink`].
    fn demote_think_blocks(&mut self, req_id: u64);

    /// Evict blocks across tiers (lowest-tier first) until at least
    /// `needed_bytes` have been freed. Returns the number of bytes freed.
    fn evict_for(&mut self, needed_bytes: u64) -> u64;

    /// Touch `block_id` — update its `last_access_ns` and reposition it in
    /// the LRU index. Cheap; called by the worker on every block read.
    fn touch(&mut self, block_id: u32);

    /// Currently used bytes across all tiers.
    fn used_bytes(&self) -> u64;

    /// Total budget in bytes.
    fn capacity_bytes(&self) -> u64;

    /// Number of free (unused) bytes available without eviction.
    fn available_bytes(&self) -> u64 {
        self.capacity_bytes().saturating_sub(self.used_bytes())
    }

    // -------------------------------------------------------------------
    // Disaggregated KV transfer (ADR-0006)
    //
    // These methods are object-safe but default to "no fabric": a manager
    // without a configured backend reports every block as `Local` and
    // returns `Error::DisaggUnavailable` for offload / ingest. The
    // `meridian-kernels::nixl::NixlBackedBlockManager` wrapper implements
    // them against the chosen fabric.
    // -------------------------------------------------------------------

    /// Serialise `block_id` to a transferable payload and drop it from the
    /// local pool. The payload is opaque to the core; fabric layers
    /// interpret it (NIXL uses raw bytes; Mooncake adds a small frame).
    ///
    /// Default implementation returns [`Error::DisaggUnavailable`].
    fn offload_block(&mut self, block_id: u32) -> Result<Vec<u8>> {
        let _ = block_id;
        Err(Error::DisaggUnavailable)
    }

    /// Reconstitute a block from a remote payload into `tier` and return
    /// its locally-assigned id.
    ///
    /// Default implementation returns [`Error::DisaggUnavailable`].
    fn ingest_block(&mut self, payload: Vec<u8>, tier: BlockTier) -> Result<u32> {
        let _ = (payload, tier);
        Err(Error::DisaggUnavailable)
    }

    /// Where the given block currently lives. The default in-process
    /// implementation reports every owned block as [`BlockLocation::Local`]
    /// and unknown ids as `Local` as well (since the caller is asking about
    /// something this manager never tracked).
    fn block_location(&self, block_id: u32) -> BlockLocation {
        let _ = block_id;
        BlockLocation::Local
    }
}

/// Concrete in-memory block manager.
#[derive(Debug)]
pub struct PhaseAwareBlockManager {
    blocks: HashMap<u32, KvBlock>,
    /// Stack of block ids that have been freed and can be reused without
    /// going through eviction. Pop is `O(1)`.
    free_list: Vec<u32>,
    /// Per-tier sorted index keyed by `(last_access_ns, block_id)`.
    eviction_queues: [BTreeSet<(u64, u32)>; 3],
    used_bytes: u64,
    capacity_bytes: u64,
    next_block_id: u32,
    block_size_bytes: u32,
    /// True if `kv_memory.aggressive_think_eviction` is set — demoted blocks
    /// are freed immediately on phase transition.
    aggressive_think_eviction: bool,
}

impl PhaseAwareBlockManager {
    /// Construct from an MLA block layout and a byte budget.
    ///
    /// `aggressive_think_eviction` mirrors `kv_memory.aggressive_think_eviction`
    /// in `meridian.toml`. When `true`, [`Self::demote_think_blocks`] frees
    /// rather than re-tiers.
    #[must_use]
    pub fn new_mla(
        mla: MlaBlockConfig,
        capacity_bytes: u64,
        aggressive_think_eviction: bool,
    ) -> Self {
        Self {
            blocks: HashMap::with_capacity(1024),
            free_list: Vec::with_capacity(256),
            eviction_queues: [BTreeSet::new(), BTreeSet::new(), BTreeSet::new()],
            used_bytes: 0,
            capacity_bytes,
            next_block_id: 0,
            block_size_bytes: mla.block_size_bytes(),
            aggressive_think_eviction,
        }
    }

    /// Construct with an explicit per-block byte cost. Use this for
    /// non-MLA models or for tests where the layout is irrelevant.
    #[must_use]
    pub fn new_with_block_size(
        block_size_bytes: u32,
        capacity_bytes: u64,
        aggressive_think_eviction: bool,
    ) -> Self {
        Self {
            blocks: HashMap::with_capacity(1024),
            free_list: Vec::with_capacity(256),
            eviction_queues: [BTreeSet::new(), BTreeSet::new(), BTreeSet::new()],
            used_bytes: 0,
            capacity_bytes,
            next_block_id: 0,
            block_size_bytes,
            aggressive_think_eviction,
        }
    }

    /// Per-block byte cost in the configured layout.
    #[must_use]
    pub fn block_size_bytes(&self) -> u32 {
        self.block_size_bytes
    }

    /// Count of blocks currently in a given tier. Useful for tests and for
    /// observability gauges.
    #[must_use]
    pub fn blocks_in_tier(&self, tier: BlockTier) -> usize {
        self.eviction_queues[tier as usize].len()
    }

    fn fresh_block_id(&mut self) -> u32 {
        if let Some(id) = self.free_list.pop() {
            return id;
        }
        let id = self.next_block_id;
        self.next_block_id = self.next_block_id.checked_add(1).unwrap_or_else(|| {
            // 4G blocks is well past anything plausible; cap and require
            // operator intervention rather than wrap silently.
            panic!("PhaseAwareBlockManager: next_block_id overflow")
        });
        id
    }

    fn insert_into_tier(&mut self, block: KvBlock) {
        let key = (block.last_access_ns, block.block_id);
        self.eviction_queues[block.tier as usize].insert(key);
        self.blocks.insert(block.block_id, block);
    }

    fn remove_from_tier_index(&mut self, block_id: u32) {
        if let Some(block) = self.blocks.get(&block_id) {
            self.eviction_queues[block.tier as usize]
                .remove(&(block.last_access_ns, block.block_id));
        }
    }
}

impl BlockManager for PhaseAwareBlockManager {
    fn allocate(&mut self, req_id: u64, tier: BlockTier, n_blocks: u32) -> Result<Vec<u32>> {
        let needed_bytes = u64::from(n_blocks) * u64::from(self.block_size_bytes);

        // Ensure capacity, evicting if necessary.
        if self.available_bytes() < needed_bytes {
            let deficit = needed_bytes - self.available_bytes();
            let freed = self.evict_for(deficit);
            if freed < deficit {
                return Err(Error::KvMemoryExhausted {
                    needed: needed_bytes,
                    freed,
                });
            }
        }

        let mut assigned = Vec::with_capacity(n_blocks as usize);
        for i in 0..n_blocks {
            let block_id = self.fresh_block_id();
            let ts = next_access_ts();
            let block = KvBlock {
                block_id,
                req_id,
                tier,
                token_range: i..(i + 1),
                size_bytes: self.block_size_bytes,
                last_access_ns: ts,
            };
            self.used_bytes += u64::from(block.size_bytes);
            self.insert_into_tier(block);
            assigned.push(block_id);
        }

        metrics::gauge!(names::BLOCK_MANAGER_USED_BYTES).set(self.used_bytes as f64);
        Ok(assigned)
    }

    fn free(&mut self, req_id: u64) {
        let to_remove: Vec<u32> = self
            .blocks
            .values()
            .filter(|b| b.req_id == req_id)
            .map(|b| b.block_id)
            .collect();

        for block_id in to_remove {
            self.remove_from_tier_index(block_id);
            if let Some(block) = self.blocks.remove(&block_id) {
                self.used_bytes = self.used_bytes.saturating_sub(u64::from(block.size_bytes));
                self.free_list.push(block_id);
            }
        }

        metrics::gauge!(names::BLOCK_MANAGER_USED_BYTES).set(self.used_bytes as f64);
    }

    fn demote_think_blocks(&mut self, req_id: u64) {
        let to_demote: Vec<u32> = self
            .blocks
            .values()
            .filter(|b| b.req_id == req_id && b.tier == BlockTier::ThinkActive)
            .map(|b| b.block_id)
            .collect();

        if self.aggressive_think_eviction {
            // Aggressive mode: free immediately rather than re-tier.
            for block_id in to_demote {
                self.remove_from_tier_index(block_id);
                if let Some(block) = self.blocks.remove(&block_id) {
                    self.used_bytes =
                        self.used_bytes.saturating_sub(u64::from(block.size_bytes));
                    self.free_list.push(block_id);
                    metrics::counter!(
                        names::BLOCK_MANAGER_EVICTIONS,
                        "tier" => "think_active_aggressive",
                    )
                    .increment(1);
                }
            }
            metrics::gauge!(names::BLOCK_MANAGER_USED_BYTES).set(self.used_bytes as f64);
            return;
        }

        for block_id in to_demote {
            self.remove_from_tier_index(block_id);
            if let Some(block) = self.blocks.get_mut(&block_id) {
                block.tier = BlockTier::ThinkComplete;
                let key = (block.last_access_ns, block.block_id);
                self.eviction_queues[BlockTier::ThinkComplete as usize].insert(key);
            }
        }
    }

    fn evict_for(&mut self, needed_bytes: u64) -> u64 {
        let mut freed: u64 = 0;

        for tier in BlockTier::eviction_order() {
            if freed >= needed_bytes {
                break;
            }

            if tier == BlockTier::OutputCritical {
                warn!(
                    freed_so_far = freed,
                    needed = needed_bytes,
                    "evicting OutputCritical KV blocks — severe memory pressure",
                );
                metrics::counter!(names::OUTPUT_CRITICAL_EVICTION).increment(1);
            }

            // Drain LRU entries from this tier until the deficit clears.
            while freed < needed_bytes {
                let key = match self.eviction_queues[tier as usize].iter().next().copied() {
                    Some(k) => k,
                    None => break,
                };
                self.eviction_queues[tier as usize].remove(&key);
                let (_, block_id) = key;

                if let Some(block) = self.blocks.remove(&block_id) {
                    let bytes = u64::from(block.size_bytes);
                    self.used_bytes = self.used_bytes.saturating_sub(bytes);
                    freed += bytes;
                    self.free_list.push(block_id);
                }
            }

            metrics::counter!(
                names::BLOCK_MANAGER_EVICTIONS,
                "tier" => tier_label(tier),
            )
            .increment(0); // touch the metric so it exists even when never fired
        }

        if freed > 0 {
            metrics::gauge!(names::BLOCK_MANAGER_USED_BYTES).set(self.used_bytes as f64);
        }
        freed
    }

    fn touch(&mut self, block_id: u32) {
        let Some(block) = self.blocks.get_mut(&block_id) else {
            return;
        };
        let old_key = (block.last_access_ns, block.block_id);
        let new_ts = next_access_ts();
        block.last_access_ns = new_ts;
        let new_key = (new_ts, block.block_id);
        let tier = block.tier;
        self.eviction_queues[tier as usize].remove(&old_key);
        self.eviction_queues[tier as usize].insert(new_key);
    }

    fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }
}

const fn tier_label(t: BlockTier) -> &'static str {
    match t {
        BlockTier::ThinkComplete => "think_complete",
        BlockTier::ThinkActive => "think_active",
        BlockTier::OutputCritical => "output_critical",
    }
}
