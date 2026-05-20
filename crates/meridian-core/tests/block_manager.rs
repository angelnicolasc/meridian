//! Integration tests for `PhaseAwareBlockManager`.
//!
//! Cover the eviction policy invariants: tier ordering, LRU within tier,
//! demote on phase transition, MLA accounting, aggressive eviction mode,
//! and KvMemoryExhausted error semantics.

use meridian_core::block_manager::{BlockManager, PhaseAwareBlockManager};
use meridian_core::error::Error;
use meridian_core::types::{BlockTier, MlaBlockConfig};
use pretty_assertions::assert_eq;

/// Helper: build a manager sized for a known number of blocks.
fn manager_with_blocks(n_blocks: u32, aggressive: bool) -> PhaseAwareBlockManager {
    // 16 bytes per block keeps arithmetic in the test obvious.
    let capacity = u64::from(n_blocks) * 16;
    PhaseAwareBlockManager::new_with_block_size(16, capacity, aggressive)
}

// ---------------------------------------------------------------------------
// allocate / free basics
// ---------------------------------------------------------------------------

#[test]
fn allocate_assigns_unique_ids() {
    let mut mgr = manager_with_blocks(8, false);
    let a = mgr.allocate(1, BlockTier::ThinkActive, 3).unwrap();
    let b = mgr.allocate(2, BlockTier::ThinkActive, 3).unwrap();
    assert_eq!(a.len(), 3);
    assert_eq!(b.len(), 3);
    for id in &a {
        assert!(!b.contains(id), "block id {id} overlapped");
    }
    assert_eq!(mgr.used_bytes(), 6 * 16);
}

#[test]
fn free_releases_blocks_and_recycles_ids() {
    let mut mgr = manager_with_blocks(4, false);
    let _ = mgr.allocate(1, BlockTier::ThinkActive, 2).unwrap();
    mgr.free(1);
    assert_eq!(mgr.used_bytes(), 0);

    // Subsequent allocation should reuse the freed ids.
    let reused = mgr.allocate(2, BlockTier::OutputCritical, 2).unwrap();
    assert_eq!(reused.len(), 2);
    assert_eq!(mgr.blocks_in_tier(BlockTier::OutputCritical), 2);
}

#[test]
fn free_is_idempotent() {
    let mut mgr = manager_with_blocks(4, false);
    let _ = mgr.allocate(7, BlockTier::ThinkActive, 2).unwrap();
    mgr.free(7);
    mgr.free(7); // second free is a no-op
    assert_eq!(mgr.used_bytes(), 0);
}

// ---------------------------------------------------------------------------
// demote_think_blocks
// ---------------------------------------------------------------------------

#[test]
fn demote_moves_think_active_to_think_complete() {
    let mut mgr = manager_with_blocks(8, false);
    let _ = mgr.allocate(1, BlockTier::ThinkActive, 3).unwrap();
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkActive), 3);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkComplete), 0);

    mgr.demote_think_blocks(1);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkActive), 0);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkComplete), 3);
}

#[test]
fn aggressive_eviction_frees_demoted_blocks() {
    let mut mgr = manager_with_blocks(8, true); // aggressive=true
    let _ = mgr.allocate(1, BlockTier::ThinkActive, 3).unwrap();
    assert_eq!(mgr.used_bytes(), 3 * 16);

    mgr.demote_think_blocks(1);
    assert_eq!(mgr.used_bytes(), 0);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkComplete), 0);
}

// ---------------------------------------------------------------------------
// Eviction order
// ---------------------------------------------------------------------------

#[test]
fn evict_for_prefers_lower_tier() {
    let mut mgr = manager_with_blocks(6, false);

    // Layout: 2 ThinkComplete, 2 ThinkActive, 2 OutputCritical. Total = 6 × 16 = 96 bytes.
    let _ = mgr.allocate(1, BlockTier::ThinkActive, 2).unwrap();
    mgr.demote_think_blocks(1); // those become ThinkComplete
    let _ = mgr.allocate(2, BlockTier::ThinkActive, 2).unwrap();
    let _ = mgr.allocate(3, BlockTier::OutputCritical, 2).unwrap();
    assert_eq!(mgr.used_bytes(), 96);

    // Free 32 bytes — must come entirely from ThinkComplete.
    let freed = mgr.evict_for(32);
    assert_eq!(freed, 32);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkComplete), 0);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkActive), 2);
    assert_eq!(mgr.blocks_in_tier(BlockTier::OutputCritical), 2);
}

#[test]
fn evict_for_falls_through_tiers_under_pressure() {
    let mut mgr = manager_with_blocks(4, false);
    // One block per tier.
    let _ = mgr.allocate(1, BlockTier::ThinkActive, 1).unwrap();
    mgr.demote_think_blocks(1);
    let _ = mgr.allocate(2, BlockTier::ThinkActive, 1).unwrap();
    let _ = mgr.allocate(3, BlockTier::OutputCritical, 1).unwrap();
    assert_eq!(mgr.used_bytes(), 48);

    // Need 48 bytes — every block must go, eviction must reach OutputCritical.
    let freed = mgr.evict_for(48);
    assert_eq!(freed, 48);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkComplete), 0);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkActive), 0);
    assert_eq!(mgr.blocks_in_tier(BlockTier::OutputCritical), 0);
}

#[test]
fn evict_for_lru_within_tier() {
    let mut mgr = manager_with_blocks(4, false);
    let a = mgr.allocate(1, BlockTier::ThinkActive, 1).unwrap()[0];
    let _b = mgr.allocate(1, BlockTier::ThinkActive, 1).unwrap()[0];
    let c = mgr.allocate(1, BlockTier::ThinkActive, 1).unwrap()[0];

    mgr.demote_think_blocks(1);
    // Touch `a` and `c`, so `b` becomes the LRU candidate.
    mgr.touch(a);
    mgr.touch(c);

    // Evict one block's worth (16 B) — should hit `b` first.
    let freed = mgr.evict_for(16);
    assert_eq!(freed, 16);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkComplete), 2);
}

// ---------------------------------------------------------------------------
// KvMemoryExhausted
// ---------------------------------------------------------------------------

#[test]
fn allocate_returns_memory_exhausted_when_no_eviction_possible() {
    let mut mgr = manager_with_blocks(2, false);
    // Fill capacity with OutputCritical blocks owned by an active request —
    // those are evictable in extremis, but if we ask for MORE than total
    // capacity, eviction can't bridge the gap.
    let _ = mgr.allocate(1, BlockTier::OutputCritical, 2).unwrap();
    let err = mgr.allocate(2, BlockTier::ThinkActive, 5).unwrap_err();
    assert!(matches!(err, Error::KvMemoryExhausted { .. }), "got {err:?}");
}

#[test]
fn allocate_succeeds_after_evicting_completed_think_blocks() {
    let mut mgr = manager_with_blocks(4, false);
    let _ = mgr.allocate(1, BlockTier::ThinkActive, 4).unwrap();
    mgr.demote_think_blocks(1);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkComplete), 4);

    // No free capacity — but ThinkComplete is fair game.
    let ids = mgr.allocate(2, BlockTier::OutputCritical, 2).unwrap();
    assert_eq!(ids.len(), 2);
    // Eviction frees only what is needed (2 of the 4 ThinkComplete), then
    // the new allocation lands on top. Final: 2 ThinkComplete + 2 OutputCritical.
    assert_eq!(mgr.used_bytes(), 4 * 16);
    assert_eq!(mgr.blocks_in_tier(BlockTier::ThinkComplete), 2);
    assert_eq!(mgr.blocks_in_tier(BlockTier::OutputCritical), 2);
}

// ---------------------------------------------------------------------------
// MLA accounting
// ---------------------------------------------------------------------------

#[test]
fn mla_block_size_drives_accounting() {
    let mla = MlaBlockConfig {
        latent_dim: 512,
        bytes_per_element: 2, // bf16
        tokens_per_block: 16,
    };
    let block_bytes = mla.block_size_bytes();
    assert_eq!(block_bytes, 512 * 2 * 16);

    let mut mgr = PhaseAwareBlockManager::new_mla(mla, u64::from(block_bytes) * 4, false);
    assert_eq!(mgr.block_size_bytes(), block_bytes);

    let _ = mgr.allocate(1, BlockTier::ThinkActive, 2).unwrap();
    assert_eq!(mgr.used_bytes(), u64::from(block_bytes) * 2);
}
