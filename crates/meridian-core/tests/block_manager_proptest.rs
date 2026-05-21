//! Property-based invariants for `PhaseAwareBlockManager`.
//!
//! These complement the example-based tests in `block_manager.rs`: rather
//! than asserting specific layouts, they drive arbitrary operation sequences
//! and assert structural invariants that must hold after every step.

use meridian_core::block_manager::{BlockManager, PhaseAwareBlockManager};
use meridian_core::types::BlockTier;
use proptest::prelude::*;

const BLOCK_BYTES: u32 = 16;
const CAPACITY_BLOCKS: u64 = 32;

#[derive(Debug, Clone)]
enum Op {
    Allocate { req: u64, tier: u8, n: u32 },
    Free { req: u64 },
    FreeById { idx: usize },
    Demote { req: u64 },
    EvictFor { bytes: u64 },
}

fn tier_of(t: u8) -> BlockTier {
    match t % 3 {
        0 => BlockTier::ThinkComplete,
        1 => BlockTier::ThinkActive,
        _ => BlockTier::OutputCritical,
    }
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0u64..6, 0u8..3, 1u32..6).prop_map(|(req, tier, n)| Op::Allocate { req, tier, n }),
        (0u64..6).prop_map(|req| Op::Free { req }),
        (0usize..40).prop_map(|idx| Op::FreeById { idx }),
        (0u64..6).prop_map(|req| Op::Demote { req }),
        (0u64..(CAPACITY_BLOCKS * u64::from(BLOCK_BYTES)))
            .prop_map(|bytes| Op::EvictFor { bytes }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// After any sequence of operations the manager's accounting and tier
    /// indices stay self-consistent.
    #[test]
    fn invariants_hold_under_arbitrary_ops(ops in prop::collection::vec(op_strategy(), 0..60)) {
        let capacity = CAPACITY_BLOCKS * u64::from(BLOCK_BYTES);
        let mut mgr = PhaseAwareBlockManager::new_with_block_size(BLOCK_BYTES, capacity, false);
        // Mirror of every live block id, so FreeById can target a real id and
        // we can cross-check `is_resident`.
        let mut live: Vec<u32> = Vec::new();

        for op in ops {
            match op {
                Op::Allocate { req, tier, n } => {
                    if let Ok(ids) = mgr.allocate(req, tier_of(tier), n) {
                        live.extend(ids);
                    }
                }
                Op::Free { req } => {
                    mgr.free(req);
                    live.retain(|id| mgr.is_resident(*id));
                }
                Op::FreeById { idx } => {
                    if !live.is_empty() {
                        let id = live[idx % live.len()];
                        let was_resident = mgr.is_resident(id);
                        let freed = mgr.free_block_by_id(id);
                        // free_block_by_id reports presence accurately.
                        prop_assert_eq!(freed, was_resident);
                        live.retain(|x| mgr.is_resident(*x));
                    }
                }
                Op::Demote { req } => {
                    mgr.demote_think_blocks(req);
                }
                Op::EvictFor { bytes } => {
                    let freed = mgr.evict_for(bytes);
                    // Eviction never reports more than was requested-plus-one-block
                    // of slack, and never exceeds the whole pool.
                    prop_assert!(freed <= capacity);
                    live.retain(|id| mgr.is_resident(*id));
                }
            }

            // --- Invariants after every operation -------------------------
            // 1. Usage never exceeds capacity.
            prop_assert!(mgr.used_bytes() <= mgr.capacity_bytes());

            // 2. used_bytes equals the sum of live blocks (one block each).
            let live_count = live.iter().filter(|id| mgr.is_resident(**id)).count() as u64;
            prop_assert_eq!(mgr.used_bytes(), live_count * u64::from(BLOCK_BYTES));

            // 3. Tier occupancy sums to the resident block count: each block
            //    is in exactly one tier index.
            let tier_total: usize = [
                BlockTier::ThinkComplete,
                BlockTier::ThinkActive,
                BlockTier::OutputCritical,
            ]
            .iter()
            .map(|t| mgr.blocks_in_tier(*t))
            .sum();
            prop_assert_eq!(tier_total as u64, live_count);
        }
    }
}
