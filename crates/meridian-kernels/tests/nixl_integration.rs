//! Integration tests for the disaggregated KV transfer layer.
#![allow(clippy::literal_string_with_formatting_args)]
//!
//! Gated by `--features nixl`. Exercises the wire protocol and the
//! `NixlBackedBlockManager` end-to-end against the in-process synthetic
//! fabric — the same code path a libnixl runtime would invoke.

#![cfg(feature = "nixl")]

use std::sync::Arc;

use meridian_core::block_manager::{BlockManager, PhaseAwareBlockManager};
use meridian_core::types::{BlockLocation, BlockTier, MlaBlockConfig};
use meridian_kernels::nixl::{
    Fabric, MooncakeAdapter, NixlBackedBlockManager, SyntheticNixlFabric, WireHeader, decode,
    encode,
};
use proptest::prelude::*;

fn any_tier() -> impl Strategy<Value = BlockTier> {
    prop_oneof![
        Just(BlockTier::ThinkComplete),
        Just(BlockTier::ThinkActive),
        Just(BlockTier::OutputCritical),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Framing is lossless: any (tier, body) survives an encode/decode cycle,
    /// and the checksum guards against silent corruption.
    #[test]
    fn encode_decode_round_trips(
        tier in any_tier(),
        body in prop::collection::vec(any::<u8>(), 0..512),
    ) {
        let framed = encode(tier, &body);
        let (decoded_tier, decoded_body) = decode(&framed).expect("decode clean payload");
        prop_assert_eq!(decoded_tier, tier);
        prop_assert_eq!(decoded_body, body);
    }

    /// The Mooncake envelope is transparent: a payload pushed and pulled back
    /// through the adapter is byte-identical to what went in.
    #[test]
    fn mooncake_push_pull_round_trips(
        tier in any_tier(),
        body in prop::collection::vec(any::<u8>(), 0..512),
    ) {
        let fabric = MooncakeAdapter::new(Arc::new(SyntheticNixlFabric::new()));
        let payload = encode(tier, &body);
        let handle = fabric.push(payload.clone()).expect("push");
        let back = fabric.pull(handle).expect("pull");
        prop_assert_eq!(back, payload);
    }
}

fn fresh_local() -> PhaseAwareBlockManager {
    PhaseAwareBlockManager::new_mla(
        MlaBlockConfig {
            latent_dim: 256,
            bytes_per_element: 2,
            tokens_per_block: 16,
        },
        256 * 2 * 16 * 128,
        false,
    )
}

fn fresh_manager() -> NixlBackedBlockManager {
    NixlBackedBlockManager::new(fresh_local(), Arc::new(SyntheticNixlFabric::new()))
}

#[test]
fn wire_header_is_thirty_two_bytes() {
    assert_eq!(
        WireHeader::SIZE,
        32,
        "ADR-0006 commits us to a 32-byte header"
    );
}

#[test]
fn version_one_protocol_is_stable() {
    assert_eq!(WireHeader::VERSION, 1);
    assert_eq!(WireHeader::MAGIC, *b"MRDN");
}

#[test]
fn end_to_end_offload_then_ingest() {
    let mut mgr = fresh_manager();
    let ids = mgr
        .allocate(1, BlockTier::ThinkActive, 4)
        .expect("allocate four blocks");
    assert_eq!(ids.len(), 4);
    for id in &ids {
        assert!(matches!(mgr.block_location(*id), BlockLocation::Local));
    }

    // Offload the first block and verify the location now reports remote.
    let _ = mgr.offload_block(ids[0]).expect("offload");
    assert!(matches!(
        mgr.block_location(ids[0]),
        BlockLocation::Remote {
            fabric: "nixl-synth"
        },
    ));

    // Untouched blocks must still report local.
    for id in &ids[1..] {
        assert!(matches!(mgr.block_location(*id), BlockLocation::Local));
    }

    // Ingest a synthesised wire payload at the receiving end.
    let body = vec![0u8; usize::try_from(mgr.local().block_size_bytes()).unwrap_or(0)];
    let payload = encode(BlockTier::ThinkComplete, &body);
    let new_id = mgr
        .ingest_block(payload, BlockTier::ThinkComplete)
        .expect("ingest");
    assert!(matches!(mgr.block_location(new_id), BlockLocation::Local));
}

#[test]
fn offload_reclaims_the_local_slot() {
    let mut mgr = fresh_manager();
    let ids = mgr
        .allocate(1, BlockTier::ThinkComplete, 4)
        .expect("allocate four blocks");
    let used_before = mgr.used_bytes();
    let block_bytes = u64::from(mgr.local().block_size_bytes());

    let _ = mgr.offload_block(ids[0]).expect("offload");

    // The offloaded block's bytes return to the local pool (D30).
    assert_eq!(mgr.used_bytes(), used_before - block_bytes);
    assert!(!mgr.local().is_resident(ids[0]));
    assert!(matches!(
        mgr.block_location(ids[0]),
        BlockLocation::Remote { fabric: "nixl-synth" },
    ));
}

#[test]
fn reused_slot_reports_local_again() {
    let mut mgr = fresh_manager();
    let ids = mgr
        .allocate(1, BlockTier::ThinkComplete, 1)
        .expect("allocate one block");
    let offloaded = ids[0];
    let _ = mgr.offload_block(offloaded).expect("offload");
    assert!(matches!(
        mgr.block_location(offloaded),
        BlockLocation::Remote { .. },
    ));

    // The freed slot id is recycled by the next allocation; the same id is now
    // a live local block and must report Local, not the stale Remote handle.
    let reused = mgr
        .allocate(2, BlockTier::OutputCritical, 1)
        .expect("reuse slot");
    assert_eq!(reused, vec![offloaded]);
    assert!(matches!(mgr.block_location(offloaded), BlockLocation::Local));
}

#[test]
fn ingest_rejects_corrupted_payload() {
    let mut mgr = fresh_manager();
    let body = b"corrupt this".to_vec();
    let mut payload = encode(BlockTier::ThinkActive, &body);
    payload[WireHeader::SIZE + 1] ^= 0xff;
    let err = mgr
        .ingest_block(payload, BlockTier::ThinkActive)
        .expect_err("must reject corrupt body");
    assert!(format!("{err}").contains("checksum"));
}

#[test]
fn synthetic_fabric_is_addressable_after_push() {
    let f = SyntheticNixlFabric::new();
    let payload = encode(BlockTier::ThinkComplete, b"alpha");
    let h = f.push(payload.clone()).expect("push");
    let back = f.pull(h).expect("pull");
    assert_eq!(back, payload);
    assert_eq!(f.label(), "nixl-synth");
    assert_eq!(f.len(), 1);
    assert!(!f.is_empty());
}

#[test]
fn mooncake_adapter_round_trips_the_meridian_payload() {
    let fabric = MooncakeAdapter::new(Arc::new(SyntheticNixlFabric::new()));
    assert_eq!(fabric.label(), "mooncake-synth");

    let payload = encode(BlockTier::ThinkComplete, b"reasoning-kv");
    let handle = fabric.push(payload.clone()).expect("push");
    let back = fabric.pull(handle).expect("pull");

    // The Mooncake envelope is stripped on pull: the inner Meridian payload
    // is byte-identical and still decodes.
    assert_eq!(back, payload);
    let (tier, body) = decode(&back).expect("decode after mooncake round trip");
    assert_eq!(tier, BlockTier::ThinkComplete);
    assert_eq!(body, b"reasoning-kv");
}

#[test]
fn block_manager_offloads_over_mooncake_fabric() {
    let mut mgr =
        NixlBackedBlockManager::new(fresh_local(), Arc::new(MooncakeAdapter::new(Arc::new(
            SyntheticNixlFabric::new(),
        ))));
    let ids = mgr
        .allocate(1, BlockTier::ThinkComplete, 2)
        .expect("allocate");

    let _ = mgr.offload_block(ids[0]).expect("offload over mooncake");
    assert!(matches!(
        mgr.block_location(ids[0]),
        BlockLocation::Remote {
            fabric: "mooncake-synth"
        },
    ));
    assert!(!mgr.local().is_resident(ids[0]));
}

#[test]
fn round_trip_preserves_tier_metadata() {
    for tier in BlockTier::eviction_order() {
        let body = vec![42u8; 64];
        let payload = encode(tier, &body);
        let (decoded_tier, decoded_body) = decode(&payload).expect("decode");
        assert_eq!(decoded_tier, tier, "tier survives round trip");
        assert_eq!(decoded_body, body);
    }
}
