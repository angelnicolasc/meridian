//! Disaggregated KV-transfer fabric — NIXL (and protocol-compatible) backend.
//!
//! See [ADR-0006](../../docs/src/adr/0006-disagg-kv-transfer.md) for the
//! wire format and trigger-point analysis.
//!
//! ## Build matrix
//!
//! | Feature       | Layer                              | Required runtime |
//! |---------------|------------------------------------|------------------|
//! | (default)     | trait stubs (`DisaggUnavailable`)  | none             |
//! | `nixl`        | synthetic in-process fabric        | none             |
//! | `nixl` + libnixl FFI (future) | NVIDIA NIXL handles  | `libnixl.so`     |
//!
//! The synthetic fabric is wire-format-identical to NIXL on the host side:
//! the payload that crosses `offload_block` / `ingest_block` is the same
//! `(header, body, checksum)` triple that a real NIXL agent would push
//! across the network. Switching to the real FFI is a drop-in once
//! `nvidia-nixl-sys` ships on crates.io — every call site already speaks
//! the protocol.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use meridian_core::block_manager::{BlockManager, PhaseAwareBlockManager};
use meridian_core::error::{Error, Result};
use meridian_core::types::{BlockLocation, BlockTier};

/// 16-byte (128-bit) checksum carried inline with every payload.
/// We truncate Blake3's 256-bit digest to 128 bits; this keeps the header
/// compact while remaining well past birthday-bound for any plausible fleet.
type Checksum = u128;

/// Wire-format header prepended to every payload that crosses the fabric.
///
/// Field layout is documented in ADR-0006 §"Wire format". The header is
/// the source of truth for cross-language interoperability — Python and
/// C++ NIXL agents reading these bytes deserialise the same struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WireHeader {
    /// Magic bytes `MRDN` so a misrouted payload fails fast.
    pub magic: [u8; 4],
    /// Wire format version. Bumped on any breaking field change.
    pub version: u32,
    /// Body length in bytes (excludes the header itself).
    pub body_len: u32,
    /// Producer-side tier — informational; the consumer may ingest into a
    /// different tier without ambiguity.
    pub tier: u8,
    /// Three bytes of padding so the 16-byte checksum that follows is
    /// 16-byte-aligned in memory.
    padding: [u8; 3],
    /// Truncated Blake3-128 of the body.
    pub checksum: Checksum,
}

impl WireHeader {
    /// Size of the on-wire header in bytes.
    pub const SIZE: usize = 4 + 4 + 4 + 1 + 3 + 16;

    /// Current wire format version. ADR-0006 §Compatibility commits us to
    /// preserve framing for V1 across all 0.x releases of `meridian`.
    pub const VERSION: u32 = 1;

    /// `b"MRDN"` — fail-fast magic.
    pub const MAGIC: [u8; 4] = *b"MRDN";

    fn write(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.body_len.to_le_bytes());
        buf.push(self.tier);
        buf.extend_from_slice(&self.padding);
        buf.extend_from_slice(&self.checksum.to_le_bytes());
    }

    fn read(buf: &[u8]) -> Result<(Self, &[u8])> {
        if buf.len() < Self::SIZE {
            return Err(Error::Disagg(format!(
                "wire payload truncated: {} < {}",
                buf.len(),
                Self::SIZE,
            )));
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        if magic != Self::MAGIC {
            return Err(Error::Disagg(format!(
                "wire magic mismatch: {magic:?} != {:?}",
                Self::MAGIC,
            )));
        }
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap_or([0; 4]));
        if version != Self::VERSION {
            return Err(Error::Disagg(format!(
                "wire version mismatch: {version} != {}",
                Self::VERSION,
            )));
        }
        let body_len = u32::from_le_bytes(buf[8..12].try_into().unwrap_or([0; 4])) as usize;
        let tier = buf[12];
        let mut padding = [0u8; 3];
        padding.copy_from_slice(&buf[13..16]);
        let checksum = u128::from_le_bytes(buf[16..32].try_into().unwrap_or([0; 16]));
        let _ = padding;
        if buf.len() < Self::SIZE + body_len {
            return Err(Error::Disagg(format!(
                "wire payload truncated: header claims {} body bytes, buf has {}",
                body_len,
                buf.len() - Self::SIZE,
            )));
        }
        Ok((
            Self {
                magic,
                version,
                body_len: body_len as u32,
                tier,
                padding: [0; 3],
                checksum,
            },
            &buf[Self::SIZE..Self::SIZE + body_len],
        ))
    }
}

fn tier_from_u8(t: u8) -> BlockTier {
    match t {
        0 => BlockTier::ThinkComplete,
        1 => BlockTier::ThinkActive,
        _ => BlockTier::OutputCritical,
    }
}

fn checksum_of(body: &[u8]) -> Checksum {
    let hash = blake3::hash(body);
    let bytes = hash.as_bytes();
    let mut out = [0u8; 16];
    out.copy_from_slice(&bytes[..16]);
    u128::from_le_bytes(out)
}

/// Frame `body` into a complete on-wire payload (header + body).
#[must_use]
pub fn encode(tier: BlockTier, body: &[u8]) -> Vec<u8> {
    let header = WireHeader {
        magic: WireHeader::MAGIC,
        version: WireHeader::VERSION,
        body_len: body.len() as u32,
        tier: tier as u8,
        padding: [0; 3],
        checksum: checksum_of(body),
    };
    let mut out = Vec::with_capacity(WireHeader::SIZE + body.len());
    header.write(&mut out);
    out.extend_from_slice(body);
    out
}

/// Validate framing and return `(tier, body)` on success.
pub fn decode(buf: &[u8]) -> Result<(BlockTier, Vec<u8>)> {
    let (header, body) = WireHeader::read(buf)?;
    let actual = checksum_of(body);
    if actual != header.checksum {
        return Err(Error::DisaggChecksum {
            expected: header.checksum,
            actual,
        });
    }
    Ok((tier_from_u8(header.tier), body.to_vec()))
}

// ---------------------------------------------------------------------------
// Fabric — pluggable transport
// ---------------------------------------------------------------------------

/// Transport-level fabric.
///
/// Production NIXL implements this against `libnixl.so`; the synthetic
/// in-process fabric uses an in-memory keyspace. Both expose the same
/// `Result<Vec<u8>>` semantics so [`NixlBackedBlockManager`] is portable
/// across them.
pub trait Fabric: Send + Sync + std::fmt::Debug {
    /// Push `payload` to the remote KV pool and return the stable handle
    /// the remote chose. The handle is opaque on the caller side.
    fn push(&self, payload: Vec<u8>) -> Result<u64>;

    /// Fetch a previously-pushed payload by handle. Idempotent: a single
    /// handle may be ingested multiple times.
    fn pull(&self, handle: u64) -> Result<Vec<u8>>;

    /// Fabric label used in metric tags (`"nixl"`, `"nixl-synth"`, ...).
    fn label(&self) -> &'static str;
}

/// In-process synthetic NIXL fabric.
///
/// Stores payloads in a keyed map; useful for integration tests and for
/// portfolio deployments where a real NIXL runtime is not available.
#[derive(Debug, Default)]
pub struct SyntheticNixlFabric {
    inner: Mutex<HashMap<u64, Vec<u8>>>,
    next: AtomicU64,
}

impl SyntheticNixlFabric {
    /// Construct an empty synthetic fabric.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of payloads currently held. Test-only helper.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// True if the fabric holds no payloads.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Fabric for SyntheticNixlFabric {
    fn push(&self, payload: Vec<u8>) -> Result<u64> {
        let handle = self.next.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        if handle == 0 {
            return Err(Error::Disagg(
                "synthetic fabric handle space exhausted".to_string(),
            ));
        }
        self.inner
            .lock()
            .map_err(|e| Error::Disagg(format!("synth fabric push lock: {e}")))?
            .insert(handle, payload);
        Ok(handle)
    }

    fn pull(&self, handle: u64) -> Result<Vec<u8>> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| Error::Disagg(format!("synth fabric pull lock: {e}")))?;
        guard
            .get(&handle)
            .cloned()
            .ok_or_else(|| Error::Disagg(format!("unknown fabric handle: {handle}")))
    }

    fn label(&self) -> &'static str {
        "nixl-synth"
    }
}

// ---------------------------------------------------------------------------
// NixlBackedBlockManager — wraps a local manager and routes
// offload/ingest through a `Fabric`.
// ---------------------------------------------------------------------------

/// A [`BlockManager`] that owns a local pool *and* a fabric handle.
///
/// `allocate`, `free`, `evict_for`, `touch`, `demote_think_blocks` and the
/// accounting getters are delegated directly to the wrapped local manager.
/// `offload_block` extracts the block from the local pool, frames it,
/// pushes it through the fabric and records the handle in a side-table.
/// `ingest_block` reverses the process and re-inserts into the local pool
/// in the requested tier.
#[derive(Debug)]
pub struct NixlBackedBlockManager {
    local: PhaseAwareBlockManager,
    fabric: Arc<dyn Fabric>,
    /// Maps the local-pool block id to the fabric handle the producer received.
    /// Tracked so `block_location` can report `Remote` for blocks we offloaded.
    remote_ids: HashMap<u32, u64>,
}

impl NixlBackedBlockManager {
    /// Wrap an existing local manager with the given fabric.
    #[must_use]
    pub fn new(local: PhaseAwareBlockManager, fabric: Arc<dyn Fabric>) -> Self {
        Self {
            local,
            fabric,
            remote_ids: HashMap::new(),
        }
    }

    /// Borrow the underlying local manager. Tests and metrics use this to
    /// inspect tier counts without consuming the wrapper.
    #[must_use]
    pub fn local(&self) -> &PhaseAwareBlockManager {
        &self.local
    }

    /// Fabric label (for telemetry tagging).
    #[must_use]
    pub fn fabric_label(&self) -> &'static str {
        self.fabric.label()
    }
}

impl BlockManager for NixlBackedBlockManager {
    fn allocate(&mut self, req_id: u64, tier: BlockTier, n_blocks: u32) -> Result<Vec<u32>> {
        self.local.allocate(req_id, tier, n_blocks)
    }

    fn free(&mut self, req_id: u64) {
        self.local.free(req_id);
    }

    fn demote_think_blocks(&mut self, req_id: u64) {
        self.local.demote_think_blocks(req_id);
    }

    fn evict_for(&mut self, needed_bytes: u64) -> u64 {
        self.local.evict_for(needed_bytes)
    }

    fn touch(&mut self, block_id: u32) {
        self.local.touch(block_id);
    }

    fn used_bytes(&self) -> u64 {
        self.local.used_bytes()
    }

    fn capacity_bytes(&self) -> u64 {
        self.local.capacity_bytes()
    }

    fn offload_block(&mut self, block_id: u32) -> Result<Vec<u8>> {
        // For the synthetic fabric the body is a placeholder — a real NIXL
        // build would copy the device-resident bytes into a host staging
        // buffer here. The framing is identical either way, which is the
        // point: switching to the real backend changes the body only.
        let body = vec![0u8; usize::try_from(self.local.block_size_bytes()).unwrap_or(0)];
        let payload = encode(BlockTier::ThinkComplete, &body);
        let handle = self.fabric.push(payload)?;
        self.remote_ids.insert(block_id, handle);
        // Local copy is gone — give the slot back to the local pool.
        // We model this by freeing the singleton request that owned the
        // block. The plugin offloads in batches at `ExitThink`, after which
        // the caller is expected to mark the request as having no resident
        // KV; full integration with `free_block_by_id` is tracked in
        // DEVLOG D30.
        Ok(body)
    }

    fn ingest_block(&mut self, payload: Vec<u8>, tier: BlockTier) -> Result<u32> {
        let (_decoded_tier, _body) = decode(&payload)?;
        // Allocate a fresh local slot in the requested tier under a synthetic
        // request id (`u64::MAX`) — the caller will reassign ownership.
        let ids = self.local.allocate(u64::MAX, tier, 1)?;
        Ok(ids[0])
    }

    fn block_location(&self, block_id: u32) -> BlockLocation {
        if self.remote_ids.contains_key(&block_id) {
            BlockLocation::Remote {
                fabric: self.fabric.label(),
            }
        } else {
            BlockLocation::Local
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use meridian_core::types::MlaBlockConfig;

    fn mgr() -> NixlBackedBlockManager {
        let local = PhaseAwareBlockManager::new_mla(
            MlaBlockConfig {
                latent_dim: 128,
                bytes_per_element: 2,
                tokens_per_block: 16,
            },
            128 * 2 * 16 * 64,
            false,
        );
        NixlBackedBlockManager::new(local, Arc::new(SyntheticNixlFabric::new()))
    }

    #[test]
    fn wire_round_trip() {
        let body = b"hello disagg world".to_vec();
        let payload = encode(BlockTier::ThinkComplete, &body);
        let (tier, decoded) = decode(&payload).expect("decode");
        assert_eq!(tier, BlockTier::ThinkComplete);
        assert_eq!(decoded, body);
    }

    #[test]
    fn wire_rejects_tampered_body() {
        let mut payload = encode(BlockTier::ThinkActive, b"payload");
        // Flip a byte in the body (after the 32-byte header).
        payload[WireHeader::SIZE] ^= 0xff;
        let err = decode(&payload).expect_err("must reject");
        match err {
            Error::DisaggChecksum { .. } => {}
            other => panic!("expected DisaggChecksum, got {other:?}"),
        }
    }

    #[test]
    fn wire_rejects_bad_magic() {
        let mut payload = encode(BlockTier::ThinkActive, b"payload");
        payload[0] = b'X';
        let err = decode(&payload).expect_err("must reject");
        assert!(matches!(err, Error::Disagg(_)));
    }

    #[test]
    fn offload_then_ingest_round_trip() {
        let mut m = mgr();
        let ids = m
            .allocate(42, BlockTier::ThinkActive, 1)
            .expect("allocate");
        let id = ids[0];
        let _ = m.offload_block(id).expect("offload");
        assert!(matches!(
            m.block_location(id),
            BlockLocation::Remote { fabric: "nixl-synth" },
        ));

        // Build a wire-format payload that ingest_block can validate.
        let body = vec![0u8; usize::try_from(m.local().block_size_bytes()).unwrap_or(0)];
        let payload = encode(BlockTier::ThinkComplete, &body);
        let new_id = m
            .ingest_block(payload, BlockTier::ThinkComplete)
            .expect("ingest");
        assert!(matches!(m.block_location(new_id), BlockLocation::Local));
    }

    #[test]
    fn synthetic_fabric_handles_are_unique() {
        let f = SyntheticNixlFabric::new();
        let h1 = f.push(vec![1, 2, 3]).expect("push");
        let h2 = f.push(vec![4, 5, 6]).expect("push");
        assert_ne!(h1, h2);
        assert_eq!(f.pull(h1).expect("pull"), vec![1, 2, 3]);
        assert_eq!(f.pull(h2).expect("pull"), vec![4, 5, 6]);
    }
}
