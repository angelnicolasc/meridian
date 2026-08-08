//! Typed error surface for `meridian-core`.
//!
//! Public APIs never panic on caller-controlled inputs; they return [`Result<T>`].
//! Internal invariants (e.g. a phase router being asked about a request it never
//! saw) use `debug_assert!` and degrade gracefully in release builds.

use std::io;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Error variants returned by `meridian-core`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The configuration file (TOML) could not be parsed.
    #[error("invalid config: {0}")]
    Config(String),

    /// A configuration value is structurally valid TOML but semantically invalid
    /// (e.g. `min_think_tokens > max_think_tokens`).
    #[error("config validation failed: {field}: {reason}")]
    ConfigValidation {
        /// Dotted field path inside the config tree, e.g. `scheduler.think_tpot_budget_ms`.
        field: &'static str,
        /// Human-readable explanation of what is wrong.
        reason: String,
    },

    /// A request identifier was used after its state was reaped.
    #[error("unknown request id: {0}")]
    UnknownRequest(u64),

    /// A KV block id was used after it was freed.
    #[error("unknown kv block id: {0}")]
    UnknownBlock(u32),

    /// Insufficient KV memory to satisfy an allocation request, even after
    /// running the eviction policy through every tier.
    #[error("kv memory exhausted: needed {needed} bytes, freed {freed}")]
    KvMemoryExhausted {
        /// Bytes the caller requested.
        needed: u64,
        /// Bytes the eviction policy was able to free.
        freed: u64,
    },

    /// Filesystem or I/O failure reading a config or model definition.
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    /// TOML deserialisation failure.
    #[error("toml parse error: {0}")]
    Toml(#[from] toml::de::Error),

    /// A disaggregated KV transfer (offload / ingest) operation failed.
    ///
    /// The default in-process [`crate::block_manager::PhaseAwareBlockManager`]
    /// does not implement a fabric and returns
    /// [`Error::DisaggUnavailable`] for any offload / ingest call. Fabric
    /// implementations such as the `nixl` feature return
    /// [`Error::Disagg`] with a transport-specific message.
    #[error("disagg fabric error: {0}")]
    Disagg(String),

    /// The active block manager has no disaggregated fabric configured.
    #[error("no disagg fabric configured (compile with --features nixl, or wire a fabric)")]
    DisaggUnavailable,

    /// A synthetic trace was used where a real measurement is required.
    ///
    /// Produced by [`crate::dspark_bridge::Provenance::into_measured`] and by
    /// every path that promotes a statistic to a publishable claim. See
    /// [`crate::dspark_bridge::provenance`] for why this is a hard error
    /// rather than a warning.
    #[error(
        "refusing to treat synthetic trace {fixture:?} as a measurement \
         (see dspark_bridge::provenance)"
    )]
    SyntheticProvenance {
        /// Name of the fixture that produced the synthetic trace.
        fixture: String,
    },

    /// Two acceptance ledgers with different provenance were merged.
    #[error("cannot merge ledgers of differing provenance: {left} into {right}")]
    ProvenanceMismatch {
        /// Provenance label of the ledger being merged into.
        left: &'static str,
        /// Provenance label of the ledger being merged from.
        right: &'static str,
    },

    /// Two acceptance ledgers with different boundary-straddle policies were
    /// merged. Their phase arms were built under incompatible attribution
    /// rules, so the combined statistic would be meaningless.
    #[error("cannot merge ledgers with differing straddle policies: {left} into {right}")]
    StraddlePolicyMismatch {
        /// Straddle policy of the ledger being merged into.
        left: &'static str,
        /// Straddle policy of the ledger being merged from.
        right: &'static str,
    },

    /// A phase-segmented statistic was requested for a phase with too few
    /// observations for it to be defined.
    #[error("insufficient observations to compute the statistic for phase {phase}")]
    InsufficientObservations {
        /// Which phase arm was short of data.
        phase: &'static str,
    },

    /// A checksum mismatch was detected when ingesting a block from a fabric.
    #[error(
        "block payload checksum mismatch on ingest (expected {expected:032x}, got {actual:032x})"
    )]
    DisaggChecksum {
        /// 128-bit checksum recorded by the producer (truncated Blake3).
        expected: u128,
        /// 128-bit checksum computed locally on ingest.
        actual: u128,
    },
}
