# Architectural Decision Records

Meridian uses [Michael Nygard's ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
to capture the *why* behind significant choices. ADRs are immutable once
"Accepted"; we supersede them rather than edit them.

Lifecycle: `Proposed` → `Accepted` → (later) `Superseded by ADR-NNNN` /
`Deprecated`.

## Index

| ID   | Status     | Title                                              |
|------|------------|----------------------------------------------------|
| 0001 | Accepted   | [Dual-queue vs. priority weights](0001-dual-queue-rationale.md) |
| 0002 | Accepted   | [Workspace tri-crate layout](0002-workspace-tri-crate.md)        |
| 0003 | Accepted   | [DashMap for per-request state](0003-dashmap-rationale.md)       |
| 0004 | Accepted   | [KV tier promotion policy](0004-kv-tier-promotion-policy.md)     |
| 0005 | Accepted   | [Benchmark methodology](0005-benchmark-methodology.md)           |
| 0006 | Accepted   | [Disagg KV transfer protocol](0006-disagg-kv-transfer.md)        |
| 0007 | Accepted   | [Release and versioning policy](0007-release-versioning-policy.md) |

## Writing a new ADR

1. Copy [`template.md`](template.md) to `NNNN-short-kebab-title.md`.
2. Open as `Proposed`; merge as `Accepted` after PR review.
3. If a later ADR supersedes this one, mark this one `Superseded by ADR-NNNN`
   in a new commit — never edit the body of an accepted ADR.
