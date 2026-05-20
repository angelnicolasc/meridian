# Security & Supply Chain

## Disclosure policy

Security vulnerabilities should be reported privately. See
[SECURITY.md](../../SECURITY.md) for the full disclosure process, severity
classification, and response SLAs.

**Summary**: critical severity (CVSS ≥ 7.0) issues receive a patch within 48
hours of confirmation. Do not open public GitHub issues for unpatched
vulnerabilities.

## Provenance attestation

Every tagged release (`v*`) generates a
[SLSA Level 2](https://slsa.dev/spec/v1.0/) provenance attestation via the
`slsa-github-generator` reusable workflow. The attestation covers:

- The `meridian-core` and `meridian-kernels` static libraries.
- The `meridian` Python wheel.

The attestation is uploaded as a GitHub release asset alongside the release
artefacts. To verify:

```bash
# Install the SLSA verifier
go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@latest

# Download the artefact and its provenance from the GitHub release.
# Then verify:
slsa-verifier verify-artifact meridian-*.whl \
    --provenance-path meridian.intoto.jsonl \
    --source-uri github.com/angelnicolasc/meridian \
    --source-tag v0.1.0
```

**What is attested**: the build provenance — that the artefact was built from
the tagged source in the GitHub Actions environment. SLSA L2 does not attest
to the security of the code itself.

## Software Bill of Materials (SBOM)

Each release includes a CycloneDX SBOM covering:

- The Rust workspace (all transitive crate dependencies).
- The Python wheel (Python package dependencies from `pyproject.toml`).

The SBOM is attached as a `.cdx.json` asset on the GitHub release. Operators
can use it with vulnerability scanning tools (Grype, Trivy, FOSSA).

## Supply-chain controls

| Control | Mechanism |
|---------|-----------|
| Dependency pinning | `Cargo.lock` and `uv.lock` committed and verified in CI |
| Dependency auditing | `cargo deny check` in the `supply-chain` CI job (licence + advisory check) |
| GitHub Actions pinning | Actions pinned to major version tags in all workflows |
| Self-hosted runner isolation | GPU runner gated to `github.repository_owner == 'angelnicolasc'` |
| DCO sign-off | All commits require `Signed-off-by` matching the commit author |
| Release provenance | SLSA L2 via `slsa-github-generator` |

## Dependency policy

New dependencies require:

1. A licence compatible with Apache-2.0 (verified by `cargo deny`).
2. No known CVEs at the time of merge (verified by `cargo deny` advisories check).
3. An entry in the SBOM at the next release.

## CI workflow permissions

All CI workflows run with minimal permissions:

| Workflow | Permissions |
|----------|-------------|
| `ci.yml` | `contents: read` |
| `release.yml` | `contents: write`, `id-token: write` (for SLSA) |
| `cuda.yml` | `contents: read` |

The `id-token: write` permission in `release.yml` is required by
`slsa-github-generator` to mint the OIDC-backed provenance token. It is
scoped only to the `build-artifacts` and `provenance` jobs.
