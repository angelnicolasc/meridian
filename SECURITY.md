# Security Policy

## Supported versions

Meridian is in pre-1.0 development. Only the latest tagged release receives
security fixes. After 1.0, the latest minor of the latest two majors will be
supported.

## Reporting a vulnerability

**Do not file a public issue for security problems.** Use GitHub's
[Private Vulnerability Reporting](https://github.com/angelnicolasc/meridian/security/advisories/new)
or email **nick.dicerutti@gmail.com** with the subject line
`[meridian-security]`.

You should expect:

| Stage                | Target time |
|----------------------|-------------|
| Acknowledgement      | 48 hours    |
| Initial assessment   | 5 business days |
| Fix or mitigation    | 30 days (critical), 90 days (high), best-effort otherwise |
| Public disclosure    | Coordinated, default 90 days after fix is available |

We credit reporters in the release notes unless anonymity is requested.

## Scope

In scope for security reporting:

- **Memory safety** in `meridian-core` and `meridian-kernels` FFI boundary —
  any reachable UB, out-of-bounds access, double-free, use-after-free.
- **CUDA kernel safety** — buffer overruns, races on shared memory, illegal
  memory access reachable from sane inputs.
- **Deserialization** of `meridian.toml`, model configs, request payloads —
  panics on adversarial input, type confusion.
- **Denial of service** — pathological requests that crash the scheduler or
  exhaust KV memory irrecoverably.
- **Supply chain** — compromised crate/wheel that ships under the Meridian name.

Out of scope:

- Vulnerabilities in upstream dependencies (file with the upstream project; we
  will track and bump promptly).
- Misconfigurations of operator-controlled deployments (e.g. exposing the
  Prometheus endpoint publicly).
- Reasoning quality degradation when budget forcing is misconfigured — this is
  a correctness concern, not a security one.

## Hardening notes

- `meridian-core` denies `unsafe_op_in_unsafe_fn` workspace-wide.
- The CUDA FFI boundary in `meridian-kernels` is the only `unsafe` surface and
  is reviewed for every change.
- Releases are signed via Sigstore cosign; artifact provenance is generated
  via SLSA Level 2.
