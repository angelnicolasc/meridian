# Glossary

Terms used throughout the Meridian documentation and codebase.

---

**TTFT** — *Time to First Token*. Wall-clock time from the moment a request
is submitted to the moment the first token is returned to the client. Dominated
by prefill latency for long prompts.

**TTOT** — *Time to Output Token*. Wall-clock time from the emission of the
`</think>` boundary token to the first user-visible output token. The metric
Meridian is specifically designed to protect.

**TPOT** — *Time Per Output Token* (also ITL: *Inter-Token Latency*). Time
between consecutive output tokens during streaming. The user-perceived "speed"
of the stream.

**ITL** — *Inter-Token Latency*. See TPOT.

**EAT** — *Entropy-Aware Termination*. A budget-forcing signal based on the
variance of the EMA of per-token Shannon entropy over the reasoning chain.
When EAT EMA variance drops below a threshold, the model is inferred to have
converged on an answer. Defined in arXiv:2509.26522.

**RPDI** — *Reasoning Phase Divergence Index*. A signal based on the ratio of
local transition-token frequency to global transition-token frequency. A high
ratio indicates the model is cycling through redundant reasoning steps
("overthinking"). Defined in arXiv:2603.14251.

**Entropy (Shannon)** — Measure of uncertainty in the next-token distribution.
Computed from the logit vector after softmax as `-Σ p_i · log(p_i)`, in nats.
High entropy = uncertain prediction; low entropy = confident prediction.

**EMA** — *Exponential Moving Average*. A smoothed average where recent values
are weighted more heavily. Controlled by `ema_alpha` (smaller = longer memory).

**KV** — *Key-Value cache*. The GPU memory store holding the attention keys and
values for each token in active requests. KV memory is the primary capacity
constraint in serving systems.

**KV block** — The unit of KV cache allocation. A fixed-size chunk (default
16 KiB) covering a fixed number of tokens (default 16). Blocks are allocated
at the request level and freed on eviction or request completion.

**ThinkComplete** — Block tier for KV blocks that belonged to a request's
reasoning phase after `</think>` has been emitted. Lowest eviction priority —
these blocks are freed first under memory pressure.

**ThinkActive** — Block tier for KV blocks belonging to a request currently
in the reasoning phase.

**OutputCritical** — Block tier for KV blocks belonging to a request in the
output phase. Highest eviction priority — evicting these causes user-visible
stream disruption. Any eviction at this tier fires `meridian.output_critical_eviction`.

**Disagg / disaggregated serving** — Prefill-decode disaggregation: the model
prefill (prompt processing) and decode (token generation) steps are executed
on separate hardware. Disagg reduces head-of-line blocking by separating the
two workloads, which have very different GPU utilisation profiles.

**NIXL** — *NVIDIA Inference eXchange Layer*. NVIDIA's reference fabric for
transferring KV blocks between prefill and decode nodes in a disaggregated
serving topology.

**Mooncake** — An open-source disaggregated serving framework with a
KV-transfer protocol. Meridian's disagg surface is documented as
Mooncake-compatible in [ADR-0006](adr/0006-disagg-kv-transfer.md).

**vLLM** — An open-source LLM serving framework. Meridian's primary integration
target. Meridian wraps vLLM's scheduler without forking the codebase.

**DashMap** — A concurrent hash map crate used for the `PhaseRouter`'s
per-request state. Provides O(1) read and write with sharded locking.
See [ADR-0003](adr/0003-dashmap-rationale.md).

**EOS** — *End of Sequence*. The special token that signals request completion.
vLLM emits an EOS event that the Meridian plugin uses to trigger request
teardown and router state reaping.

**Conventional Commits** — A commit message standard used throughout this
repository. Format: `<type>(<scope>): <summary>`. See
[conventionalcommits.org](https://www.conventionalcommits.org/en/v1.0.0/).

**DCO** — *Developer Certificate of Origin*. A sign-off mechanism (`git commit -s`)
that certifies the contributor has the right to submit the code under the
project's license. Required for all contributions — see [CONTRIBUTING.md](../../CONTRIBUTING.md).

**SLSA** — *Supply-chain Levels for Software Artifacts*. A framework for
supply-chain security. Meridian attests Level 2 provenance on every tagged
release via `slsa-github-generator`. See [ADR-0007](adr/0007-release-versioning-policy.md).

**SBOM** — *Software Bill of Materials*. A machine-readable inventory of
software components and their licenses. Meridian generates a CycloneDX SBOM
for each release, attached as a GitHub release asset.
