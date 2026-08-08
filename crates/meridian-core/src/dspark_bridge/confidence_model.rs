//! Formal model of DSpark's confidence head and Meridian's EAT/RPDI signal,
//! expressed in one shared vocabulary so the two can be compared as
//! mathematics rather than as analogy.
//!
//! This module is the executable form of work items **MER-P0.1**, **MER-P0.2**
//! and **MER-P0.3**. The prose version, with citations, is
//! [`docs/src/notes/phase-conditioned-speculation.md`][note].
//!
//! # The shared object
//!
//! At decode step `k` a target model induces a predictive distribution `pᵗ`
//! over the vocabulary. A speculative drafter induces its own `pᵈ`. Everything
//! below is a functional of one or both.
//!
//! | Quantity | Reads | Defined by |
//! |---|---|---|
//! | DSpark confidence `c_k` | `pᵈ`, `pᵗ` (via a learned head on draft hidden state) | DSpark Eq. 7 |
//! | DSpark supervision target `c*_k` | `pᵈ`, `pᵗ` | DSpark Eq. 8 |
//! | Meridian EAT `e_k` | `pᵗ` only | [`crate::types::EntropySignal::eat`] |
//! | Meridian token entropy `H_k` | `pᵗ` only | [`crate::types::EntropySignal::token_entropy`] |
//!
//! # MER-P0.1 — what DSpark's confidence head computes
//!
//! DSpark attaches a confidence head to the draft backbone. For drafted
//! position `k` it emits
//!
//! ```text
//! c_k = σ( wᵀ [ h_k ; W₁[x_{k-1}] ] )            (DSpark Eq. 7)
//! ```
//!
//! where `h_k` is the backbone hidden state and `W₁[x_{k-1}]` the Markov
//! embedding of the previously drafted token. `c_k ∈ (0,1)` models the
//! probability that the draft token at `k` survives verification *given every
//! preceding draft token was accepted*, so prefix survival is the cumulative
//! product `S_k = ∏_{i ≤ k} c_i`.
//!
//! The head is trained by position-weighted binary cross-entropy (Eq. 11)
//! against an **analytical** target rather than a sampled accept/reject
//! outcome:
//!
//! ```text
//! c*_k = 1 - ½ ‖ pᵈ_k - pᵗ_k ‖₁                  (DSpark Eq. 8)
//! ```
//!
//! `½‖·‖₁` is total variation distance, so `c* = 1 - TV(pᵈ, pᵗ)` — exactly the
//! acceptance probability of one step of speculative sampling under the
//! standard accept/reject rule. Scores are calibrated post hoc by sequential
//! temperature scaling, minimising expected calibration error of the
//! *cumulative product*, position by position.
//!
//! Downstream, DSpark's prefix scheduler picks a verification length per
//! request by maximising system throughput `Θ = τ · SPS(B)` — accepted tokens
//! per cycle times steps-per-second at the resulting batch size.
//!
//! # MER-P0.2 — what Meridian's EAT/RPDI signal computes
//!
//! Meridian's probe reads the same `pᵗ` and emits [`EntropySignal`]:
//!
//! ```text
//! e_k = Σ_{v ∈ E} pᵗ_k(v)         EAT: mass on the think-end token set E
//! H_k = -Σ_v pᵗ_k(v) ln pᵗ_k(v)   Shannon entropy, nats
//! ```
//!
//! smoothed into `μ_k = α e_k + (1-α) μ_{k-1}` with a companion second-moment
//! EMA so `Var[e] ≈ ν_k - μ_k²` is available without a sample window. RPDI
//! compares a short-window EMA of the transition indicator `1[H_k > θ]`
//! against its running global mean. Both feed one decision:
//! [`BudgetForceReason`](crate::types::BudgetForceReason).
//!
//! # MER-P0.3 — where the two signals actually meet
//!
//! The comparison is not a metaphor. Five results, each with a test in this
//! module.
//!
//! **Proposition 1 (point-mass identity).** For a deterministic drafter
//! `pᵈ = δ_x̂`, `c* = pᵗ(x̂)`.
//! *Proof.* `‖δ_x̂ - pᵗ‖₁ = (1 - pᵗ(x̂)) + Σ_{v≠x̂} pᵗ(v) = 2(1 - pᵗ(x̂))`;
//! halve and subtract from one. ∎
//!
//! **Corollary 2 (EAT *is* an acceptance rate).** Meridian's `e_k` is the
//! DSpark supervision target `c*_k` for the drafter that always proposes the
//! think-end token. The two projects were computing the same functional for
//! different reasons.
//!
//! **Proposition 3 (EAT ceilings any boundary-supported drafter).** If
//! `supp(pᵈ) ⊆ E` then `c* ≤ e`, with equality iff `pᵈ(v) ≥ pᵗ(v)` for all
//! `v ∈ E`.
//! *Proof.* `TV(pᵈ,pᵗ) = sup_A |pᵈ(A) - pᵗ(A)| ≥ pᵈ(E) - pᵗ(E) = 1 - e`. ∎
//!
//! **Proposition 4 (entropy brackets the target mode).** Write
//! `M = max_v pᵗ(v)` and let `V` be the vocabulary size. Then
//!
//! ```text
//! e^{-H}  ≤  M  ≤  p*(H, V)
//! ```
//!
//! where `p*` is the unique root on `[1/V, 1]` of
//! `H = H_b(p) + (1-p)·ln(V-1)`.
//! *Proof.* Lower: `H = -Σ p ln p ≥ -ln M`. Upper: the right-hand side is the
//! maximum entropy attainable by any distribution over `V` symbols whose
//! largest mass is `p` (a spike over a uniform tail), and it is strictly
//! decreasing in `p`; feasibility of the observed `H` therefore forces
//! `M ≤ p*`. ∎
//!
//! **Corollary 5 (the actionable one).** Every deterministic drafter satisfies
//! `c* = pᵗ(x̂) ≤ M ≤ p*(H, V)`. High target entropy at a step is a *proof*
//! that no deterministic drafter can be accepted often there — independent of
//! which drafter is deployed, and computable from a signal Meridian already
//! has. Note the direction: this bound can only ever justify drafting
//! **less**. It can never justify drafting more, which is exactly the safety
//! property [`super::hook`] relies on.
//!
//! # The limit of what Phase 0 can claim
//!
//! `c*` depends on **both** `pᵈ` and `pᵗ`. Every Meridian signal depends on
//! `pᵗ` alone. Meridian can therefore *bound* draft acceptance from quantities
//! it already computes, and can never *predict* it without running the
//! drafter. That is the precise reason this work ships as a bounds provider
//! plus a data-ready hook, and the precise reason the empirical question in
//! Section 5 of the blueprint needs a GPU.
//!
//! [note]: https://github.com/angelnicolasc/meridian/blob/main/docs/src/notes/phase-conditioned-speculation.md

use crate::types::EntropySignal;

// ---------------------------------------------------------------------------
// Distribution-level primitives
// ---------------------------------------------------------------------------

/// Total variation distance `½‖p - q‖₁` between two discrete distributions.
///
/// Inputs are used as given; no renormalisation is applied, because silently
/// repairing a malformed distribution would hide a caller bug. Slices of
/// unequal length are handled by treating the shorter as zero-padded, which is
/// the mathematically correct reading rather than a truncation.
#[must_use]
pub fn total_variation(p: &[f32], q: &[f32]) -> f32 {
    let common = p.len().min(q.len());
    let mut l1 = 0.0_f64;
    for i in 0..common {
        l1 += f64::from(p[i] - q[i]).abs();
    }
    for &x in &p[common..] {
        l1 += f64::from(x).abs();
    }
    for &x in &q[common..] {
        l1 += f64::from(x).abs();
    }
    (0.5 * l1) as f32
}

/// The DSpark analytical acceptance rate `c* = 1 - TV(pᵈ, pᵗ)` (Eq. 8).
///
/// This is the quantity DSpark's confidence head is trained to regress, and
/// the single-step acceptance probability of speculative sampling.
///
/// # Examples
///
/// ```
/// use meridian_core::dspark_bridge::analytical_acceptance;
///
/// // A drafter that matches the target exactly is always accepted.
/// let target = [0.7_f32, 0.2, 0.1];
/// assert!((analytical_acceptance(&target, &target) - 1.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn analytical_acceptance(draft: &[f32], target: &[f32]) -> f32 {
    (1.0 - total_variation(draft, target)).clamp(0.0, 1.0)
}

/// Proposition 1: the analytical acceptance rate of a **deterministic**
/// drafter equals the target's probability mass on the token it proposed.
///
/// The whole bridge between the two projects rests on this identity — see
/// [`eat_as_acceptance_rate`].
///
/// # Examples
///
/// ```
/// use meridian_core::dspark_bridge::{acceptance_of_point_mass_draft, analytical_acceptance};
///
/// let target = [0.6_f32, 0.3, 0.1];
/// let greedy = [1.0_f32, 0.0, 0.0];
/// let via_identity = acceptance_of_point_mass_draft(target[0]);
/// let via_definition = analytical_acceptance(&greedy, &target);
/// assert!((via_identity - via_definition).abs() < 1e-6);
/// ```
#[must_use]
pub fn acceptance_of_point_mass_draft(target_mass_on_drafted_token: f32) -> f32 {
    target_mass_on_drafted_token.clamp(0.0, 1.0)
}

/// Corollary 2, named for what it means: Meridian's EAT signal *is* the DSpark
/// analytical acceptance rate of a drafter that always proposes `</think>`.
///
/// Numerically a pure rename of [`acceptance_of_point_mass_draft`], kept as
/// its own symbol so call sites read as the claim they rely on.
#[must_use]
pub fn eat_as_acceptance_rate(eat: f32) -> f32 {
    acceptance_of_point_mass_draft(eat)
}

/// Proposition 3: the largest analytical acceptance rate achievable by any
/// drafter whose proposal is supported on the think-end token set.
///
/// Equals [`eat_as_acceptance_rate`] numerically; a distinct name because this
/// is an *upper bound over a family* of drafters rather than the exact value
/// for one of them.
#[must_use]
pub fn boundary_drafter_acceptance_ceiling(eat: f32) -> f32 {
    eat.clamp(0.0, 1.0)
}

/// Proposition 4, lower half: `e^{-H} ≤ max_v pᵗ(v)`.
///
/// Read as *headroom*: some deterministic drafter — the one proposing the
/// target's mode — achieves at least this acceptance rate. It says nothing
/// about the drafter actually deployed, which does not know `pᵗ`.
///
/// `entropy_nats` must be in nats, the unit
/// [`EntropySignal::token_entropy`] is documented to use.
#[must_use]
pub fn mode_mass_lower_bound(entropy_nats: f32) -> f32 {
    if !entropy_nats.is_finite() || entropy_nats < 0.0 {
        return 0.0;
    }
    (-entropy_nats).exp().clamp(0.0, 1.0)
}

/// Proposition 4, upper half: `max_v pᵗ(v) ≤ p*(H, V)`.
///
/// Solves `H = H_b(p) + (1-p)·ln(V-1)` for `p ∈ [1/V, 1]` by bisection. The
/// right-hand side is the maximum entropy attainable by any distribution over
/// `V` symbols whose largest mass is `p`, and is strictly decreasing there, so
/// the root is unique.
///
/// By Corollary 5 this is a hard ceiling on the single-step acceptance rate of
/// **any** deterministic drafter at a step with this entropy — the bound
/// [`super::hook`] uses, because a ceiling can only ever shrink a draft
/// budget.
///
/// # Examples
///
/// ```
/// use meridian_core::dspark_bridge::{deterministic_acceptance_ceiling, mode_mass_lower_bound};
///
/// // Uniform over 4 symbols has H = ln 4; the mode mass is pinned at 1/4.
/// let h = 4.0_f32.ln();
/// assert!((deterministic_acceptance_ceiling(h, 4) - 0.25).abs() < 1e-4);
/// // The bracket is well-ordered.
/// assert!(deterministic_acceptance_ceiling(h, 4) >= mode_mass_lower_bound(h));
/// ```
#[must_use]
pub fn deterministic_acceptance_ceiling(entropy_nats: f32, vocab_size: u32) -> f32 {
    if !entropy_nats.is_finite() || entropy_nats <= 0.0 {
        return 1.0;
    }
    if vocab_size <= 1 {
        return 1.0;
    }
    let vocab = f64::from(vocab_size);
    let max_entropy = vocab.ln();
    // Entropy above ln V is unattainable; clamp rather than fail, since probe
    // noise can push a measured value a hair over the ceiling.
    let target_h = f64::from(entropy_nats).min(max_entropy);

    let (mut lo, mut hi) = (1.0 / vocab, 1.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if max_entropy_given_mode_mass(mid, vocab) > target_h {
            // This `p` still admits more entropy than observed, so the
            // feasible mode mass lies above it.
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)) as f32
}

/// Maximum Shannon entropy (nats) attainable over `vocab` symbols by a
/// distribution whose largest probability mass is exactly `p`: a spike of `p`
/// with the remainder spread uniformly over the other `V - 1` symbols.
fn max_entropy_given_mode_mass(p: f64, vocab: f64) -> f64 {
    if p >= 1.0 {
        return 0.0;
    }
    let rest = 1.0 - p;
    let spike = if p > 0.0 { -p * p.ln() } else { 0.0 };
    spike - rest * (rest / (vocab - 1.0)).ln()
}

/// Prefix survival probability `S_γ = ∏_{i ≤ γ} c_i`.
///
/// The composition rule DSpark's scheduler applies to per-position confidences
/// before choosing a verification depth.
#[must_use]
pub fn prefix_survival(per_position_confidence: &[f32]) -> f32 {
    per_position_confidence
        .iter()
        .fold(1.0_f64, |acc, &c| acc * f64::from(c.clamp(0.0, 1.0))) as f32
}

/// Expected tokens committed per verification cycle, `τ(a, γ)`.
///
/// `τ(a, γ) = (1 - a^{γ+1}) / (1 - a)` under an i.i.d. acceptance model — the
/// classical speculative decoding result, including the target-generated bonus
/// token.
///
/// # Model caveat
///
/// The i.i.d. assumption is the weak link. Real drafters exhibit suffix decay —
/// acceptance falls with position — which is precisely what DSpark's sequential
/// head exists to mitigate and why DeepSpec's harness reports
/// `accept_rates_by_position`. This function is a first-order planner for
/// [`super::hook`], not a claim about measured behaviour;
/// [`super::ledger`] measures the real quantity once data exists.
#[must_use]
pub fn expected_accepted_length(acceptance: f32, proposal_len: u32) -> f32 {
    let a = f64::from(acceptance.clamp(0.0, 1.0));
    let gamma = f64::from(proposal_len);
    if (1.0 - a).abs() < 1e-9 {
        return (gamma + 1.0) as f32;
    }
    ((1.0 - a.powf(gamma + 1.0)) / (1.0 - a)) as f32
}

// ---------------------------------------------------------------------------
// Bounds derived from a live Meridian signal
// ---------------------------------------------------------------------------

/// What Meridian's already-computed signal provably implies about draft
/// acceptance at one decode step.
///
/// Every field is a *bound*, never an estimate. Nothing here predicts what a
/// specific drafter will do, because nothing here has seen a drafter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptanceBounds {
    /// Corollary 5 ceiling: no deterministic drafter exceeds this single-step
    /// acceptance rate at a step with this entropy.
    pub deterministic_ceiling: f32,
    /// Proposition 4 floor on the mode mass. Headroom, not a guarantee for the
    /// deployed drafter — retained so the bracket is auditable at a glance.
    pub achievable_floor: f32,
    /// Proposition 3 ceiling for drafters proposing only think-end tokens;
    /// by Corollary 2, also the exact acceptance rate of the drafter that
    /// always proposes `</think>`.
    pub boundary_ceiling: f32,
}

impl AcceptanceBounds {
    /// Derive the bounds from a probe sample and the model's vocabulary size.
    #[must_use]
    pub fn from_signal(signal: &EntropySignal, vocab_size: u32) -> Self {
        Self {
            deterministic_ceiling: deterministic_acceptance_ceiling(
                signal.token_entropy,
                vocab_size,
            ),
            achievable_floor: mode_mass_lower_bound(signal.token_entropy),
            boundary_ceiling: boundary_drafter_acceptance_ceiling(signal.eat),
        }
    }

    /// Width of the entropy bracket on the mode mass. A wide bracket means the
    /// entropy signal is weakly informative at this step.
    #[must_use]
    pub fn bracket_width(&self) -> f32 {
        self.deterministic_ceiling - self.achievable_floor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn point_mass(n: usize, at: usize) -> Vec<f32> {
        let mut v = vec![0.0; n];
        v[at] = 1.0;
        v
    }

    fn entropy_nats(p: &[f32]) -> f32 {
        -p.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| f64::from(x) * f64::from(x).ln())
            .sum::<f64>() as f32
    }

    /// Deterministic xorshift — reproducible sweeps without a dev-dependency.
    fn sampler() -> impl FnMut() -> f32 {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0
        }
    }

    // -- Total variation / acceptance ------------------------------------

    #[test]
    fn tv_of_identical_distributions_is_zero() {
        let p = [0.5_f32, 0.3, 0.2];
        assert!(total_variation(&p, &p).abs() < TOL);
        assert!((analytical_acceptance(&p, &p) - 1.0).abs() < TOL);
    }

    #[test]
    fn tv_of_disjoint_point_masses_is_one() {
        let p = point_mass(4, 0);
        let q = point_mass(4, 3);
        assert!((total_variation(&p, &q) - 1.0).abs() < TOL);
        assert!(analytical_acceptance(&p, &q).abs() < TOL);
    }

    #[test]
    fn tv_is_symmetric() {
        let p = [0.1_f32, 0.6, 0.3];
        let q = [0.4_f32, 0.4, 0.2];
        assert!((total_variation(&p, &q) - total_variation(&q, &p)).abs() < TOL);
    }

    /// Shorter slices are treated as zero-padded, not truncated.
    #[test]
    fn tv_handles_ragged_slices_as_zero_padded() {
        let short = [1.0_f32];
        let long = [0.5_f32, 0.5];
        // ½(|1 - 0.5| + 0.5) = 0.5
        assert!((total_variation(&short, &long) - 0.5).abs() < TOL);
    }

    // -- Proposition 1 / Corollary 2 -------------------------------------

    #[test]
    fn proposition_1_point_mass_identity_holds_over_a_sweep() {
        let vocab = 8;
        for target_mass in [0.05_f32, 0.25, 0.5, 0.75, 0.99] {
            let mut target = vec![(1.0 - target_mass) / (vocab - 1) as f32; vocab];
            target[0] = target_mass;
            let draft = point_mass(vocab, 0);

            let by_definition = analytical_acceptance(&draft, &target);
            let by_identity = acceptance_of_point_mass_draft(target_mass);
            assert!(
                (by_definition - by_identity).abs() < 1e-4,
                "mass={target_mass}: definition={by_definition}, identity={by_identity}",
            );
        }
    }

    #[test]
    fn corollary_2_eat_equals_acceptance_of_a_think_end_drafter() {
        let vocab = 16;
        let think_end_idx = 3;
        for eat in [0.01_f32, 0.2, 0.6, 0.95] {
            let mut target = vec![(1.0 - eat) / (vocab - 1) as f32; vocab];
            target[think_end_idx] = eat;
            let drafter_always_proposing_think_end = point_mass(vocab, think_end_idx);

            let measured = analytical_acceptance(&drafter_always_proposing_think_end, &target);
            assert!(
                (measured - eat_as_acceptance_rate(eat)).abs() < 1e-4,
                "eat={eat}: measured={measured}",
            );
        }
    }

    // -- Proposition 3 ---------------------------------------------------

    #[test]
    fn proposition_3_eat_ceilings_boundary_supported_drafters() {
        let vocab = 10;
        // Boundary set E = {2, 5}; EAT is the combined mass on those.
        let (e0, e1) = (0.18_f32, 0.12);
        let eat = e0 + e1;
        let mut target = vec![(1.0 - eat) / (vocab - 2) as f32; vocab];
        target[2] = e0;
        target[5] = e1;

        for split in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let mut draft = vec![0.0_f32; vocab];
            draft[2] = split;
            draft[5] = 1.0 - split;
            let acceptance = analytical_acceptance(&draft, &target);
            assert!(
                acceptance <= boundary_drafter_acceptance_ceiling(eat) + 1e-5,
                "split={split}: acceptance={acceptance} exceeded ceiling {eat}",
            );
        }

        // Equality case: a singleton boundary set, where the point-mass draft
        // trivially dominates the target on E.
        let singleton_target = {
            let mut t = vec![(1.0 - e0) / (vocab - 1) as f32; vocab];
            t[2] = e0;
            t
        };
        let draft = point_mass(vocab, 2);
        assert!((analytical_acceptance(&draft, &singleton_target) - e0).abs() < 1e-4);
    }

    // -- Proposition 4 / Corollary 5 -------------------------------------

    /// The bracket `e^{-H} ≤ M ≤ p*(H,V)` must hold for arbitrary
    /// distributions, in both directions.
    #[test]
    fn proposition_4_brackets_the_mode_mass_for_random_distributions() {
        let mut next = sampler();
        let vocab = 32_usize;

        for _ in 0..300 {
            // Vary concentration so the sweep covers both near-uniform and
            // spiky regimes rather than only the middle.
            let sharpness = 1.0 + 12.0 * next();
            let raw: Vec<f32> = (0..vocab)
                .map(|_| (next() + 1e-3).powf(sharpness))
                .collect();
            let sum: f32 = raw.iter().sum();
            let p: Vec<f32> = raw.iter().map(|x| x / sum).collect();

            let h = entropy_nats(&p);
            let mode = p.iter().copied().fold(0.0_f32, f32::max);

            let floor = mode_mass_lower_bound(h);
            let ceiling = deterministic_acceptance_ceiling(h, vocab as u32);

            assert!(
                mode >= floor - 1e-4,
                "mode={mode} below floor={floor} (H={h})"
            );
            assert!(
                mode <= ceiling + 1e-4,
                "mode={mode} above ceiling={ceiling} (H={h})",
            );
            assert!(ceiling >= floor - 1e-4, "bracket inverted at H={h}");
        }
    }

    /// Corollary 5 is the statement the hook depends on: no deterministic
    /// drafter — not even an adversarially lucky one — beats the ceiling.
    #[test]
    fn corollary_5_ceilings_every_deterministic_drafter() {
        let mut next = sampler();
        let vocab = 24_usize;

        for _ in 0..100 {
            let raw: Vec<f32> = (0..vocab).map(|_| next() + 1e-3).collect();
            let sum: f32 = raw.iter().sum();
            let target: Vec<f32> = raw.iter().map(|x| x / sum).collect();
            let ceiling = deterministic_acceptance_ceiling(entropy_nats(&target), vocab as u32);

            for drafted in 0..vocab {
                let acceptance = analytical_acceptance(&point_mass(vocab, drafted), &target);
                assert!(
                    acceptance <= ceiling + 1e-4,
                    "drafting {drafted} achieved {acceptance} over ceiling {ceiling}",
                );
            }
        }
    }

    #[test]
    fn ceiling_is_exact_on_the_uniform_distribution() {
        for vocab in [2_u32, 4, 10, 128, 151_936] {
            let h = f64::from(vocab).ln() as f32;
            let ceiling = deterministic_acceptance_ceiling(h, vocab);
            let expected = 1.0 / vocab as f32;
            assert!(
                (ceiling - expected).abs() < 1e-4,
                "V={vocab}: ceiling={ceiling}, want {expected}",
            );
        }
    }

    #[test]
    fn ceiling_is_exact_on_the_spike_plus_uniform_family() {
        // The extremal family attains the bound with equality by construction.
        let vocab = 64_usize;
        for spike in [0.1_f32, 0.35, 0.7, 0.9] {
            let mut p = vec![(1.0 - spike) / (vocab - 1) as f32; vocab];
            p[0] = spike;
            let recovered = deterministic_acceptance_ceiling(entropy_nats(&p), vocab as u32);
            assert!(
                (recovered - spike).abs() < 1e-3,
                "spike={spike}: recovered={recovered}",
            );
        }
    }

    #[test]
    fn zero_entropy_implies_certain_acceptance_is_possible() {
        assert!((mode_mass_lower_bound(0.0) - 1.0).abs() < TOL);
        assert!((deterministic_acceptance_ceiling(0.0, 1_000) - 1.0).abs() < TOL);
    }

    #[test]
    fn bounds_are_monotone_decreasing_in_entropy() {
        let mut previous_floor = 1.1_f32;
        let mut previous_ceiling = 1.1_f32;
        for step in 0..20 {
            let h = step as f32 * 0.25;
            let floor = mode_mass_lower_bound(h);
            let ceiling = deterministic_acceptance_ceiling(h, 1_024);
            assert!(floor <= previous_floor + 1e-6);
            assert!(ceiling <= previous_ceiling + 1e-6);
            previous_floor = floor;
            previous_ceiling = ceiling;
        }
    }

    #[test]
    fn degenerate_inputs_do_not_panic_or_produce_nan() {
        for h in [f32::NAN, f32::INFINITY, -1.0, 0.0, 1e30] {
            let floor = mode_mass_lower_bound(h);
            let ceiling = deterministic_acceptance_ceiling(h, 32);
            assert!(floor.is_finite() && (0.0..=1.0).contains(&floor), "h={h}");
            assert!(
                ceiling.is_finite() && (0.0..=1.0).contains(&ceiling),
                "h={h}"
            );
        }
        assert!((deterministic_acceptance_ceiling(1.0, 0) - 1.0).abs() < TOL);
        assert!((deterministic_acceptance_ceiling(1.0, 1) - 1.0).abs() < TOL);
    }

    // -- Composition rules -----------------------------------------------

    #[test]
    fn prefix_survival_is_the_cumulative_product() {
        assert!((prefix_survival(&[]) - 1.0).abs() < TOL);
        assert!((prefix_survival(&[0.9, 0.8, 0.5]) - 0.36).abs() < 1e-4);
        // A single certain rejection collapses the whole prefix.
        assert!(prefix_survival(&[0.99, 0.0, 0.99]).abs() < TOL);
    }

    #[test]
    fn expected_accepted_length_matches_closed_forms() {
        // a = 0 → only the bonus token survives.
        assert!((expected_accepted_length(0.0, 7) - 1.0).abs() < TOL);
        // a = 1 → the whole block plus the bonus token.
        assert!((expected_accepted_length(1.0, 7) - 8.0).abs() < TOL);
        // a = 0.5, γ = 3 → (1 - 0.5⁴)/0.5 = 1.875.
        assert!((expected_accepted_length(0.5, 3) - 1.875).abs() < 1e-4);
    }

    #[test]
    fn expected_accepted_length_is_monotone_in_both_arguments() {
        for gamma in 1_u32..8 {
            let mut previous = 0.0;
            for step in 0..=10 {
                let a = step as f32 / 10.0;
                let tau = expected_accepted_length(a, gamma);
                assert!(tau >= previous - 1e-5, "a={a}, γ={gamma}");
                previous = tau;
            }
        }
        for step in 0..=10 {
            let a = step as f32 / 10.0;
            let mut previous = 0.0;
            for gamma in 1_u32..8 {
                let tau = expected_accepted_length(a, gamma);
                assert!(tau >= previous - 1e-5, "a={a}, γ={gamma}");
                previous = tau;
            }
        }
    }

    // -- Bounds bundle ---------------------------------------------------

    #[test]
    fn bounds_from_signal_wires_the_right_fields() {
        let signal = EntropySignal {
            token_entropy: 2.0_f32.ln(),
            eat: 0.42,
            eat_ema: 0.40,
            eat_ema_variance: 0.001,
        };
        let bounds = AcceptanceBounds::from_signal(&signal, 4);

        assert!((bounds.boundary_ceiling - 0.42).abs() < TOL);
        assert!((bounds.achievable_floor - 0.5).abs() < 1e-4);
        assert!(bounds.deterministic_ceiling >= bounds.achievable_floor);
        assert!(bounds.bracket_width() >= 0.0);
    }

    #[test]
    fn uniform_distribution_yields_the_tightest_possible_ceiling() {
        let vocab = 151_936_u32; // Qwen3's vocabulary.
        let signal = EntropySignal {
            token_entropy: f64::from(vocab).ln() as f32,
            eat: 0.0,
            eat_ema: 0.0,
            eat_ema_variance: 0.0,
        };
        let bounds = AcceptanceBounds::from_signal(&signal, vocab);
        assert!(bounds.deterministic_ceiling < 1e-4);
        assert!(bounds.boundary_ceiling.abs() < TOL);
    }
}
