//! Small, dependency-free statistics kernel used by [`super::ledger`].
//!
//! Phase 1 of the phase-conditioned-speculation work (see
//! [ADR-0009](https://github.com/angelnicolasc/meridian/blob/main/docs/src/adr/0009-phase-conditioned-speculation.md))
//! compares mean accepted draft length between two unpaired samples — think-phase
//! verification steps and output-phase verification steps — drawn from the *same*
//! responses. Those two samples have neither equal variance nor equal size, so the
//! correct test is **Welch's unequal-variance _t_-test**, not Student's pooled test.
//!
//! Everything here is `f64`. The signal path elsewhere in the crate is `f32`
//! because it runs per token; this runs once per report.
//!
//! ## Why hand-rolled
//!
//! `meridian-core` has no numeric dependency and is not going to grow one for a
//! single test statistic. The three routines below (`ln_gamma`, `betacf`,
//! `regularized_incomplete_beta`) are textbook, and the module test suite pins
//! them against closed forms that are exact for `df = 1` (Cauchy) and `df = 2`,
//! plus published critical values elsewhere. See the tests at the bottom of this
//! file.

// ---------------------------------------------------------------------------
// Summary statistics
// ---------------------------------------------------------------------------

/// Streaming mean/variance accumulator (Welford's online algorithm).
///
/// Welford is used rather than the naive sum-of-squares form because accepted
/// lengths are small integers accumulated over potentially millions of
/// verification steps, where `Σx²` loses precision against `(Σx)²/n`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Moments {
    count: u64,
    mean: f64,
    /// Sum of squared deviations from the running mean (`M2` in Welford).
    m2: f64,
}

impl Moments {
    /// An empty accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Fold one observation in.
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Merge another accumulator (Chan et al. parallel variance combination).
    ///
    /// Used to combine per-request or per-rank partial ledgers without
    /// replaying the underlying observations.
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = *other;
            return;
        }
        let n_a = self.count as f64;
        let n_b = other.count as f64;
        let total = n_a + n_b;
        let delta = other.mean - self.mean;
        self.mean = (n_a * self.mean + n_b * other.mean) / total;
        self.m2 += other.m2 + delta * delta * n_a * n_b / total;
        self.count += other.count;
    }

    /// Number of observations folded in.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Sample mean. `0.0` when empty — callers gate on [`Self::count`].
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Unbiased sample variance (`n - 1` denominator). `None` below two
    /// observations, where the quantity is undefined rather than zero.
    #[must_use]
    pub fn variance(&self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        Some(self.m2 / (self.count - 1) as f64)
    }

    /// Unbiased sample standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }
}

// ---------------------------------------------------------------------------
// Welch's t-test
// ---------------------------------------------------------------------------

/// Outcome of a two-sample Welch test.
///
/// The sign convention is **`a - b`**: in [`super::ledger`] the first sample is
/// the think phase and the second is the output phase, so a negative
/// [`Self::mean_difference`] is the direction the Section 5 hypothesis predicts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelchResult {
    /// `mean(a) - mean(b)`.
    pub mean_difference: f64,
    /// Standard error of the difference, `sqrt(s_a²/n_a + s_b²/n_b)`.
    pub std_error: f64,
    /// Welch's _t_ statistic.
    pub t_statistic: f64,
    /// Welch–Satterthwaite effective degrees of freedom (non-integer).
    pub degrees_of_freedom: f64,
    /// Two-sided _p_-value under the _t_ distribution with
    /// [`Self::degrees_of_freedom`].
    pub p_value: f64,
    /// Cohen's _d_ on the pooled standard deviation. Reported alongside the
    /// _p_-value because at the sample sizes a Phase 1 run produces (10⁴–10⁶
    /// verification steps) statistical significance is nearly free and effect
    /// size is the quantity that decides whether the result matters.
    pub cohens_d: f64,
    /// Half-width of the 95 % confidence interval on
    /// [`Self::mean_difference`], using the _t_ critical value at
    /// [`Self::degrees_of_freedom`].
    pub ci95_half_width: f64,
}

impl WelchResult {
    /// Lower bound of the 95 % confidence interval on the mean difference.
    #[must_use]
    pub fn ci95_lower(&self) -> f64 {
        self.mean_difference - self.ci95_half_width
    }

    /// Upper bound of the 95 % confidence interval on the mean difference.
    #[must_use]
    pub fn ci95_upper(&self) -> f64 {
        self.mean_difference + self.ci95_half_width
    }

    /// `true` when the 95 % interval excludes zero.
    ///
    /// Deliberately named after what it checks rather than "significant" —
    /// the decision rule in the Phase 1 protocol is an effect-size threshold,
    /// not a _p_ < 0.05 gate.
    #[must_use]
    pub fn ci95_excludes_zero(&self) -> bool {
        self.ci95_lower() > 0.0 || self.ci95_upper() < 0.0
    }
}

/// Run Welch's unequal-variance _t_-test on two summarised samples.
///
/// Returns `None` when either sample has fewer than two observations, or when
/// both variances are zero (the statistic is undefined, not infinite — a
/// degenerate input must not be reported as an infinitely strong effect).
#[must_use]
pub fn welch_t_test(a: &Moments, b: &Moments) -> Option<WelchResult> {
    let (var_a, var_b) = (a.variance()?, b.variance()?);
    let (n_a, n_b) = (a.count() as f64, b.count() as f64);

    let se_sq = var_a / n_a + var_b / n_b;
    if se_sq <= 0.0 {
        return None;
    }
    let std_error = se_sq.sqrt();
    let mean_difference = a.mean() - b.mean();
    let t_statistic = mean_difference / std_error;

    // Welch–Satterthwaite.
    let term_a = var_a / n_a;
    let term_b = var_b / n_b;
    let degrees_of_freedom =
        se_sq * se_sq / (term_a * term_a / (n_a - 1.0) + term_b * term_b / (n_b - 1.0));

    let p_value = students_t_two_sided_p(t_statistic, degrees_of_freedom);

    // Pooled SD for Cohen's d. Uses the classic pooled estimator even though
    // the test itself does not assume equal variance: d is a descriptive
    // standardisation, and the pooled form is what readers expect.
    let pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0);
    let cohens_d = if pooled_var > 0.0 {
        mean_difference / pooled_var.sqrt()
    } else {
        0.0
    };

    let ci95_half_width = students_t_critical(degrees_of_freedom, 0.05) * std_error;

    Some(WelchResult {
        mean_difference,
        std_error,
        t_statistic,
        degrees_of_freedom,
        p_value,
        cohens_d,
        ci95_half_width,
    })
}

// ---------------------------------------------------------------------------
// Student's t distribution
// ---------------------------------------------------------------------------

/// Two-sided _p_-value of `t` under Student's _t_ with `df` degrees of freedom.
///
/// Uses the standard identity
/// `p = I_{df/(df + t²)}(df/2, 1/2)`
/// where `I` is the regularized incomplete beta function.
#[must_use]
pub fn students_t_two_sided_p(t: f64, df: f64) -> f64 {
    if !t.is_finite() || !df.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    regularized_incomplete_beta(0.5 * df, 0.5, x).clamp(0.0, 1.0)
}

/// Two-sided critical value `t*` such that `P(|T| > t*) = alpha`.
///
/// Found by bisection on [`students_t_two_sided_p`], which is strictly
/// decreasing in `t` — 200 iterations is far more than the ~60 needed to
/// exhaust `f64` on the bracket, and costs nothing at report cadence.
#[must_use]
pub fn students_t_critical(df: f64, alpha: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 || !(0.0..1.0).contains(&alpha) {
        return f64::NAN;
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0e4_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if students_t_two_sided_p(mid, df) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Regularized incomplete beta function `I_x(a, b)`.
///
/// Continued-fraction evaluation with the standard symmetry reflection so the
/// fraction is always evaluated in its rapidly-converging regime.
#[must_use]
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let front = ln_front.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for the incomplete beta, evaluated with the modified
/// Lentz algorithm.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 300;
    const EPS: f64 = 3.0e-16;
    const TINY: f64 = 1.0e-300;

    let sum_ab = a + b;
    let a_plus_one = a + 1.0;
    let a_minus_one = a - 1.0;

    let mut numer = 1.0;
    let mut denom = 1.0 - sum_ab * x / a_plus_one;
    if denom.abs() < TINY {
        denom = TINY;
    }
    denom = 1.0 / denom;
    let mut value = denom;

    for step in 1..=MAX_ITER {
        let m = step as f64;
        let m2 = 2.0 * m;

        // Even step.
        let coeff = m * (b - m) * x / ((a_minus_one + m2) * (a + m2));
        denom = 1.0 + coeff * denom;
        if denom.abs() < TINY {
            denom = TINY;
        }
        numer = 1.0 + coeff / numer;
        if numer.abs() < TINY {
            numer = TINY;
        }
        denom = 1.0 / denom;
        value *= denom * numer;

        // Odd step.
        let coeff = -(a + m) * (sum_ab + m) * x / ((a + m2) * (a_plus_one + m2));
        denom = 1.0 + coeff * denom;
        if denom.abs() < TINY {
            denom = TINY;
        }
        numer = 1.0 + coeff / numer;
        if numer.abs() < TINY {
            numer = TINY;
        }
        denom = 1.0 / denom;
        let delta = denom * numer;
        value *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    value
}

/// Natural log of the gamma function (Lanczos approximation, g = 7).
///
/// Accurate to ~15 significant digits over the positive reals, which is the
/// only domain this module evaluates it on.
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection: Γ(x)Γ(1-x) = π / sin(πx).
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut acc = COEFFS[0];
    for (i, coeff) in COEFFS.iter().enumerate().skip(1) {
        acc += coeff / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for comparisons against closed forms.
    const TOL: f64 = 1e-9;

    fn moments_of(xs: &[f64]) -> Moments {
        let mut m = Moments::new();
        for &x in xs {
            m.push(x);
        }
        m
    }

    #[test]
    fn welford_matches_textbook_variance() {
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let m = moments_of(&xs);
        assert_eq!(m.count(), 8);
        assert!((m.mean() - 5.0).abs() < TOL);
        // Unbiased variance of this classic sample is 32/7.
        assert!((m.variance().unwrap() - 32.0 / 7.0).abs() < TOL);
    }

    #[test]
    fn merge_is_equivalent_to_sequential_push() {
        let left = moments_of(&[1.0, 2.0, 3.0, 4.0]);
        let right = moments_of(&[10.0, 11.0, 12.0]);
        let mut merged = left;
        merged.merge(&right);

        let sequential = moments_of(&[1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]);
        assert_eq!(merged.count(), sequential.count());
        assert!((merged.mean() - sequential.mean()).abs() < TOL);
        assert!((merged.variance().unwrap() - sequential.variance().unwrap()).abs() < 1e-12);
    }

    #[test]
    fn merge_with_empty_is_identity_in_both_directions() {
        let filled = moments_of(&[3.0, 5.0, 8.0]);
        let empty = Moments::new();

        let mut a = filled;
        a.merge(&empty);
        assert_eq!(a, filled);

        let mut b = empty;
        b.merge(&filled);
        assert_eq!(b, filled);
    }

    #[test]
    fn variance_undefined_below_two_observations() {
        assert!(Moments::new().variance().is_none());
        assert!(moments_of(&[1.0]).variance().is_none());
        assert!(moments_of(&[1.0, 2.0]).variance().is_some());
    }

    // -- t distribution --------------------------------------------------

    /// For df = 1 the t distribution is standard Cauchy, whose two-sided tail
    /// has the closed form `1 - (2/π)·arctan(t)`.
    #[test]
    fn t_matches_cauchy_closed_form_at_df_1() {
        for &t in &[0.25_f64, 0.5, 1.0, 2.0, 5.0, 20.0] {
            let expected = 1.0 - (2.0 / std::f64::consts::PI) * t.atan();
            let actual = students_t_two_sided_p(t, 1.0);
            assert!(
                (actual - expected).abs() < 1e-10,
                "t={t}: got {actual}, want {expected}",
            );
        }
    }

    /// For df = 2 the closed form is `1 - t/sqrt(t² + 2)`.
    #[test]
    fn t_matches_closed_form_at_df_2() {
        for &t in &[0.1_f64, 1.0, 2.5, 10.0] {
            let expected = 1.0 - t / (t * t + 2.0).sqrt();
            let actual = students_t_two_sided_p(t, 2.0);
            assert!(
                (actual - expected).abs() < 1e-10,
                "t={t}: got {actual}, want {expected}",
            );
        }
    }

    /// Published two-sided 5 % critical values.
    #[test]
    fn t_critical_matches_published_tables() {
        // (df, t*) from standard tables, quoted to 3 decimal places.
        for &(df, expected) in &[(5.0, 2.571), (10.0, 2.228), (30.0, 2.042), (100.0, 1.984)] {
            let actual = students_t_critical(df, 0.05);
            assert!(
                (actual - expected).abs() < 5e-4,
                "df={df}: got {actual}, want {expected}",
            );
        }
    }

    /// As df → ∞ the t distribution converges to the standard normal, whose
    /// two-sided 5 % critical value is 1.959_964.
    #[test]
    fn t_converges_to_normal_for_large_df() {
        let actual = students_t_critical(1.0e7, 0.05);
        assert!((actual - 1.959_964).abs() < 1e-4, "got {actual}");
    }

    #[test]
    fn t_is_symmetric_and_monotone() {
        for &t in &[0.5_f64, 1.0, 3.0] {
            let p_pos = students_t_two_sided_p(t, 12.0);
            let p_neg = students_t_two_sided_p(-t, 12.0);
            assert!((p_pos - p_neg).abs() < 1e-15);
        }
        assert!(students_t_two_sided_p(0.5, 12.0) > students_t_two_sided_p(3.0, 12.0));
        assert!((students_t_two_sided_p(0.0, 12.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        // Γ(n) = (n-1)!
        for (n, factorial) in [(1.0, 1.0), (2.0, 1.0), (5.0, 24.0), (11.0, 3_628_800.0)] {
            let actual: f64 = ln_gamma(n).exp();
            assert!(
                (actual - factorial).abs() / factorial < 1e-12,
                "Γ({n}) = {actual}, want {factorial}",
            );
        }
        // Γ(1/2) = sqrt(π).
        assert!((ln_gamma(0.5).exp() - std::f64::consts::PI.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn incomplete_beta_endpoints_and_symmetry() {
        assert!((regularized_incomplete_beta(2.0, 3.0, 0.0) - 0.0).abs() < TOL);
        assert!((regularized_incomplete_beta(2.0, 3.0, 1.0) - 1.0).abs() < TOL);
        // I_x(a,b) = 1 - I_{1-x}(b,a).
        let lhs = regularized_incomplete_beta(2.5, 4.5, 0.3);
        let rhs = 1.0 - regularized_incomplete_beta(4.5, 2.5, 0.7);
        assert!((lhs - rhs).abs() < 1e-12);
    }

    // -- Welch -----------------------------------------------------------

    /// Worked example against values derived by hand from the definitions, so
    /// a reviewer can re-check them without running the code:
    ///
    /// ```text
    /// mean_a = 163.0/8 = 20.375      var_a = 93.1149/7 = 13.30213
    /// mean_b = 188.0/8 = 23.5        var_b =  30.06/7  =  4.29429
    /// se     = sqrt(13.30213/8 + 4.29429/8) = 1.483089
    /// t      = (20.375 - 23.5) / 1.483089 = -2.10704
    /// df     = se⁴ / (…) = 11.0929      (Welch–Satterthwaite)
    /// ```
    #[test]
    fn welch_recovers_known_statistic() {
        let a = moments_of(&[27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1]);
        let b = moments_of(&[27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0]);
        let r = welch_t_test(&a, &b).unwrap();

        assert!(
            (r.mean_difference + 3.125).abs() < 1e-9,
            "Δ={}",
            r.mean_difference
        );
        assert!((r.std_error - 1.483_089).abs() < 1e-5, "se={}", r.std_error);
        assert!(
            (r.t_statistic + 2.107_04).abs() < 1e-4,
            "t={}",
            r.t_statistic
        );
        assert!(
            (r.degrees_of_freedom - 11.0929).abs() < 1e-3,
            "df={}",
            r.degrees_of_freedom,
        );
        // Two-sided p at t = 2.107, df = 11.09 sits just above the 5 % line.
        assert!(r.p_value > 0.05 && r.p_value < 0.07, "p={}", r.p_value);
    }

    #[test]
    fn welch_sign_convention_is_a_minus_b() {
        let low = moments_of(&[1.0, 2.0, 3.0, 2.0, 1.0]);
        let high = moments_of(&[10.0, 11.0, 12.0, 11.0, 10.0]);
        let r = welch_t_test(&low, &high).unwrap();
        assert!(r.mean_difference < 0.0);
        assert!(r.t_statistic < 0.0);
        assert!(r.cohens_d < 0.0);
    }

    #[test]
    fn welch_ci_brackets_the_difference_and_excludes_zero_when_separated() {
        let low = moments_of(&[1.0, 2.0, 3.0, 2.0, 1.0]);
        let high = moments_of(&[10.0, 11.0, 12.0, 11.0, 10.0]);
        let r = welch_t_test(&low, &high).unwrap();
        assert!(r.ci95_lower() < r.mean_difference);
        assert!(r.ci95_upper() > r.mean_difference);
        assert!(r.ci95_excludes_zero());
    }

    #[test]
    fn welch_ci_includes_zero_for_identical_samples() {
        let a = moments_of(&[4.0, 5.0, 6.0, 5.0, 4.0]);
        let b = moments_of(&[4.0, 5.0, 6.0, 5.0, 4.0]);
        let r = welch_t_test(&a, &b).unwrap();
        assert!((r.mean_difference).abs() < TOL);
        assert!(!r.ci95_excludes_zero());
        assert!((r.p_value - 1.0).abs() < TOL);
    }

    /// A degenerate sample (zero variance on both sides) must yield `None`
    /// rather than an infinite statistic.
    #[test]
    fn welch_rejects_zero_variance_inputs() {
        let a = moments_of(&[3.0, 3.0, 3.0]);
        let b = moments_of(&[7.0, 7.0, 7.0]);
        assert!(welch_t_test(&a, &b).is_none());
    }

    #[test]
    fn welch_rejects_undersized_samples() {
        let a = moments_of(&[1.0]);
        let b = moments_of(&[1.0, 2.0, 3.0]);
        assert!(welch_t_test(&a, &b).is_none());
        assert!(welch_t_test(&Moments::new(), &b).is_none());
    }
}
