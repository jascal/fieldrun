//! Balanced-ternary expansion of integer weights — the runnable, byte-identical mirror of the
//! **"lossless via expansion" lemma** (see TERNARY de-risk). Every integer `n` has a unique balanced-
//! ternary representation `n = Σ_j t_j · 3^j` with `t_j ∈ {−1, 0, +1}`; by linearity over a commutative
//! ring, an integer-weight dot distributes EXACTLY into a power-of-3-weighted sum of ternary dots:
//!
//! > `Σ_i w_i x_i  =  Σ_j 3^j · (Σ_i t_{ij} x_i)`   — exact, for *any* `x`.
//!
//! This module is pure integer arithmetic (no model, no I/O), so the lemma is unit-testable exhaustively;
//! `--verify-ternary` then runs it against real int8 weight codes and reports the **trit sparsity** — the
//! baseline for the optimization (minimize total nonzero trits preserving behaviour; zeros are free in
//! Datalog's closed world). The lossless half is the easy half — this nails it in fieldrun's currency
//! (byte-identical, like `--verify-cache`); the value is then in shrinking the blowup.

/// Trits needed to represent any integer with `|n| ≤ q`: `K = ⌈log₃(2q+1)⌉` (`cap_k = (3^k−1)/2`).
pub fn trits_for(q: i64) -> usize {
    let (mut k, mut cap) = (0usize, 0i64); // cap = max |n| representable in k balanced trits
    while cap < q.abs() {
        k += 1;
        cap = cap * 3 + 1; // cap_k = 3·cap_{k-1} + 1 = (3^k − 1)/2
    }
    k.max(1)
}

/// Balanced-ternary digits of `n` (LSB first), exactly `k` of them. `Σ out[j]·3^j == n`.
pub fn to_trits(mut n: i64, k: usize) -> Vec<i8> {
    let mut out = vec![0i8; k];
    for d in out.iter_mut() {
        let r = ((n % 3) + 3) % 3; // 0, 1, 2
        let t: i64 = if r == 2 { -1 } else { r }; // 2 ↦ −1 (carry), else the residue
        *d = t as i8;
        n = (n - t) / 3; // exact: (n − t) ≡ 0 (mod 3)
    }
    debug_assert_eq!(n, 0, "k={k} trits too few for the value");
    out
}

/// Reconstruct `Σ t_j · 3^j` from balanced trits.
pub fn from_trits(t: &[i8]) -> i64 {
    let (mut acc, mut p) = (0i64, 1i64);
    for &d in t {
        acc += d as i64 * p;
        p *= 3;
    }
    acc
}

/// The lossless-distribution identity on a real integer weight row `w` against activation `x`:
/// returns `(lhs = Σ w_i x_i, rhs = Σ_j 3^j Σ_i t_{ij} x_i)`. They MUST be equal (i64, exact).
pub fn distribute(w: &[i64], x: &[i64], k: usize) -> (i64, i64) {
    let lhs: i64 = w.iter().zip(x).map(|(&wi, &xi)| wi * xi).sum();
    let trits: Vec<Vec<i8>> = w.iter().map(|&wi| to_trits(wi, k)).collect();
    let (mut rhs, mut p) = (0i64, 1i64);
    for j in 0..k {
        let inner: i64 = trits.iter().zip(x).map(|(t, &xi)| t[j] as i64 * xi).sum();
        rhs += p * inner;
        p *= 3;
    }
    (lhs, rhs)
}

/// Trit-sparsity stats over a weight slice — the optimization baseline.
#[derive(Default)]
pub struct TritStats {
    pub n_weights: usize,
    pub k: usize,
    pub total_trits: usize,   // n_weights · k (the uniform/dense expansion)
    pub nonzero_trits: usize, // the sparse cost — zeros are absent facts in Datalog
    pub used_len: Vec<usize>, // used_len[j] = #weights whose highest nonzero trit is at position j (1..=k); [0] = exact zero
}

pub fn trit_stats(w: &[i64], k: usize) -> TritStats {
    let mut s = TritStats { k, used_len: vec![0; k + 1], ..Default::default() };
    for &wi in w {
        let t = to_trits(wi, k);
        s.n_weights += 1;
        s.total_trits += k;
        s.nonzero_trits += t.iter().filter(|&&x| x != 0).count();
        let top = t.iter().rposition(|&x| x != 0).map(|p| p + 1).unwrap_or(0);
        s.used_len[top] += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trits_for_known_bounds() {
        assert_eq!(trits_for(0), 1);
        assert_eq!(trits_for(1), 1); // cap_1 = 1
        assert_eq!(trits_for(7), 3); // int4: cap_2=4 < 7 ≤ cap_3=13
        assert_eq!(trits_for(13), 3); // exactly cap_3
        assert_eq!(trits_for(14), 4);
        assert_eq!(trits_for(127), 6); // int8: cap_5=121 < 127 ≤ cap_6=364
        assert_eq!(trits_for(-127), 6); // sign-agnostic
    }

    #[test]
    fn roundtrip_exhaustive_over_int8_expansion_range() {
        // The load-bearing lemma, exhaustively: every integer in the full 6-trit range round-trips, and
        // every digit is a valid balanced trit. cap_6 = (3^6 − 1)/2 = 364.
        let k = 6;
        for n in -364..=364 {
            let t = to_trits(n, k);
            assert!(t.iter().all(|&d| (-1..=1).contains(&d)), "non-trit digit for {n}: {t:?}");
            assert_eq!(from_trits(&t), n, "round-trip failed for {n}");
        }
    }

    #[test]
    fn distribute_is_exact() {
        // Σ w·x  ==  Σ_j 3^j Σ t·x  for real-ish integer rows (the byte-identical distribution).
        let w: Vec<i64> = (-9..=9).collect(); // includes the boundary signs
        let x: Vec<i64> = (0..w.len()).map(|i| (i as i64 * 37 % 2001) - 1000).collect();
        let k = trits_for(*w.iter().map(|v| v).max_by_key(|v| v.abs()).unwrap());
        let (lhs, rhs) = distribute(&w, &x, k);
        assert_eq!(lhs, rhs, "distribution not exact: {lhs} != {rhs}");
    }

    #[test]
    fn sparsity_small_weights_use_fewer_trits() {
        // 0 uses 0 trits, ±1 uses 1, ±13 uses 3 — the variable-length headroom the optimizer exploits.
        let st = trit_stats(&[0, 1, -1, 13, -13], 6);
        assert_eq!(st.used_len[0], 1); // the single 0
        assert_eq!(st.used_len[1], 2); // ±1
        assert_eq!(st.used_len[3], 2); // ±13
        assert!(st.nonzero_trits < st.total_trits, "zeros must reduce the nonzero count");
    }
}
