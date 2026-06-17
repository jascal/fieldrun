//! Tropical geometry of the decision surface — the shared nearest-facet kernel for the power-diagram
//! probes (`--probe-facet`, `--probe-tropical`). See [`TROPICAL_PROPOSAL.md`] §2/§11.
//!
//! The token cells in residual space are the **Laguerre power diagram** of the unembedding frame
//! `{U_v}`; the facet between cells `t` and `v` is the bisector `{r : L_t(r) = L_v(r)}`, and the
//! normalized margin `(L_t − L_v)/‖U_t − U_v‖` is the **exact Euclidean distance** from `r` to that
//! facet (TT1/TT2). This module holds the pure geometry — no forward pass, no I/O — so `--probe-facet`
//! and `--probe-tropical` share one verified kernel (and `headgate.rs`'s head-gating geometry could
//! adopt it next). Inputs are precomputed elsewhere: the logits `l[v] = ⟨U_v, r⟩`, the squared norms
//! `unorm[v] = ‖U_v‖²`, and a single Gram row for the winner `g[v] = ⟨U_v, U_t⟩`, so that
//! `‖U_t − U_v‖² = ‖U_t‖² + ‖U_v‖² − 2⟨U_v, U_t⟩` needs no extra `weight_row` calls.

/// Parallel/degenerate-facet guard: `‖U_t − U_v‖² ≤ DEGEN` ⇒ no facet (matches `--probe-facet` and
/// `headgate.rs`, where degenerate rows are sent to `−∞`).
pub const DEGEN: f32 = 1e-4;

/// The binding facet of the tropical hypersurface `T(M)` nearest to a residual with logits `l`, winner `t`.
#[derive(Clone, Copy, Debug)]
pub struct Facet {
    /// nearest competing token `v*` (the facet `r` is closest to crossing).
    pub vstar: usize,
    /// exact Euclidean distance from `r` to the `t`–`v*` bisector = the normalized margin (TT2).
    pub dist: f32,
    /// logit runner-up `argmax_{v≠t} l[v]` (the cheap proxy `v*` usually — but not always — equals).
    pub ru: usize,
    /// `cos∠(U_t, U_v*)` — the crossing sharpness, the `T→0` image of PIC's `ρ` (TT6); `NaN` if undefined.
    pub angle: f32,
}

/// Compute the nearest facet over the whole vocabulary (the exact `--probe-facet` kernel).
/// `l`, `unorm`, `g` are all indexed by token; `g[v] = ⟨U_v, U_t⟩` is the winner's Gram row.
pub fn nearest_facet(l: &[f32], t: usize, unorm: &[f32], g: &[f32]) -> Facet {
    let (mut best_d, mut vstar) = (f32::INFINITY, t);
    let (mut best_l, mut ru) = (f32::NEG_INFINITY, t);
    for v in 0..l.len() {
        if v == t {
            continue;
        }
        if l[v] > best_l {
            best_l = l[v];
            ru = v;
        }
        let dvv2 = unorm[t] + unorm[v] - 2.0 * g[v]; // ‖U_t − U_v‖²
        if dvv2 > DEGEN {
            let dv = (l[t] - l[v]) / dvv2.sqrt(); // exact distance to the t–v bisector facet
            if dv < best_d {
                best_d = dv;
                vstar = v;
            }
        }
    }
    let denom = (unorm[t] * unorm[vstar]).sqrt();
    let angle = if vstar != t && denom > 1e-12 {
        g[vstar] / denom
    } else {
        f32::NAN
    };
    Facet { vstar, dist: best_d, ru, angle }
}

/// Local active-monomial count near the winning cell: `#{v : L_t − L_v ≤ eps}` (includes the winner).
/// A cheap local-tropical-rank proxy (TROPICAL_PROPOSAL §11.1) — larger ⇒ more monomials crowd the cell.
pub fn local_rank(l: &[f32], t: usize, eps: f32) -> usize {
    let lt = l[t];
    l.iter().filter(|&&lv| lt - lv <= eps).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Orthogonal frame U_0=(1,0), U_1=(0,1), U_2=(-1,0); residual r=(2,0.1) ⇒ winner t=0.
    // l = ⟨U_v,r⟩ = [2, 0.1, -2]; unorm = [1,1,1]; g[v] = ⟨U_v,U_0⟩ = [1, 0, -1].
    #[test]
    fn nearest_facet_basic() {
        let l = [2.0_f32, 0.1, -2.0];
        let unorm = [1.0_f32, 1.0, 1.0];
        let g = [1.0_f32, 0.0, -1.0];
        let f = nearest_facet(&l, 0, &unorm, &g);
        // v=1: ‖U_0−U_1‖²=2, dist=(2−0.1)/√2≈1.3435; v=2: ‖U_0−U_2‖²=4, dist=(2+2)/2=2.0 ⇒ nearest is v=1.
        assert_eq!(f.vstar, 1);
        assert!((f.dist - 1.343_502_9).abs() < 1e-3, "dist={}", f.dist);
        assert_eq!(f.ru, 1, "logit runner-up is token 1");
        assert!(f.angle.abs() < 1e-6, "U_0 ⟂ U_1 ⇒ cos angle 0, got {}", f.angle);
    }

    // Degenerate guard + ru≠v* divergence: token 1 is the logit runner-up but is PARALLEL to the winner
    // (‖U_t−U_1‖²=0), so it has no facet and is skipped — the nearest facet is the farther-in-logit token 2.
    #[test]
    fn nearest_facet_skips_degenerate() {
        let l = [5.0_f32, 3.0, 1.0];
        let unorm = [1.0_f32, 1.0, 1.0];
        let g = [1.0_f32, 1.0, 0.0]; // g[1]=1 ⇒ ‖U_0−U_1‖² = 1+1−2 = 0 (degenerate); g[2]=0 ⇒ ‖·‖²=2
        let f = nearest_facet(&l, 0, &unorm, &g);
        assert_eq!(f.ru, 1, "logit runner-up is still token 1");
        assert_eq!(f.vstar, 2, "token 1 has no facet (parallel) ⇒ nearest facet is token 2");
        assert!((f.dist - (4.0 / 2.0_f32.sqrt())).abs() < 1e-4, "dist={}", f.dist);
    }

    #[test]
    fn local_rank_counts_within_eps() {
        let l = [2.0_f32, 0.1, -2.0];
        assert_eq!(local_rank(&l, 0, 1.0), 1, "only the winner is within 1.0 of the max");
        assert_eq!(local_rank(&l, 0, 2.0), 2, "winner + token 1 (gap 1.9) within 2.0");
        assert_eq!(local_rank(&l, 0, 100.0), 3, "all three within a wide eps");
    }
}
