//! TurboQuant codec — a structured random rotation (SRHT) + a data-free per-coordinate Lloyd–Max scalar
//! quantizer (TURBOQUANT.md; Zandieh–Daliri–Hadian–Mirrokni, arXiv:2504.19874). Pure geometry — no model,
//! no I/O — so it is unit-testable in isolation, and it is the codec behind `--probe-distortion` (E-TQ2:
//! flip-rate vs facet margin / distortion) and, later, the optional KV-cache mode.
//!
//! The point of the random rotation: it **isotropizes** the quantization distortion. A unit vector's
//! coordinates become ~`N(0, 1/d)` after rotation, so a per-coordinate scalar quantizer is near-optimal,
//! AND the residual `x̂ − x` is isotropic — which is exactly why the tropical facet margin (the *normalized*
//! distance to the decision boundary, `tropical::nearest_facet`, TT2) is the flip-stability predictor with
//! a closed-form threshold `ρ(b,d) ≈ √(√3π/2 / d)·2⁻ᵇ` (the `‖U_t−U_v‖` cancels).

use std::f32::consts::PI;

/// In-place fast Walsh–Hadamard transform (radix-2 butterfly). `a.len()` must be a power of two.
/// Unnormalized: applying it twice yields `n·identity`.
fn fwht(a: &mut [f32]) {
    let n = a.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (x, y) = (a[j], a[j + h]);
                a[j] = x + y;
                a[j + h] = x - y;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// A structured orthogonal rotation `R = (1/√dpad)·H·D`: a seeded ±1 diagonal sign flip `D` then a
/// normalized Walsh–Hadamard transform `H`. `O(d log d)`, exactly invertible (up to f32), and it operates
/// on a power-of-two-padded vector so any `d` is supported.
pub struct Rotation {
    signs: Vec<i8>,
    pub dpad: usize,
}

impl Rotation {
    pub fn new(seed: u64, d: usize) -> Rotation {
        let dpad = d.next_power_of_two();
        let mut s = seed | 1;
        let signs = (0..dpad)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if s & 1 == 0 { 1i8 } else { -1 }
            })
            .collect();
        Rotation { signs, dpad }
    }

    /// Rotate `x` (len ≤ dpad): pad, sign-flip, FWHT, normalize. Returns the len-`dpad` rotated vector.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut v = vec![0f32; self.dpad];
        for (i, &xi) in x.iter().enumerate() {
            v[i] = xi * self.signs[i] as f32;
        }
        fwht(&mut v);
        let inv = 1.0 / (self.dpad as f32).sqrt();
        for vi in v.iter_mut() {
            *vi *= inv;
        }
        v
    }

    /// Inverse: FWHT, normalize, then undo the sign flip; return the first `d` coordinates.
    pub fn invert(&self, v: &[f32], d: usize) -> Vec<f32> {
        let mut w = v.to_vec();
        fwht(&mut w);
        let inv = 1.0 / (self.dpad as f32).sqrt();
        (0..d).map(|i| w[i] * inv * self.signs[i] as f32).collect()
    }
}

/// Data-free Lloyd–Max levels for a unit-variance `N(0,1)`: the `2^bits` MMSE quantizer levels, computed by
/// the Lloyd iteration over a numerical density (prefix-summed cells). Scale by `σ` for `N(0,σ²)`.
pub fn lloyd_max_gaussian(bits: u8) -> Vec<f32> {
    let n = 1usize << bits;
    let (lo, hi, steps) = (-8.0f64, 8.0f64, 16_001usize);
    let h = (hi - lo) / (steps - 1) as f64;
    // prefix sums of pdf and x·pdf over the grid (unnormalized Gaussian; ratios are all we need)
    let (mut cp, mut cxp) = (vec![0f64; steps + 1], vec![0f64; steps + 1]);
    for i in 0..steps {
        let x = lo + i as f64 * h;
        let p = (-0.5 * x * x).exp();
        cp[i + 1] = cp[i] + p;
        cxp[i + 1] = cxp[i] + x * p;
    }
    let idx = |x: f64| -> usize { (((x - lo) / h).round() as isize).clamp(0, steps as isize) as usize };
    let mut lev: Vec<f64> = (0..n).map(|k| -2.5 + 5.0 * (k as f64 + 0.5) / n as f64).collect();
    for _ in 0..200 {
        let mut bnd = vec![0f64; n + 1];
        bnd[0] = lo;
        bnd[n] = hi;
        for k in 1..n {
            bnd[k] = 0.5 * (lev[k - 1] + lev[k]);
        }
        let mut delta = 0f64;
        for k in 0..n {
            let (a, b) = (idx(bnd[k]), idx(bnd[k + 1]));
            let den = cp[b] - cp[a];
            if den > 0.0 {
                let m = (cxp[b] - cxp[a]) / den;
                delta += (m - lev[k]).abs();
                lev[k] = m;
            }
        }
        if delta < 1e-7 {
            break;
        }
    }
    // The MMSE quantizer of a symmetric density is symmetric — enforce it exactly (the numerical Lloyd on a
    // finite grid is only symmetric to grid resolution). antisymmetrize: out[k] = −out[n−1−k].
    (0..n).map(|k| ((lev[k] - lev[n - 1 - k]) * 0.5) as f32).collect()
}

/// Index of the nearest level (levels sorted ascending).
fn nearest_level(x: f32, levels: &[f32]) -> u8 {
    match levels.binary_search_by(|l| l.partial_cmp(&x).unwrap()) {
        Ok(i) => i as u8,
        Err(i) => {
            if i == 0 {
                0
            } else if i >= levels.len() {
                (levels.len() - 1) as u8
            } else if (x - levels[i - 1]).abs() <= (levels[i] - x).abs() {
                (i - 1) as u8
            } else {
                i as u8
            }
        }
    }
}

/// The TurboQuant MSE codec: rotation + scaled Lloyd–Max levels. `encode→decode` reconstructs `x` up to the
/// near-optimal distortion `D_mse ≤ (√3π/2)·4⁻ᵇ`. (The unbiased `prod`/QJL mode is a future addition.)
pub struct Codec {
    pub bits: u8,
    levels: Vec<f32>,
    rot: Rotation,
    d: usize,
}

impl Codec {
    pub fn new(bits: u8, seed: u64, d: usize) -> Codec {
        let rot = Rotation::new(seed, d);
        let sigma = 1.0 / (rot.dpad as f32).sqrt(); // post-rotation per-coord std for a unit vector
        let levels = lloyd_max_gaussian(bits).iter().map(|l| l * sigma).collect();
        Codec { bits, levels, rot, d }
    }

    /// Encode `x` (len d) → (`bits`-bit codes over dpad coords, scale = ‖x‖).
    pub fn encode(&self, x: &[f32]) -> (Vec<u8>, f32) {
        let scale = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        let u: Vec<f32> = x.iter().map(|v| v / scale).collect();
        let r = self.rot.apply(&u);
        let codes = r.iter().map(|&ri| nearest_level(ri, &self.levels)).collect();
        (codes, scale)
    }

    /// Decode → `x̂` (len d), with `E[x̂] ≈ x` and `‖x̂ − x‖/‖x‖` bounded by the distortion rate.
    pub fn decode(&self, codes: &[u8], scale: f32) -> Vec<f32> {
        let rhat: Vec<f32> = codes.iter().map(|&c| self.levels[c as usize]).collect();
        let uhat = self.rot.invert(&rhat, self.d);
        uhat.iter().map(|v| v * scale).collect()
    }

    /// Convenience: round-trip `x` and return `x̂`.
    pub fn roundtrip(&self, x: &[f32]) -> Vec<f32> {
        let (c, s) = self.encode(x);
        self.decode(&c, s)
    }
}

/// Per-coordinate RMS distortion `ρ(b,d) ≈ √(√3π/2 / d)·2⁻ᵇ` — the *isotropic* facet-displacement scale a
/// `b`-bit codec induces. A decision with normalized facet margin `m > z·ρ` is stable under TurboQuant.
pub fn rho(bits: u8, d: usize) -> f32 {
    ((3f32.sqrt() * PI / 2.0) / d as f32).sqrt() * 2f32.powi(-(bits as i32))
}

/// Inner-product distortion variance `D_prod = (√3·π²·‖y‖²/d)·4⁻ᵇ` (the logit-error variance for a row `y`).
pub fn d_prod(bits: u8, d: usize, ynorm2: f32) -> f32 {
    3f32.sqrt() * PI * PI * ynorm2 / d as f32 * 4f32.powi(-(bits as i32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn fwht_twice_is_n_identity() {
        let mut a = [1.0f32, 2.0, -3.0, 0.5];
        let orig = a;
        fwht(&mut a);
        fwht(&mut a);
        for (x, o) in a.iter().zip(orig) {
            assert!((x - 4.0 * o).abs() < 1e-4, "H² should be n·I");
        }
    }

    #[test]
    fn rotation_is_orthogonal_and_invertible() {
        let rot = Rotation::new(0xABCD, 6); // dpad = 8
        let x = [0.3f32, -1.2, 0.7, 2.1, -0.4, 0.9];
        let r = rot.apply(&x);
        assert!((norm(&r) - norm(&x)).abs() < 1e-4, "rotation preserves norm");
        let back = rot.invert(&r, x.len());
        for (a, b) in back.iter().zip(x) {
            assert!((a - b).abs() < 1e-4, "invert∘apply = identity");
        }
    }

    #[test]
    fn rotation_isotropizes_a_spike() {
        // e_0 (all energy in one coord) → rotated mass spread roughly evenly (|coord| ≈ 1/√dpad each).
        let rot = Rotation::new(7, 16);
        let mut e0 = vec![0f32; 16];
        e0[0] = 1.0;
        let r = rot.apply(&e0);
        let maxabs = r.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!(maxabs < 0.5, "spike should spread (max |coord| {maxabs} ≪ 1)");
        assert!((norm(&r) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn lloyd_levels_symmetric_and_monotone() {
        let lv = lloyd_max_gaussian(4); // 16 levels
        assert_eq!(lv.len(), 16);
        for w in lv.windows(2) {
            assert!(w[1] > w[0], "levels strictly increasing");
        }
        // symmetric about 0: level[k] ≈ −level[n−1−k]
        let n = lv.len();
        for k in 0..n {
            assert!((lv[k] + lv[n - 1 - k]).abs() < 1e-3, "levels symmetric");
        }
    }

    #[test]
    fn codec_roundtrip_error_falls_with_bits() {
        // a deterministic unit-ish vector in d=64; relative MSE must drop ~4× per bit and stay under the
        // near-optimality bound (√3π/2)·4^-b with slack for the finite Lloyd grid.
        let d = 64;
        let x: Vec<f32> = (0..d).map(|i| ((i as f32 * 0.37).sin() + 0.1 * i as f32).rem_euclid(3.0) - 1.0).collect();
        let mut prev = f32::INFINITY;
        for bits in [2u8, 4, 6] {
            let c = Codec::new(bits, 0x1234, d);
            let xh = c.roundtrip(&x);
            let err2: f32 = x.iter().zip(&xh).map(|(a, b)| (a - b) * (a - b)).sum();
            let rel = err2 / x.iter().map(|v| v * v).sum::<f32>();
            let bound = 3f32.sqrt() * PI / 2.0 * 4f32.powi(-(bits as i32));
            assert!(rel < 3.0 * bound, "bits={bits}: rel MSE {rel} exceeds 3×bound {bound}");
            assert!(rel < prev, "more bits must reduce distortion");
            prev = rel;
        }
    }

    #[test]
    fn roundtrip_nonpow2_large_d() {
        // robustness: d not a power of two (300 → dpad 512), higher bits — len preserved, bound holds.
        let d = 300;
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.91).cos() * 1.7 + 0.05 * (i as f32 - 150.0)).collect();
        let denom: f32 = x.iter().map(|v| v * v).sum();
        for bits in [4u8, 6, 8] {
            let c = Codec::new(bits, 0xDEAD_BEEF, d);
            let xh = c.roundtrip(&x);
            assert_eq!(xh.len(), d, "decode preserves length for non-pow2 d");
            let err2: f32 = x.iter().zip(&xh).map(|(a, b)| (a - b) * (a - b)).sum();
            let rel = err2 / denom;
            let bound = 3f32.sqrt() * PI / 2.0 * 4f32.powi(-(bits as i32));
            assert!(rel < 3.0 * bound, "d={d} bits={bits}: rel {rel} > 3×bound {bound}");
        }
    }

    #[test]
    fn rho_and_dprod_scaling() {
        assert!((rho(5, 100) / rho(6, 100) - 2.0).abs() < 1e-3, "ρ halves per bit");
        assert!((d_prod(5, 100, 1.0) / d_prod(6, 100, 1.0) - 4.0).abs() < 1e-3, "D_prod quarters per bit");
        assert!((d_prod(4, 100, 1.0) / d_prod(4, 400, 1.0) - 4.0).abs() < 1e-3, "D_prod ∝ 1/d");
    }
}
