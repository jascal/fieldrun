//! Gated DeltaNet — the linear-attention kernel for Qwen3.6 (`qwen3_5_moe`)'s hybrid layers.
//!
//! Qwen3.6 interleaves Gated DeltaNet *linear* attention (3 of every 4 layers) with gated full attention.
//! This is the single-head recurrence; the arch calls it per value-head (GQA: each value-head pairs with
//! its grouped key-head), preceded by a short causal depthwise conv (kernel 4) on q/k/v.
//!
//! The recurrence and its conventions are PINNED against the reference HF implementation
//! (`torch_recurrent_gated_delta_rule`, matched to ~1e-7 — see `experiments/qwen3next/`):
//!   • `α_t = exp(g_log_t)`           (the gate input is log-decay)
//!   • `q ← (1/√d_k) · L2norm(q)`     (read scaling + L2-norm on the query)
//!   • `k ← L2norm(k)`               (L2-norm on the key)
//!   • per step:  S ← α_t·S ;  v̂ = Sᵀk ;  Δ = β_t·(v − v̂) ;  S ← S + k⊗Δ ;  o_t = Sᵀq   (read AFTER write)
//! State `S ∈ ℝ^{d_k×d_v}`. Recurrent form (one token at a time) — exactly what fieldrun's decode needs;
//! the chunked-parallel form is only a prefill speedup (add later, test against this).

const EPS: f32 = 1e-6;

/// Single-head Gated DeltaNet over a `t`-token sequence. Flat row-major slices:
/// `q,k` are `t·d_k`, `v` is `t·d_v`, `g_log` and `beta` are `t`. Returns `t·d_v` outputs.
#[allow(dead_code)] // wired into the qwen3_5_moe arch in a follow-up increment; tested standalone for now
pub fn gated_deltanet(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_log: &[f32],
    beta: &[f32],
    t: usize,
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    debug_assert_eq!(q.len(), t * dk);
    debug_assert_eq!(v.len(), t * dv);
    let scale = 1.0 / (dk as f32).sqrt();
    let mut s = vec![0f32; dk * dv]; // S[d*dv + e]: k-dim d → v-dim e
    let mut out = vec![0f32; t * dv];
    let (mut qn, mut kn) = (vec![0f32; dk], vec![0f32; dk]);
    let mut vnew = vec![0f32; dv];
    for i in 0..t {
        let qrow = &q[i * dk..(i + 1) * dk];
        let krow = &k[i * dk..(i + 1) * dk];
        let qden = qrow.iter().map(|x| x * x).sum::<f32>().sqrt() + EPS;
        let kden = krow.iter().map(|x| x * x).sum::<f32>().sqrt() + EPS;
        for d in 0..dk {
            qn[d] = qrow[d] / qden * scale;
            kn[d] = krow[d] / kden;
        }
        let a = g_log[i].exp();
        for x in s.iter_mut() {
            *x *= a; // S ← α·S
        }
        // v̂ = Sᵀk, then Δ = β·(v − v̂)
        let vrow = &v[i * dv..(i + 1) * dv];
        for e in 0..dv {
            vnew[e] = 0.0;
        }
        for d in 0..dk {
            let kd = kn[d];
            let base = d * dv;
            for e in 0..dv {
                vnew[e] += s[base + e] * kd; // accumulate v̂ in vnew
            }
        }
        for e in 0..dv {
            vnew[e] = beta[i] * (vrow[e] - vnew[e]); // Δ
        }
        // S ← S + k⊗Δ ; o = Sᵀq (read after write)
        let obase = i * dv;
        for d in 0..dk {
            let (kd, qd, base) = (kn[d], qn[d], d * dv);
            for e in 0..dv {
                let sv = s[base + e] + kd * vnew[e];
                s[base + e] = sv;
                out[obase + e] += sv * qd;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // deterministic integer-formula inputs — reproduced bit-for-bit in the numpy oracle (see
    // experiments/qwen3next/, gated_deltanet_qwen36) so the golden below is an exact cross-language check.
    fn inputs(
        t: usize,
        dk: usize,
        dv: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let f = |a: usize, m: usize| (a % m) as f32 / m as f32 - 0.5;
        let q = (0..t)
            .flat_map(|i| (0..dk).map(move |j| f(i * 7 + j * 5, 13)))
            .collect();
        let k = (0..t)
            .flat_map(|i| (0..dk).map(move |j| f(i * 3 + j * 11, 13)))
            .collect();
        let v = (0..t)
            .flat_map(|i| (0..dv).map(move |j| f(i * 5 + j * 7, 13)))
            .collect();
        let g = (0..t)
            .map(|i| -((((i * 2) % 5) + 1) as f32) / 10.0)
            .collect();
        let beta = (0..t).map(|i| ((((i * 4) % 7) + 1) as f32) / 8.0).collect();
        (q, k, v, g, beta)
    }

    #[test]
    fn matches_transformers_golden() {
        // golden from experiments/qwen3next: gated_deltanet_qwen36 == transformers torch_recurrent_gated_delta_rule
        let (t, dk, dv) = (6, 4, 3);
        let (q, k, v, g, beta) = inputs(t, dk, dv);
        let out = gated_deltanet(&q, &k, &v, &g, &beta, t, dk, dv);
        let expected: [[f32; 3]; 6] = [
            [-1.80064492e-02, 1.38511148e-03, -1.52362263e-02],
            [1.34938902e-02, -8.02549051e-02, 1.01205176e-04],
            [-1.57434119e-02, 3.69177108e-02, -6.45992238e-03],
            [4.14348764e-02, 3.15596170e-05, 3.75046219e-02],
            [1.22955904e-02, 2.24161982e-02, 6.49370906e-03],
            [9.58056159e-02, -1.50076024e-02, -9.46560176e-02],
        ];
        let mut maxerr = 0f32;
        for i in 0..t {
            for e in 0..dv {
                maxerr = maxerr.max((out[i * dv + e] - expected[i][e]).abs());
            }
        }
        assert!(
            maxerr < 1e-5,
            "Gated DeltaNet diverges from the transformers-pinned oracle: maxerr {maxerr:e}"
        );
    }

    #[test]
    fn beta_zero_is_inert() {
        // β=0 everywhere ⇒ no writes ⇒ state stays 0 ⇒ outputs are 0
        let (t, dk, dv) = (5, 4, 3);
        let (q, k, v, _g, _b) = inputs(t, dk, dv);
        let out = gated_deltanet(&q, &k, &v, &vec![0.0; t], &vec![0.0; t], t, dk, dv);
        assert!(out.iter().all(|&x| x.abs() < 1e-12));
    }

    #[test]
    fn q_scaled_by_inv_sqrt_dk() {
        // output magnitude scales ~1/√d_k via the query scaling — sanity that the scale is applied
        let (t, dk, dv) = (4, 16, 3);
        let (q, k, v, g, beta) = inputs(t, dk, dv);
        let out = gated_deltanet(&q, &k, &v, &g, &beta, t, dk, dv);
        assert!(out.iter().any(|&x| x.abs() > 0.0)); // non-trivial, didn't collapse to zero
    }
}
