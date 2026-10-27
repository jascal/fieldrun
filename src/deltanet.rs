//! Gated DeltaNet — the linear-attention kernel for Qwen3.6's hybrid layers.
//! (`qwen3_5_moe` is the HF/transformers `model_type` for the Qwen3.6 / Qwen3-Next MoE family; the two
//! names refer to the same architecture throughout this crate.)
//!
//! Qwen3.6 interleaves Gated DeltaNet *linear* attention (3 of every 4 layers) with gated full attention.
//! This is the single-head recurrence; the arch calls it per value-head (GQA: each value-head pairs with
//! its grouped key-head), preceded by a short causal depthwise conv (kernel 4) on q/k/v.
//!
//! The recurrence and its conventions are PINNED against the reference HF implementation
//! (`torch_recurrent_gated_delta_rule`; L2norm is x/sqrt(sum(x*x)+1e-6)).
//! The discriminating low-norm pin is `experiments/pdc_qwen35_replay_engineering/normalization_golden.json`:
//!   • `α_t = exp(g_log_t)`           (the gate input is log-decay)
//!   • `q ← (1/√d_k) · L2norm(q)`     (read scaling + L2-norm on the query)
//!   • `k ← L2norm(k)`               (L2-norm on the key)
//!   • per step:  S ← α_t·S ;  v̂ = Sᵀk ;  Δ = β_t·(v − v̂) ;  S ← S + k⊗Δ ;  o_t = Sᵀq   (read AFTER write)
//! State `S ∈ ℝ^{d_k×d_v}`. Recurrent form (one token at a time) — exactly what fieldrun's decode needs;
//! the chunked-parallel form is only a prefill speedup (add later, test against this).

/// Squared-norm regularizer, matching HF `x * rsqrt(sum(x*x) + eps)`.
/// Epsilon belongs inside the square root; zero q/k rows remain zero.
const EPS: f32 = 1e-6;

/// SiLU / swish: `z·σ(z)`. f32-graceful at ±inf.
#[inline]
fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

/// Numerically-stable softplus `ln(1+eᶻ)` (= `logaddexp(0, z)`).
#[inline]
fn stable_softplus(z: f32) -> f32 {
    if z > 0.0 {
        z + (1.0 + (-z).exp()).ln()
    } else {
        (1.0 + z.exp()).ln()
    }
}

/// Single-head Gated DeltaNet over a `t`-token sequence. Flat row-major slices:
/// `q,k` are `t·d_k`, `v` is `t·d_v`, `g_log` and `beta` are `t`. Returns `t·d_v` outputs.
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
    gated_deltanet_state(q, k, v, g_log, beta, t, dk, dv, &mut vec![0f32; dk * dv])
}

/// [`gated_deltanet`] with CALLER-OWNED state — the decode kernel. `s` is `dk·dv` row-major
/// (`s[d·dv + e]`), read in and written out, so one token per call carries the recurrence forward.
/// Starting from a zeroed `s` and feeding the whole sequence in one call is the same arithmetic in the
/// same order, so a decode split is bit-identical to the full-sequence form (the `t=1` step applies the
/// identical `α`-decay → `v̂` → `Δ` → write → read sequence per token).
pub fn gated_deltanet_state(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_log: &[f32],
    beta: &[f32],
    t: usize,
    dk: usize,
    dv: usize,
    s: &mut [f32],
) -> Vec<f32> {
    debug_assert_eq!(q.len(), t * dk);
    debug_assert_eq!(v.len(), t * dv);
    debug_assert_eq!(s.len(), dk * dv);
    let scale = 1.0 / (dk as f32).sqrt();
    let s = &mut *s; // S[d*dv + e]: k-dim d → v-dim e
    let mut out = vec![0f32; t * dv];
    let (mut qn, mut kn) = (vec![0f32; dk], vec![0f32; dk]);
    let mut vnew = vec![0f32; dv];
    for i in 0..t {
        let qrow = &q[i * dk..(i + 1) * dk];
        let krow = &k[i * dk..(i + 1) * dk];
        let qden = (qrow.iter().map(|x| x * x).sum::<f32>() + EPS).sqrt();
        let kden = (krow.iter().map(|x| x * x).sum::<f32>() + EPS).sqrt();
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

/// Short causal depthwise conv1d + SiLU on the concatenated `[q,k,v]` channels (Qwen3.6's linear layer
/// prelude). `x` is row-major `t·conv_dim`; `weight` is `conv_dim·k` (per-channel kernel), `bias` is
/// `conv_dim`. Causal: out[t,c] depends on x[t-(k-1)..t, c] (left-pad zeros). Matches `nn.Conv1d(groups=
/// conv_dim, padding=k-1)[:, :, :t]` then SiLU — verified against torch (test `conv_matches_torch_golden`).
pub fn causal_conv1d_silu(x: &[f32], weight: &[f32], bias: &[f32], t: usize, conv_dim: usize,
                          k: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), t * conv_dim);
    let mut out = vec![0f32; t * conv_dim];
    for ti in 0..t {
        for c in 0..conv_dim {
            let mut acc = bias[c];
            for j in 0..k {
                let src = ti as isize - (k as isize - 1) + j as isize; // causal: left-pad zeros
                if src >= 0 {
                    acc += x[src as usize * conv_dim + c] * weight[c * k + j];
                }
            }
            out[ti * conv_dim + c] = silu(acc);
        }
    }
    out
}

/// [`causal_conv1d_silu`] with a rolling window of the previous `k-1` input rows, for incremental decode.
///
/// `hist` holds the most recent `(k-1)·conv_dim` input values (row-major, oldest first) and is updated
/// in place to the last `k-1` rows of `x` afterwards. The conv is causal with left zero-padding, so an
/// output row depends only on `x[t-(k-1)..t]` — prepending the retained history reproduces the
/// full-sequence window exactly, in the same accumulation order, hence bit-identical. A short first call
/// (`t < k-1`, e.g. a 3-token prompt at `k=4`) zero-pads the remainder of the window, matching the
/// full-sequence left-pad.
pub fn causal_conv1d_silu_hist(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    t: usize,
    conv_dim: usize,
    k: usize,
    hist: &mut Vec<f32>,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), t * conv_dim);
    let keep_rows = k - 1;
    let keep = keep_rows * conv_dim; // retained history, in VALUES
    if t >= keep_rows {
        // every output window lies entirely inside `x` — no history needed
        let out = causal_conv1d_silu(x, weight, bias, t, conv_dim, k);
        hist.clear();
        hist.extend_from_slice(&x[x.len() - keep..]);
        return out;
    }
    // prepend the retained window (zero-filled on the first call), then drop the rows we re-computed
    let mut buf = vec![0f32; (keep_rows + t) * conv_dim];
    let have = (hist.len() / conv_dim).min(keep_rows) * conv_dim;
    buf[keep - have..keep].copy_from_slice(&hist[hist.len() - have..]);
    buf[keep..].copy_from_slice(x);
    let out = causal_conv1d_silu(&buf, weight, bias, keep_rows + t, conv_dim, k);
    hist.clear();
    hist.extend_from_slice(&buf[buf.len() - keep..]);
    out[keep..].to_vec()
}

/// Multi-head GQA Gated DeltaNet: runs `gated_deltanet` per value-head, each paired with its grouped key-
/// head (transformers `repeat_interleave(num_v/num_k)`). `q,k` are `t·(n_k_heads·hk)`, `v` is
/// `t·(n_v_heads·hv)`, `g,beta` are `t·n_v_heads` (per value-head). Returns `t·(n_v_heads·hv)`.
pub fn gated_deltanet_mha(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], t: usize,
                          n_k_heads: usize, n_v_heads: usize, hk: usize, hv: usize) -> Vec<f32> {
    let mut s: Vec<Vec<f32>> = (0..n_v_heads).map(|_| vec![0f32; hk * hv]).collect();
    gated_deltanet_mha_state(q, k, v, g, beta, t, n_k_heads, n_v_heads, hk, hv, &mut s)
}

/// [`gated_deltanet_mha`] with per-value-head CALLER-OWNED state `s` (one `hk·hv` matrix per value head,
/// advanced in place). Each value head is an independent recurrence over the same inputs, so splitting
/// the sequence across calls changes nothing but where the loop boundary sits — bit-identical output.
pub fn gated_deltanet_mha_state(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], t: usize,
                                n_k_heads: usize, n_v_heads: usize, hk: usize, hv: usize,
                                s: &mut [Vec<f32>]) -> Vec<f32> {
    let r = n_v_heads / n_k_heads; // value-heads per key-head (repeat_interleave factor)
    debug_assert_eq!(s.len(), n_v_heads);
    let mut out = vec![0f32; t * n_v_heads * hv];
    let (mut qh, mut kh) = (vec![0f32; t * hk], vec![0f32; t * hk]);
    let (mut vh, mut gh, mut bh) = (vec![0f32; t * hv], vec![0f32; t], vec![0f32; t]);
    for vhead in 0..n_v_heads {
        let khead = vhead / r; // GQA: value-head vhead reads key-head vhead//r
        for ti in 0..t {
            let qbase = ti * n_k_heads * hk + khead * hk;
            let vbase = ti * n_v_heads * hv + vhead * hv;
            qh[ti * hk..ti * hk + hk].copy_from_slice(&q[qbase..qbase + hk]);
            kh[ti * hk..ti * hk + hk].copy_from_slice(&k[qbase..qbase + hk]);
            vh[ti * hv..ti * hv + hv].copy_from_slice(&v[vbase..vbase + hv]);
            gh[ti] = g[ti * n_v_heads + vhead];
            bh[ti] = beta[ti * n_v_heads + vhead];
        }
        let oh = gated_deltanet_state(&qh, &kh, &vh, &gh, &bh, t, hk, hv, &mut s[vhead]);
        for ti in 0..t {
            let ob = ti * n_v_heads * hv + vhead * hv;
            out[ob..ob + hv].copy_from_slice(&oh[ti * hv..ti * hv + hv]);
        }
    }
    out
}

/// Gated RMSNorm — Qwen3.6's linear-layer output gate (`Qwen3_5MoeRMSNormGated`). Per row of size `d`
/// (a value-head's `head_v_dim`): `out = weight · (x · rsqrt(mean(x²)+eps)) · silu(gate)`. Note the order
/// — RMS-normalize FIRST, then apply the SiLU gate. `weight` is `d`; `x`/`gate` are row-major `rows·d`.
pub fn rmsnorm_gated(x: &[f32], weight: &[f32], gate: &[f32], rows: usize, d: usize,
                     eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * d];
    for r in 0..rows {
        let base = r * d;
        let ms = (0..d).map(|i| x[base + i] * x[base + i]).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..d {
            let z = gate[base + i];
            out[base + i] = weight[i] * (x[base + i] * inv) * silu(z);
        }
    }
    out
}

/// The DeltaNet decay gate: `g = −exp(a_log)·softplus(a + dt_bias)`, per value-head — produces the
/// `g_log` the kernel consumes (`α = exp(g)`). `a` is `t·n_v_heads` (the dt projection); `a_log`,
/// `dt_bias` are `n_v_heads` per-head params. softplus is the numerically-stable `ln(1+eᶻ)`.
pub fn delta_gate(a: &[f32], a_log: &[f32], dt_bias: &[f32], t: usize, n_v_heads: usize) -> Vec<f32> {
    let mut g = vec![0f32; t * n_v_heads];
    for i in 0..t {
        for vh in 0..n_v_heads {
            let z = a[i * n_v_heads + vh] + dt_bias[vh];
            g[i * n_v_heads + vh] = -a_log[vh].exp() * stable_softplus(z);
        }
    }
    g
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
        // Historical O(1) reference: this loose tolerance alone cannot distinguish epsilon placement.
        // low_norm_qk_matches_transformers_regularized_squared_norm below is the binding normalization pin.
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
    fn low_norm_qk_matches_transformers_regularized_squared_norm() {
        // External official-kernel fixture; tiny q/k exposes where epsilon belongs.
        let f: serde_json::Value = serde_json::from_str(include_str!(
            "../experiments/pdc_qwen35_replay_engineering/normalization_golden.json")).unwrap();
        let matrix = |key: &str| -> Vec<f32> { f[key].as_array().unwrap().iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32)).collect() };
        let vector = |key: &str| -> Vec<f32> { f[key].as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap() as f32).collect() };
        let got = gated_deltanet(&matrix("q"), &matrix("k"), &matrix("v"),
            &vector("g"), &vector("beta"), 3, 4, 2);
        let expected = vector("expected");
        let error = got.iter().zip(expected.iter()).map(|(a,b)|(a-b).abs()).fold(0.0f32, f32::max);
        assert!(error < 1e-7, "official low-norm DeltaNet golden error={error:e}");
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

    #[test]
    fn single_token_decode_closed_form() {
        // t=1 is THE decode step. With S₀=0: v̂=0, Δ=β·v, S=k̂⊗Δ, o = (k̂·q̂)·β·v exactly.
        let (dk, dv) = (8, 4);
        let (q, k, v, _g, beta) = inputs(1, dk, dv);
        let out = gated_deltanet(&q, &k, &v, &[(-0.3f32)], &beta, 1, dk, dv);
        // recompute q̂ (l2norm·1/√dk) and k̂ (l2norm) to form the closed form
        let qd = (q.iter().map(|x| x * x).sum::<f32>() + EPS).sqrt();
        let kd = (k.iter().map(|x| x * x).sum::<f32>() + EPS).sqrt();
        let scale = 1.0 / (dk as f32).sqrt();
        let kq: f32 = (0..dk).map(|d| (k[d] / kd) * (q[d] / qd * scale)).sum();
        for e in 0..dv {
            let want = kq * beta[0] * v[e];
            assert!((out[e] - want).abs() < 1e-6, "t=1 closed form mismatch at {e}");
        }
    }

    #[test]
    fn full_decay_is_memoryless() {
        // g_log → −∞ (α→0): the decayed state vanishes each step, so every output equals its own
        // single-step closed form (no carry-over) — stresses the decay path + read-after-write order.
        let (t, dk, dv) = (5, 6, 3);
        let (q, k, v, _g, beta) = inputs(t, dk, dv);
        let out = gated_deltanet(&q, &k, &v, &vec![-40.0; t], &beta, t, dk, dv);
        let scale = 1.0 / (dk as f32).sqrt();
        for i in 0..t {
            let (qr, kr, vr) = (&q[i * dk..(i + 1) * dk], &k[i * dk..(i + 1) * dk], &v[i * dv..(i + 1) * dv]);
            let qd = (qr.iter().map(|x| x * x).sum::<f32>() + EPS).sqrt();
            let kd = (kr.iter().map(|x| x * x).sum::<f32>() + EPS).sqrt();
            let kq: f32 = (0..dk).map(|d| (kr[d] / kd) * (qr[d] / qd * scale)).sum();
            for e in 0..dv {
                assert!((out[i * dv + e] - kq * beta[i] * vr[e]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn conv_matches_torch_golden() {
        // golden from torch nn.Conv1d(groups=conv_dim, padding=k-1)[:, :, :t] + SiLU (experiments/qwen3next)
        let (t, cd, k) = (5, 3, 4);
        let g = |a: usize, m: usize| (a % m) as f32 / m as f32 - 0.5;
        let x: Vec<f32> = (0..t).flat_map(|ti| (0..cd).map(move |c| g(c * 3 + ti * 5, 13))).collect();
        let w: Vec<f32> = (0..cd).flat_map(|c| (0..k).map(move |j| g(c * 2 + j * 7, 11))).collect();
        let b: Vec<f32> = (0..cd).map(|c| g(c * 5, 7)).collect();
        let out = causal_conv1d_silu(&x, &w, &b, t, cd, k);
        let expected: [[f32; 3]; 5] = [
            [-2.33067304e-01, 1.88297376e-01, -3.03615537e-02],
            [-1.70510843e-01, 9.76778418e-02, -7.16514513e-02],
            [-1.70003459e-01, 1.90605924e-01, 2.88861338e-02],
            [-1.80367693e-01, 2.44066134e-01, -1.28440201e-01],
            [-1.31578267e-01, -4.13411260e-02, 7.79500380e-02],
        ];
        let mut e = 0f32;
        for ti in 0..t {
            for c in 0..cd {
                e = e.max((out[ti * cd + c] - expected[ti][c]).abs());
            }
        }
        assert!(e < 1e-5, "causal conv1d+silu diverges from torch: maxerr {e:e}");
    }

    #[test]
    fn mha_matches_oracle_golden() {
        // golden: per value-head gated_deltanet_qwen36 with GQA repeat_interleave (experiments/qwen3next)
        let (t, nk, nv, hk, hv) = (4, 2, 4, 3, 2);
        let f = |a: usize, m: usize| (a % m) as f32 / m as f32 - 0.5;
        let q: Vec<f32> = (0..t).flat_map(|i| (0..nk * hk).map(move |j| f(i * 7 + j * 5, 13))).collect();
        let k: Vec<f32> = (0..t).flat_map(|i| (0..nk * hk).map(move |j| f(i * 3 + j * 11, 13))).collect();
        let v: Vec<f32> = (0..t).flat_map(|i| (0..nv * hv).map(move |j| f(i * 5 + j * 7, 13))).collect();
        let g: Vec<f32> = (0..t)
            .flat_map(|i| (0..nv).map(move |vh| -((((i * 2 + vh) % 5) + 1) as f32) / 10.0))
            .collect();
        let beta: Vec<f32> = (0..t)
            .flat_map(|i| (0..nv).map(move |vh| ((((i * 4 + vh) % 7) + 1) as f32) / 8.0))
            .collect();
        let out = gated_deltanet_mha(&q, &k, &v, &g, &beta, t, nk, nv, hk, hv);
        let expected: [[f32; 8]; 4] = [
            [-2.55630077e-02, 1.96638521e-03, -4.32604745e-02, 1.17983112e-02, 6.09471667e-02, -3.38595371e-02, 6.32044692e-02, -6.32044692e-02],
            [2.62730114e-02, -1.34264722e-01, -2.19625037e-03, 1.96182288e-01, -2.12298473e-02, 5.25713204e-03, -3.36882808e-02, 3.31517910e-02],
            [-4.62090139e-02, 8.21297941e-02, -6.60442062e-02, -1.26936124e-01, -3.81816041e-03, 4.33765223e-02, -7.71871087e-02, 4.38512297e-02],
            [-2.63870312e-03, 3.43525633e-02, 2.61399669e-02, -7.29618900e-02, -7.80981770e-02, 1.21141041e-02, 9.21856499e-02, 1.19378679e-02],
        ];
        let mut e = 0f32;
        for i in 0..t {
            for j in 0..nv * hv {
                e = e.max((out[i * nv * hv + j] - expected[i][j]).abs());
            }
        }
        assert!(e < 1e-5, "multi-head GQA DeltaNet diverges from the oracle: maxerr {e:e}");
    }

    #[test]
    fn rmsnorm_gated_matches_torch_golden() {
        // golden from the exact Qwen3_5MoeRMSNormGated torch ops (experiments/qwen3next)
        let (rows, d) = (3, 4);
        let f = |a: usize, m: usize| (a % m) as f32 / m as f32 - 0.5;
        let x: Vec<f32> = (0..rows).flat_map(|r| (0..d).map(move |i| f(r * 5 + i * 3, 13))).collect();
        let w: Vec<f32> = (0..d).map(|i| 0.5 + f(i * 7, 11)).collect();
        let gate: Vec<f32> = (0..rows).flat_map(|r| (0..d).map(move |i| f(r * 3 + i * 5, 13))).collect();
        let out = rmsnorm_gated(&x, &w, &gate, rows, d, 1e-6);
        let expected: [[f32; 4]; 3] = [
            [0.0, 3.10081746e-02, -5.32961125e-03, -8.34661350e-02],
            [0.0, 1.57070104e-02, -6.24770932e-02, 7.33087957e-02],
            [0.0, -2.04900041e-01, 2.71954276e-02, -6.77670771e-03],
        ];
        let mut e = 0f32;
        for r in 0..rows {
            for i in 0..d {
                e = e.max((out[r * d + i] - expected[r][i]).abs());
            }
        }
        assert!(e < 1e-5, "gated RMSNorm diverges from torch: maxerr {e:e}");
    }

    #[test]
    fn delta_gate_matches_golden() {
        // golden: g = -exp(a_log)·softplus(a + dt_bias) (experiments/qwen3next)
        let (t, nv) = (3, 4);
        let f = |a: usize, m: usize| (a % m) as f32 / m as f32 - 0.5;
        let a: Vec<f32> = (0..t).flat_map(|i| (0..nv).map(move |vh| f(i * 5 + vh * 3, 13) * 2.0)).collect();
        let a_log: Vec<f32> = (0..nv).map(|vh| f(vh * 7, 11) - 0.5).collect();
        let dt_bias: Vec<f32> = (0..nv).map(|vh| f(vh * 3, 7)).collect();
        let g = delta_gate(&a, &a_log, &dt_bias, t, nv);
        let expected: [[f32; 4]; 3] = [
            [-7.40958041e-02, -3.01688733e-01, -4.07378321e-01, -7.13984646e-01],
            [-1.44606430e-01, -5.39423235e-01, -6.52185451e-01, -2.71522122e-01],
            [-2.62137223e-01, -2.04752970e-01, -2.93120362e-01, -5.09606860e-01],
        ];
        let mut e = 0f32;
        for i in 0..t {
            for vh in 0..nv {
                e = e.max((g[i * nv + vh] - expected[i][vh]).abs());
            }
        }
        assert!(e < 1e-6, "delta gate diverges from golden: maxerr {e:e}");
    }

    // --- incremental-decode equivalence -------------------------------------------------
    // The hybrid KV cache is only allowed to exist if feeding a sequence one token at a time through
    // the stateful kernels gives BYTE-IDENTICAL results to one full-sequence call. These pin that at
    // the kernel level, before the arch wires them together.

    /// Feeding `t` tokens one at a time through `gated_deltanet_state` must reproduce the single
    /// full-sequence call exactly — not to a tolerance, but bit-for-bit (`to_bits`).
    #[test]
    fn state_split_is_bit_identical() {
        let (t, dk, dv) = (7, 6, 4);
        let (q, k, v, g, beta) = inputs(t, dk, dv);
        let full = gated_deltanet(&q, &k, &v, &g, &beta, t, dk, dv);
        let mut s = vec![0f32; dk * dv];
        let mut split = Vec::with_capacity(t * dv);
        for i in 0..t {
            split.extend_from_slice(&gated_deltanet_state(
                &q[i * dk..(i + 1) * dk],
                &k[i * dk..(i + 1) * dk],
                &v[i * dv..(i + 1) * dv],
                &g[i..i + 1],
                &beta[i..i + 1],
                1,
                dk,
                dv,
                &mut s,
            ));
        }
        assert_eq!(full.len(), split.len());
        for i in 0..full.len() {
            assert_eq!(
                full[i].to_bits(),
                split[i].to_bits(),
                "token-split DeltaNet diverged at {i}: {:e} vs {:e}",
                full[i],
                split[i]
            );
        }
    }

    /// Same, through the multi-head GQA wrapper with one state matrix per value head.
    #[test]
    fn mha_state_split_is_bit_identical() {
        let (t, nk, nv, hk, hv) = (5, 2, 4, 3, 2);
        let f = |a: usize, m: usize| (a % m) as f32 / m as f32 - 0.5;
        let q: Vec<f32> = (0..t).flat_map(|i| (0..nk * hk).map(move |j| f(i * 7 + j * 5, 13))).collect();
        let k: Vec<f32> = (0..t).flat_map(|i| (0..nk * hk).map(move |j| f(i * 3 + j * 11, 13))).collect();
        let v: Vec<f32> = (0..t).flat_map(|i| (0..nv * hv).map(move |j| f(i * 5 + j * 7, 13))).collect();
        let g: Vec<f32> = (0..t)
            .flat_map(|i| (0..nv).map(move |vh| -((((i * 2 + vh) % 5) + 1) as f32) / 10.0))
            .collect();
        let beta: Vec<f32> = (0..t)
            .flat_map(|i| (0..nv).map(move |vh| ((((i * 4 + vh) % 7) + 1) as f32) / 8.0))
            .collect();
        let full = gated_deltanet_mha(&q, &k, &v, &g, &beta, t, nk, nv, hk, hv);
        let mut s: Vec<Vec<f32>> = (0..nv).map(|_| vec![0f32; hk * hv]).collect();
        let mut split = Vec::new();
        for i in 0..t {
            split.extend_from_slice(&gated_deltanet_mha_state(
                &q[i * nk * hk..(i + 1) * nk * hk],
                &k[i * nk * hk..(i + 1) * nk * hk],
                &v[i * nv * hv..(i + 1) * nv * hv],
                &g[i * nv..(i + 1) * nv],
                &beta[i * nv..(i + 1) * nv],
                1,
                nk,
                nv,
                hk,
                hv,
                &mut s,
            ));
        }
        assert_eq!(full.len(), split.len());
        for i in 0..full.len() {
            assert_eq!(full[i].to_bits(), split[i].to_bits(), "MHA split diverged at {i}");
        }
    }

    /// The conv rolling window must reproduce the full-sequence conv bit-for-bit, INCLUDING a first
    /// call shorter than the kernel (t < k-1), where the full-sequence form zero-pads on the left.
    #[test]
    fn conv_hist_split_is_bit_identical() {
        let (cd, k) = (3, 4);
        let f = |a: usize, m: usize| (a % m) as f32 / m as f32 - 0.5;
        let w: Vec<f32> = (0..cd).flat_map(|c| (0..k).map(move |j| f(c * 2 + j * 7, 11))).collect();
        let b: Vec<f32> = (0..cd).map(|c| f(c * 5, 7)).collect();
        // two cases: a short first block (2 < k-1, forces the zero-padded window) then more singles
        for first in [1usize, 2, 3, 4] {
            let total = first + 5;
            let x: Vec<f32> =
                (0..total).flat_map(|ti| (0..cd).map(move |c| f(c * 3 + ti * 5, 13))).collect();
            let full = causal_conv1d_silu(&x, &w, &b, total, cd, k);
            let mut hist: Vec<f32> = Vec::new();
            let mut split = Vec::new();
            // prefill block of `first` rows, then one row at a time
            split.extend_from_slice(&causal_conv1d_silu_hist(
                &x[..first * cd], &w, &b, first, cd, k, &mut hist,
            ));
            for i in first..total {
                split.extend_from_slice(&causal_conv1d_silu_hist(
                    &x[i * cd..(i + 1) * cd], &w, &b, 1, cd, k, &mut hist,
                ));
            }
            assert_eq!(full.len(), split.len(), "first={first}");
            for i in 0..full.len() {
                assert_eq!(
                    full[i].to_bits(),
                    split[i].to_bits(),
                    "conv rolling-window split diverged at {i} (first={first})"
                );
            }
        }
    }
}
