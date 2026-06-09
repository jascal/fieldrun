//! Tier B — composition, Gemma 4 (dense text path: Per-Layer Embeddings on, MoE off). A faithful Rust port of
//! `Gemma4ForCausalLM`'s text model. On top of the Gemma-3 backbone it adds:
//!   - RMSNorm uses the stored weight *directly* (NOT (1+w) — Gemma 4 inits norm weights to 1.0);
//!   - per-head **value-norm** (RMS over head_dim, no learnable weight) alongside the q/k norms;
//!   - attention **scaling = 1.0** (QK-norm makes the 1/√d unnecessary);
//!   - a **different head_dim on global (full) layers** (so q/k/v/o shapes differ per layer type);
//!   - **partial-rotary "proportional" RoPE** on global layers (only the first ⌊0.25·hd/2⌋ frequency pairs rotate; the
//!     rest are zero-padded → identity), local layers full-rotate at the lower base;
//!   - the **Per-Layer-Embedding (PLE) gated-residual block** per layer: a token-identity aux embedding + a context
//!     projection of the input embedding, gated by the post-FFN hidden and added back.
//! Validated top-1 against a tiny random-init `Gemma4ForCausalLM` (the faithfulness gate). MoE / attention_k_eq_v /
//! KV-sharing are not yet implemented (the convert asserts they're off).

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Gemma4 {
    b: Bundle,
    n_layer: usize,
    h: usize,
    nkv: usize,
    hd_local: usize,
    hd_global: usize,
    d: usize,
    ple: usize,
    eps: f32,
    escale: f32,     // √d, the input-embedding scale
    ple_escale: f32, // √ple, the PLE token-identity embedding scale
    proj_scale: f32, // 1/√d, the PLE context-projection scale
    inv_local: Vec<f32>,
    inv_global: Vec<f32>,
    window: usize,
    sliding: Vec<bool>,
    tied: bool,
}

fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn softmax_rows(a: &mut Array2<f32>) {
    for mut row in a.rows_mut() {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        row.mapv_inplace(|v| v / s);
    }
}

impl Gemma4 {
    pub fn new(b: Bundle, _route: f32, _kv_int8: bool) -> Gemma4 {
        // config: [nl, nh, nkv, nkv_g, hd_local, hd_global, d, ffn, vocab, tied, window, ple, <nl sliding flags>]
        let c = &b.config;
        let (n_layer, h, nkv) = (c[0] as usize, c[1] as usize, c[2] as usize);
        let (hd_local, hd_global, d) = (c[4] as usize, c[5] as usize, c[6] as usize);
        let tied = c[9] != 0;
        let window = c[10] as usize;
        let ple = c[11] as usize;
        let sliding: Vec<bool> = (0..n_layer).map(|l| c[12 + l] != 0).collect();
        // config_f: [theta_local, theta_global, eps, partial_rotary_factor]
        let (theta_local, theta_global, eps, prf) =
            (b.config_f[0] as f32, b.config_f[1] as f32, b.config_f[2] as f32, b.config_f[3] as f32);
        // local: full rotation at the low base over hd_local
        let inv_local = (0..hd_local / 2).map(|j| 1.0 / theta_local.powf(2.0 * j as f32 / hd_local as f32)).collect();
        // global: "proportional" partial rotary — first `angles` pairs at the high base, the rest zero (un-rotated)
        let angles = (prf * (hd_global as f32) / 2.0) as usize;
        let inv_global = (0..hd_global / 2)
            .map(|j| if j < angles { 1.0 / theta_global.powf(2.0 * j as f32 / hd_global as f32) } else { 0.0 })
            .collect();
        Gemma4 {
            b, n_layer, h, nkv, hd_local, hd_global, d, ple, eps,
            escale: (d as f32).sqrt(), ple_escale: (ple as f32).sqrt(), proj_scale: (d as f32).powf(-0.5),
            inv_local, inv_global, window, sliding, tied,
        }
    }

    fn unembed(&self) -> &str {
        if self.tied { "embed" } else { "lm_head" }
    }

    fn hd_of(&self, l: usize) -> usize {
        if self.sliding[l] { self.hd_local } else { self.hd_global }
    }

    fn inv_for(&self, l: usize) -> &[f32] {
        if self.sliding[l] { &self.inv_local } else { &self.inv_global }
    }

    /// RMSNorm with the stored weight applied directly (Gemma 4 has no (1+w) bake). `w_opt = None` → no weight
    /// (value-norm / with_scale=False). Normalises each row over its full width.
    fn rmsnorm(&self, x: &Array2<f32>, w_opt: Option<&[f32]>) -> Array2<f32> {
        let mut out = x.clone();
        for mut row in out.rows_mut() {
            let n = row.len() as f32;
            let ms = row.iter().map(|v| v * v).sum::<f32>() / n;
            let inv = 1.0 / (ms + self.eps).sqrt();
            match w_opt {
                Some(w) => for (i, v) in row.iter_mut().enumerate() { *v = *v * inv * w[i]; },
                None => for v in row.iter_mut() { *v *= inv; },
            }
        }
        out
    }

    fn norm(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let w = self.b.arr1o(name);
        self.rmsnorm(x, Some(w.as_slice().unwrap()))
    }

    /// Per-head RMSNorm over the layer's head_dim. `name = None` → value-norm (no weight).
    fn head_norm(&self, x: &mut Array2<f32>, name: Option<&str>, n_heads: usize, hd: usize) {
        let w = name.map(|n| self.b.arr1o(n));
        let w = w.as_ref().map(|a| a.as_slice().unwrap());
        for mut row in x.rows_mut() {
            for head in 0..n_heads {
                let base = head * hd;
                let ms = (0..hd).map(|c| { let v = row[base + c]; v * v }).sum::<f32>() / hd as f32;
                let inv = 1.0 / (ms + self.eps).sqrt();
                match w {
                    Some(w) => for c in 0..hd { row[base + c] = row[base + c] * inv * w[c]; },
                    None => for c in 0..hd { row[base + c] *= inv; },
                }
            }
        }
    }

    fn rope(&self, x: &mut Array2<f32>, n_heads: usize, hd: usize, inv: &[f32]) {
        let half = hd / 2;
        for (i, mut row) in x.rows_mut().into_iter().enumerate() {
            for head in 0..n_heads {
                let base = head * hd;
                for j in 0..half {
                    let ang = i as f32 * inv[j];
                    let (cs, sn) = (ang.cos(), ang.sin());
                    let (a, b) = (row[base + j], row[base + j + half]);
                    row[base + j] = a * cs - b * sn;
                    row[base + j + half] = b * cs + a * sn;
                }
            }
        }
    }

    /// Per-Layer-Embedding inputs: per_layer_inputs[l] is (seq, ple). Combines the token-identity aux embedding
    /// (`embed_per_layer`, scaled √ple) with the context projection of the √d-scaled input embedding
    /// (`per_layer_model_projection` · 1/√d, RMSNorm'd over ple), then `(proj + tok) · 1/√2`.
    fn per_layer_inputs(&self, ids: &[i64], emb_scaled: &Array2<f32>) -> Vec<Array2<f32>> {
        let seq = ids.len();
        let tok = self.b.rows_f32("embed_per_layer", ids); // (seq, nl*ple), un-scaled
        let proj = self.b.mm(emb_scaled, "per_layer_model_projection"); // (seq, nl*ple)
        let pnorm = self.b.arr1o("per_layer_projection_norm");
        let pnorm = pnorm.as_slice().unwrap();
        let inv2 = 1.0 / 2.0f32.sqrt();
        (0..self.n_layer)
            .map(|l| {
                let c0 = l * self.ple;
                let mut out = Array2::<f32>::zeros((seq, self.ple));
                for t in 0..seq {
                    // context projection slice, RMSNorm over ple with the (no-bake) weight
                    let mut ctx: Vec<f32> = (0..self.ple).map(|c| proj[[t, c0 + c]] * self.proj_scale).collect();
                    let ms = ctx.iter().map(|v| v * v).sum::<f32>() / self.ple as f32;
                    let rinv = 1.0 / (ms + self.eps).sqrt();
                    for (c, v) in ctx.iter_mut().enumerate() { *v = *v * rinv * pnorm[c]; }
                    // token-identity slice (scaled √ple), combine
                    for c in 0..self.ple {
                        out[[t, c]] = (ctx[c] + tok[[t, c0 + c]] * self.ple_escale) * inv2;
                    }
                }
                out
            })
            .collect()
    }

    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let (h, nkv) = (self.h, self.nkv);
        let emb = self.b.rows_f32("embed", ids);
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale));
        }
        let pli = self.per_layer_inputs(ids, &x);

        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let rep = h / nkv;
            // --- attention ---
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let mut v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.head_norm(&mut q, Some(&format!("{p}self_attn.q_norm")), h, hd);
            self.head_norm(&mut k, Some(&format!("{p}self_attn.k_norm")), nkv, hd);
            self.head_norm(&mut v, None, nkv, hd); // value-norm: RMS only, no weight
            let inv = self.inv_for(l);
            self.rope(&mut q, h, hd, inv);
            self.rope(&mut k, nkv, hd, inv);
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()); // scaling = 1.0 in Gemma 4
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i || (sliding && j + self.window <= i) {
                            scores[[i, j]] = -1e30;
                        }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            x = &x + &self.norm(&o, &format!("{p}post_attention_layernorm"));

            // --- MLP ---
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = gelu_tanh(*hv) * uv;
            }
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            x = &x + &self.norm(&mlp, &format!("{p}post_feedforward_layernorm"));

            // --- PLE gated-residual block ---
            let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate")); // (seq, ple)
            g.mapv_inplace(gelu_tanh);
            g = &g * &pli[l]; // gate by the per-layer embedding
            let proj = self.b.mm(&g, &format!("{p}per_layer_projection")); // (seq, d)
            x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
        }
        self.norm(&x, "norm")
    }
}

impl Model for Gemma4 {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32(self.unembed(), &last);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}
