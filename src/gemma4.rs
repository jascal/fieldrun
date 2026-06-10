//! Tier B — composition, Gemma 4 (text path: Per-Layer Embeddings on; dense FFN, with an optional summed top-k MoE
//! expert branch). A faithful Rust port of `Gemma4ForCausalLM`'s text model. On top of the Gemma-3 backbone it adds:
//!   - RMSNorm uses the stored weight *directly* (NOT (1+w) — Gemma 4 inits norm weights to 1.0);
//!   - per-head **value-norm** (RMS over head_dim, no learnable weight) alongside the q/k norms;
//!   - attention **scaling = 1.0** (QK-norm makes the 1/√d unnecessary);
//!   - a **different head_dim on global (full) layers** (so q/k/v/o shapes differ per layer type);
//!   - **partial-rotary "proportional" RoPE** on global layers (only the first ⌊0.25·hd/2⌋ frequency pairs rotate; the
//!     rest are zero-padded → identity), local layers full-rotate at the lower base;
//!   - the **Per-Layer-Embedding (PLE) gated-residual block** per layer: a token-identity aux embedding + a context
//!     projection of the input embedding, gated by the post-FFN hidden and added back.
//!   - a per-layer **`layer_scalar`** (a persistent buffer, default 1.0) multiplied into the residual as the LAST op of
//!     each decoder layer; read from the checkpoint into config_f so a non-1.0 value runs faithfully.
//! Validated top-1 against a tiny random-init `Gemma4ForCausalLM` (the faithfulness gate), dense and MoE. Incremental
//! KV-cache `generate`/`generate_stream` (f32 + int8-KV; per-layer GQA width, since local/global head_dim differ) and
//! `explain` are wired. attention_k_eq_v / KV-sharing are not yet implemented (the convert asserts they're off).

use std::collections::HashMap;

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
    moe: bool,        // MoE-FFN layers (dense MLP + a sparse top-k expert branch, summed)
    n_exp: usize,
    topk: usize,
    moe_inter: usize,
    layer_scalar: Vec<f32>, // per-layer output multiplier applied as the LAST op of each layer (default 1.0)
    kv_int8: bool,    // store the KV cache (per-layer GQA width) as int8 with a per-kv-head scale during generate
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
    pub fn new(b: Bundle, _route: f32, kv_int8: bool) -> Gemma4 {
        // config: [nl, nh, nkv, nkv_g, hd_local, hd_global, d, ffn, vocab, tied, window, ple,
        //          moe, n_exp, topk, moe_inter, <nl sliding flags>]
        let c = &b.config;
        let (n_layer, h, nkv) = (c[0] as usize, c[1] as usize, c[2] as usize);
        let (hd_local, hd_global, d) = (c[4] as usize, c[5] as usize, c[6] as usize);
        let tied = c[9] != 0;
        let window = c[10] as usize;
        let ple = c[11] as usize;
        let (moe, n_exp, topk, moe_inter) = (c[12] != 0, c[13] as usize, c[14] as usize, c[15] as usize);
        let sliding: Vec<bool> = (0..n_layer).map(|l| c[16 + l] != 0).collect();
        // config_f: [theta_local, theta_global, eps, partial_rotary_factor, <nl layer_scalar>]
        let (theta_local, theta_global, eps, prf) =
            (b.config_f[0] as f32, b.config_f[1] as f32, b.config_f[2] as f32, b.config_f[3] as f32);
        // per-layer output scalar; default 1.0 for older bundles that didn't record it.
        let layer_scalar: Vec<f32> = (0..n_layer)
            .map(|l| if b.config_f.len() > 4 + l { b.config_f[4 + l] as f32 } else { 1.0 })
            .collect();
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
            inv_local, inv_global, window, sliding, tied, moe, n_exp, topk, moe_inter, layer_scalar, kv_int8,
        }
    }

    /// The MoE-FFN expert branch for one layer: route each token to its top-k experts, then for each *active* expert
    /// dequantise its weights once (paged in from the mmap) and run its assigned tokens. `x_pre` is the pre-FFN hidden
    /// (= the residual the router and experts both read). Returns the (already pre/post-normed) expert contribution.
    fn moe_branch(&self, l: usize, x_pre: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.");
        let seq = x_pre.nrows();
        // --- router (norm has no weight; then * scale * 1/√d; proj; softmax; top-k; renorm; per-expert scale) ---
        let hn = self.rmsnorm(x_pre, None);
        let scale = self.b.arr1o(&format!("{p}router.scale"));
        let mut hs = hn;
        let dscale = (self.d as f32).powf(-0.5);
        for mut row in hs.rows_mut() {
            for (i, v) in row.iter_mut().enumerate() { *v = *v * scale[i] * dscale; }
        }
        let scores = self.b.mm(&hs, &format!("{p}router.proj")); // (seq, E)
        let pes = self.b.arr1o(&format!("{p}router.per_expert_scale"));
        // per token → its top-k (expert, weight); group tokens by expert for one dequant per active expert
        let mut assign: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for t in 0..seq {
            let row = scores.row(t);
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / denom).collect();
            let mut idx: Vec<usize> = (0..self.n_exp).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            idx.truncate(self.topk);
            let tksum: f32 = idx.iter().map(|&e| probs[e]).sum();
            for &e in &idx {
                let w = probs[e] / tksum * pes[e];
                assign.entry(e).or_default().push((t, w));
            }
        }
        // --- experts (pre-norm’d input; each active expert dequantised once from the mmap) ---
        let h2_in = self.norm(x_pre, &format!("{p}pre_feedforward_layernorm_2"));
        let mut h2 = Array2::<f32>::zeros((seq, self.d));
        let mi = self.moe_inter;
        for (e, toks) in &assign {
            let mut rows = Array2::<f32>::zeros((toks.len(), self.d));
            for (i, &(t, _)) in toks.iter().enumerate() { rows.row_mut(i).assign(&h2_in.row(t)); }
            let gate_up = self.b.expert_mm(&rows, &format!("{p}experts.{e}.gate_up")); // (ntok, 2*mi)
            let mut hh = Array2::<f32>::zeros((toks.len(), mi));
            for i in 0..toks.len() {
                for c in 0..mi {
                    hh[[i, c]] = gelu_tanh(gate_up[[i, c]]) * gate_up[[i, c + mi]];
                }
            }
            let outp = self.b.expert_mm(&hh, &format!("{p}experts.{e}.down")); // (ntok, d)
            for (i, &(t, w)) in toks.iter().enumerate() {
                for c in 0..self.d { h2[[t, c]] += w * outp[[i, c]]; }
            }
        }
        self.norm(&h2, &format!("{p}post_feedforward_layernorm_2"))
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

    fn rope(&self, x: &mut Array2<f32>, n_heads: usize, hd: usize, pos0: usize, inv: &[f32]) {
        let half = hd / 2;
        for (i, mut row) in x.rows_mut().into_iter().enumerate() {
            let pos = pos0 + i;
            for head in 0..n_heads {
                let base = head * hd;
                for j in 0..half {
                    let ang = pos as f32 * inv[j];
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
            self.rope(&mut q, h, hd, 0, inv);
            self.rope(&mut k, nkv, hd, 0, inv);
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

            // --- MLP (dense; + a summed MoE expert branch on MoE layers) ---
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = gelu_tanh(*hv) * uv;
            }
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            let combined = if self.moe {
                // dense path normed by post_ffn_norm_1, expert path normed by post_ffn_norm_2 (inside moe_branch), summed
                let h1 = self.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &self.moe_branch(l, &x)
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));

            // --- PLE gated-residual block ---
            let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate")); // (seq, ple)
            g.mapv_inplace(gelu_tanh);
            g = &g * &pli[l]; // gate by the per-layer embedding
            let proj = self.b.mm(&g, &format!("{p}per_layer_projection")); // (seq, d)
            x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            x *= self.layer_scalar[l]; // per-layer output scalar — the last op of the layer (Gemma4ForCausalLM)
        }
        self.norm(&x, "norm")
    }

    fn head_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let logits = self.b.rowdot_f32(self.unembed(), &xfn.row(xfn.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn explanation(&self, ids: &[i64]) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let (h, nkv) = (self.h, self.nkv);
        let emb = self.b.rows_f32("embed", ids);
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale));
        }
        let pli = self.per_layer_inputs(ids, &x);
        let mut att_last: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut head_act: Vec<Vec<f32>> = Vec::new();
        let mut head_hd: Vec<usize> = Vec::new(); // per-layer head_dim (local vs global) — for head DLA
        let mut mlp_h: Vec<Vec<f32>> = Vec::new();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let mut v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.head_norm(&mut q, Some(&format!("{p}self_attn.q_norm")), h, hd);
            self.head_norm(&mut k, Some(&format!("{p}self_attn.k_norm")), nkv, hd);
            self.head_norm(&mut v, None, nkv, hd);
            let inv = self.inv_for(l);
            self.rope(&mut q, h, hd, 0, inv);
            self.rope(&mut k, nkv, hd, 0, inv);
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            let mut layer_att = Vec::with_capacity(h);
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()); // scaling = 1.0
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i || (sliding && j + self.window <= i) { scores[[i, j]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                layer_att.push(scores.row(seq - 1).to_vec());
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            att_last.push(layer_att);
            head_act.push(attn_out.row(seq - 1).to_vec());
            head_hd.push(hd);
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            x = &x + &self.norm(&o, &format!("{p}post_attention_layernorm"));
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) { *hv = gelu_tanh(*hv) * uv; }
            mlp_h.push(hidden.row(seq - 1).to_vec());
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            let combined = if self.moe {
                let h1 = self.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &self.moe_branch(l, &x)
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
            g.mapv_inplace(gelu_tanh);
            g = &g * &pli[l];
            let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
            x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            x *= self.layer_scalar[l]; // per-layer output scalar — the last op of the layer (Gemma4ForCausalLM)
        }
        let xf = self.norm(&x, "norm");
        let un = self.unembed();
        let lg = self.b.rowdot_f32(un, &xf.row(seq - 1).to_vec());
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let gain = self.b.arr1("norm").to_vec();
        assemble(
            ids,
            &att_last,
            &mlp_h,
            model_predicts,
            |l, n, act| {
                let w_out = self.b.weight_row(&format!("l{l}.mlp.down_proj"), n);
                top_promoted(&self.b.rowdot_f32(un, &w_out), act, 5)
            },
            |l, head| head_dla(&self.b, &format!("l{l}.self_attn.o_proj"), un, &head_act[l], head, head_hd[l], &gain, false, 5),
        )
    }

    /// Run `m` new positions through the layers, caching K/V (post value-norm / RoPE; per-layer GQA width nkv*hd_of(l),
    /// which differs between local and global layers) and attending over the whole cache (causal + per-layer sliding
    /// window). The PLE gated-residual block is per-token, so it is recomputed for the new rows only. cur = absolute
    /// position of the first new row.
    fn forward_block(&self, ids: &[i64], emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let (h, nkv) = (self.h, self.nkv);
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let pli = self.per_layer_inputs(ids, &x);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let mut v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.head_norm(&mut q, Some(&format!("{p}self_attn.q_norm")), h, hd);
            self.head_norm(&mut k, Some(&format!("{p}self_attn.k_norm")), nkv, hd);
            self.head_norm(&mut v, None, nkv, hd);
            let inv = self.inv_for(l);
            self.rope(&mut q, h, hd, cur, inv);
            self.rope(&mut k, nkv, hd, cur, inv);
            kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
            vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = kc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let vh = vc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t());
                for i in 0..m {
                    let abs = cur + i;
                    for j in 0..klen {
                        if j > abs || (sliding && j + self.window <= abs) { scores[[i, j]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            x = &x + &self.norm(&o, &format!("{p}post_attention_layernorm"));
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) { *hv = gelu_tanh(*hv) * uv; }
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            let combined = if self.moe {
                let h1 = self.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &self.moe_branch(l, &x)
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
            g.mapv_inplace(gelu_tanh);
            g = &g * &pli[l];
            let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
            x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            x *= self.layer_scalar[l]; // per-layer output scalar — the last op of the layer (Gemma4ForCausalLM)
        }
        self.norm(&x, "norm")
    }

    /// `forward_block` with an int8 KV cache (per-layer GQA width, per-kv-head scale): ~4x smaller cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_q(&self, ids: &[i64], emb: &Array2<f32>, cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>],
                       vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]) -> Array2<f32> {
        let (h, nkv) = (self.h, self.nkv);
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let pli = self.per_layer_inputs(ids, &x);
        let q8 = |v: f32, sc: f32| (v / sc).round().clamp(-127.0, 127.0) as i8;
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let rep = h / nkv;
            let kvdim = nkv * hd;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let mut v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.head_norm(&mut q, Some(&format!("{p}self_attn.q_norm")), h, hd);
            self.head_norm(&mut k, Some(&format!("{p}self_attn.k_norm")), nkv, hd);
            self.head_norm(&mut v, None, nkv, hd);
            let inv = self.inv_for(l);
            self.rope(&mut q, h, hd, cur, inv);
            self.rope(&mut k, nkv, hd, cur, inv);
            for i in 0..m {
                let pos = cur + i;
                for kh in 0..nkv {
                    let base = kh * hd;
                    let sck = ((0..hd).fold(0f32, |mx, c| mx.max(k[[i, base + c]].abs())) / 127.0).max(1e-8);
                    let scv = ((0..hd).fold(0f32, |mx, c| mx.max(v[[i, base + c]].abs())) / 127.0).max(1e-8);
                    ks[l][pos * nkv + kh] = sck;
                    vs[l][pos * nkv + kh] = scv;
                    for c in 0..hd {
                        kc[l][pos * kvdim + base + c] = q8(k[[i, base + c]], sck);
                        vc[l][pos * kvdim + base + c] = q8(v[[i, base + c]], scv);
                    }
                }
            }
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let mut kh_a = Array2::<f32>::zeros((klen, hd));
                let mut vh_a = Array2::<f32>::zeros((klen, hd));
                for pos in 0..klen {
                    let (sck, scv) = (ks[l][pos * nkv + kv], vs[l][pos * nkv + kv]);
                    for c in 0..hd {
                        kh_a[[pos, c]] = kc[l][pos * kvdim + kv * hd + c] as f32 * sck;
                        vh_a[[pos, c]] = vc[l][pos * kvdim + kv * hd + c] as f32 * scv;
                    }
                }
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let mut scores = qh.dot(&kh_a.t());
                for i in 0..m {
                    let abs = cur + i;
                    for j in 0..klen {
                        if j > abs || (sliding && j + self.window <= abs) { scores[[i, j]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh_a));
            }
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            x = &x + &self.norm(&o, &format!("{p}post_attention_layernorm"));
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) { *hv = gelu_tanh(*hv) * uv; }
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            let combined = if self.moe {
                let h1 = self.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &self.moe_branch(l, &x)
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
            g.mapv_inplace(gelu_tanh);
            g = &g * &pli[l];
            let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
            x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            x *= self.layer_scalar[l]; // per-layer output scalar — the last op of the layer (Gemma4ForCausalLM)
        }
        self.norm(&x, "norm")
    }

    fn generate_kv_int8(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let total = prompt.len() + n_new;
        let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|l| vec![0i8; total * self.nkv * self.hd_of(l)]).collect();
        let mut vc = kc.clone();
        let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|_| vec![0f32; total * self.nkv]).collect();
        let mut vs = ks.clone();
        let mut emb = self.b.rows_f32("embed", prompt) * self.escale;
        let xb = self.forward_block_q(prompt, &emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            emb = self.b.rows_f32("embed", &[next]) * self.escale;
            let xb = self.forward_block_q(&[next], &emb, pos, &mut kc, &mut ks, &mut vc, &mut vs);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }
}

impl Model for Gemma4 {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32(self.unembed(), &last);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn explain(&self, ids: &[i64]) -> Option<crate::explain::Explanation> {
        Some(self.explanation(ids))
    }

    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        if self.kv_int8 {
            return self.generate_kv_int8(prompt, n_new);
        }
        let total = prompt.len() + n_new;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|l| Array2::zeros((total, self.nkv * self.hd_of(l)))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|l| Array2::zeros((total, self.nkv * self.hd_of(l)))).collect();
        let mut emb = self.b.rows_f32("embed", prompt) * self.escale;
        let xb = self.forward_block(prompt, &emb, 0, &mut kc, &mut vc);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            emb = self.b.rows_f32("embed", &[next]) * self.escale;
            let xb = self.forward_block(&[next], &emb, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }

    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let total = prompt.len() + max_tokens;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|l| Array2::zeros((total, self.nkv * self.hd_of(l)))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|l| Array2::zeros((total, self.nkv * self.hd_of(l)))).collect();
        let mut emb = self.b.rows_f32("embed", prompt) * self.escale;
        let xb = self.forward_block(prompt, &emb, 0, &mut kc, &mut vc);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::new();
        let mut pos = prompt.len();
        loop {
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            if !emit(next) || out.len() == max_tokens {
                break;
            }
            emb = self.b.rows_f32("embed", &[next]) * self.escale;
            let xb = self.forward_block(&[next], &emb, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
        out
    }
}
