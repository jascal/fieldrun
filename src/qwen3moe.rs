//! Tier B — composition, Qwen3-MoE. The RoPE backbone (RMSNorm, single-base RoPE, GQA, the standard two-norm pre-norm
//! block) plus the two things Qwen3 adds: **QK-norm** (a per-head RMSNorm on q/k after projection, before RoPE) and a
//! per-layer **MoE FFN** — a plain-gate router (softmax → top-k → optional renorm, no scales) over packed experts,
//! running on the post-attention-normed hidden; layers not selected by `decoder_sparse_step` / `mlp_only_layers` use a
//! dense SwiGLU. Experts are read on demand from the mmap (the offload path), so a Qwen3-MoE whose expert weights far
//! exceed RAM still runs. Sliding-window attention (`use_sliding_window`) applies one window size to EVERY layer (no
//! per-layer pattern, unlike Gemma 3; single RoPE base) — window 0 in the bundle means full attention. No attention
//! bias, no embed scale, no soft-capping. A faithful port of `Qwen3MoeForCausalLM`, validated top-1 against a tiny
//! random-init instance.

use std::collections::HashMap;

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Qwen3Moe {
    b: Bundle,
    n_layer: usize,
    h: usize,
    nkv: usize,
    hd: usize,
    d: usize,
    eps: f32,
    scale: f32, // head_dim^-0.5
    inv: Vec<f32>,
    n_exp: usize,
    topk: usize,
    moe_inter: usize,
    norm_topk: bool,
    moe: Vec<bool>,
    window: usize, // sliding-window size, all layers (0 = full attention)
    tied: bool,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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

impl Qwen3Moe {
    pub fn new(b: Bundle, _route: f32, _kv_int8: bool) -> Qwen3Moe {
        // config: [nl, nh, nkv, hd, d, ffn, vocab, tied, n_exp, topk, moe_inter, norm_topk, <nl moe flags>, window]
        let c = &b.config;
        let (n_layer, h, nkv, hd, d) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize, c[4] as usize);
        let tied = c[7] != 0;
        let (n_exp, topk, moe_inter, norm_topk) = (c[8] as usize, c[9] as usize, c[10] as usize, c[11] != 0);
        let moe: Vec<bool> = (0..n_layer).map(|l| c[12 + l] != 0).collect();
        let window = if c.len() > 12 + n_layer { c[12 + n_layer] as usize } else { 0 }; // absent in older bundles
        let (theta, eps) = (b.config_f[0] as f32, b.config_f[1] as f32);
        let inv = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        Qwen3Moe { b, n_layer, h, nkv, hd, d, eps, scale: (hd as f32).powf(-0.5), inv,
                   n_exp, topk, moe_inter, norm_topk, moe, window, tied }
    }

    fn unembed(&self) -> &str {
        if self.tied { "embed" } else { "lm_head" }
    }

    fn norm(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let w = self.b.arr1o(name);
        let mut out = x.clone();
        for mut row in out.rows_mut() {
            let n = row.len() as f32;
            let ms = row.iter().map(|v| v * v).sum::<f32>() / n;
            let inv = 1.0 / (ms + self.eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() { *v = *v * inv * w[i]; }
        }
        out
    }

    /// Per-head RMSNorm over head_dim (QK-norm), weight applied directly.
    fn head_norm(&self, x: &mut Array2<f32>, name: &str, n_heads: usize) {
        let w = self.b.arr1o(name);
        let hd = self.hd;
        for mut row in x.rows_mut() {
            for head in 0..n_heads {
                let base = head * hd;
                let ms = (0..hd).map(|c| { let v = row[base + c]; v * v }).sum::<f32>() / hd as f32;
                let inv = 1.0 / (ms + self.eps).sqrt();
                for c in 0..hd { row[base + c] = row[base + c] * inv * w[c]; }
            }
        }
    }

    fn rope(&self, x: &mut Array2<f32>, n_heads: usize) {
        let (hd, half) = (self.hd, self.hd / 2);
        for (i, mut row) in x.rows_mut().into_iter().enumerate() {
            for head in 0..n_heads {
                let base = head * hd;
                for j in 0..half {
                    let ang = i as f32 * self.inv[j];
                    let (c, s) = (ang.cos(), ang.sin());
                    let (a, b) = (row[base + j], row[base + j + half]);
                    row[base + j] = a * c - b * s;
                    row[base + j + half] = b * c + a * s;
                }
            }
        }
    }

    /// MoE FFN for one layer over the post-attention-normed hidden `a2`: route each token to its top-k experts (softmax
    /// over all experts → top-k → optional renorm), then for each *active* expert dequantise its weights once (paged in
    /// from the mmap) and run its assigned tokens (SwiGLU), weighted-summed.
    fn moe_branch(&self, l: usize, a2: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.");
        let seq = a2.nrows();
        let mut logits = self.b.mm(a2, &format!("{p}gate")); // (seq, n_exp)
        softmax_rows(&mut logits);
        let mut assign: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for t in 0..seq {
            let row = logits.row(t);
            let mut idx: Vec<usize> = (0..self.n_exp).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            idx.truncate(self.topk);
            let denom: f32 = if self.norm_topk { idx.iter().map(|&e| row[e]).sum() } else { 1.0 };
            for &e in &idx {
                assign.entry(e).or_default().push((t, row[e] / denom));
            }
        }
        let mut out = Array2::<f32>::zeros((seq, self.d));
        for (e, toks) in &assign {
            let mut rows = Array2::<f32>::zeros((toks.len(), self.d));
            for (i, &(t, _)) in toks.iter().enumerate() { rows.row_mut(i).assign(&a2.row(t)); }
            // SwiGLU from per-expert gate/up/down (each paged in from the mmap on demand)
            let gate = self.b.expert_mm(&rows, &format!("{p}experts.{e}.gate")); // (ntok, mi)
            let up = self.b.expert_mm(&rows, &format!("{p}experts.{e}.up"));      // (ntok, mi)
            let mut hh = gate;
            for (hv, uv) in hh.iter_mut().zip(up.iter()) { *hv = silu(*hv) * uv; }
            let down = self.b.expert_mm(&hh, &format!("{p}experts.{e}.down"));     // (ntok, d)
            for (i, &(t, wgt)) in toks.iter().enumerate() {
                for c in 0..self.d { out[[t, c]] += wgt * down[[i, c]]; }
            }
        }
        out
    }

    fn dense_mlp(&self, l: usize, a2: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.");
        let gate = self.b.mm(a2, &format!("{p}mlp.gate_proj"));
        let up = self.b.mm(a2, &format!("{p}mlp.up_proj"));
        let mut hidden = gate;
        for (hv, uv) in hidden.iter_mut().zip(up.iter()) { *hv = silu(*hv) * uv; }
        self.b.mm(&hidden, &format!("{p}mlp.down_proj"))
    }

    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids); // no embed scale in Qwen
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.head_norm(&mut q, &format!("{p}q_norm"), h);   // QK-norm before RoPE
            self.head_norm(&mut k, &format!("{p}k_norm"), nkv);
            self.rope(&mut q, h);
            self.rope(&mut k, nkv);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i || (self.window > 0 && j + self.window <= i) { scores[[i, j]] = -1e30; } // causal + window
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));

            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let mlp = if self.moe[l] { self.moe_branch(l, &a2) } else { self.dense_mlp(l, &a2) };
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }
}

impl Model for Qwen3Moe {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32(self.unembed(), &last);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}
