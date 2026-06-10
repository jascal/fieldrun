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
    kv_int8: bool, // store the KV cache (GQA width) as int8 with a per-kv-head scale during generate
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
    pub fn new(b: Bundle, _route: f32, kv_int8: bool) -> Qwen3Moe {
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
                   n_exp, topk, moe_inter, norm_topk, moe, window, tied, kv_int8 }
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

    /// Rotary embedding on a (m, n_heads*hd) block; row i is absolute position `pos0 + i` (pos0 > 0 for KV-cache decode).
    fn rope(&self, x: &mut Array2<f32>, n_heads: usize, pos0: usize) {
        let (hd, half) = (self.hd, self.hd / 2);
        for (i, mut row) in x.rows_mut().into_iter().enumerate() {
            let pos = pos0 + i;
            for head in 0..n_heads {
                let base = head * hd;
                for j in 0..half {
                    let ang = pos as f32 * self.inv[j];
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
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
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

    fn unembed_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let logits = self.b.rowdot_f32(self.unembed(), &xfn.row(xfn.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    /// For explain: on a MoE layer, the dominant expert (highest softmax route) for one token's post-attention-normed
    /// hidden `row` (a (1, d) array) and that expert's SwiGLU hidden — names the layer's "MLP feature".
    fn top_expert_feature(&self, l: usize, row: &Array2<f32>) -> (usize, Vec<f32>) {
        let p = format!("l{l}.");
        let mut logits = self.b.mm(row, &format!("{p}gate"));
        softmax_rows(&mut logits);
        let r = logits.row(0);
        let e = (0..self.n_exp).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap();
        let gate = self.b.expert_mm(row, &format!("{p}experts.{e}.gate"));
        let up = self.b.expert_mm(row, &format!("{p}experts.{e}.up"));
        let mut hh = gate;
        for (h, u) in hh.iter_mut().zip(up.iter()) { *h = silu(*h) * u; }
        (e, hh.row(0).to_vec())
    }

    fn explanation(&self, ids: &[i64]) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        let mut att_last: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut head_act: Vec<Vec<f32>> = Vec::new();
        let mut mlp_h: Vec<Vec<f32>> = Vec::new();
        let mut feat_src: Vec<Option<usize>> = Vec::new(); // Some(e) = MoE expert e; None = dense mlp.down_proj
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.head_norm(&mut q, &format!("{p}q_norm"), h);
            self.head_norm(&mut k, &format!("{p}k_norm"), nkv);
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            let mut layer_att = Vec::with_capacity(h);
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i || (self.window > 0 && j + self.window <= i) { scores[[i, j]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                layer_att.push(scores.row(seq - 1).to_vec());
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            att_last.push(layer_att);
            head_act.push(attn_out.row(seq - 1).to_vec());
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            if self.moe[l] {
                let last = a2.slice(s![seq - 1..seq, ..]).to_owned();
                let (e, hidden) = self.top_expert_feature(l, &last);
                feat_src.push(Some(e));
                mlp_h.push(hidden);
            } else {
                let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
                let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
                let mut hidden = gate;
                for (hv, uv) in hidden.iter_mut().zip(up.iter()) { *hv = silu(*hv) * uv; }
                feat_src.push(None);
                mlp_h.push(hidden.row(seq - 1).to_vec());
            }
            let mlp = if self.moe[l] { self.moe_branch(l, &a2) } else { self.dense_mlp(l, &a2) };
            x = &x + &mlp;
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
                let w_out = match feat_src[l] {
                    Some(e) => self.b.expert_row(&format!("l{l}.experts.{e}.down"), n),
                    None => self.b.weight_row(&format!("l{l}.mlp.down_proj"), n),
                };
                top_promoted(&self.b.rowdot_f32(un, &w_out), act, 5)
            },
            |l, head| head_dla(&self.b, &format!("l{l}.self_attn.o_proj"), un, &head_act[l], head, hd, &gain, false, 5),
        )
    }

    /// Run `m` new positions through the layers, caching K/V (post-RoPE/QK-norm, GQA width nkv*hd) and attending over the
    /// whole cache (causal + sliding window). cur = absolute position of the first new row.
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.head_norm(&mut q, &format!("{p}q_norm"), h);
            self.head_norm(&mut k, &format!("{p}k_norm"), nkv);
            self.rope(&mut q, h, cur);
            self.rope(&mut k, nkv, cur);
            kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
            vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = kc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let vh = vc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..m {
                    let abs = cur + i;
                    for j in 0..klen {
                        if j > abs || (self.window > 0 && j + self.window <= abs) { scores[[i, j]] = -1e30; }
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

    /// `forward_block` with an int8 KV cache (GQA width, per-kv-head scale): ~4x smaller cache, ~identical tokens.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_q(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>],
                       vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]) -> Array2<f32> {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let kvdim = nkv * hd;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let q8 = |v: f32, sc: f32| (v / sc).round().clamp(-127.0, 127.0) as i8;
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.head_norm(&mut q, &format!("{p}q_norm"), h);
            self.head_norm(&mut k, &format!("{p}k_norm"), nkv);
            self.rope(&mut q, h, cur);
            self.rope(&mut k, nkv, cur);
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
                let mut scores = qh.dot(&kh_a.t()) * self.scale;
                for i in 0..m {
                    let abs = cur + i;
                    for j in 0..klen {
                        if j > abs || (self.window > 0 && j + self.window <= abs) { scores[[i, j]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh_a));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let mlp = if self.moe[l] { self.moe_branch(l, &a2) } else { self.dense_mlp(l, &a2) };
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    fn generate_kv_int8(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let total = prompt.len() + n_new;
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|_| vec![0i8; total * kvdim]).collect();
        let mut vc = kc.clone();
        let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|_| vec![0f32; total * self.nkv]).collect();
        let mut vs = ks.clone();
        let emb = self.b.rows_f32("embed", prompt);
        let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
        let mut next = self.unembed_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
            next = self.unembed_argmax(&xb);
            pos += 1;
        }
    }
}

impl Model for Qwen3Moe {
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
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let emb = self.b.rows_f32("embed", prompt);
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = self.unembed_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.unembed_argmax(&xb);
            pos += 1;
        }
    }

    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let total = prompt.len() + max_tokens;
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc = kc.clone();
        let emb = self.b.rows_f32("embed", prompt);
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = self.unembed_argmax(&xb);
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
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.unembed_argmax(&xb);
            pos += 1;
        }
        out
    }
}
