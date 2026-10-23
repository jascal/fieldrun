//! Tier B — composition, Qwen3-MoE. The RoPE backbone (RMSNorm, single-base RoPE, GQA, the standard two-norm pre-norm
//! block) plus the two things Qwen3 adds: **QK-norm** (a per-head RMSNorm on q/k after projection, before RoPE) and a
//! per-layer **MoE FFN** — a plain-gate router (softmax → top-k → optional renorm, no scales) over packed experts,
//! running on the post-attention-normed hidden; layers not selected by `decoder_sparse_step` / `mlp_only_layers` use a
//! dense SwiGLU. Experts are read on demand from the mmap (the offload path), so a Qwen3-MoE whose expert weights far
//! exceed RAM still runs. Sliding-window attention (`use_sliding_window`) applies one window size to EVERY layer (no
//! per-layer pattern, unlike Gemma 3; single RoPE base) — window 0 in the bundle means full attention. No Qwen3 attention
//! bias, no embed scale, no soft-capping. A faithful port of `Qwen3MoeForCausalLM`, validated top-1 against a tiny
//! random-init instance.
//!
//! The same kernel also carries two more sparse-MoE families, which differ from Qwen3-MoE as follows:
//! **Mixtral** (no QK-norm; `block_sparse_moe` experts; every layer MoE; the top-k is always renormalised) and
//! **Qwen2-MoE** (no QK-norm; q/k/v projection biases; plus an always-on shared expert whose SwiGLU output is scaled per token by
//! `sigmoid(shared_expert_gate)` and added to the routed mixture). Both facts come from the CONVERT, which probes the
//! checkpoint for the `q_norm`/`shared_expert` tensors; nothing here is inferred at runtime.

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
    qk_norm: bool, // per-head RMSNorm on q/k (Qwen3 ships it; Mixtral/Qwen2-MoE do not)
    shared_inter: usize, // always-on shared expert width, 0 = none (Qwen2-MoE has one, Mixtral/Qwen3-MoE do not)
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
        // `qk_norm` and the shared-expert width are APPENDED after `window` and read defensively: every bundle written
        // before the Mixtral/Qwen2-MoE landing has QK-norm and no shared expert, so absent means (true, 0).
        let qk_norm = if c.len() > 13 + n_layer { c[13 + n_layer] != 0 } else { true };
        let shared_inter = if c.len() > 15 + n_layer { c[15 + n_layer] as usize } else { 0 };
        let (theta, eps) = (b.config_f[0] as f32, b.config_f[1] as f32);
        let inv = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        Qwen3Moe { b, n_layer, h, nkv, hd, d, eps, scale: (hd as f32).powf(-0.5), inv,
                   n_exp, topk, moe_inter, norm_topk, moe, window, qk_norm, shared_inter, tied, kv_int8 }
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

    /// Optional projection biases are present in Qwen2-MoE checkpoints; older bundles omit them.
    fn attention_proj(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let bias = format!("{name}.bias");
        if self.b.has(&bias) { self.b.mm_bias(x, name, &bias) } else { self.b.mm(x, name) }
    }

    /// Apply QK-norm to a layer's q/k, guarded by `qk_norm`. ONE call at each forward site, so a family that ships no
    /// `q_norm`/`k_norm` tensors (Mixtral, Qwen2-MoE) can never half-apply it — `head_norm` reads a REQUIRED array, so
    /// an unguarded call on such a bundle would panic rather than degrade.
    fn qk_norm_apply(&self, p: &str, q: &mut Array2<f32>, k: &mut Array2<f32>) {
        if !self.qk_norm { return; }
        self.head_norm(q, &format!("{p}q_norm"), self.h);
        self.head_norm(k, &format!("{p}k_norm"), self.nkv);
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
    ///
    /// `scan = Some((expert, neuron))` additionally returns the per-row value of that ONE routed unit —
    /// `w_router · silu(gate[n]) · up[n]` — recorded in the same pass so the expert's weights are read once. Rows
    /// whose top-k does not contain `expert` stay `NaN`: the unit is not computed there at all, so it has no value
    /// rather than a value of zero (see `Model::blind_moe_*`).
    fn moe_branch(&self, l: usize, a2: &Array2<f32>, scan: Option<(usize, usize)>) -> (Array2<f32>, Vec<f32>) {
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
        let mut scan_out: Vec<f32> = scan.map(|_| vec![f32::NAN; seq]).unwrap_or_default();
        // Prefetch every active expert's weights up front (MADV_WILLNEED) so the OS pages experts 2..k from the mmap
        // while expert 1 is computed — overlapping the per-token page-in stalls that bound MoE decode under offload.
        for &e in assign.keys() {
            self.b.prefetch(&format!("{p}experts.{e}.gate"));
            self.b.prefetch(&format!("{p}experts.{e}.up"));
            self.b.prefetch(&format!("{p}experts.{e}.down"));
        }
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
                if let Some((se, sn)) = scan {
                    if *e == se && sn < hh.ncols() {
                        scan_out[t] = wgt * hh[[i, sn]];
                    }
                }
                for c in 0..self.d { out[[t, c]] += wgt * down[[i, c]]; }
            }
        }
        // Always-on shared expert (Qwen2-MoE): the same SwiGLU over the same post-attention-normed hidden, scaled per
        // token by sigmoid(shared_expert_gate(h)) and ADDED to the routed mixture — not concatenated, not weighted into
        // the top-k. Mixtral/Qwen3-MoE have no shared expert (shared_inter == 0) and skip this entirely.
        if self.shared_inter > 0 {
            let gate = self.b.mm(a2, &format!("{p}shared.gate_proj"));
            let up = self.b.mm(a2, &format!("{p}shared.up_proj"));
            let mut hh = gate;
            for (hv, uv) in hh.iter_mut().zip(up.iter()) { *hv = silu(*hv) * uv; }
            let sd = self.b.mm(&hh, &format!("{p}shared.down_proj")); // (seq, d)
            let sg = self.b.mm(a2, &format!("{p}shared_gate"));       // (seq, 1)
            for t in 0..seq {
                let g = 1.0 / (1.0 + (-sg[[t, 0]]).exp());
                for c in 0..self.d { out[[t, c]] += g * sd[[t, c]]; }
            }
        }
        (out, scan_out)
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
            let mut q = self.attention_proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.attention_proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.attention_proj(&a, &format!("{p}self_attn.v_proj"));
            self.qk_norm_apply(&p, &mut q, &mut k); // QK-norm before RoPE (absent on Mixtral/Qwen2-MoE)
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
            x = &x + &self.attention_proj(&attn_out, &format!("{p}self_attn.o_proj"));

            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let (mlp, _) = if self.moe[l] { self.moe_branch(l, &a2, None) } else { (self.dense_mlp(l, &a2), vec![]) };
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    /// Forward capturing the recursion substrate (mirrors `Rope::recursion_capture` for the Qwen3-MoE backbone):
    /// per-layer post-block residual (for the per-layer logit-lens) + the element-wise max over late-layer heads of
    /// the attention matrix (the binding signal; late = last third). MoE/dense FFN per layer exactly as in `hidden`.
    fn recursion_capture(&self, ids: &[i64]) -> (Vec<Array2<f32>>, Array2<f32>) {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let late0 = 2 * self.n_layer / 3;
        let mut x = self.b.rows_f32("embed", ids);
        let mut resids: Vec<Array2<f32>> = Vec::with_capacity(self.n_layer);
        let mut maxback = Array2::<f32>::zeros((seq, seq));
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.attention_proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.attention_proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.attention_proj(&a, &format!("{p}self_attn.v_proj"));
            self.qk_norm_apply(&p, &mut q, &mut k);
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
                        if j > i || (self.window > 0 && j + self.window <= i) { scores[[i, j]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                if l >= late0 {
                    for i in 0..seq {
                        for j in 0..=i {
                            if scores[[i, j]] > maxback[[i, j]] { maxback[[i, j]] = scores[[i, j]]; }
                        }
                    }
                }
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &self.attention_proj(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let (mlp, _) = if self.moe[l] { self.moe_branch(l, &a2, None) } else { (self.dense_mlp(l, &a2), vec![]) };
            x = &x + &mlp;
            resids.push(x.clone());
        }
        (resids, maxback)
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
            let mut q = self.attention_proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.attention_proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.attention_proj(&a, &format!("{p}self_attn.v_proj"));
            self.qk_norm_apply(&p, &mut q, &mut k);
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
            x = &x + &self.attention_proj(&attn_out, &format!("{p}self_attn.o_proj"));
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
            let (mlp, _) = if self.moe[l] { self.moe_branch(l, &a2, None) } else { (self.dense_mlp(l, &a2), vec![]) };
            x = &x + &mlp;
        }
        let xf = self.norm(&x, "norm");
        let x_last = x.row(seq - 1).to_vec(); // residual either side of the final norm — recovers its frozen scale
        let xf_last = xf.row(seq - 1).to_vec();
        let un = self.unembed();
        let lg = self.b.rowdot_f32(un, &xf_last);
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let gain = self.b.arr1("norm").to_vec();
        let u_pred = self.b.weight_row(un, model_predicts as usize);
        assemble(
            ids,
            &att_last,
            &head_act,
            &mlp_h,
            &lg,
            model_predicts,
            &gain,
            false,
            &[],
            &x_last,
            &xf_last,
            &u_pred,
            0,
            &|_v| Vec::new(),
            |l, n| match feat_src[l] {
                Some(e) => self.b.expert_row(&format!("l{l}.experts.{e}.down"), n),
                None => self.b.weight_row(&format!("l{l}.mlp.down_proj"), n),
            },
            |l, head| head_raw_contrib(&self.b, &format!("l{l}.self_attn.o_proj"), &head_act[l], head, hd),
            |c| self.b.rowdot_f32(un, c),
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
            let mut q = self.attention_proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.attention_proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.attention_proj(&a, &format!("{p}self_attn.v_proj"));
            self.qk_norm_apply(&p, &mut q, &mut k);
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
            x = &x + &self.attention_proj(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let (mlp, _) = if self.moe[l] { self.moe_branch(l, &a2, None) } else { (self.dense_mlp(l, &a2), vec![]) };
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    /// `forward_block` that also returns the per-row value of ONE routed unit at the target layer — the
    /// blind-max scoring kernel. Identical walk and arithmetic (same QK-norm guard, same RoPE positions,
    /// same window, same routed+shared FFN), so its value is bit-identical to a naive per-position forward;
    /// the difference is only that it ADVANCES the KV cache, which is why the scorer snapshots before probing.
    fn forward_block_scan(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>],
                          scan: Option<(usize, usize, usize)>) -> Vec<f32> {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let mut scan_out: Vec<f32> = Vec::new();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.attention_proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.attention_proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.attention_proj(&a, &format!("{p}self_attn.v_proj"));
            self.qk_norm_apply(&p, &mut q, &mut k);
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
            x = &x + &self.attention_proj(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let (mlp, vals) = if self.moe[l] {
                let sc = scan.filter(|(sl, ..)| *sl == l).map(|(_, e, n)| (e, n));
                self.moe_branch(l, &a2, sc)
            } else {
                (self.dense_mlp(l, &a2), Vec::new())
            };
            if !vals.is_empty() { scan_out = vals; }
            x = &x + &mlp;
        }
        scan_out
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
            let mut q = self.attention_proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.attention_proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.attention_proj(&a, &format!("{p}self_attn.v_proj"));
            self.qk_norm_apply(&p, &mut q, &mut k);
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
            x = &x + &self.attention_proj(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let (mlp, _) = if self.moe[l] { self.moe_branch(l, &a2, None) } else { (self.dense_mlp(l, &a2), vec![]) };
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

/// The blind-max scorer for this arch: a KV-cache twin of [`Qwen3Moe::forward_block_scan`] held at position `pos`,
/// with a snapshot so candidate tokens never see each other's writes. (Qwen3.5 needs a hybrid DeltaNet state
/// instead; here the cache is row-indexed, so a clone/restore is all the isolation a probe needs.)
struct Qwen3Scorer {
    kc: Vec<Array2<f32>>,
    vc: Vec<Array2<f32>>,
    snap_kc: Vec<Array2<f32>>,
    snap_vc: Vec<Array2<f32>>,
    pos: usize,
}

impl Qwen3Scorer {
    /// Prefill `prefix`, then score candidates at the next position.
    fn new(m: &Qwen3Moe, prefix: &[i64], total: usize) -> Option<Qwen3Scorer> {
        if prefix.is_empty() || prefix.len() >= total {
            return None;
        }
        let kvdim = m.nkv * m.hd;
        let mut kc: Vec<Array2<f32>> = (0..m.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc: Vec<Array2<f32>> = (0..m.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let emb = m.b.rows_f32("embed", prefix);
        m.forward_block(&emb, 0, &mut kc, &mut vc);
        let snap_kc = kc.clone();
        let snap_vc = vc.clone();
        Some(Qwen3Scorer { kc, vc, snap_kc, snap_vc, pos: prefix.len() })
    }

    /// An empty scorer at position 0 — no seed context, for the corpus scan, which places every token.
    fn empty(m: &Qwen3Moe, total: usize) -> Qwen3Scorer {
        let kvdim = m.nkv * m.hd;
        let kc: Vec<Array2<f32>> = (0..m.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let vc = kc.clone();
        let snap_kc = kc.clone();
        let snap_vc = vc.clone();
        Qwen3Scorer { kc, vc, snap_kc, snap_vc, pos: 0 }
    }

    /// Score `token` at `pos` by the ROUTED unit's `w_router · act`, WITHOUT committing. `None` for a bad
    /// unit; `Some(NaN)` when the router did not select that expert here — the unit has no value, which is
    /// not the same as a value of zero.
    fn try_token_moe(&mut self, m: &Qwen3Moe, token: i64, layer: usize, expert: usize, neuron: usize) -> Option<f32> {
        if layer >= m.n_layer || expert >= m.n_exp { return None; }
        self.kc.clone_from(&self.snap_kc); // candidates must not see each other's writes
        self.vc.clone_from(&self.snap_vc);
        let emb = m.b.rows_f32("embed", &[token]);
        let vals = m.forward_block_scan(&emb, self.pos, &mut self.kc, &mut self.vc, Some((layer, expert, neuron)));
        vals.first().copied()
    }

    /// Advance the cache by `token` (the winner's value was already read by the probe).
    fn commit(&mut self, m: &Qwen3Moe, token: i64) {
        let emb = m.b.rows_f32("embed", &[token]);
        m.forward_block(&emb, self.pos, &mut self.kc, &mut self.vc);
        self.pos += 1;
        self.snap_kc.clone_from(&self.kc);
        self.snap_vc.clone_from(&self.vc);
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

    fn logits(&self, ids: &[i64]) -> Option<Vec<f32>> {
        let xf = self.hidden(ids);
        Some(self.b.rowdot_f32(self.unembed(), &xf.row(ids.len() - 1).to_vec()))
    }

    fn dims(&self) -> Option<(usize, usize)> {
        Some((self.n_layer, self.h))
    }

    /// The scorer IS implemented here, so a MoE bundle must not be reported as `NoScorer` — it is
    /// `RoutedLayers`: a token's router picks which experts run, so "neuron n" of a layer has no fixed
    /// meaning across positions, while expert `e`'s neuron `n` DOES (`--blind-max L,E,N`). Naming which
    /// of the two problems it is, is the whole point of this method.
    fn blind_support(&self) -> crate::model::BlindSupport {
        if self.n_exp != 0 { crate::model::BlindSupport::RoutedLayers } else { crate::model::BlindSupport::Supported }
    }

    /// One ROUTED unit's self-description: expert `e` neuron `n`'s own output row (`experts.{e}.down`),
    /// scaled by the final-norm gain and projected through the unembed — the same construction
    /// `blind_proposals` uses for a dense `mlp.down_proj`, so the two are comparable.
    fn blind_moe_proposals(&self, layer: usize, expert: usize, neuron: usize, k: usize, polarity: f32) -> Option<Vec<i64>> {
        if layer >= self.n_layer || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        // Expert weights are PAGED, not resident: `weight_row` cannot see them, so go through
        // `expert_f32`, which returns (shape, values) as (in, out) row-major. down is (mi, d), so neuron
        // `n`'s output direction is one contiguous d-wide row of it.
        let (shape, vals) = self.b.expert_f32(&format!("l{layer}.experts.{expert}.down"));
        let d = shape[1];
        if neuron >= shape[0] {
            return None;
        }
        let gain = self.b.arr1("norm").to_vec();
        let mut w = vals[neuron * d..(neuron + 1) * d].to_vec();
        w.iter_mut().zip(gain.iter()).for_each(|(v, g)| *v *= g);
        let proj = self.b.rowdot_f32(self.unembed(), &w);
        let mut idx: Vec<usize> = (0..proj.len()).collect();
        idx.sort_by(|&a, &b| {
            (polarity * proj[b]).partial_cmp(&(polarity * proj[a])).unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(idx.into_iter().take(k).map(|i| i as i64).collect())
    }

    /// Greedy hill-climb for one routed unit, scored by `w_router · act`. Candidates the router does not
    /// select score `NaN` and are skipped rather than treated as zero, so the climb never "wins" by
    /// steering the router away from the expert it is trying to maximise.
    fn blind_moe_climb(&self, layer: usize, expert: usize, neuron: usize, prefix: &[i64], cands: &[i64],
                       steps: usize, polarity: f32) -> Option<crate::model::BlindArm> {
        if cands.is_empty() || layer >= self.n_layer || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let mut sc = Qwen3Scorer::new(self, prefix, prefix.len() + steps + 1)?;
        let mut ids = prefix.to_vec();
        let mut best = f32::NEG_INFINITY;
        for _ in 0..steps {
            let mut pick = None;
            let mut pick_obj = f32::NEG_INFINITY;
            for &t in cands {
                match sc.try_token_moe(self, t, layer, expert, neuron) {
                    Some(v) if !v.is_nan() && polarity * v > pick_obj => {
                        pick_obj = polarity * v;
                        pick = Some(t);
                    }
                    _ => {}
                }
            }
            let Some(t) = pick else { break }; // no candidate reaches this expert on this position
            sc.commit(self, t);
            ids.push(t);
            best = best.max(pick_obj);
        }
        if best == f32::NEG_INFINITY {
            return None;
        }
        Some(crate::model::BlindArm { ids, obj: best, pos: None })
    }

    /// Oracle for one routed unit over a corpus: best `w_router · act` among positions that route to
    /// `expert`. Chunked exactly like the dense arm (one weight read amortised over CORPUS_CHUNK
    /// positions, each still scored with true causal history) and advancing the same KV cache.
    fn blind_moe_corpus_best(&self, layer: usize, expert: usize, neuron: usize, ids: &[i64],
                             polarity: f32) -> Option<crate::model::BlindArm> {
        if ids.len() < 2 || layer >= self.n_layer || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let mut sc = Qwen3Scorer::empty(self, ids.len());
        let (mut best_obj, mut best_idx, mut routed) = (f32::NEG_INFINITY, 0usize, 0usize);
        for chunk in ids.chunks(crate::model::CORPUS_CHUNK) {
            let start = sc.pos;
            let emb = self.b.rows_f32("embed", chunk);
            let vals = self.forward_block_scan(&emb, start, &mut sc.kc, &mut sc.vc, Some((layer, expert, neuron)));
            sc.pos += chunk.len();
            if vals.len() != chunk.len() {
                continue;
            }
            for (j, &v) in vals.iter().enumerate() {
                if v.is_nan() {
                    continue; // expert not in this row's top-k: unit not computed there
                }
                routed += 1;
                let o = polarity * v;
                if o > best_obj {
                    best_obj = o;
                    best_idx = start + j;
                }
            }
        }
        if routed == 0 {
            return None; // this expert is never selected anywhere in the corpus
        }
        let lo = best_idx.saturating_sub(crate::model::CORPUS_CTX - 1);
        Some(crate::model::BlindArm { ids: ids[lo..=best_idx].to_vec(), obj: best_obj, pos: Some(best_idx) })
    }

    fn recursion_trace(&self, ids: &[i64]) -> Option<Vec<crate::model::RecPos>> {
        if ids.len() < 3 { return Some(vec![]); }
        let (resids, maxback) = self.recursion_capture(ids);
        let un = self.unembed();
        // arch-specific lens only: final RMSNorm ("norm") then unembed argmax; the rest is shared (build_rec_trace).
        Some(crate::model::build_rec_trace(&resids, maxback, 2 * self.n_layer / 3, |_l, resid| {
            let normed = self.norm(resid, "norm");
            (0..normed.nrows())
                .map(|p| {
                    let lg = self.b.rowdot_f32(un, &normed.row(p).to_vec());
                    lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
                })
                .collect()
        }))
    }

    fn recursion_lens_at(&self, ids: &[i64], positions: &[usize]) -> Option<Vec<Vec<(usize, i64)>>> {
        let (resids, _) = self.recursion_capture(ids);
        let un = self.unembed();
        Some(crate::model::build_rec_lens_at(&resids, positions, 2 * self.n_layer / 3, |resid, p| {
            let normed = self.norm(&resid.slice(s![p..p + 1, ..]).to_owned(), "norm");
            let lg = self.b.rowdot_f32(un, &normed.row(0).to_vec());
            lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
        }))
    }

    /// Per-block DLA decomposition (the `--pil-dump` seam for the Qwen3-MoE family). Same rope-style RMSNorm
    /// geometry as `Rope::residual_decomp` — embedding, then each layer's attention `o_proj` and FFN writes (the
    /// FFN is the routed-expert mixture `moe_branch` for MoE layers, the dense MLP otherwise) sum to the pre-final-
    /// norm residual, so per-block contribution = `inv_rms · Σ_d gain_d write_b_d U_v_d`. Block-level: the whole
    /// MoE layer counts as one "mlp" block (per-expert is a finer follow-up).
    fn residual_decomp(&self, ids: &[i64], toks: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let last = seq - 1;
        let mut x = self.b.rows_f32("embed", ids);
        let mut labels: Vec<String> = vec!["embed".into()];
        let mut writes: Vec<Vec<f32>> = vec![x.row(last).to_vec()];
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.attention_proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.attention_proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.attention_proj(&a, &format!("{p}self_attn.v_proj"));
            self.qk_norm_apply(&p, &mut q, &mut k);
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
                        if j > i || (self.window > 0 && j + self.window <= i) { scores[[i, j]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            let aw = self.attention_proj(&attn_out, &format!("{p}self_attn.o_proj"));
            writes.push(aw.row(last).to_vec());
            labels.push(format!("L{l}.attn"));
            x = &x + &aw;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let (mw, _) = if self.moe[l] { self.moe_branch(l, &a2, None) } else { (self.dense_mlp(l, &a2), vec![]) };
            writes.push(mw.row(last).to_vec());
            labels.push(format!("L{l}.{}", if self.moe[l] { "moe" } else { "mlp" }));
            x = &x + &mw;
        }
        let xpre = x.row(last).to_vec();
        let d = xpre.len();
        let inv_rms = 1.0 / (xpre.iter().map(|v| v * v).sum::<f32>() / d as f32 + self.eps).sqrt();
        let gain = self.b.arr1("norm");
        let un = self.unembed();
        let urows: Vec<Vec<f32>> = toks.iter().map(|&t| self.b.weight_row(un, t as usize)).collect();
        let contrib: Vec<Vec<f32>> = writes
            .iter()
            .map(|w| urows.iter().map(|u| inv_rms * (0..d).map(|i| gain[i] * w[i] * u[i]).sum::<f32>()).collect())
            .collect();
        Some((labels, contrib))
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
        if self.kv_int8 {
            // int8 KV cache for chat/serve — 4x smaller cache (longer context in the same budget); lossy by design.
            let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|_| vec![0i8; total * kvdim]).collect();
            let mut vc = kc.clone();
            let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|_| vec![0f32; total * self.nkv]).collect();
            let mut vs = ks.clone();
            let emb = self.b.rows_f32("embed", prompt);
            let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
            let mut next = self.unembed_argmax(&xb);
            let mut out = Vec::new();
            let mut pos = prompt.len();
            loop {
                if eos.contains(&next) { break; }
                out.push(next);
                if !emit(next) || out.len() == max_tokens { break; }
                let e = self.b.rows_f32("embed", &[next]);
                let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
                next = self.unembed_argmax(&xb);
                pos += 1;
            }
            return out;
        }
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

    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool, cache: &mut crate::model::PrefixKv) -> Vec<i64> {
        if self.kv_int8 {
            let (kvdim, nkv, n_layer) = (self.nkv * self.hd, self.nkv, self.n_layer);
            let alloc = |total: usize| {
                let kc: Vec<Vec<i8>> = (0..n_layer).map(|_| vec![0i8; total * kvdim]).collect();
                let vc = kc.clone();
                let ks: Vec<Vec<f32>> = (0..n_layer).map(|_| vec![0f32; total * nkv]).collect();
                let vs = ks.clone();
                (kc, vc, ks, vs)
            };
            let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>], vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]| {
                let emb = self.b.rows_f32("embed", ids);
                self.forward_block_q(&emb, cur, kc, ks, vc, vs)
            };
            return crate::model::prefix_generate_q(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, _ctx| self.unembed_argmax(xb));
        }
        let (kvdim, n_layer) = (self.nkv * self.hd, self.n_layer);
        let alloc = |total: usize| {
            let kc: Vec<Array2<f32>> = (0..n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
            let vc = kc.clone();
            (kc, vc)
        };
        let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]| {
            let emb = self.b.rows_f32("embed", ids);
            self.forward_block(&emb, cur, kc, vc)
        };
        crate::model::prefix_generate(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, _ctx| self.unembed_argmax(xb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::Bundle;
    use crate::model::BlindSupport;

    fn v(seed: usize, n: usize) -> Vec<f32> {
        (0..n).map(|i| (((seed * 41 + i * 23) % 43) as f32) / 20.0 - 1.0).collect()
    }

    /// A tiny `qwen3moe` bundle written straight to disk, with knobs for the three family-specific bits so
    /// one fixture can stand for Qwen3-MoE (QK-norm), Mixtral (no QK-norm) and Qwen2-MoE (no QK-norm +
    /// a gated shared expert). Each test gets its OWN path — `cargo test` runs these concurrently.
    fn build_tiny(tag: &str, moe: bool, qk: bool, shared: usize) -> Qwen3Moe {
        let (nl, nh, nkv, hd, d, vocab, ffn) = (4usize, 4, 2, 8, 32, 24, 16);
        let n_exp = if moe { 4usize } else { 0 };
        let mi = 32usize;
        let dir = std::env::temp_dir().join(format!("fr_bm_q3m_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stem = dir.join("b").to_string_lossy().into_owned();
        let mut bin: Vec<u8> = Vec::new();
        let mut specs: Vec<serde_json::Value> = Vec::new();
        let mut add = |name: String, shape: Vec<usize>, data: Vec<f32>| {
            let offset = bin.len();
            for x in &data { bin.extend_from_slice(&x.to_le_bytes()); }
            specs.push(serde_json::json!({ "name": name, "dtype": "f32", "shape": shape,
                "offset": offset, "bytes": bin.len() - offset }));
        };
        add("embed".into(), vec![vocab, d], v(1, vocab * d));
        add("norm".into(), vec![d], v(2, d));
        add("lm_head".into(), vec![vocab, d], v(3, vocab * d));
        for l in 0..nl {
            let p = format!("l{l}.");
            add(format!("{p}in_ln"), vec![d], v(10 + l, d));
            add(format!("{p}post_ln"), vec![d], v(20 + l, d));
            add(format!("{p}self_attn.q_proj"), vec![d, nh * hd], v(300 + l, d * nh * hd));
            add(format!("{p}self_attn.k_proj"), vec![d, nkv * hd], v(310 + l, d * nkv * hd));
            add(format!("{p}self_attn.v_proj"), vec![d, nkv * hd], v(320 + l, d * nkv * hd));
            add(format!("{p}self_attn.o_proj"), vec![nh * hd, d], v(330 + l, nh * hd * d));
            if shared > 0 {
                for (proj, width, seed) in [("q_proj", nh * hd, 360), ("k_proj", nkv * hd, 370), ("v_proj", nkv * hd, 380)] {
                    add(format!("{p}self_attn.{proj}.bias"), vec![width], v(seed + l, width));
                }
            }
            if qk {
                add(format!("{p}q_norm"), vec![hd], v(340 + l, hd));
                add(format!("{p}k_norm"), vec![hd], v(350 + l, hd));
            }
            if moe {
                add(format!("{p}gate"), vec![d, n_exp], v(100 + l, d * n_exp));
                for e in 0..n_exp {
                    // `.experts.` in the name is what puts these in the PAGED map `expert_f32` reads.
                    add(format!("{p}experts.{e}.gate"), vec![d, mi], v(110 + e, d * mi));
                    add(format!("{p}experts.{e}.up"), vec![d, mi], v(120 + e, d * mi));
                    add(format!("{p}experts.{e}.down"), vec![mi, d], v(130 + e, mi * d));
                }
                if shared > 0 {
                    add(format!("{p}shared.gate_proj"), vec![d, shared], v(140 + l, d * shared));
                    add(format!("{p}shared.up_proj"), vec![d, shared], v(150 + l, d * shared));
                    add(format!("{p}shared.down_proj"), vec![shared, d], v(160 + l, shared * d));
                    add(format!("{p}shared_gate"), vec![d, 1], v(170 + l, d));
                }
            } else {
                add(format!("{p}mlp.gate_proj"), vec![d, ffn], v(70 + l, d * ffn));
                add(format!("{p}mlp.up_proj"), vec![d, ffn], v(80 + l, d * ffn));
                add(format!("{p}mlp.down_proj"), vec![ffn, d], v(90 + l, ffn * d));
            }
        }
        let mut cfg: Vec<i64> = vec![nl as i64, nh as i64, nkv as i64, hd as i64, d as i64, ffn as i64,
                                    vocab as i64, 0, n_exp as i64, 2, mi as i64, 1];
        for _ in 0..nl { cfg.push(if moe { 1 } else { 0 }); }
        cfg.push(0); // window: full attention
        cfg.push(qk as i64);
        cfg.push((shared > 0) as i64);
        cfg.push(shared as i64);
        let manifest = serde_json::json!({ "format": crate::bundle::FORMAT, "version": crate::bundle::VERSION,
            "arch": "qwen3moe", "config": cfg, "config_f": [10000.0, 1e-5], "arrays": specs });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
        Qwen3Moe::new(Bundle::load(&stem).unwrap(), 0.0, false)
    }

    /// Naive re-walk: the post-attention-normed hidden entering layer `l`'s FFN, with NO cache. This is
    /// deliberately an independent copy of the attention walk rather than a call into the cached path —
    /// the point of the test below is that the cached scorer agrees with a from-scratch forward.
    fn naive_a2(m: &Qwen3Moe, ids: &[i64], l: usize) -> Array2<f32> {
        let seq = ids.len();
        let (h, nkv, hd) = (m.h, m.nkv, m.hd);
        let rep = h / nkv;
        let mut x = m.b.rows_f32("embed", ids);
        for j in 0..l {
            let p = format!("l{j}.");
            let a = m.norm(&x, &format!("{p}in_ln"));
            let mut q = m.attention_proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = m.attention_proj(&a, &format!("{p}self_attn.k_proj"));
            let v = m.attention_proj(&a, &format!("{p}self_attn.v_proj"));
            m.qk_norm_apply(&p, &mut q, &mut k);
            m.rope(&mut q, h, 0);
            m.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * m.scale;
                for i in 0..seq {
                    for j2 in 0..seq {
                        if j2 > i { scores[[i, j2]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &m.attention_proj(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = m.norm(&x, &format!("{p}post_ln"));
            let (mlp, _) = if m.moe[j] { m.moe_branch(j, &a2, None) } else { (m.dense_mlp(j, &a2), vec![]) };
            x = &x + &mlp;
        }
        let p = format!("l{l}.");
        let a = m.norm(&x, &format!("{p}in_ln"));
        let mut q = m.attention_proj(&a, &format!("{p}self_attn.q_proj"));
        let mut k = m.attention_proj(&a, &format!("{p}self_attn.k_proj"));
        let v = m.attention_proj(&a, &format!("{p}self_attn.v_proj"));
        m.qk_norm_apply(&p, &mut q, &mut k);
        m.rope(&mut q, h, 0);
        m.rope(&mut k, nkv, 0);
        let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
        for head in 0..h {
            let kv = head / rep;
            let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
            let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
            let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
            let mut scores = qh.dot(&kh.t()) * m.scale;
            for i in 0..seq {
                for j2 in 0..seq {
                    if j2 > i { scores[[i, j2]] = -1e30; }
                }
            }
            softmax_rows(&mut scores);
            attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
        }
        x = &x + &m.attention_proj(&attn_out, &format!("{p}self_attn.o_proj"));
        m.norm(&x, &format!("{p}post_ln"))
    }

    /// The whole routed probe rests on this: one cached scorer step over the KV state must equal a naive
    /// from-scratch forward over the same ids, bit for bit. If the snapshot/restore or the QK-norm guard
    /// drifted by even one ULP, every blind-max ratio on this arch would be measured against a fiction.
    #[test]
    fn qwen3_scorer_step_matches_naive_bitwise() {
        for (tag, qk, shared) in [("qk", true, 0usize), ("noqk", false, 0), ("shared", false, 24)] {
            let m = build_tiny(tag, true, qk, shared);
            let ids: Vec<i64> = vec![1, 9, 4, 7, 2, 5];
            let total = ids.len() + 6;
            for layer in [0usize, 2, 3] {
                let mut sc = Qwen3Scorer::new(&m, &ids, total).unwrap();
                assert_eq!(sc.pos, ids.len());
                for (i, &tok) in [7i64, 3, 11].iter().enumerate() {
                    let (e, n) = ((i + layer) % m.n_exp, (i * 5) % 32);
                    let got = sc.try_token_moe(&m, tok, layer, e, n);
                    let mut want_ids = ids.clone();
                    want_ids.push(tok);
                    let a2 = naive_a2(&m, &want_ids, layer);
                    let (_, want) = m.moe_branch(layer, &a2, Some((e, n)));
                    let want = want[want.len() - 1];
                    let got = got.unwrap();
                    if want.is_nan() {
                        assert!(got.is_nan(), "{tag} L{layer}: expert not selected, so both must be NaN");
                    } else {
                        assert_eq!(got.to_bits(), want.to_bits(),
                            "{tag} layer {layer} tok {tok}: scorer {:e} != naive {:e}", got, want);
                    }
                }
            }
        }
    }

    /// Candidates must be INDEPENDENT: probing B after A must equal probing B on a fresh scorer. Without
    /// the snapshot restore, A's cache writes would leak into B.
    #[test]
    fn qwen3_candidates_are_independent() {
        let m = build_tiny("indep", true, true, 0);
        let ids: Vec<i64> = vec![2, 4, 6];
        let (l, e, n) = (1usize, 2usize, 3usize);
        let mut sc = Qwen3Scorer::new(&m, &ids, 32).unwrap();
        let _ = sc.try_token_moe(&m, 11, l, e, n);
        let after_a = sc.try_token_moe(&m, 7, l, e, n).unwrap();
        let mut fresh = Qwen3Scorer::new(&m, &ids, 32).unwrap();
        let on_fresh = fresh.try_token_moe(&m, 7, l, e, n).unwrap();
        assert_eq!(after_a.to_bits(), on_fresh.to_bits(),
            "probing A then B differs from probing B alone: candidates leak into each other");
    }

    /// The routed objective must equal `w_router · silu(gate_n) · up_n` written out by hand, and must be NaN
    /// (not zero) exactly where the router did not select that expert.
    #[test]
    fn qwen3_routed_objective_matches_reference() {
        let m = build_tiny("ref", true, false, 24);
        let (l, e, n) = (1usize, 2usize, 7usize);
        let stream: Vec<i64> = (0..6).map(|i: i64| i).collect();
        let a2 = naive_a2(&m, &stream, l);
        let (_out, got) = m.moe_branch(l, &a2, Some((e, n)));
        let mut logits = m.b.mm(&a2, &format!("l{l}.gate"));
        softmax_rows(&mut logits);
        let (_, gv) = m.b.expert_f32(&format!("l{l}.experts.{e}.gate"));
        let (_, uv) = m.b.expert_f32(&format!("l{l}.experts.{e}.up"));
        let (dsh, _) = m.b.expert_f32(&format!("l{l}.experts.{e}.down"));
        let (d, mi) = (dsh[1], dsh[0]);
        assert_eq!(got.len(), stream.len(), "one value per position");
        let mut n_routed = 0;
        for t in 0..stream.len() {
            let row = logits.row(t).to_vec();
            let mut idx: Vec<usize> = (0..m.n_exp).collect();
            idx.sort_by(|&x, &y| row[y].partial_cmp(&row[x]).unwrap());
            idx.truncate(m.topk);
            if !idx.contains(&e) {
                assert!(got[t].is_nan(), "t={t}: expert not selected, so the unit has NO value there");
                continue;
            }
            n_routed += 1;
            let denom: f32 = if m.norm_topk { idx.iter().map(|&x| row[x]).sum::<f32>().max(1e-20) } else { 1.0 };
            let w = row[e] / denom;
            // matmul first, THEN silu: gate/up are (d, mi) and project the input, they do not scale it
            let (mut g, mut u) = (0f32, 0f32);
            for i in 0..d {
                g += a2[[t, i]] * gv[i * mi + n];
                u += a2[[t, i]] * uv[i * mi + n];
            }
            let want = w * silu(g) * u;
            // Tolerance, not bits: this reference re-does the dot product in its own summation order, and
            // `expert_mm` sums it in the GEMM's (a couple of ULP apart on a length-32 dot). What this test
            // pins is the OBJECTIVE — `w_router · silu(gate_n) · up_n` — not the GEMM's associativity; the
            // bit-exact invariant is the cached-vs-naive scorer test above, where both sides share expert_mm.
            assert!((got[t] - want).abs() <= 1e-5 * want.abs().max(1e-3),
                "t={t}: branch {:e} != reference {:e}", got[t], want);
        }
        assert!(n_routed > 0, "no position selected the expert — the reference proves nothing");
    }

    /// A MoE bundle must report `RoutedLayers`, NOT `NoScorer`: the scorer IS implemented here, and calling
    /// it unsupported is the exact conflation this enum exists to prevent.
    #[test]
    fn qwen3_moe_bundle_reports_routed_layers() {
        let moe = build_tiny("decl", true, true, 0);
        assert_eq!(moe.blind_support(), BlindSupport::RoutedLayers);
        let dense = build_tiny("decl2", false, true, 0);
        assert_eq!(dense.blind_support(), BlindSupport::Supported);
    }

    /// The QK-norm bit must actually gate: with it off, a bundle that DOES carry q_norm/k_norm must not
    /// apply them, so the two must disagree. Otherwise the guard is dead code and the six call sites could
    /// regress to an unconditional norm without any test noticing.
    #[test]
    fn qwen3_qk_norm_bit_is_load_bearing() {
        let on = build_tiny("qkon", true, true, 0);
        let mut off = build_tiny("qkoff", true, true, 0);
        assert!(on.qk_norm && off.qk_norm, "fixture should write the bit on for both");
        off.qk_norm = false;
        let ids: Vec<i64> = vec![1, 9, 4, 7, 2, 5];
        let (a, b) = (on.hidden(&ids), off.hidden(&ids));
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(diff > 1e-6, "the QK-norm bit changed nothing ({diff:e}) — the guard is dead code");
    }
}
