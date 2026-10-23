//! Tier B — composition, GPT-OSS (`model_type: gpt_oss`). A faithful port of `GptOssForCausalLM`.
//!
//! Four things make this arch different from every other MoE kernel in the tree, and all four are easy to
//! get subtly wrong (each "works" — produces numbers, loses top-1 agreement — rather than crashing):
//!
//! 1. **The router has a bias and takes the top-k on the RAW logits.** `GptOssTopKRouter` computes
//!    `logits = h·Wᵀ + b`, takes `topk` over those logits, and only then softmaxes **the top-k values
//!    among themselves**. It is NOT a softmax over all experts, and there is no correction bias: the
//!    weight is `softmax(top-k logits)`, not a normalised subset of a full softmax.
//! 2. **The expert gate/up are INTERLEAVED**, not concatenated: `gate, up = gate_up[..., ::2], gate_up[...,
//!    1::2]`, so gate `j` lives at column `2j` and up `j` at `2j+1`. Every other arch here splits
//!    `gate_up` in half.
//! 3. **The clamped SwiGLU is not `silu(clamp(g))·clamp(u)`.** It is `glu = clamp(g,max=limit)·sigmoid(
//!    clamp(g,max=limit)·alpha)` and then `(clamp(u,±limit) + 1)·glu` — the `+1` bias on the up branch is
//!    load-bearing. `alpha = 1.702`, `limit = 7.0`. Both the routed experts and the biases
//!    (`gate_up_proj_bias`, `down_proj_bias`) are per-expert.
//! 4. **Attention has a per-head learned SINK** and the attention projections carry **biases**. The sink
//!    is concatenated to the logits, the softmax runs over `[scores; sink]`, and the sink column is then
//!    DROPPED — so it absorbs probability mass rather than being renormalised away. A zero sink is not a
//!    no-op.
//!
//! Attention layers ALTERNATE: `layer_types[i] = "sliding_attention" if (i+1)%2 else "full_attention"`, so
//! layer 0 SLIDES and layer 1 is full. RoPE is **YaRN** (factor 32, beta_fast 32, beta_slow 1, truncate
//! False, original max 4096) over the FULL head_dim. Norms use the weight DIRECTLY (`GptOssRMSNorm` is
//! "T5LayerNorm", not Llama's `1+w`). Experts are written one array each so the mmap-offload contract holds.
use std::collections::HashMap;

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct GptOss {
    b: Bundle,
    n_layer: usize,
    h: usize,
    nkv: usize,
    hd: usize,
    d: usize,
    eps: f32,
    scale: f32,
    inv: Vec<f32>,
    att_factor: f32,
    alpha: f32,
    limit: f32,
    n_exp: usize,
    topk: usize,
    inter: usize,
    window: usize,
    sliding: Vec<bool>,
    tied: bool,
    kv_int8: bool,
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax over ONE row in place. Used for the router (over the top-k values only) and for attention
/// (over `[scores; sink]`).
fn softmax_row(a: &mut [f32]) {
    let m = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in a.iter_mut() { *v = (*v - m).exp(); s += *v; }
    let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
    for v in a.iter_mut() { *v *= inv; }
}

impl GptOss {
    pub fn new(b: Bundle, _route: f32, kv_int8: bool) -> GptOss {
        // config: [nl, nh, nkv, hd, d, vocab, tied, n_exp, topk, inter, <nl sliding flags>]
        let c = &b.config;
        let (n_layer, h, nkv, hd, d) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize, c[4] as usize);
        let tied = c[6] != 0;
        let (n_exp, topk, inter) = (c[7] as usize, c[8] as usize, c[9] as usize);
        let sliding: Vec<bool> = (0..n_layer).map(|l| c[10 + l] != 0).collect();
        let window = if c.len() > 10 + n_layer { c[10 + n_layer] as usize } else { 0 };
        // config_f: [theta, eps, alpha, limit, attn_factor, yarn_factor, beta_fast, beta_slow, orig_max]
        let (theta, eps, alpha, limit, att_factor) =
            (b.config_f[0], b.config_f[1], b.config_f[2], b.config_f[3], b.config_f[4]);
        let (eps, alpha, limit, att_factor) = (eps as f32, alpha as f32, limit as f32, att_factor as f32);
        let inv = crate::mla::yarn_inv_freq(
            theta, hd, b.config_f[5], b.config_f[6], b.config_f[7], b.config_f[8], false);
        // `attention_factor` scales cos/sin ONLY. `GptOssAttention.scaling` is a bare `head_dim**-0.5`,
        // with NO attention factor in it (DeepSeek-V3 is the one that adds `mscale_all_dim**2` to the
        // softmax scale; this arch does not). Folding it into the softmax scale instead — the obvious
        // "both" reading — still produces a full-looking logit vector and only loses top-1 positions.
        let scale = (hd as f32).powf(-0.5);
        GptOss { b, n_layer, h, nkv, hd, d, eps, scale, inv, att_factor, alpha, limit, n_exp, topk, inter,
                 window, sliding, tied, kv_int8 }
    }

    fn unembed(&self) -> &str { if self.tied { "embed" } else { "lm_head" } }

    /// RMSNorm with the weight used DIRECTLY — `GptOssRMSNorm` is T5-style, the "main diff with Llama".
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

    /// Causal (+ window) attention scores with the per-head **sink appended as the last column**.
    ///
    /// The sink must live INSIDE the softmax and then be dropped — that is what makes it a sink. Renormalising
    /// the real scores after masking, or folding the sink back in afterwards, both change the answer while
    /// still producing plausible numbers. `window = 0` means full attention (this family ALTERNATES layers).
    fn attn_scores(&self, qh: &ndarray::ArrayView2<f32>, kh: &ndarray::ArrayView2<f32>, sink: f32,
                   rows: usize, klen: usize, pos0: usize, window: usize) -> Array2<f32> {
        let mut sc = Array2::<f32>::zeros((rows, klen + 1));
        sc.slice_mut(s![.., ..klen]).assign(&(qh.dot(&kh.t()) * self.scale));
        for i in 0..rows {
            // NB the sink column is deliberately NEVER masked: it is concatenated, softmaxed WITH the
            // scores, and dropped afterwards. Masking it (a tempting `j == klen` in the loop below) turns
            // the sink into a no-op — its whole purpose is to absorb probability mass.
            sc[[i, klen]] = sink;
            let abs = pos0 + i;
            for j in 0..klen {
                if j > abs || (window > 0 && j + window <= abs) { sc[[i, j]] = -1e30; }
            }
        }
        sc
    }

    /// Rotary embedding on a (m, n_heads*hd) block; row i is absolute position `pos0 + i`. Split-half,
    /// matching `_apply_rotary_emb`'s `chunk(x, 2, dim=-1)`. `inv` already carries the YaRN ramp.
    ///
    /// YaRN's `attention_factor` multiplies cos/sin (so it scales the rotated vectors), which is where it
    /// lands in transformers: `cos = emb.cos() * self.attention_scaling` inside the rotary embedding.
    fn rope(&self, x: &mut Array2<f32>, n_heads: usize, pos0: usize) {
        let half = self.hd / 2;
        for (i, mut row) in x.rows_mut().into_iter().enumerate() {
            let pos = pos0 + i;
            for head in 0..n_heads {
                let base = head * self.hd;
                for j in 0..half {
                    let ang = pos as f32 * self.inv[j];
                    let (c, s) = (ang.cos() * self.att_factor, ang.sin() * self.att_factor);
                    let (a, bb) = (row[base + j], row[base + j + half]);
                    row[base + j] = a * c - bb * s;
                    row[base + j + half] = bb * c + a * s;
                }
            }
        }
    }

    /// Router: top-k on the RAW (biased) logits, then a softmax over just those top-k values.
    /// `scan = Some((expert, neuron))` also returns the per-row `w · act` for that one routed unit, where
    /// `act` is the clamped SwiGLU hidden. Rows whose top-k excludes the expert stay `NaN` — the unit is
    /// not computed there, so it has no value rather than a value of zero.
    fn moe(&self, l: usize, a2: &Array2<f32>, scan: Option<(usize, usize)>) -> (Array2<f32>, Vec<f32>) {
        let p = format!("l{l}.");
        let seq = a2.nrows();
        let mut logits = self.b.mm(a2, &format!("{p}router"));          // (seq, n_exp)
        let rbias = self.b.arr1o(&format!("{p}router.bias"));            // (n_exp)
        for mut row in logits.rows_mut() {
            for (i, v) in row.iter_mut().enumerate() { *v += rbias[i]; }
        }
        let mut assign: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for t in 0..seq {
            let raw = logits.row(t).to_vec();
            let mut idx: Vec<usize> = (0..self.n_exp).collect();
            idx.sort_by(|&a, &b| raw[b].partial_cmp(&raw[a]).unwrap());
            idx.truncate(self.topk);
            let mut kept: Vec<f32> = idx.iter().map(|&e| raw[e]).collect();
            softmax_row(&mut kept);   // softmax over the TOP-K VALUES ONLY
            for (i, &e) in idx.iter().enumerate() {
                assign.entry(e).or_default().push((t, kept[i]));
            }
        }
        let mut out = Array2::<f32>::zeros((seq, self.d));
        let mut scan_out: Vec<f32> = scan.map(|_| vec![f32::NAN; seq]).unwrap_or_default();
        for &e in assign.keys() {
            self.b.prefetch(&format!("l{l}.experts.{e}.gate_up"));
            self.b.prefetch(&format!("l{l}.experts.{e}.down"));
        }
        for (e, toks) in &assign {
            let mut rows = Array2::<f32>::zeros((toks.len(), self.d));
            for (i, &(t, _)) in toks.iter().enumerate() { rows.row_mut(i).assign(&a2.row(t)); }
            let (outp, glu) = self.expert(l, *e, &rows);
            // the scanned neuron, if this expert is the scanned one; rows whose top-k EXCLUDES it never reach
            // this loop, so their scan_out stays NaN — the unit is not computed there, which is not value 0.
            let sn = scan.filter(|(se, _)| se == e).map(|(_, n)| n).filter(|&n| n < self.inter);
            for (i, &(t, w)) in toks.iter().enumerate() {
                if let Some(n) = sn { scan_out[t] = w * glu[[i, n]]; }
                for c in 0..self.d { out[[t, c]] += w * outp[[i, c]]; }
            }
        }
        (out, scan_out)
    }

    /// One expert: `down((up+1) · glu) + down_bias`, with the INTERLEAVED gate/up and the gpt-oss clamp.
    /// Returns `(down, glu)` — `glu` is the clamped hidden the scan and `explain` both read, and it is already
    /// materialised here, so handing it back costs a move rather than a recompute.
    fn expert(&self, l: usize, e: usize, rows: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let prefix = format!("l{l}.experts.{e}.");
        let gu = self.b.expert_mm_bias(rows, &format!("{prefix}gate_up"), &format!("l{l}.expertbias.{e}.gate_up"));
        let mi = self.inter;
        let mut glu = Array2::<f32>::zeros((gu.nrows(), mi));
        for i in 0..gu.nrows() {
            for j in 0..mi {
                // INTERLEAVED: gate at 2j, up at 2j+1 — NOT a half/half split
                let g = gu[[i, 2 * j]].min(self.limit);
                let u = gu[[i, 2 * j + 1]].clamp(-self.limit, self.limit);
                glu[[i, j]] = (u + 1.0) * (g * sigmoid(g * self.alpha));
            }
        }
        let down = self.b.expert_mm_bias(&glu, &format!("{prefix}down"), &format!("l{l}.expertbias.{e}.down"));
        (down, glu)
    }

    /// Naive full forward (no cache) returning the post-final-norm hidden.
    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm_bias(&a, &format!("{p}self_attn.q_proj"), &format!("{p}self_attn.q_proj.bias"));
            let mut k = self.b.mm_bias(&a, &format!("{p}self_attn.k_proj"), &format!("{p}self_attn.k_proj.bias"));
            let v = self.b.mm_bias(&a, &format!("{p}self_attn.v_proj"), &format!("{p}self_attn.v_proj.bias"));
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let sinks = self.b.arr1o(&format!("{p}self_attn.sinks"));
            let window = if self.sliding[l] { self.window } else { 0 };
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let sc = self.attn_scores(&qh, &kh, sinks[head], seq, seq, 0, window);
                let mut owned = sc.to_owned();
                for mut row in owned.rows_mut() { softmax_row(row.as_slice_mut().unwrap()); }
                let real = owned.slice(s![.., ..seq]);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&real.dot(&vh));
            }
            let ao = self.b.mm_bias(&attn_out, &format!("{p}self_attn.o_proj"), &format!("{p}self_attn.o_proj.bias"));
            x = &x + &ao;
            let a2 = self.norm(&x, &format!("{p}post_attention_layernorm"));
            let (mlp, _) = self.moe(l, &a2, None);
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    /// `hidden` that also returns the per-row value of ONE routed unit at the target layer — the blind-max
    /// scoring kernel. A copy of the walk above with the MoE scan plumbed in, because a probe that went
    /// through a DIFFERENT walk than the model's is exactly how a scorer silently stops meaning anything.
    /// The twin's agreement with `hidden` is asserted in this file's tests.
    fn hidden_scan(&self, ids: &[i64], scan: (usize, usize, usize)) -> Vec<f32> {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        let mut scan_out: Vec<f32> = Vec::new();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm_bias(&a, &format!("{p}self_attn.q_proj"), &format!("{p}self_attn.q_proj.bias"));
            let mut k = self.b.mm_bias(&a, &format!("{p}self_attn.k_proj"), &format!("{p}self_attn.k_proj.bias"));
            let v = self.b.mm_bias(&a, &format!("{p}self_attn.v_proj"), &format!("{p}self_attn.v_proj.bias"));
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let sinks = self.b.arr1o(&format!("{p}self_attn.sinks"));
            let window = if self.sliding[l] { self.window } else { 0 };
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let sc = self.attn_scores(&qh, &kh, sinks[head], seq, seq, 0, window);
                let mut owned = sc.to_owned();
                for mut row in owned.rows_mut() { softmax_row(row.as_slice_mut().unwrap()); }
                let real = owned.slice(s![.., ..seq]);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&real.dot(&vh));
            }
            let ao = self.b.mm_bias(&attn_out, &format!("{p}self_attn.o_proj"), &format!("{p}self_attn.o_proj.bias"));
            x = &x + &ao;
            let a2 = self.norm(&x, &format!("{p}post_attention_layernorm"));
            let (mlp, vals) = self.moe(l, &a2, if scan.0 == l { Some((scan.1, scan.2)) } else { None });
            if !vals.is_empty() { scan_out = vals; }
            x = &x + &mlp;
        }
        scan_out
    }

    fn unembed_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let logits = self.b.rowdot_f32(self.unembed(), &xfn.row(xfn.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    /// Run `m` new positions through the layers, caching K/V (post-RoPE, GQA width `nkv*hd`) and attending over
    /// the whole cache. `cur` is the absolute position of the first new row.
    ///
    /// The sink is a function of the whole row, so it cannot be folded into the incremental path the way a plain
    /// score can — it is re-materialised per head here exactly as in `hidden`, which is why the cached decode is
    /// byte-identical to the naive recompute (asserted in `validate_all.sh`).
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm_bias(&a, &format!("{p}self_attn.q_proj"), &format!("{p}self_attn.q_proj.bias"));
            let mut k = self.b.mm_bias(&a, &format!("{p}self_attn.k_proj"), &format!("{p}self_attn.k_proj.bias"));
            let v = self.b.mm_bias(&a, &format!("{p}self_attn.v_proj"), &format!("{p}self_attn.v_proj.bias"));
            self.rope(&mut q, h, cur);
            self.rope(&mut k, nkv, cur);
            kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
            vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            let sinks = self.b.arr1o(&format!("{p}self_attn.sinks"));
            let window = if self.sliding[l] { self.window } else { 0 };
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = kc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let vh = vc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let sc = self.attn_scores(&qh, &kh, sinks[head], m, klen, cur, window);
                let mut owned = sc.to_owned();
                for mut row in owned.rows_mut() { softmax_row(row.as_slice_mut().unwrap()); }
                let real = owned.slice(s![.., ..klen]);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&real.dot(&vh));
            }
            let ao = self.b.mm_bias(&attn_out, &format!("{p}self_attn.o_proj"), &format!("{p}self_attn.o_proj.bias"));
            x = &x + &ao;
            let a2 = self.norm(&x, &format!("{p}post_attention_layernorm"));
            let (mlp, _) = self.moe(l, &a2, None);
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    /// `forward_block` with an int8 KV cache (GQA width, per-kv-head scale): ~4x smaller, lossy by design.
    /// Lossy here in the same way as everywhere else: the sink and the scores stay f32, only K/V are quantised.
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
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm_bias(&a, &format!("{p}self_attn.q_proj"), &format!("{p}self_attn.q_proj.bias"));
            let mut k = self.b.mm_bias(&a, &format!("{p}self_attn.k_proj"), &format!("{p}self_attn.k_proj.bias"));
            let v = self.b.mm_bias(&a, &format!("{p}self_attn.v_proj"), &format!("{p}self_attn.v_proj.bias"));
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
            let sinks = self.b.arr1o(&format!("{p}self_attn.sinks"));
            let window = if self.sliding[l] { self.window } else { 0 };
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
                let sc = self.attn_scores(&qh, &kh_a.view(), sinks[head], m, klen, cur, window);
                let mut owned = sc.to_owned();
                for mut row in owned.rows_mut() { softmax_row(row.as_slice_mut().unwrap()); }
                let real = owned.slice(s![.., ..klen]);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&real.dot(&vh_a));
            }
            let ao = self.b.mm_bias(&attn_out, &format!("{p}self_attn.o_proj"), &format!("{p}self_attn.o_proj.bias"));
            x = &x + &ao;
            let a2 = self.norm(&x, &format!("{p}post_attention_layernorm"));
            let (mlp, _) = self.moe(l, &a2, None);
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    /// The dominant expert at one row (highest RAW router logit — the same quantity the top-k selects on) and
    /// its clamped-SwiGLU hidden, which is what `explain` prints as this layer's `promotes` vector.
    fn top_expert_feature(&self, l: usize, row: &Array2<f32>) -> (usize, Vec<f32>) {
        let p = format!("l{l}.");
        let mut logits = self.b.mm(row, &format!("{p}router"));
        let rbias = self.b.arr1o(&format!("{p}router.bias"));
        for mut r in logits.rows_mut() {
            for (i, v) in r.iter_mut().enumerate() { *v += rbias[i]; }
        }
        let r = logits.row(0);
        let e = (0..self.n_exp).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap();
        let (_, glu) = self.expert(l, e, row);
        (e, glu.row(0).to_vec())
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
        let mut feat_src: Vec<usize> = Vec::new(); // the dominant expert at the final row
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm_bias(&a, &format!("{p}self_attn.q_proj"), &format!("{p}self_attn.q_proj.bias"));
            let mut k = self.b.mm_bias(&a, &format!("{p}self_attn.k_proj"), &format!("{p}self_attn.k_proj.bias"));
            let v = self.b.mm_bias(&a, &format!("{p}self_attn.v_proj"), &format!("{p}self_attn.v_proj.bias"));
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let sinks = self.b.arr1o(&format!("{p}self_attn.sinks"));
            let window = if self.sliding[l] { self.window } else { 0 };
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            let mut layer_att = Vec::with_capacity(h);
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let sc = self.attn_scores(&qh, &kh, sinks[head], seq, seq, 0, window);
                let mut owned = sc.to_owned();
                for mut row in owned.rows_mut() { softmax_row(row.as_slice_mut().unwrap()); }
                // The SINK column is dropped here exactly as the model drops it, so `mass` in the readout is the
                // REAL attention mass (<1 for a sink-heavy head). A head whose learned sink absorbs the row
                // therefore reads as diffuse/NO-OP rather than as "attends to position 0" — which is the truth.
                layer_att.push(owned.row(seq - 1).slice(s![..seq]).to_vec());
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd])
                    .assign(&owned.slice(s![.., ..seq]).dot(&vh));
            }
            att_last.push(layer_att);
            head_act.push(attn_out.row(seq - 1).to_vec());
            let ao = self.b.mm_bias(&attn_out, &format!("{p}self_attn.o_proj"), &format!("{p}self_attn.o_proj.bias"));
            x = &x + &ao;
            let a2 = self.norm(&x, &format!("{p}post_attention_layernorm"));
            let last = a2.slice(s![seq - 1..seq, ..]).to_owned();
            let (e, hidden) = self.top_expert_feature(l, &last);
            feat_src.push(e);
            mlp_h.push(hidden);
            let (mlp, _) = self.moe(l, &a2, None);
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
            // experts are PAGED, so the neuron's own write row must be read through expert_f32, not weight_row
            |l, n| {
                let (shape, vals) = self.b.expert_f32(&format!("l{}.experts.{}.down", l, feat_src[l]));
                let w = shape[1];
                vals[n * w..(n + 1) * w].to_vec()
            },
            // o_proj's BIAS is a constant added to every head, so it cancels in a per-head contribution.
            |l, head| head_raw_contrib(&self.b, &format!("l{l}.self_attn.o_proj"), &head_act[l], head, hd),
            |c| self.b.rowdot_f32(un, c),
        )
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
            if out.len() == n_new { return out; }
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
            next = self.unembed_argmax(&xb);
            pos += 1;
        }
    }
}

impl Model for GptOss {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let logits = self.b.rowdot_f32(self.unembed(), &xf.row(ids.len() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn logits(&self, ids: &[i64]) -> Option<Vec<f32>> {
        let xf = self.hidden(ids);
        Some(self.b.rowdot_f32(self.unembed(), &xf.row(ids.len() - 1).to_vec()))
    }

    fn dims(&self) -> Option<(usize, usize)> { Some((self.n_layer, self.h)) }

    fn explain(&self, ids: &[i64]) -> Option<crate::explain::Explanation> { Some(self.explanation(ids)) }

    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        if prompt.is_empty() || n_new == 0 { return Vec::new(); }
        if self.kv_int8 { return self.generate_kv_int8(prompt, n_new); }
        let total = prompt.len() + n_new;
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc = kc.clone();
        let emb = self.b.rows_f32("embed", prompt);
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = self.unembed_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new { return out; }
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.unembed_argmax(&xb);
            pos += 1;
        }
    }

    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        if prompt.is_empty() || max_tokens == 0 { return Vec::new(); }
        let total = prompt.len() + max_tokens;
        let kvdim = self.nkv * self.hd;
        if self.kv_int8 {
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
            if eos.contains(&next) { break; }
            out.push(next);
            if !emit(next) || out.len() == max_tokens { break; }
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.unembed_argmax(&xb);
            pos += 1;
        }
        out
    }

    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool,
                               cache: &mut crate::model::PrefixKv) -> Vec<i64> {
        let (kvdim, nkv, n_layer) = (self.nkv * self.hd, self.nkv, self.n_layer);
        if self.kv_int8 {
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
            return crate::model::prefix_generate_q(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd,
                                                   &|xb, _ctx| self.unembed_argmax(xb));
        }
        let alloc = |total: usize| {
            let kc: Vec<Array2<f32>> = (0..n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
            let vc = kc.clone();
            (kc, vc)
        };
        let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]| {
            let emb = self.b.rows_f32("embed", ids);
            self.forward_block(&emb, cur, kc, vc)
        };
        crate::model::prefix_generate(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd,
                                      &|xb, _ctx| self.unembed_argmax(xb))
    }

    /// `blind_support` is coherent with the arms below: the scorer IS implemented for this arch, so a MoE
    /// bundle reports `RoutedLayers` rather than the `NoScorer` default.
    fn blind_support(&self) -> crate::model::BlindSupport {
        if self.n_exp != 0 { crate::model::BlindSupport::RoutedLayers } else { crate::model::BlindSupport::NoScorer }
    }

    /// One ROUTED unit's self-description: expert `e` neuron `n`'s own output row (`experts.{e}.down`),
    /// scaled by the final-norm gain and projected through the unembed.
    fn blind_moe_proposals(&self, layer: usize, expert: usize, neuron: usize, k: usize, polarity: f32) -> Option<Vec<i64>> {
        if layer >= self.n_layer || self.n_exp == 0 || expert >= self.n_exp { return None; }
        let (shape, vals) = self.b.expert_f32(&format!("l{layer}.experts.{expert}.down"));
        let d = shape[1];
        if neuron >= shape[0] { return None; }
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
    /// select score `NaN` and are SKIPPED rather than treated as zero — otherwise the climb would "win" by
    /// steering the router away from the expert it is trying to maximise.
    fn blind_moe_climb(&self, layer: usize, expert: usize, neuron: usize, prefix: &[i64], cands: &[i64],
                       steps: usize, polarity: f32) -> Option<crate::model::BlindArm> {
        if cands.is_empty() || layer >= self.n_layer || self.n_exp == 0 || expert >= self.n_exp { return None; }
        let mut ids = prefix.to_vec();
        let mut best = f32::NEG_INFINITY;
        for _ in 0..steps {
            let mut pick = None;
            let mut pick_obj = f32::NEG_INFINITY;
            for &t in cands {
                let mut probe = ids.clone();
                probe.push(t);
                let vals = self.hidden_scan(&probe, (layer, expert, neuron));
                if let Some(&v) = vals.last() {
                    if !v.is_nan() && polarity * v > pick_obj { pick_obj = polarity * v; pick = Some(t); }
                }
            }
            let Some(t) = pick else { break }; // no candidate reaches this expert on this position
            ids.push(t);
            best = best.max(pick_obj);
        }
        if best == f32::NEG_INFINITY { return None; }
        Some(crate::model::BlindArm { ids, obj: best, pos: None })
    }

    /// Oracle for one routed unit over a corpus: best `w_router · act` among positions that route to
    /// `expert`, chunked exactly like the dense arm. Each chunk is a causal sweep — every position in it
    /// still sees its true history — and one weight read is amortised over CORPUS_CHUNK positions.
    fn blind_moe_corpus_best(&self, layer: usize, expert: usize, neuron: usize, ids: &[i64],
                             polarity: f32) -> Option<crate::model::BlindArm> {
        if ids.len() < 2 || layer >= self.n_layer || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let (mut best_obj, mut best_idx, mut routed) = (f32::NEG_INFINITY, 0usize, 0usize);
        // carried context: each chunk attends over everything before it, as the dense arm does
        let mut carried: Vec<i64> = Vec::new();
        for chunk in ids.chunks(crate::model::CORPUS_CHUNK) {
            let start = carried.len();
            let mut window = carried.clone();
            window.extend_from_slice(chunk);
            let vals = self.hidden_scan(&window, (layer, expert, neuron));
            let tail = &vals[vals.len().saturating_sub(chunk.len())..];
            if tail.len() == chunk.len() {
                for (j, &v) in tail.iter().enumerate() {
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
            carried = window;
        }
        if routed == 0 {
            return None; // this expert is never selected anywhere in the corpus
        }
        let lo = best_idx.saturating_sub(crate::model::CORPUS_CTX - 1);
        Some(crate::model::BlindArm { ids: ids[lo..=best_idx].to_vec(), obj: best_obj, pos: Some(best_idx) })
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

    /// A tiny `gptoss` bundle written straight to disk. `sink0` sets the per-head attention sink; `window`
    /// is the sliding width. config: [nl, nh, nkv, hd, d, vocab, tied, n_exp, topk, inter, <sliding>, window]
    /// config_f: [theta, eps, alpha, limit, att_factor, yarn_factor, beta_fast, beta_slow, orig_max]
    fn build_tiny(tag: &str, window: usize, sink0: f32) -> GptOss {
        let (nl, nh, nkv, hd, d, vocab, mi) = (4usize, 4, 2, 8, 32, 24, 16usize);
        let (n_exp, topk) = (4usize, 2usize);
        let dir = std::env::temp_dir().join(format!("fr_bm_gptoss_{tag}_{}", std::process::id()));
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
            add(format!("{p}input_layernorm"), vec![d], v(10 + l, d));
            add(format!("{p}post_attention_layernorm"), vec![d], v(20 + l, d));
            add(format!("{p}self_attn.q_proj"), vec![d, nh * hd], v(300 + l, d * nh * hd));
            add(format!("{p}self_attn.k_proj"), vec![d, nkv * hd], v(310 + l, d * nkv * hd));
            add(format!("{p}self_attn.v_proj"), vec![d, nkv * hd], v(320 + l, d * nkv * hd));
            add(format!("{p}self_attn.o_proj"), vec![nh * hd, d], v(330 + l, nh * hd * d));
            add(format!("{p}self_attn.q_proj.bias"), vec![nh * hd], v(340 + l, nh * hd));
            add(format!("{p}self_attn.k_proj.bias"), vec![nkv * hd], v(350 + l, nkv * hd));
            add(format!("{p}self_attn.v_proj.bias"), vec![nkv * hd], v(360 + l, nkv * hd));
            add(format!("{p}self_attn.o_proj.bias"), vec![d], v(370 + l, d));
            add(format!("{p}self_attn.sinks"), vec![nh], vec![sink0; nh]);
            add(format!("{p}router"), vec![d, n_exp], v(100 + l, d * n_exp));
            add(format!("{p}router.bias"), vec![n_exp], v(105 + l, n_exp));
            for e in 0..n_exp {
                // `.experts.` in the name routes into the PAGED map `expert_f32` reads; the per-expert biases
                // are deliberately NOT named that way — they must stay resident.
                add(format!("{p}experts.{e}.gate_up"), vec![d, 2 * mi], v(110 + e, d * 2 * mi));
                add(format!("{p}expertbias.{e}.gate_up"), vec![2 * mi], v(115 + e, 2 * mi));
                add(format!("{p}experts.{e}.down"), vec![mi, d], v(120 + e, mi * d));
                add(format!("{p}expertbias.{e}.down"), vec![d], v(125 + e, d));
            }
        }
        let mut cfg: Vec<i64> = vec![nl as i64, nh as i64, nkv as i64, hd as i64, d as i64, vocab as i64,
                                    0, n_exp as i64, topk as i64, mi as i64];
        // the real family ALTERNATES: layer 0 slides, layer 1 is full
        for l in 0..nl { cfg.push(((l + 1) % 2 != 0) as i64); }
        cfg.push(window as i64);
        let manifest = serde_json::json!({ "format": crate::bundle::FORMAT, "version": crate::bundle::VERSION,
            "arch": "gptoss", "config": cfg,
            "config_f": [150000.0, 1e-5, 1.702, 7.0, 1.0, 32.0, 32.0, 1.0, 4096.0], "arrays": specs });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
        GptOss::new(Bundle::load(&stem).unwrap(), 0.0, false)
    }

    /// The post-attention-normed hidden entering layer `l`'s FFN, walked from scratch (no cache, no scan).
    /// Deliberately an independent copy of the walk rather than a call into `hidden`/`hidden_scan`: the tests
    /// below are only meaningful if the reference and the thing under test are two different walks.
    fn naive_a2(m: &GptOss, ids: &[i64], l: usize) -> Array2<f32> {
        let seq = ids.len();
        let (h, nkv, hd) = (m.h, m.nkv, m.hd);
        let rep = h / nkv;
        let mut x = m.b.rows_f32("embed", ids);
        for ll in 0..=l {
            let p = format!("l{ll}.");
            let a = m.norm(&x, &format!("{p}input_layernorm"));
            let mut q = m.b.mm_bias(&a, &format!("{p}self_attn.q_proj"), &format!("{p}self_attn.q_proj.bias"));
            let mut k = m.b.mm_bias(&a, &format!("{p}self_attn.k_proj"), &format!("{p}self_attn.k_proj.bias"));
            let v = m.b.mm_bias(&a, &format!("{p}self_attn.v_proj"), &format!("{p}self_attn.v_proj.bias"));
            m.rope(&mut q, h, 0);
            m.rope(&mut k, nkv, 0);
            let sinks = m.b.arr1o(&format!("{p}self_attn.sinks"));
            let window = if m.sliding[ll] { m.window } else { 0 };
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let sc = m.attn_scores(&qh, &kh, sinks[head], seq, seq, 0, window);
                let mut own = sc.to_owned();
                for mut row in own.rows_mut() { softmax_row(row.as_slice_mut().unwrap()); }
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd])
                    .assign(&own.slice(s![.., ..seq]).dot(&vh));
            }
            let ao = m.b.mm_bias(&attn_out, &format!("{p}self_attn.o_proj"), &format!("{p}self_attn.o_proj.bias"));
            x = &x + &ao;
            if ll == l { break; }
            let (mlp, _) = m.moe(ll, &m.norm(&x, &format!("{p}post_attention_layernorm")), None);
            x = &x + &mlp;
        }
        m.norm(&x, &format!("l{l}.post_attention_layernorm"))
    }

    /// The CACHED decode must be byte-identical to the naive full recompute. The sink is the thing that could
    /// break this: it is a per-row column concatenated to the scores, so a cache that stored it wrong (or
    /// folded it into the real scores) would still emit plausible tokens and fail only here.
    #[test]
    fn gptoss_kv_decode_is_bit_identical_to_naive() {
        let m = build_tiny("kv", 3, 0.7);
        let prompt: Vec<i64> = (0..7).collect();
        let mut naive = Vec::new();
        let mut ids = prompt.clone();
        for _ in 0..6 {
            let n = m.predict(&ids);
            naive.push(n);
            ids.push(n);
        }
        let total = prompt.len() + 6;
        let kvdim = m.nkv * m.hd;
        let mut kc: Vec<Array2<f32>> = (0..m.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc = kc.clone();
        let emb = m.b.rows_f32("embed", &prompt);
        let xb = m.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = m.unembed_argmax(&xb);
        let mut via_cache = Vec::new();
        let mut pos = prompt.len();
        for _ in 0..6 {
            via_cache.push(next);
            let e = m.b.rows_f32("embed", &[next]);
            let xb = m.forward_block(&e, pos, &mut kc, &mut vc);
            next = m.unembed_argmax(&xb);
            pos += 1;
        }
        assert_eq!(via_cache, naive, "KV decode diverged from the naive full recompute");
        assert_eq!(via_cache, m.generate(&prompt, 6), "generate() must take the cached path");
    }

    /// `hidden_scan` is the blind-max scorer. It must agree BITWISE with the same unit scored off an
    /// independently walked `a2` — otherwise the probe measures a different model than the one that predicts.
    #[test]
    fn gptoss_scan_twin_matches_an_independent_walk() {
        let m = build_tiny("scan", 3, 0.7);
        let ids: Vec<i64> = (0..8).collect();
        let (l, e, n) = (1usize, 2usize, 5usize);
        let got = m.hidden_scan(&ids, (l, e, n));
        let a2 = naive_a2(&m, &ids, l);
        let (_o, want) = m.moe(l, &a2, Some((e, n)));
        assert_eq!(got.len(), want.len());
        for t in 0..got.len() {
            assert_eq!(got[t].to_bits(), want[t].to_bits(),
                "t={t}: twin {:e} != independent walk {:e}", got[t], want[t]);
        }
        assert!(got.iter().any(|v| v.is_nan()),
            "with top-k 2 of 4 some positions must skip the expert, and those must be NaN");
        assert!(got.iter().any(|v| !v.is_nan()), "no position selected the expert — proves nothing");
    }

    /// Candidates must not see each other. The scorer re-walks from the same prefix every probe, so asking for
    /// the same candidate twice must give the identical number; if the probe advanced any shared state, the
    /// second answer would be taken against a longer history than the first.
    #[test]
    fn gptoss_candidates_are_independent() {
        let m = build_tiny("indep", 3, 0.7);
        let (l, e, n) = (2usize, 1usize, 3usize);
        let prefix: Vec<i64> = vec![3, 9, 4];
        let cands: Vec<i64> = vec![1, 7, 12, 20];
        let pass = |cands: &[i64]| -> Vec<f32> {
            cands.iter().map(|&t| {
                let mut p = prefix.clone();
                p.push(t);
                m.hidden_scan(&p, (l, e, n)).last().copied().unwrap()
            }).collect()
        };
        let first = pass(&cands);
        let second = pass(&cands);
        for (i, (a, b)) in first.iter().zip(&second).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "candidate {i} changed between passes: {a} vs {b}");
        }
    }

    /// The routed objective is `w · glu_n` with `w = softmax(top-k RAW biased logits)[rank of e]`. Two
    /// near-misses both "work": softmaxing over ALL experts (DeepSeek/MiniMax's choice-vs-weight split is NOT
    /// this arch's), or taking the top-k after a softmax. This pins the actual form.
    #[test]
    fn gptoss_routed_objective_matches_reference() {
        let m = build_tiny("ref", 3, 0.7);
        let (l, e, n) = (1usize, 3usize, 4usize);
        let ids: Vec<i64> = (0..6).collect();
        let a2 = naive_a2(&m, &ids, l);
        let (_o, got) = m.moe(l, &a2, Some((e, n)));
        let mut logits = m.b.mm(&a2, &format!("l{l}.router"));
        let rb = m.b.arr1o(&format!("l{l}.router.bias"));
        for mut r in logits.rows_mut() { for (i, val) in r.iter_mut().enumerate() { *val += rb[i]; } }
        let mut n_routed = 0;
        for t in 0..a2.nrows() {
            let raw = logits.row(t).to_vec();
            let mut idx: Vec<usize> = (0..m.n_exp).collect();
            idx.sort_by(|&a, &b| raw[b].partial_cmp(&raw[a]).unwrap());
            idx.truncate(m.topk);
            if !idx.contains(&e) {
                assert!(got[t].is_nan(), "t={t}: expert not selected -> NO value, not zero");
                continue;
            }
            n_routed += 1;
            // softmax over the TOP-K VALUES ONLY
            let kept: Vec<f32> = idx.iter().map(|&x| raw[x]).collect();
            let mx = kept.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = kept.iter().map(|v| (v - mx).exp()).collect();
            let w = ex[idx.iter().position(|&x| x == e).unwrap()] / ex.iter().sum::<f32>();
            let (_, glu) = m.expert(l, e, &a2.slice(s![t..t + 1, ..]).to_owned());
            let want = w * glu[[0, n]];
            assert!((got[t] - want).abs() <= 1e-5 * want.abs().max(1e-3),
                "t={t}: {:e} != reference {:e}", got[t], want);
        }
        assert!(n_routed > 0, "no position selected the expert — the reference proves nothing");
    }

    /// A MoE bundle must report `RoutedLayers`, NOT `NoScorer`: the scorer IS implemented here, so claiming
    /// unsupported is the exact conflation this enum exists to prevent.
    #[test]
    fn gptoss_bundle_reports_routed_layers() {
        let m = build_tiny("decl", 3, 0.7);
        assert_eq!(m.blind_support(), BlindSupport::RoutedLayers);
    }

    /// The attention SINK is not a no-op: masking it (folding it out of the softmax) must change the forward.
    /// This is the one GPT-OSS-specific behaviour that is cheap to get wrong, because a zero sink would pass.
    /// Both bundles are built from the same deterministic weights, so the sink is the only difference.
    #[test]
    fn gptoss_sink_is_not_a_no_op() {
        let ids: Vec<i64> = (0..9).collect();
        let with = build_tiny("sinkA", 3, 0.9);
        let zeroed = build_tiny("sinkB", 3, 0.0);
        assert_ne!(with.hidden(&ids), zeroed.hidden(&ids),
            "the sink had no effect — it is not reaching the softmax at all");
    }

    /// The sliding window must apply on the SLIDING layers only. Applying it everywhere (or nowhere) both look
    /// like "attention with a window"; only the alternating pattern matches the reference.
    #[test]
    fn gptoss_alternating_window_is_load_bearing() {
        let m = build_tiny("alt", 3, 0.5);
        assert!(m.sliding[0] && !m.sliding[1] && m.sliding[2] && !m.sliding[3],
            "the family alternates: 0 sliding, 1 full, 2 sliding, 3 full");
        assert_eq!(m.window, 3);
    }
    #[test]
    fn gptoss_corpus_oracle_matches_causal_positions() {
        let m = build_tiny("oracle", 3, 0.7);
        let ids: Vec<i64> = (0..40).map(|i| (i * 7 % 24) as i64).collect();
        let (layer, neuron) = (1, 5);
        for expert in 0..m.n_exp {
            for polarity in [1.0, -1.0] {
                let mut expected = None;
                let mut best = f32::NEG_INFINITY;
                for pos in 0..ids.len() {
                    let vals = m.hidden_scan(&ids[..=pos], (layer, expert, neuron));
                    let obj = polarity * vals[pos];
                    if obj > best { best = obj; expected = Some(pos); }
                }
                let got = m.blind_moe_corpus_best(layer, expert, neuron, &ids, polarity);
                assert_eq!(got.as_ref().and_then(|a| a.pos), expected);
                if let Some(arm) = got {
                    assert!((arm.obj - best).abs() <= 1e-5 * best.abs().max(1.0));
                    assert_eq!(*arm.ids.last().unwrap(), ids[expected.unwrap()]);
                }
            }
        }
        assert!(m.blind_moe_corpus_best(layer, m.n_exp, neuron, &ids, 1.0).is_none());
    }

    #[test]
    fn gptoss_zero_token_generation_is_empty() {
        let mut m = build_tiny("empty", 3, 0.7);
        for quantized in [false, true] {
            m.kv_int8 = quantized;
            assert!(m.generate(&[1, 2], 0).is_empty());
            assert!(m.generate(&[], 2).is_empty());
            let mut emit = |_| panic!("zero-token generation must not emit");
            assert!(m.generate_stream(&[1, 2], 0, &[], &mut emit).is_empty());
            assert!(m.generate_stream(&[], 2, &[], &mut emit).is_empty());
        }
    }

}
