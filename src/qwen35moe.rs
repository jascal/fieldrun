//! Qwen3.5 hybrid Gated-DeltaNet / full-attention, with dense SwiGLU or MoE FFNs. Layout: the `3×linear
//! → 1×full` pattern (per `layer_types`); 3 of every 4 layers are Gated DeltaNet *linear*
//! attention (`crate::deltanet`, golden-tested vs transformers in #112–#114), the 4th is full GQA attention.
//!
//! Architecture quirks this port implements (each verified against `Qwen3_5MoeForCausalLM`):
//!   • full attention is GATED: `q_proj → [query | gate]` per head; `attn_out ·= sigmoid(gate)` before o_proj
//!   • PARTIAL RoPE: only the first `rotary_dim = head_dim·partial_rotary_factor` of each head is rotated
//!   • MoE variant: softmax top-k, ALWAYS-renormalized weights (independent of norm_topk_prob) + a
//!     sigmoid-gated shared expert; dense variant: ordinary SwiGLU
//!   • RMSNorm is GEMMA-style `·(1+weight)` (the gating bug that took the longest); the linear path's output
//!     gate is the separate `Qwen3_5MoeRMSNormGated` = plain `·weight` (see `deltanet::rmsnorm_gated`)
//!
//! Validated to DECODE PARITY vs transformers on a tiny model (`experiments/qwen3next/`): the full-attention
//! op matches to 9.7e-9 (`full_attn_golden.py`), and end-to-end logits argmax = 100% (max diff 1.27e-3),
//! per-layer residual ≤ 7e-3 — so the MoE renorm, gated attention, shared expert, and partial RoPE are
//! confirmed correct, not just asserted. Harness: make_tiny → convert → `--hidden-dump`/`--logits-dump` → compare.
//! Dense Qwen3.5 text (MiMo-V2.6-Distill-Qwen-9B's backbone) is gated 60/60 top-1 against torch by
//! `scripts/qwen35_dense_ref.py`; the MoE variant is gated argmax-for-argmax against transformers by
//! `experiments/qwen3next/`.
//!
//! DECODE is incremental: `HybridState` below carries the Gated DeltaNet recurrence (per-value-head `S`
//! plus the conv1d rolling window) alongside the full-attention K/V cache, so each token is O(1) instead of
//! re-running the prefix — 6.0x on Qwen3.5-0.8B, byte-identical to `predict`. `--kv-int8` quantises the
//! full-attention K/V only; the recurrent state stays f32 (there is no natural per-token scale for it).
//! Because the state is RECURRENT rather than row-indexed, cross-turn prefix reuse can only resume when the
//! new prompt extends the cached one exactly; anything else falls back to a cold prefill (correct, just not
//! faster). `validate_all.sh` gates this and, because token equality alone is satisfiable by a cold prefill,
//! also asserts the resume counter advanced — see that script's "HYBRID decode gate".

use std::collections::HashMap;

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::deltanet::{
    causal_conv1d_silu, causal_conv1d_silu_hist, delta_gate, gated_deltanet_mha,
    gated_deltanet_mha_state, rmsnorm_gated,
};
use crate::model::{Model, PrefixKv};

/// Counts how many times a hybrid decode actually RESUMED from a persisted DeltaNet state, versus falling
/// back to a cold prefill. Token equality alone can't tell those apart (a cold prefill is correct too), so
/// the `--gen-prefix` gate prints this to prove the prefix path engaged instead of silently not running.
pub static QWEN35_RESUMES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub struct Qwen35Moe {
    b: Bundle,
    n_layer: usize,
    h: usize,
    nkv: usize,
    hd: usize,
    d: usize,
    eps: f32,
    scale: f32,
    rotary_dim: usize, // partial RoPE: only the first rotary_dim of each head_dim is rotated
    inv: Vec<f32>,
    n_exp: usize,
    topk: usize,
    shared_inter: usize,
    nvh: usize,
    nkh: usize,
    hkd: usize,
    hvd: usize,
    conv_k: usize,
    linear: Vec<bool>, // layer_types: true = Gated DeltaNet linear layer, false = full attention
    tied: bool,
    kv_int8: bool,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

/// K/V store for the cached full-attention layers. `F32` is the exact cache — one row per position, so a
/// prefix can be copied by row-slice and a decode step is byte-identical to the naive recompute. `Q8` is
/// the `--kv-int8` cache (per-kv-head absmax scale, ~4× smaller, lossy by design). One variant per model,
/// fixed for its lifetime.
enum Kv {
    F32 { kc: Vec<Array2<f32>>, vc: Vec<Array2<f32>> },
    Q8 { kc: Vec<Vec<i8>>, vc: Vec<Vec<i8>>, ks: Vec<Vec<f32>>, vs: Vec<Vec<f32>>, kvdim: usize },
}

/// Decode state for the hybrid stack: the full-attention K/V cache plus, for each *linear* layer, the
/// Gated-DeltaNet recurrence state `S` (one `hkd·hvd` matrix per value head) and the conv1d rolling window.
/// Sized once by [`HybridState::new`], then advanced a row at a time.
///
/// Deliberately kept separate from the naive `full_attn`/`linear_attn` pair above: those stay the untouched
/// top-1-vs-torch reference, so `generate` (cached) vs `predict` (naive) is a real comparison rather than
/// the cache checking itself.
struct HybridState {
    kv: Kv,
    /// per layer, LINEAR layers only: `nvh` state matrices of `hkd·hvd`
    s: Vec<Vec<Vec<f32>>>,
    /// per layer, LINEAR layers only: the retained `(conv_k-1)` pre-conv qkv rows
    hist: Vec<Vec<f32>>,
}

impl HybridState {
    fn new(m: &Qwen35Moe, total: usize) -> HybridState {
        let kvdim = m.nkv * m.hd;
        // full-attention layers get a growing K/V cache; linear layers get a 0-row placeholder — their
        // history is the fixed-size DeltaNet state, so allocating rows for them would waste ~2·nkv·hd·total
        // floats per layer (GBs on a 27B).
        let kv = if m.kv_int8 {
            let z = || vec![0i8; total * kvdim];
            Kv::Q8 {
                kc: (0..m.n_layer).map(|l| if m.linear[l] { Vec::new() } else { z() }).collect(),
                vc: (0..m.n_layer).map(|l| if m.linear[l] { Vec::new() } else { z() }).collect(),
                ks: (0..m.n_layer).map(|l| if m.linear[l] { Vec::new() } else { vec![0f32; total * m.nkv] }).collect(),
                vs: (0..m.n_layer).map(|l| if m.linear[l] { Vec::new() } else { vec![0f32; total * m.nkv] }).collect(),
                kvdim,
            }
        } else {
            // 0 rows for a linear layer, `total` for a full-attention one
            let z = |l: usize| if m.linear[l] { Array2::zeros((0, kvdim)) } else { Array2::zeros((total, kvdim)) };
            Kv::F32 {
                kc: (0..m.n_layer).map(z).collect(),
                vc: (0..m.n_layer).map(z).collect(),
            }
        };
        let s = (0..m.n_layer)
            .map(|l| if m.linear[l] { vec![vec![0f32; m.hkd * m.hvd]; m.nvh] } else { Vec::new() })
            .collect();
        let conv_dim = m.nkh * m.hkd * 2 + m.nvh * m.hvd;
        let hist = (0..m.n_layer)
            .map(|l| if m.linear[l] { vec![0f32; (m.conv_k - 1) * conv_dim] } else { Vec::new() })
            .collect();
        HybridState { kv, s, hist }
    }
}

impl Kv {
    /// Store `m` freshly-computed K/V rows (post-RoPE, GQA width `nkv·hd`) at absolute positions `[cur..cur+m)`.
    fn put(&mut self, l: usize, cur: usize, k: &Array2<f32>, v: &Array2<f32>, nkv: usize, hd: usize) {
        let m = k.nrows();
        match self {
            Kv::F32 { kc, vc } => {
                kc[l].slice_mut(s![cur..cur + m, ..]).assign(k);
                vc[l].slice_mut(s![cur..cur + m, ..]).assign(v);
            }
            Kv::Q8 { kc, vc, ks, vs, kvdim } => {
                let kvdim = *kvdim;
                let q8 = |v: f32, sc: f32| (v / sc).round().clamp(-127.0, 127.0) as i8;
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
            }
        }
    }

    /// Dequantised (or raw) K/V for kv-head `kv` over positions `[0..klen)`, into caller-owned scratch
    /// `(klen, hd)` buffers so the per-head loop doesn't re-allocate every layer of every token.
    fn load_head(&self, l: usize, kv: usize, klen: usize, hd: usize, nkv: usize, kh: &mut Array2<f32>, vh: &mut Array2<f32>) {
        match self {
            Kv::F32 { kc, vc } => {
                let base = kv * hd;
                kh.slice_mut(s![0..klen, ..]).assign(&kc[l].slice(s![0..klen, base..base + hd]));
                vh.slice_mut(s![0..klen, ..]).assign(&vc[l].slice(s![0..klen, base..base + hd]));
            }
            Kv::Q8 { kc, vc, ks, vs, kvdim } => {
                for pos in 0..klen {
                    let (sck, scv) = (ks[l][pos * nkv + kv], vs[l][pos * nkv + kv]);
                    for c in 0..hd {
                        kh[[pos, c]] = kc[l][pos * kvdim + kv * hd + c] as f32 * sck;
                        vh[[pos, c]] = vc[l][pos * kvdim + kv * hd + c] as f32 * scv;
                    }
                }
            }
        }
    }
}

impl Qwen35Moe {
    pub fn new(b: Bundle, _route: f32, kv_int8: bool) -> Qwen35Moe {
        // config_i: [nl,nh,nkv,hd,d,vocab,tied,n_exp,topk,moe_inter,shared_inter,norm_topk,
        //            nvh,nkh,hkd,hvd,conv_k, <nl layer_types: 0=full 1=linear>]
        let c = &b.config;
        let (n_layer, h, nkv, hd, d) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize, c[4] as usize);
        let tied = c[6] != 0;
        let (n_exp, topk, shared_inter) = (c[7] as usize, c[8] as usize, c[10] as usize); // c[11]=norm_topk unused (always renorm)
        let (nvh, nkh, hkd, hvd, conv_k) =
            (c[12] as usize, c[13] as usize, c[14] as usize, c[15] as usize, c[16] as usize);
        let linear: Vec<bool> = (0..n_layer).map(|l| c[17 + l] != 0).collect();
        let rotary_dim = if c.len() > 17 + n_layer { c[17 + n_layer] as usize } else { hd }; // default = full
        let (theta, eps) = (b.config_f[0] as f32, b.config_f[1] as f32);
        // inv frequencies over the ROTARY dim (partial RoPE), rotary_dim/2 entries
        let inv = (0..rotary_dim / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / rotary_dim as f32)).collect();
        Qwen35Moe { b, n_layer, h, nkv, hd, d, eps, scale: (hd as f32).powf(-0.5), rotary_dim, inv, n_exp,
                    topk, shared_inter, nvh, nkh, hkd, hvd, conv_k, linear, tied, kv_int8 }
    }

    fn unembed(&self) -> &str {
        if self.tied { "embed" } else { "lm_head" }
    }

    // Qwen3.5 RMSNorm is GEMMA-style: out = normed · (1 + weight) (weight init 0), NOT the plain `· weight`
    // of Qwen3/Llama. (The linear path's gated norm is plain `· weight` — see deltanet::rmsnorm_gated.)
    fn norm(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let w = self.b.arr1o(name);
        let mut out = x.clone();
        for mut row in out.rows_mut() {
            let n = row.len() as f32;
            let ms = row.iter().map(|v| v * v).sum::<f32>() / n;
            let inv = 1.0 / (ms + self.eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = *v * inv * (1.0 + w[i]);
            }
        }
        out
    }

    fn head_norm(&self, x: &mut Array2<f32>, name: &str, n_heads: usize) {
        let w = self.b.arr1o(name);
        let hd = self.hd;
        for mut row in x.rows_mut() {
            for head in 0..n_heads {
                let base = head * hd;
                let ms = (0..hd).map(|c| row[base + c] * row[base + c]).sum::<f32>() / hd as f32;
                let inv = 1.0 / (ms + self.eps).sqrt();
                for c in 0..hd {
                    row[base + c] = row[base + c] * inv * (1.0 + w[c]);
                }
            }
        }
    }

    /// Partial RoPE (Qwen3.5): rotate only the first `rotary_dim` of each head_dim via rotate_half over that
    /// sub-block (pairs (j, j+rotary_dim/2)); dims [rotary_dim..head_dim] pass through un-rotated.
    fn rope(&self, x: &mut Array2<f32>, n_heads: usize, pos0: usize) {
        let (hd, half) = (self.hd, self.rotary_dim / 2);
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

    /// Full GQA attention layer (non-cached): returns the o_proj output to add to the residual.
    fn full_attn(&self, l: usize, a: &Array2<f32>) -> Array2<f32> {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let seq = a.nrows();
        let p = format!("l{l}.");
        // q_proj is GATED: it outputs h*head_dim*2 = per head [query(hd) | gate(hd)]; the gate sigmoid-scales
        // the attention output (Qwen3_5MoeAttention). Split per head into query and gate.
        let qg = self.b.mm(a, &format!("{p}self_attn.q_proj")); // (seq, h*2*hd)
        let mut q = Array2::<f32>::zeros((seq, h * hd));
        let mut gate = Array2::<f32>::zeros((seq, h * hd));
        for t in 0..seq {
            for head in 0..h {
                let src = head * 2 * hd;
                for c in 0..hd {
                    q[[t, head * hd + c]] = qg[[t, src + c]];
                    gate[[t, head * hd + c]] = qg[[t, src + hd + c]];
                }
            }
        }
        let mut k = self.b.mm(a, &format!("{p}self_attn.k_proj"));
        let v = self.b.mm(a, &format!("{p}self_attn.v_proj"));
        self.head_norm(&mut q, &format!("{p}q_norm"), h);
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
                    if j > i {
                        scores[[i, j]] = -1e30; // causal
                    }
                }
            }
            softmax_rows(&mut scores);
            attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
        }
        for (o, g) in attn_out.iter_mut().zip(gate.iter()) {
            *o *= sigmoid(*g); // gated attention
        }
        self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"))
    }

    /// Gated DeltaNet linear-attention layer (non-cached): returns the out_proj output to add to the residual.
    fn linear_attn(&self, l: usize, a: &Array2<f32>) -> Array2<f32> {
        let seq = a.nrows();
        let p = format!("l{l}.linear_attn.");
        let key_dim = self.nkh * self.hkd;
        let value_dim = self.nvh * self.hvd;
        let conv_dim = key_dim * 2 + value_dim;
        let qkv = self.b.mm(a, &format!("{p}in_proj_qkv")); // (seq, conv_dim)
        let z = self.b.mm(a, &format!("{p}in_proj_z")); // (seq, value_dim) — output gate
        let bb = self.b.mm(a, &format!("{p}in_proj_b")); // (seq, nvh) — beta logits
        let aa = self.b.mm(a, &format!("{p}in_proj_a")); // (seq, nvh) — dt
        // causal depthwise conv (no bias) + SiLU on the concatenated [q,k,v] channels
        let cw = self.b.arr2o(&format!("{p}conv1d")); // (conv_dim, conv_k)
        let conv_w: Vec<f32> = cw.iter().cloned().collect();
        let qkv_v: Vec<f32> = qkv.iter().cloned().collect();
        let conv = causal_conv1d_silu(&qkv_v, &conv_w, &vec![0f32; conv_dim], seq, conv_dim, self.conv_k);
        // split per timestep: [q: key_dim | k: key_dim | v: value_dim]
        let (mut q, mut k, mut v) = (vec![0f32; seq * key_dim], vec![0f32; seq * key_dim], vec![0f32; seq * value_dim]);
        for t in 0..seq {
            let base = t * conv_dim;
            q[t * key_dim..(t + 1) * key_dim].copy_from_slice(&conv[base..base + key_dim]);
            k[t * key_dim..(t + 1) * key_dim].copy_from_slice(&conv[base + key_dim..base + 2 * key_dim]);
            v[t * value_dim..(t + 1) * value_dim].copy_from_slice(&conv[base + 2 * key_dim..base + conv_dim]);
        }
        // decay gate g (log-space) and beta, per value-head
        let a_log = self.b.arr1o(&format!("{p}A_log"));
        let dt_bias = self.b.arr1o(&format!("{p}dt_bias"));
        let a_dt: Vec<f32> = aa.iter().cloned().collect();
        let g = delta_gate(&a_dt, a_log.as_slice().unwrap(), dt_bias.as_slice().unwrap(), seq, self.nvh);
        let beta: Vec<f32> = bb.iter().map(|&x| sigmoid(x)).collect();
        // multi-head GQA Gated DeltaNet
        let core = gated_deltanet_mha(&q, &k, &v, &g, &beta, seq, self.nkh, self.nvh, self.hkd, self.hvd);
        // gated RMSNorm per (token, value-head) over head_v_dim, gated by z; then out_proj
        let norm_w = self.b.arr1o(&format!("{p}norm"));
        let z_v: Vec<f32> = z.iter().cloned().collect();
        let gated = rmsnorm_gated(&core, norm_w.as_slice().unwrap(), &z_v, seq * self.nvh, self.hvd, self.eps);
        let gated = Array2::from_shape_vec((seq, value_dim), gated).unwrap();
        self.b.mm(&gated, &format!("{p}out_proj"))
    }

    /// SparseMoeBlock: softmax top-k routed experts + a sigmoid-gated shared expert.
    fn moe_branch(&self, l: usize, a2: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.");
        let seq = a2.nrows();
        let mut logits = self.b.mm(a2, &format!("{p}gate"));
        softmax_rows(&mut logits);
        let mut assign: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for t in 0..seq {
            let row = logits.row(t);
            let mut idx: Vec<usize> = (0..self.n_exp).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            idx.truncate(self.topk);
            // Qwen3.5 router ALWAYS renormalizes the top-k weights (router_top_value /= sum), independent of
            // norm_topk_prob — see Qwen3_5MoeTopKRouter.forward.
            let denom: f32 = idx.iter().map(|&e| row[e]).sum::<f32>().max(1e-20);
            for &e in &idx {
                assign.entry(e).or_default().push((t, row[e] / denom));
            }
        }
        let mut out = Array2::<f32>::zeros((seq, self.d));
        for &e in assign.keys() {
            self.b.prefetch(&format!("{p}experts.{e}.gate"));
            self.b.prefetch(&format!("{p}experts.{e}.up"));
            self.b.prefetch(&format!("{p}experts.{e}.down"));
        }
        for (e, toks) in &assign {
            let mut rows = Array2::<f32>::zeros((toks.len(), self.d));
            for (i, &(t, _)) in toks.iter().enumerate() {
                rows.row_mut(i).assign(&a2.row(t));
            }
            let gate = self.b.expert_mm(&rows, &format!("{p}experts.{e}.gate"));
            let up = self.b.expert_mm(&rows, &format!("{p}experts.{e}.up"));
            let mut hh = gate;
            for (hv, uv) in hh.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            let down = self.b.expert_mm(&hh, &format!("{p}experts.{e}.down"));
            for (i, &(t, wgt)) in toks.iter().enumerate() {
                for c in 0..self.d {
                    out[[t, c]] += wgt * down[[i, c]];
                }
            }
        }
        // shared expert (applied to all tokens), sigmoid-gated
        if self.shared_inter > 0 {
            let sg = self.b.mm(a2, &format!("{p}shared.gate_proj"));
            let su = self.b.mm(a2, &format!("{p}shared.up_proj"));
            let mut sh = sg;
            for (hv, uv) in sh.iter_mut().zip(su.iter()) {
                *hv = silu(*hv) * uv;
            }
            let sd = self.b.mm(&sh, &format!("{p}shared.down_proj"));
            let g = self.b.mm(a2, &format!("{p}shared_gate")); // (seq, 1)
            for t in 0..seq {
                let gt = sigmoid(g[[t, 0]]);
                for c in 0..self.d {
                    out[[t, c]] += gt * sd[[t, c]];
                }
            }
        }
        out
    }

    fn ffn(&self, l: usize, x: &Array2<f32>) -> Array2<f32> {
        if self.n_exp != 0 {
            return self.moe_branch(l, x);
        }
        let p = format!("l{l}.mlp.");
        let gate = self.b.mm(x, &format!("{p}gate_proj"));
        let up = self.b.mm(x, &format!("{p}up_proj"));
        let mut hidden = gate;
        for (v, u) in hidden.iter_mut().zip(up.iter()) {
            *v = silu(*v) * u;
        }
        self.b.mm(&hidden, &format!("{p}down_proj"))
    }

    /// Cached twin of [`Qwen35Moe::full_attn`]: same math, but K/V are written into the store at absolute
    /// position `cur` and the query attends over `[0..cur+m)` instead of recomputing the whole prefix.
    fn full_attn_cached(&self, l: usize, a: &Array2<f32>, cur: usize, st: &mut HybridState) -> Array2<f32> {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let m = a.nrows();
        let klen = cur + m;
        let p = format!("l{l}.");
        let qg = self.b.mm(a, &format!("{p}self_attn.q_proj")); // (m, h*2*hd) — [query | gate] per head
        let mut q = Array2::<f32>::zeros((m, h * hd));
        let mut gate = Array2::<f32>::zeros((m, h * hd));
        for t in 0..m {
            for head in 0..h {
                let src = head * 2 * hd;
                for c in 0..hd {
                    q[[t, head * hd + c]] = qg[[t, src + c]];
                    gate[[t, head * hd + c]] = qg[[t, src + hd + c]];
                }
            }
        }
        let mut k = self.b.mm(a, &format!("{p}self_attn.k_proj"));
        let v = self.b.mm(a, &format!("{p}self_attn.v_proj"));
        self.head_norm(&mut q, &format!("{p}q_norm"), h);
        self.head_norm(&mut k, &format!("{p}k_norm"), nkv);
        self.rope(&mut q, h, cur); // absolute positions
        self.rope(&mut k, nkv, cur);
        st.kv.put(l, cur, &k, &v, nkv, hd);
        let mut attn_out = Array2::<f32>::zeros((m, h * hd));
        let (mut kh, mut vh) = (Array2::<f32>::zeros((klen, hd)), Array2::<f32>::zeros((klen, hd)));
        for head in 0..h {
            let kv = head / rep;
            st.kv.load_head(l, kv, klen, hd, nkv, &mut kh, &mut vh);
            let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
            let mut scores = qh.dot(&kh.t()) * self.scale;
            for i in 0..m {
                let abs = cur + i;
                for j in 0..klen {
                    if j > abs {
                        scores[[i, j]] = -1e30;
                    }
                }
            }
            softmax_rows(&mut scores);
            attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
        }
        for (o, g) in attn_out.iter_mut().zip(gate.iter()) {
            *o *= sigmoid(*g);
        }
        self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"))
    }

    /// Cached twin of [`Qwen35Moe::linear_attn`]: identical math, but the conv1d keeps a rolling window
    /// and the DeltaNet recurrence keeps its per-value-head state, so `m` new rows cost O(m) not O(prefix).
    fn linear_attn_cached(&self, l: usize, a: &Array2<f32>, st: &mut HybridState) -> Array2<f32> {
        let m = a.nrows();
        let p = format!("l{l}.linear_attn.");
        let key_dim = self.nkh * self.hkd;
        let value_dim = self.nvh * self.hvd;
        let conv_dim = key_dim * 2 + value_dim;
        let qkv = self.b.mm(a, &format!("{p}in_proj_qkv"));
        let z = self.b.mm(a, &format!("{p}in_proj_z"));
        let bb = self.b.mm(a, &format!("{p}in_proj_b"));
        let aa = self.b.mm(a, &format!("{p}in_proj_a"));
        let cw = self.b.arr2o(&format!("{p}conv1d"));
        let conv_w: Vec<f32> = cw.iter().cloned().collect();
        let qkv_v: Vec<f32> = qkv.iter().cloned().collect();
        let conv = causal_conv1d_silu_hist(
            &qkv_v, &conv_w, &vec![0f32; conv_dim], m, conv_dim, self.conv_k, &mut st.hist[l],
        );
        let (mut q, mut k, mut v) = (vec![0f32; m * key_dim], vec![0f32; m * key_dim], vec![0f32; m * value_dim]);
        for t in 0..m {
            let base = t * conv_dim;
            q[t * key_dim..(t + 1) * key_dim].copy_from_slice(&conv[base..base + key_dim]);
            k[t * key_dim..(t + 1) * key_dim].copy_from_slice(&conv[base + key_dim..base + 2 * key_dim]);
            v[t * value_dim..(t + 1) * value_dim].copy_from_slice(&conv[base + 2 * key_dim..base + conv_dim]);
        }
        let a_log = self.b.arr1o(&format!("{p}A_log"));
        let dt_bias = self.b.arr1o(&format!("{p}dt_bias"));
        let a_dt: Vec<f32> = aa.iter().cloned().collect();
        let g = delta_gate(&a_dt, a_log.as_slice().unwrap(), dt_bias.as_slice().unwrap(), m, self.nvh);
        let beta: Vec<f32> = bb.iter().map(|&x| sigmoid(x)).collect();
        let core = gated_deltanet_mha_state(
            &q, &k, &v, &g, &beta, m, self.nkh, self.nvh, self.hkd, self.hvd, &mut st.s[l],
        );
        let norm_w = self.b.arr1o(&format!("{p}norm"));
        let z_v: Vec<f32> = z.iter().cloned().collect();
        let gated = rmsnorm_gated(&core, norm_w.as_slice().unwrap(), &z_v, m * self.nvh, self.hvd, self.eps);
        let gated = Array2::from_shape_vec((m, value_dim), gated).unwrap();
        self.b.mm(&gated, &format!("{p}out_proj"))
    }

    /// Cached twin of [`Qwen35Moe::hidden`]: run `m` new rows (placed at absolute position `cur`) through the
    /// layers, advancing `st` instead of recomputing the prefix.
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, st: &mut HybridState) -> Array2<f32> {
        let mut x = emb.clone();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let attn = if self.linear[l] { self.linear_attn_cached(l, &a, st) } else { self.full_attn_cached(l, &a, cur, st) };
            x = &x + &attn;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            x = &x + &self.ffn(l, &a2);
        }
        self.norm(&x, "norm")
    }

    fn head_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let last = xfn.row(xfn.nrows() - 1).to_vec();
        self.b
            .rowdot_f32(self.unembed(), &last)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i64
    }

    /// The greedy loop shared by `generate_stream`/`generate_stream_prefix`: prefill `ids` (placed at absolute
    /// position `cur`) into `st`, then advance one token at a time. `cur` is 0 for a cold start and the reuse
    /// point for a resumed one — it positions both the RoPE and the attention length, and the DeltaNet
    /// state in `st` must already be advanced to `cur`.
    ///
    /// Returns the emitted tokens AND the position the stack actually reached. The loop breaks on
    /// `out.len() == max_tokens` BEFORE forwarding that last token, so the state sits at `pos` while `out`
    /// holds one more token than was fed through — a mismatch a row-indexed KV cache can absorb (the tail
    /// just gets re-forwarded) but a RECURRENT state cannot. Callers persisting `st` must truncate their ids
    /// to `pos` so the saved state and the saved id count agree.
    #[allow(clippy::type_complexity)]
    fn decode_loop(
        &self,
        ids: &[i64],
        cur: usize,
        max_tokens: usize,
        eos: &[i64],
        emit: &mut dyn FnMut(i64) -> bool,
        st: &mut HybridState,
    ) -> (Vec<i64>, usize) {
        let emb = self.b.rows_f32("embed", ids);
        let xf = self.forward_block(&emb, cur, st);
        let mut next = self.head_argmax(&xf);
        let mut out = Vec::new();
        let mut pos = cur + ids.len();
        loop {
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            if !emit(next) || out.len() == max_tokens {
                break;
            }
            let emb = self.b.rows_f32("embed", &[next]);
            let xf = self.forward_block(&emb, pos, st);
            next = self.head_argmax(&xf);
            pos += 1;
        }
        (out, pos)
    }

    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let attn = if self.linear[l] { self.linear_attn(l, &a) } else { self.full_attn(l, &a) };
            x = &x + &attn;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            x = &x + &self.ffn(l, &a2);
        }
        self.norm(&x, "norm")
    }

    /// Debug: residual-stream snapshots in transformers `output_hidden_states` convention — [input-to-L0
    /// (=embed), input-to-L1, …, input-to-L{n-1}, POST-final-norm] (each flat seq·d, row-major). The last
    /// entry is post-final-norm (HF appends it), so it lines up with the reference's last hidden state.
    pub fn hiddens(&self, ids: &[i64]) -> Vec<Vec<f32>> {
        let mut snaps = Vec::with_capacity(self.n_layer + 1);
        let mut x = self.b.rows_f32("embed", ids);
        snaps.push(x.iter().cloned().collect()); // hs[0] = embed = input to L0
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let attn = if self.linear[l] { self.linear_attn(l, &a) } else { self.full_attn(l, &a) };
            x = &x + &attn;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            x = &x + &self.ffn(l, &a2);
            if l + 1 < self.n_layer {
                snaps.push(x.iter().cloned().collect()); // input to L{l+1}
            }
        }
        snaps.push(self.norm(&x, "norm").iter().cloned().collect()); // hs[N] = post-final-norm
        snaps
    }
}

impl Model for Qwen35Moe {
    fn predict(&self, ids: &[i64]) -> i64 {
        self.logits(ids).map(|lg| lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
            .unwrap_or(0)
    }

    fn logits(&self, ids: &[i64]) -> Option<Vec<f32>> {
        let xf = self.hidden(ids);
        Some(self.b.rowdot_f32(self.unembed(), &xf.row(ids.len() - 1).to_vec()))
    }

    fn dims(&self) -> Option<(usize, usize)> {
        Some((self.n_layer, self.h))
    }

    /// Hybrid incremental decode. The DeltaNet layers keep their recurrent state and the full-attention
    /// layers their K/V cache, so each token is O(1) instead of re-running the whole prefix — the
    /// validation gate holds this byte-identical to `predict` (the naive full-recompute path).
    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let mut st = HybridState::new(self, prompt.len() + n_new);
        // fixed-length, no early stop: `generate` is the CLI KV-vs-naive benchmark, and `predict` below is
        // its naive arm, so both must run exactly n_new tokens for the comparison to mean anything.
        let mut ctx = prompt.to_vec();
        let emb = self.b.rows_f32("embed", &ctx);
        let xf = self.forward_block(&emb, 0, &mut st);
        let mut next = self.head_argmax(&xf);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let emb = self.b.rows_f32("embed", &[next]);
            let xf = self.forward_block(&emb, pos, &mut st);
            next = self.head_argmax(&xf);
            ctx.push(next);
            pos += 1;
        }
    }

    /// KV-cached streaming (early-stop at eos + per-token emit) for chat/serve.
    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let mut st = HybridState::new(self, prompt.len() + max_tokens);
        self.decode_loop(prompt, 0, max_tokens, eos, emit, &mut st).0
    }

    /// Cross-turn prefix reuse. The K/V rows are indexed per position, but the DeltaNet state is a
    /// *recurrent* value with no per-position copy — so it can only be carried over when the new prompt
    /// extends the cached one exactly (the dominant multi-turn case: cache a turn, then prefill just the new
    /// suffix). Any earlier divergence falls back to a cold prefill, which is correct, just not faster.
    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool, cache: &mut PrefixKv) -> Vec<i64> {
        let total = prompt.len() + max_tokens;
        // `prefix_generate` copies rows [0..l) where l = reuse_len; a hybrid state snapshot is only valid at
        // exactly `cache.ids.len()`, so require the reuse point to land there.
        let l = cache.reuse_len(prompt);
        let resumable = l > 0 && l == cache.ids.len() && cache.lin_s.len() == self.n_layer && !self.kv_int8;
        if resumable {
            QWEN35_RESUMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let mut st = HybridState::new(self, total);
        if resumable {
            for layer in 0..self.n_layer {
                if self.linear[layer] {
                    st.s[layer] = std::mem::take(&mut cache.lin_s[layer]);
                    st.hist[layer] = std::mem::take(&mut cache.lin_hist[layer]);
                } else if let Kv::F32 { kc, vc } = &mut st.kv {
                    kc[layer].slice_mut(s![0..l, ..]).assign(&cache.kc[layer].slice(s![0..l, ..]));
                    vc[layer].slice_mut(s![0..l, ..]).assign(&cache.vc[layer].slice(s![0..l, ..]));
                }
            }
        }
        let (out, pos) = if resumable {
            self.decode_loop(&prompt[l..], l, max_tokens, eos, emit, &mut st)
        } else {
            self.decode_loop(prompt, 0, max_tokens, eos, emit, &mut st)
        };
        // Persist for the next turn: K/V + ids + the hybrid state, which is now positioned after
        // `pos` tokens. `ids` is TRUNCATED to `pos` so the saved id count and the recurrent state agree —
        // the loop breaks before forwarding the final emitted token, so `out` is one longer than what the
        // stack consumed. The saved KV rows are exactly [0..pos), so a later turn resumes consistently.
        if !self.kv_int8 {
            if let Kv::F32 { kc, vc } = st.kv {
                let mut ids: Vec<i64> = Vec::with_capacity(pos);
                ids.extend_from_slice(prompt);
                ids.extend_from_slice(&out);
                ids.truncate(pos);
                cache.ids = ids;
                cache.kc = kc;
                cache.vc = vc;
                // built fresh (not indexed in place) — `lin_s` is empty on a cold start.
                let mut lin_s: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.n_layer);
                let mut lin_hist: Vec<Vec<f32>> = Vec::with_capacity(self.n_layer);
                for layer in 0..self.n_layer {
                    lin_s.push(std::mem::take(&mut st.s[layer]));
                    lin_hist.push(std::mem::take(&mut st.hist[layer]));
                }
                cache.lin_s = lin_s;
                cache.lin_hist = lin_hist;
            }
        } else {
            // int8: keep the existing single-slot contract (no prefix reuse for the hybrid state).
            cache.ids.clear();
            cache.kc.clear();
            cache.vc.clear();
            cache.lin_s.clear();
            cache.lin_hist.clear();
        }
        out
    }
}
