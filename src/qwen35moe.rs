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
///
/// A **validation-gate diagnostic only** — deliberately a process-wide counter, read as a delta around a
/// single call. It is not a serving metric: under concurrent requests the global total is uninformative,
/// so don't build reuse-rate telemetry on it. If the chat/serve path ever needs per-request accounting,
/// thread a counter through `PrefixKv` instead.
pub static QWEN35_RESUMES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Counts turns that had a NON-EMPTY prefix cache but still cold-prefilled, because the new prompt diverged
/// from the cached one before its end (a recurrent state can only resume at the exact cached length). This is
/// the reuse-rate signal the exact-extension rule costs you — a large number here means turns are re-prefilling
/// more than the caller expects. Empty-cache first turns are not counted; those are normal.
pub static QWEN35_PREFIX_FALLBACKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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
#[derive(Clone)]
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
#[derive(Clone)]
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

/// Blind-max scorer for the hybrid stack.
///
/// The rope scorer can overwrite a single K/V cache row per candidate. That trick does NOT transfer: the
/// DeltaNet layers carry an ACCUMULATOR (per-value-head `S` plus the conv window), so probing a candidate
/// mutates state the next candidate needs to start from. Each hill-climb step therefore takes ONE snapshot
/// of the whole hybrid state, and every candidate is evaluated by restoring that snapshot, advancing a single
/// row, and reading the target unit. Restoring is what makes candidates independent; snapshotting once per
/// STEP (not per candidate) is what keeps it affordable — the recurrent state is ~19 MB on a 0.8B.
pub struct Qwen35Scorer {
    st: HybridState,   // working state, mutated by each probe
    snap: HybridState, // restore point at `pos`, refreshed only by `commit`
    pos: usize,
}

impl Qwen35Scorer {
    /// Prefill `prefix`, then score candidates at the next position.
    pub fn new(m: &Qwen35Moe, prefix: &[i64], total: usize) -> Option<Qwen35Scorer> {
        let st = m.blind_score_state(prefix, total)?;
        let pos = prefix.len();
        Some(Qwen35Scorer { snap: st.clone(), st, pos })
    }

    /// An empty scorer at position 0 — no seed context, for the corpus scan, which places every token.
    /// `snap` is a clone of an empty state and is only meaningful after a `commit`; the scan path never
    /// reads it (it has no candidates to isolate).
    pub fn empty(m: &Qwen35Moe, total: usize) -> Qwen35Scorer {
        let st = HybridState::new(m, total);
        Qwen35Scorer { snap: st.clone(), st, pos: 0 }
    }

    /// Score `token` at `pos` WITHOUT committing it. None for a bad layer or an out-of-range unit.
    pub fn try_token(&mut self, m: &Qwen35Moe, token: i64, layer: usize, neuron: usize) -> Option<f32> {
        self.st = self.snap.clone(); // candidates must not see each other's writes
        let emb = m.b.rows_f32("embed", &[token]);
        let (mlp_h, _, _) = m.forward_block_capture_mlp(&emb, self.pos, &mut self.st, None, None);
        mlp_h.get(layer)?.get(neuron).copied()
    }

    /// Score `token` at the next position by the ROUTED unit's `w_router · act`, WITHOUT committing.
    /// `None` for a bad unit; `Some(NaN)` when the router did not select that expert for this position —
    /// the unit genuinely has no value there, which is not the same as a value of zero.
    pub fn try_token_moe(
        &mut self,
        m: &Qwen35Moe,
        token: i64,
        layer: usize,
        expert: usize,
        neuron: usize,
    ) -> Option<f32> {
        self.st = self.snap.clone(); // candidates must not see each other's writes
        let emb = m.b.rows_f32("embed", &[token]);
        let (_h, _rows, vals) = m.forward_block_capture_mlp(&emb, self.pos, &mut self.st, None, Some((layer, expert, neuron)));
        vals.first().copied()
    }

    /// Accept `token`: re-probe from the snapshot, advance `pos`, re-take the snapshot there.
    pub fn commit(&mut self, m: &Qwen35Moe, token: i64, layer: usize, neuron: usize) -> Option<f32> {
        let a = self.try_token(m, token, layer, neuron);
        self.pos += 1;
        self.snap = self.st.clone();
        a
    }

    pub fn pos(&self) -> usize {
        self.pos
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
    fn moe_branch(&self, l: usize, a2: &Array2<f32>, scan: Option<(usize, usize)>) -> (Array2<f32>, Vec<f32>) {
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
        // Per-row value of ONE routed unit, recorded in the same pass so the expert's weights are read
        // once. Empty where the expert is not in that row's top-k: the unit is not computed there at
        // all, so it has no value rather than a value of zero — see `Model::blind_moe_*`.
        let mut scan_out: Vec<f32> = scan.map(|_| vec![f32::NAN; seq]).unwrap_or_default();
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
                if let Some((se, sn)) = scan {
                    if *e == se && sn < hh.ncols() {
                        scan_out[t] = wgt * hh[[i, sn]];
                    }
                }
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
        (out, scan_out)
    }

    fn ffn(&self, l: usize, x: &Array2<f32>) -> Array2<f32> {
        if self.n_exp != 0 {
            return self.moe_branch(l, x, None).0;
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

    /// Pre-fill `prefix` into a hybrid state that a [`Qwen35Scorer`] can probe from.
    pub(crate) fn blind_score_state(&self, prefix: &[i64], total: usize) -> Option<HybridState> {
        if prefix.is_empty() || prefix.len() >= total {
            return None;
        }
        let mut st = HybridState::new(self, total);
        let emb = self.b.rows_f32("embed", prefix);
        self.forward_block(&emb, 0, &mut st);
        Some(st)
    }

    /// `forward_block` with the last-row post-FFN activations captured per layer — the blind-max scoring
    /// kernel. Identical walk and arithmetic, so its `mlp_h[layer][neuron]` is bit-identical to the naive
    /// `mlp_act_last` over the same ids; the difference is that it ADVANCES `st` (DeltaNet recurrence and
    /// conv window included), which is why the scorer snapshots before probing a candidate.
    fn forward_block_capture_mlp(&self, emb: &Array2<f32>, cur: usize, st: &mut HybridState, scan: Option<(usize, usize)>, moe_scan: Option<(usize, usize, usize)>) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let last = emb.nrows() - 1;
        let mut x = emb.clone();
        let mut mlp_h: Vec<Vec<f32>> = Vec::with_capacity(self.n_layer);
        // per-ROW activations of one unit, for the corpus scan (see `Model::blind_corpus_best`)
        let mut scan_out: Vec<f32> = Vec::new();
        // per-ROW router-weighted activation of one ROUTED unit, for the MoE blind-max arms
        let mut moe_out: Vec<f32> = Vec::new();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let attn = if self.linear[l] { self.linear_attn_cached(l, &a, st) } else { self.full_attn_cached(l, &a, cur, st) };
            x = &x + &attn;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            // dense FFN only — a MoE layer's activation depends on which experts the router picks, so there
            // is no single "neuron n" to report here (see `mlp_act_last`)
            if self.n_exp == 0 {
                let q = format!("l{l}.mlp.");
                let gate = self.b.mm(&a2, &format!("{q}gate_proj"));
                let up = self.b.mm(&a2, &format!("{q}up_proj"));
                let mut hid = gate;
                for (hv, uv) in hid.iter_mut().zip(up.iter()) {
                    *hv = silu(*hv) * uv;
                }
                mlp_h.push(hid.row(last).to_vec());
                if let Some((sl, sn)) = scan {
                    // `column` panics out of range, and a bad --neuron must not take the process down
                    if l == sl && sn < hid.ncols() {
                        scan_out = hid.column(sn).to_vec();
                    }
                }
                x = &x + &self.b.mm(&hid, &format!("{q}down_proj"));
            } else if let Some((ml, me, mn)) = moe_scan.filter(|(ml, ..)| *ml == l) {
                // one pass: the branch's output AND this row's routed-unit value
                let (mo, vals) = self.moe_branch(l, &a2, Some((me, mn)));
                moe_out = vals;
                mlp_h.push(Vec::new());
                x = &x + &mo;
            } else {
                mlp_h.push(Vec::new());
                x = &x + &self.ffn(l, &a2);
            }
        }
        (mlp_h, scan_out, moe_out)
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
        self.hidden_pdc(ids, false).0
    }

    // Observe the actual residual additions, without changing the default arithmetic.
    fn hidden_pdc(&self, ids: &[i64], capture: bool) -> (Array2<f32>, Vec<f32>, Vec<serde_json::Value>) {
        let mut x = self.b.rows_f32("embed", ids);
        let last = ids.len() - 1;
        let mut sources = Vec::new();
        if capture { sources.push(serde_json::json!({"id":"embed", "write":x.row(last).to_vec()})); }
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let attn = if self.linear[l] { self.linear_attn(l, &a) } else { self.full_attn(l, &a) };
            if capture { sources.push(serde_json::json!({"id":format!("L{l}.attn"),
                "mechanism":if self.linear[l] {"gated_deltanet"} else {"gated_full_attention"},
                "write":attn.row(last).to_vec()})); }
            x = &x + &attn;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let mlp = self.ffn(l, &a2);
            if capture { sources.push(serde_json::json!({"id":format!("L{l}.mlp"),"write":mlp.row(last).to_vec()})); }
            x = &x + &mlp;
        }
        let residual = if capture { x.row(last).to_vec() } else { Vec::new() };
        (self.norm(&x, "norm"), residual, sources)
    }

    pub fn pdc_source_dump(&self, selection: &serde_json::Value) -> Result<serde_json::Value, String> {
        if self.n_exp != 0 || self.kv_int8 { return Err("PDC Qwen capture requires unpruned weights and f32 state".into()); }
        let contexts: Vec<Vec<i64>> = serde_json::from_value(selection["contexts"].clone()).map_err(|e|e.to_string())?;
        let candidates: Vec<i64> = serde_json::from_value(selection["candidate_tokens"].clone()).map_err(|e|e.to_string())?;
        let vocab = self.b.config[5];
        if contexts.is_empty() || contexts.iter().any(|c| c.is_empty() || c.iter().any(|&t|t<0 || t>=vocab)) {
            return Err("nonempty in-vocabulary contexts required".into());
        }
        if candidates.is_empty() || candidates.iter().any(|&t|t<0 || t>=vocab)
            || candidates.iter().collect::<std::collections::HashSet<_>>().len()!=candidates.len() {
            return Err("nonempty unique in-vocabulary candidates required".into());
        }
        let check = selection.get("check_observer").and_then(|v|v.as_bool()).unwrap_or(true);
        let mut cases = Vec::new();
        for ids in contexts {
            let started = std::time::Instant::now();
            let (hidden, residual, sources) = self.hidden_pdc(&ids, true);
            let last_hidden = hidden.row(ids.len()-1).to_vec();
            let scores = self.b.rowdot_f32(self.unembed(), &last_hidden);
            let winner = scores.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0;
            let elapsed = started.elapsed().as_micros();
            let observer_error = if check {
                let plain = self.hidden(&ids);
                Some(hidden.iter().zip(plain.iter()).map(|(a,b)|(a-b).abs()).fold(0.0f32, f32::max))
            } else { None };
            if observer_error.is_some_and(|e|e!=0.0) { return Err("observer changed native hidden states".into()); }
            cases.push(serde_json::json!({"ids":ids,"sources":sources,"final_residual":residual,
                "final_hidden":last_hidden,"prediction":winner,"candidate_tokens":candidates,
                "candidate_scores":candidates.iter().map(|&t|scores[t as usize]).collect::<Vec<_>>(),
                "scoring_scope":"full_vocabulary","native_output_row_calls":vocab,
                "observer_hidden_max_error":observer_error,"observer_check_performed":check,
                "elapsed_forward_decode_us":elapsed}));
        }
        Ok(serde_json::json!({"schema":"pdc.qwen35-source-capture.v1","arch":self.b.arch,
            "config":self.b.config,"config_f":self.b.config_f,"cases":cases}))
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
    /// Naive post-FFN activations for every neuron of `layer` at the last token of `ids`. Same walk as
    /// `hidden`, same order, exiting early at `layer` (exact — no later layer feeds back into an earlier one).
    ///
    /// DENSE-ONLY by construction: on a MoE model a layer's FFN is `moe_branch`, so "neuron n" depends on
    /// which experts the router picks for the token being scored — there is no fixed unit to report. Returns
    /// None there rather than inventing one.
    fn mlp_act_last(&self, ids: &[i64], layer: usize) -> Option<Vec<f32>> {
        if layer >= self.n_layer || ids.is_empty() || self.n_exp != 0 {
            return None;
        }
        let last = ids.len() - 1;
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..=layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let attn = if self.linear[l] { self.linear_attn(l, &a) } else { self.full_attn(l, &a) };
            x = &x + &attn;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let q = format!("l{l}.mlp.");
            let gate = self.b.mm(&a2, &format!("{q}gate_proj"));
            let up = self.b.mm(&a2, &format!("{q}up_proj"));
            let mut hid = gate;
            for (hv, uv) in hid.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            if l == layer {
                return Some(hid.row(last).to_vec());
            }
            x = &x + &self.b.mm(&hid, &format!("{q}down_proj"));
        }
        None
    }

    /// The unit's own projected vocabulary — the SAME readout `explain` prints as `promotes` (down-projection
    /// · final-norm gain · unembed), signed so the NEGATIVE pole is reachable. Qwen3.5's full-attention
    /// layers sit every 4th layer; the FFN below them is dense SwiGLU, so the unit is a real fixed unit here.
    fn blind_support(&self) -> crate::model::BlindSupport {
        // The scorer is implemented here for both releases; only the MoE one declines, and for a
        // different reason than "unsupported arch". Saying so is the whole point of this method.
        if self.n_exp != 0 {
            crate::model::BlindSupport::RoutedLayers
        } else {
            crate::model::BlindSupport::Supported
        }
    }

    fn blind_proposals(&self, layer: usize, neuron: usize, k: usize, polarity: f32) -> Option<Vec<i64>> {
        if layer >= self.n_layer || self.n_exp != 0 {
            return None;
        }
        let gain = self.b.arr1("norm").to_vec();
        let mut w = self.b.weight_row(&format!("l{layer}.mlp.down_proj"), neuron);
        w.iter_mut().zip(gain.iter()).for_each(|(v, g)| *v *= g);
        let proj = self.b.rowdot_f32(self.unembed(), &w);
        let mut idx: Vec<usize> = (0..proj.len()).collect();
        idx.sort_by(|&a, &b| {
            (polarity * proj[b]).partial_cmp(&(polarity * proj[a])).unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(idx.into_iter().take(k).map(|i| i as i64).collect())
    }

    /// One ROUTED unit's self-description: expert `e` neuron `n`'s own output row
    /// (`experts.{e}.down`), scaled by the final-norm gain and projected through the unembed — exactly
    /// the construction `blind_proposals` uses for a dense `mlp.down_proj`, so the two are comparable.
    fn blind_moe_proposals(&self, layer: usize, expert: usize, neuron: usize, k: usize, polarity: f32) -> Option<Vec<i64>> {
        if layer >= self.n_layer || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        // Expert weights are PAGED, not resident: `weight_row` cannot see them, so go through
        // `expert_f32`, which returns (shape, values) as (in, out) row-major. down_proj is (mi, d),
        // so neuron `n`'s output direction is one contiguous d-wide row of it.
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
    fn blind_moe_climb(
        &self,
        layer: usize,
        expert: usize,
        neuron: usize,
        prefix: &[i64],
        cands: &[i64],
        steps: usize,
        polarity: f32,
    ) -> Option<crate::model::BlindArm> {
        if cands.is_empty() || layer >= self.n_layer || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let mut sc = Qwen35Scorer::new(self, prefix, prefix.len() + steps + 1)?;
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
            // advance the hybrid state by re-scoring the winner from the snapshot; its value was
            // already read by the probe above
            sc.commit(self, t, 0, 0);
            ids.push(t);
            best = best.max(pick_obj);
        }
        if best == f32::NEG_INFINITY {
            return None;
        }
        Some(crate::model::BlindArm { ids, obj: best, pos: None })
    }

    /// Oracle for one routed unit over a corpus: best `w_router · act` among positions that route to
    /// `expert`. Chunked exactly like the dense arm, and using the same hybrid scorer, so the recurrent
    /// state advances identically across chunk boundaries.
    fn blind_moe_corpus_best(
        &self,
        layer: usize,
        expert: usize,
        neuron: usize,
        ids: &[i64],
        polarity: f32,
    ) -> Option<crate::model::BlindArm> {
        if ids.len() < 2 || layer >= self.n_layer || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let mut sc = Qwen35Scorer::empty(self, ids.len());
        let (mut best_obj, mut best_idx, mut routed) = (f32::NEG_INFINITY, 0usize, 0usize);
        for chunk in ids.chunks(crate::model::CORPUS_CHUNK) {
            let start = sc.pos();
            let emb = self.b.rows_f32("embed", chunk);
            let (_h, _rows, vals) =
                self.forward_block_capture_mlp(&emb, start, &mut sc.st, None, Some((layer, expert, neuron)));
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

    /// Greedy hill-climb over those proposals, scored by RAW ACTIVATION through [`Qwen35Scorer`].
    fn blind_climb(
        &self,
        layer: usize,
        neuron: usize,
        prefix: &[i64],
        cands: &[i64],
        steps: usize,
        polarity: f32,
    ) -> Option<crate::model::BlindArm> {
        if cands.is_empty() || layer >= self.n_layer || self.n_exp != 0 {
            return None;
        }
        let mut sc = Qwen35Scorer::new(self, prefix, prefix.len() + steps + 1)?;
        let mut ids = prefix.to_vec();
        let mut best = f32::NEG_INFINITY;
        for _ in 0..steps {
            let (mut pick, mut pick_obj) = (cands[0], f32::NEG_INFINITY);
            for &t in cands {
                if let Some(a) = sc.try_token(self, t, layer, neuron) {
                    let o = polarity * a;
                    if o > pick_obj {
                        pick_obj = o;
                        pick = t;
                    }
                }
            }
            sc.commit(self, pick, layer, neuron);
            ids.push(pick);
            best = pick_obj;
        }
        Some(crate::model::BlindArm { ids, obj: best, pos: None })
    }

    /// Oracle arm — scored by the NAIVE path so the baseline stays independent of the search cache.
    fn blind_corpus_best(
        &self,
        layer: usize,
        neuron: usize,
        ids: &[i64],
        polarity: f32,
    ) -> Option<crate::model::BlindArm> {
        if ids.len() < 2 || layer >= self.n_layer || self.n_exp != 0 {
            return None;
        }
        // Chunked, not one token at a time: at batch 1 every step re-reads the whole (dense) weight set,
        // which measured ~0.9 s/token on a 1.5B and made a real corpus infeasible. `CORPUS_CHUNK` rows per
        // block amortises one weight read over 32 positions while the hybrid state carries forward, so
        // each position still sees its true causal history. Every position is scored exactly once.
        let mut sc = Qwen35Scorer::empty(self, ids.len());
        let (mut best_obj, mut best_idx) = (f32::NEG_INFINITY, 0usize);
        for chunk in ids.chunks(crate::model::CORPUS_CHUNK) {
            let start = sc.pos();
            let emb = self.b.rows_f32("embed", chunk);
            let (_mlp_h, rows, _) = self.forward_block_capture_mlp(&emb, start, &mut sc.st, Some((layer, neuron)), None);
            sc.pos += chunk.len();
            if rows.len() != chunk.len() {
                continue; // unit out of range for this layer's FFN width
            }
            for (j, &a) in rows.iter().enumerate() {
                let o = polarity * a;
                if o > best_obj {
                    best_obj = o;
                    best_idx = start + j;
                }
            }
        }
        if best_obj == f32::NEG_INFINITY {
            return None;
        }
        let lo = best_idx.saturating_sub(crate::model::CORPUS_CTX - 1);
        Some(crate::model::BlindArm {
            ids: ids[lo..=best_idx].to_vec(),
            obj: best_obj,
            pos: Some(best_idx),
        })
    }

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
        } else if !cache.ids.is_empty() {
            // had a cache, couldn't use it — surface it rather than let a cold prefill look like a hit
            QWEN35_PREFIX_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
                ids.truncate(pos); // NOT a bug: see above — the recurrent state sits at `pos`, so the saved
                                    // id count must too, even though `out` is one token longer. A row-indexed
                                    // KV cache could ignore this mismatch; a recurrent one cannot.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BlindArm, BlindSupport, Model, POLE_NEG, POLE_POS};

    fn v(seed: usize, n: usize) -> Vec<f32> {
        (0..n).map(|i| (((seed * 41 + i * 23) % 43) as f32) / 20.0 - 1.0).collect()
    }

    /// A 4-layer hybrid qwen35 bundle: layer_types = [linear, linear, linear, full], dense SwiGLU FFN.
    /// `moe = true` swaps in the packed expert layout so the MoE decline path has real weights to decline on.
    fn build_tiny(tag: &str, moe: bool) -> Qwen35Moe {
        let (nl, nh, nkv, hd, d, vocab, ffn) = (4usize, 4, 2, 8, 32, 24, 16);
        let (nvh, nkh, hkd, hvd, conv_k) = (4usize, 2, 8, 8, 4);
        let n_exp = if moe { 4usize } else { 0 };
        let (moe_inter, shared_inter) = (32usize, 32usize);
        let dir = std::env::temp_dir().join(format!("fr_bm35_{tag}_{}", std::process::id()));
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
        let key_dim = nkh * hkd;
        let value_dim = nvh * hvd;
        for l in 0..nl {
            let linear = l != nl - 1;
            let p = format!("l{l}.");
            add(format!("{p}in_ln"), vec![d], v(10 + l, d));
            add(format!("{p}post_ln"), vec![d], v(20 + l, d));
            add(format!("{p}mlp.gate_proj"), vec![d, ffn], v(70 + l, d * ffn));
            add(format!("{p}mlp.up_proj"), vec![d, ffn], v(80 + l, d * ffn));
            add(format!("{p}mlp.down_proj"), vec![ffn, d], v(90 + l, ffn * d));
            if moe {
                // RUNTIME names, not HF's packed ones. These used to be written as
                // `mlp.gate.weight` / `mlp.experts.gate_up_proj` / … which is what `convert` emits on the
                // way IN — the loader never reads them, so a MoE bundle built here had no readable experts
                // and any test that actually ran a MoE forward would panic on a missing array. The decline
                // tests passed only because they short-circuit before touching a single expert.
                add(format!("{p}gate"), vec![d, n_exp], v(100 + l, d * n_exp));
                for e in 0..n_exp {
                    // `.experts.` in the name is what puts these in the PAGED map `expert_f32` reads.
                    add(format!("{p}experts.{e}.gate"), vec![d, moe_inter], v(110 + e, d * moe_inter));
                    add(format!("{p}experts.{e}.up"), vec![d, moe_inter], v(120 + e, d * moe_inter));
                    add(format!("{p}experts.{e}.down"), vec![moe_inter, d], v(130 + e, moe_inter * d));
                }
                add(format!("{p}shared.gate_proj"), vec![d, shared_inter], v(140 + l, d * shared_inter));
                add(format!("{p}shared.up_proj"), vec![d, shared_inter], v(150 + l, d * shared_inter));
                add(format!("{p}shared.down_proj"), vec![shared_inter, d], v(160 + l, shared_inter * d));
                // (d, 1): the router sigmoid gate is read through `mm`, which needs 2-D
                add(format!("{p}shared_gate"), vec![d, 1], v(170 + l, d));
            }
            if linear {
                add(format!("{p}linear_attn.in_proj_qkv"), vec![d, 2 * key_dim + value_dim], v(200 + l, d * (2 * key_dim + value_dim)));
                add(format!("{p}linear_attn.in_proj_z"), vec![d, value_dim], v(210 + l, d * value_dim));
                add(format!("{p}linear_attn.in_proj_b"), vec![d, nvh], v(220 + l, d * nvh));
                add(format!("{p}linear_attn.in_proj_a"), vec![d, nvh], v(230 + l, d * nvh));
                add(format!("{p}linear_attn.conv1d"), vec![2 * key_dim + value_dim, conv_k], v(240 + l, (2 * key_dim + value_dim) * conv_k));
                add(format!("{p}linear_attn.A_log"), vec![nvh], v(250 + l, nvh));
                add(format!("{p}linear_attn.dt_bias"), vec![nvh], v(260 + l, nvh));
                add(format!("{p}linear_attn.norm"), vec![hvd], v(270 + l, hvd));
                add(format!("{p}linear_attn.out_proj"), vec![value_dim, d], v(280 + l, value_dim * d));
            } else {
                add(format!("{p}self_attn.q_proj"), vec![d, 2 * nh * hd], v(300 + l, d * 2 * nh * hd));
                add(format!("{p}self_attn.k_proj"), vec![d, nkv * hd], v(310 + l, d * nkv * hd));
                add(format!("{p}self_attn.v_proj"), vec![d, nkv * hd], v(320 + l, d * nkv * hd));
                add(format!("{p}self_attn.o_proj"), vec![nh * hd, d], v(330 + l, nh * hd * d));
                add(format!("{p}q_norm"), vec![hd], v(340 + l, hd));
                add(format!("{p}k_norm"), vec![hd], v(350 + l, hd));
            }
        }
        let mut cfg: Vec<i64> = vec![nl as i64, nh as i64, nkv as i64, hd as i64, d as i64, vocab as i64,
                                    0, n_exp as i64, 2, moe_inter as i64, shared_inter as i64, 0,
                                    nvh as i64, nkh as i64, hkd as i64, hvd as i64, conv_k as i64];
        for l in 0..nl { cfg.push(if l != nl - 1 { 1 } else { 0 }); }
        let manifest = serde_json::json!({ "format": crate::bundle::FORMAT, "version": crate::bundle::VERSION,
            "arch": "qwen35", "config": cfg, "config_f": [10000.0, 1e-5, 0.25], "arrays": specs });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
        Qwen35Moe::new(Bundle::load(&stem).unwrap(), 0.0, false)
    }

    /// Each test gets its OWN on-disk bundle: `cargo test` runs these concurrently, and sharing a path
    /// would let one test load another's half-written file.
    fn load_tiny(tag: &str) -> Qwen35Moe { build_tiny(tag, false) }
    fn load_tiny_moe(tag: &str) -> Qwen35Moe { build_tiny(tag, true) }
    /// FFN width, fixed by `build_tiny` (matches the `mlp.gate_proj` second dim).
    fn m_ffn(_m: &Qwen35Moe) -> usize { 16 }

    /// The whole port rests on this: a single cached scorer step over the HYBRID state must equal the naive
    /// `mlp_act_last` over the same ids, bit for bit. If the snapshot/restore path drifted from the naive walk
    /// by even one ULP, every blind-max ratio on this arch would be measured against a fiction.
    #[test]
    fn qwen35_scorer_step_matches_naive_bitwise() {
        let m = load_tiny("step");
        let ids: Vec<i64> = vec![1, 9, 4, 7, 2, 5];
        let total = ids.len() + 6;
        for layer in [0usize, 1, 3] { // a linear layer, a linear layer, the full-attention layer
            let mut sc = Qwen35Scorer::new(&m, &ids, total).unwrap();
            assert_eq!(sc.pos(), ids.len());
            for (i, &tok) in [7i64, 3, 11].iter().enumerate() {
                let n = (i * 5) % m_ffn(&m);
                let got = sc.try_token(&m, tok, layer, n).unwrap();
                let mut want_ids = ids.clone();
                want_ids.push(tok);
                let want = m.mlp_act_last(&want_ids, layer).unwrap()[n];
                assert_eq!(
                    got.to_bits(), want.to_bits(),
                    "layer {layer} tok {tok}: scorer {:e} != naive {:e}", got, want
                );
            }
        }
    }

    /// Candidates must be INDEPENDENT: probing token B after token A must give the same value as probing B
    /// on a fresh scorer. Without the restore, the DeltaNet accumulator would carry A's write into B.
    #[test]
    fn qwen35_candidates_are_independent() {
        let m = load_tiny("indep");
        let ids: Vec<i64> = vec![2, 6, 1, 8];
        let total = ids.len() + 4;
        let mut sc = Qwen35Scorer::new(&m, &ids, total).unwrap();
        let b_first = sc.try_token(&m, 5, 0, 3).unwrap();
        let _ = sc.try_token(&m, 13, 0, 3).unwrap(); // must not perturb the next probe
        let _ = sc.try_token(&m, 1, 0, 3).unwrap();
        let b_again = sc.try_token(&m, 5, 0, 3).unwrap();
        assert_eq!(b_first.to_bits(), b_again.to_bits(), "a prior probe leaked into the next candidate");
    }

    /// After commit, the state must have advanced exactly one position and the next probe must match a naive
    /// recompute over prefix ++ [committed, next].
    #[test]
    fn qwen35_commit_advances_state() {
        let m = load_tiny("commit");
        let ids: Vec<i64> = vec![3, 9, 4];
        let total = ids.len() + 4;
        let mut sc = Qwen35Scorer::new(&m, &ids, total).unwrap();
        sc.commit(&m, 6, 0, 2);
        assert_eq!(sc.pos(), ids.len() + 1);
        let got = sc.try_token(&m, 8, 0, 2).unwrap();
        let want = m.mlp_act_last(&[3, 9, 4, 6, 8], 0).unwrap()[2];
        assert_eq!(got.to_bits(), want.to_bits(), "post-commit probe {got:e} != naive {want:e}");
    }

    /// The climb's reported objective must equal a NAIVE recompute of the sequence it claims to have found.
    #[test]
    fn qwen35_climb_objective_matches_naive() {
        let m = load_tiny("climb");
        let ids: Vec<i64> = vec![1, 5, 3];
        for pol in [POLE_POS, POLE_NEG] {
            let cands = m.blind_proposals(0, 4, 8, pol).unwrap();
            assert!(!cands.is_empty());
            let arm: BlindArm = m.blind_climb(0, 4, &ids, &cands, 3, pol).unwrap();
            let act = m.mlp_act_last(&arm.ids, 0).unwrap()[4];
            assert_eq!((pol * act).to_bits(), arm.obj.to_bits(),
                       "pole {pol}: reported {:e} but naive gives {:e}", arm.obj, pol * act);
        }
    }

    /// A chunked scan must equal a single-token walk of the same stream. The corpus scan feeds `CORPUS_CHUNK`
    /// rows per forward so one weight read is amortised over 32 positions; that is only sound if a block
    /// and a sequence of single steps produce the same numbers. This is that claim, bit for bit — and it
    /// also pins the DeltaNet recurrence + conv window across chunk boundaries, which is the part of the
    /// hybrid state most likely to drift when rows are fed in bulk.
    #[test]
    fn qwen35_chunked_scan_matches_single_token_walk() {
        let m = load_tiny("chunk");
        let stream: Vec<i64> = vec![3, 9, 4, 6, 8, 1, 7, 5, 2, 11, 0, 6];
        let total = stream.len();
        // chunked, but forced to one row per block so the two paths differ only in block shape
        let mut sc = Qwen35Scorer::empty(&m, total);
        for chunk in stream.chunks(1) {
            let start = sc.pos();
            let emb = m.b.rows_f32("embed", chunk);
            let (_h, rows, _) = m.forward_block_capture_mlp(&emb, start, &mut sc.st, Some((0, 2)), None);
            sc.pos += chunk.len();
            for (j, &a) in rows.iter().enumerate() {
                let want = m.mlp_act_last(&stream[..=start + j], 0).unwrap()[2];
                assert_eq!(a.to_bits(), want.to_bits(), "block at {} row {} : {a:e} != naive {want:e}", start, j);
            }
        }
        // and the real chunked arm agrees with a naive per-position max
        let arm: BlindArm = m.blind_corpus_best(0, 2, &stream, POLE_POS).unwrap();
        let brute: Vec<f32> = (0..stream.len()).map(|i| m.mlp_act_last(&stream[..=i], 0).unwrap()[2]).collect();
        let want = brute.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(arm.obj.to_bits(), want.to_bits(), "chunked arm {:e} != naive {:e}", arm.obj, want);
    }

    /// The corpus arm must equal the brute-force max over every position of a naive `mlp_act_last`,
    /// bit for bit — and must report the argmax position, not just its value.
    #[test]
    fn qwen35_corpus_best_matches_naive_per_position_max() {
        let m = load_tiny("corpus");
        let (l, n) = (0usize, 4usize);
        let stream: Vec<i64> = vec![1, 5, 3, 9, 2, 8, 6, 4, 0, 7];
        for pol in [POLE_POS, POLE_NEG] {
            let arm: BlindArm = m.blind_corpus_best(l, n, &stream, pol).unwrap();
            let brute: Vec<f32> = (0..stream.len()).map(|i| pol * m.mlp_act_last(&stream[..=i], l).unwrap()[n]).collect();
            let want = brute.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(arm.obj.to_bits(), want.to_bits(),
                       "pole {pol}: streamed {:e} vs naive {:e}", arm.obj, want);
            let at = brute.iter().position(|&x| x == want).expect("max must occur");
            assert_eq!(arm.pos, Some(at), "reported position must be the argmax position");
            assert_eq!(arm.ids.last().copied(), Some(stream[at]), "ids must end at the best position");
        }
    }

    /// The routed oracle's value must equal an INDEPENDENT recompute.
    ///
    /// The sweep runs through the cached, chunked, hybrid-state path. So the reference here goes around
    /// it: take the residual ENTERING the layer from `hiddens`, apply the post-norm, and run the router
    /// and that one expert by hand. If the two disagree, the oracle is measuring something the search
    /// cannot see — and a ratio whose denominator came from a different instrument than its numerator is
    /// fiction. This is the whole point of picking `w_router · act`: it must be reproducible from the
    /// layer input alone, with no scorer state involved.
    /// The routed OBJECTIVE must be reproducible from the layer's post-norm input alone.
    ///
    /// This is the claim that makes `w_router · act` a legible score rather than an opaque one, and it
    /// double-checks the two things most likely to be wrong: that the router's ALWAYS-renormalised top-k
    /// weights are applied, and that the expert projections are matmuls (sum over inputs BEFORE the silu,
    /// not an elementwise product). Both mistakes produce plausible-looking numbers, so they are worth a
    /// hand-written reference.
    ///
    /// Deliberately driven from a CONTROLLED `a2` rather than the model's own residual. The FFN sees the
    /// post-ATTENTION residual, which is not reachable without re-running attention, so deriving it from
    /// `hiddens` would mean testing something other than what the sweep reads. What this test owns is the
    /// objective; the residual itself is gated by the cached-vs-naive tests on the dense arms.
    #[test]
    fn qwen35_routed_objective_matches_reference() {
        let m = load_tiny_moe("moeref");
        let (l, e, n) = (1usize, 0usize, 1usize);
        let stream: Vec<i64> = (0..6).map(|i: i64| i).collect();
        let xin = m.hiddens(&stream)[l].clone();
        let (seq, d) = (stream.len(), m.d);
        let mut rows = Array2::<f32>::zeros((seq, d));
        for t in 0..seq {
            for c in 0..d {
                rows[[t, c]] = xin[t * d + c];
            }
        }
        let a2 = m.norm(&rows, &format!("l{l}.post_ln"));

        // what the branch reports for this unit, on this exact input
        let (_out, got) = m.moe_branch(l, &a2, Some((e, n)));

        // ...and the same thing written out by hand
        let mut logits = m.b.mm(&a2, &format!("l{l}.gate"));
        softmax_rows(&mut logits);
        let (_, gv) = m.b.expert_f32(&format!("l{l}.experts.{e}.gate"));
        let (_, uv) = m.b.expert_f32(&format!("l{l}.experts.{e}.up"));
        let (dsh, _) = m.b.expert_f32(&format!("l{l}.experts.{e}.down"));
        let mi = dsh[0];
        assert_eq!(got.len(), seq, "one value per position");
        let mut n_routed = 0;
        for t in 0..seq {
            let row = logits.row(t).to_vec();
            // Qwen3.5 router: softmax -> top-k -> RENORMALISE the kept weights
            let mut idx: Vec<usize> = (0..m.n_exp).collect();
            idx.sort_by(|&x, &y| row[y].partial_cmp(&row[x]).unwrap());
            idx.truncate(m.topk);
            if !idx.contains(&e) {
                assert!(got[t].is_nan(), "t={t}: expert not selected, so the unit has NO value there");
                continue;
            }
            n_routed += 1;
            let denom: f32 = idx.iter().map(|&x| row[x]).sum::<f32>().max(1e-20);
            let w = row[e] / denom;
            // matmul first, THEN silu: gate/up are (d, mi) and project the input, they do not scale it
            let (mut g, mut u) = (0f32, 0f32);
            for i in 0..d {
                g += a2[[t, i]] * gv[i * mi + n];
                u += a2[[t, i]] * uv[i * mi + n];
            }
            let want = w * silu(g) * u;
            // NOT bit-equality, and deliberately: this reference re-derives the same arithmetic in a
            // different summation order than `expert_mm`, which costs the last ulp or two of f32. Asserting
            // bit-equality here would be asserting float noise. What this test owns is the OBJECTIVE's
            // semantics — renormalised routing, matmul-before-silu, w·act — and those differ by orders of
            // magnitude if any of the three is wrong. The BIT-EXACT claim is carried by the chunked ==
            // single-token test, which walks the identical code path.
            let rel = ((got[t] - want) / want.abs().max(1e-12)).abs();
            assert!(rel < 1e-5, "t={t}: routed objective {:e} vs reference {:e} (rel {rel:e})", got[t], want);
        }
        assert!(n_routed > 0, "fixture must route this expert somewhere");
    }

    /// The corpus sweep visits positions in `CORPUS_CHUNK`-row blocks; forcing it to one row per block
    /// must give BIT-IDENTICAL results. This is the bit-exact claim the chunked architecture rests on for
    /// routed units too — if the carried hybrid state drifted across a block boundary, this is where it
    /// would show.
    #[test]
    fn qwen35_routed_chunked_equals_single_token_walk() {
        let m = load_tiny_moe("moechunk");
        let (l, e, n) = (1usize, 2usize, 4usize);
        let stream: Vec<i64> = (0..70).map(|i: i64| (i * 3 + 2) % 24).collect();
        let chunked = m.blind_moe_corpus_best(l, e, n, &stream, POLE_POS).unwrap();
        let mut sc = Qwen35Scorer::empty(&m, stream.len());
        let mut best = f32::NEG_INFINITY;
        for (i, &t) in stream.iter().enumerate() {
            let emb = m.b.rows_f32("embed", &[t]);
            let (_h, _r, vals) = m.forward_block_capture_mlp(&emb, i, &mut sc.st, None, Some((l, e, n)));
            if let Some(v) = vals.first().copied().filter(|v| !v.is_nan()) {
                best = best.max(v);
            }
        }
        assert_eq!(chunked.obj.to_bits(), best.to_bits(),
                   "chunked {:e} vs single-token walk {:e}", chunked.obj, best);
    }

    /// A MoE layer has no fixed "neuron n" — the routed expert decides. Every entry point must decline
    /// rather than silently reporting something.
    #[test]
    fn qwen35_moe_layers_decline() {
        let m = load_tiny_moe("moe");
        assert!(m.mlp_act_last(&[1, 2, 3], 0).is_none(), "MoE mlp_act_last must be None");
        assert!(m.blind_proposals(0, 1, 8, POLE_POS).is_none(), "MoE blind_proposals must be None");
        assert!(m.blind_climb(0, 1, &[1, 2], &[5, 6], 2, POLE_POS).is_none(), "MoE climb must be None");
        assert!(m.blind_corpus_best(0, 1, &[1, 2, 3, 4], POLE_POS).is_none(), "MoE corpus arm must be None");
        // ...and it must say WHY. Reporting "no cached scorer" here would be false: the scorer is
        // implemented for this arch, and the dense release runs it. A user who cannot tell those apart
        // concludes the arch is unsupported and stops.
        assert_eq!(m.blind_support(), BlindSupport::RoutedLayers, "MoE must report RoutedLayers, not NoScorer");
    }

    /// The dense release of the SAME arch must report Supported. If this ever reports anything else the
    /// user is told a supported model is not supported — the exact false claim this method exists to stop.
    #[test]
    fn qwen35_dense_reports_supported() {
        let m = load_tiny("support");
        assert_eq!(m.blind_support(), BlindSupport::Supported, "dense qwen3_5 must report Supported");
        assert_ne!(m.blind_support(), load_tiny_moe("support_moe").blind_support(),
                   "dense and MoE releases of one arch must not report the same support");
    }
    #[test]
    fn pdc_observer_preserves_hybrid_walk_and_complete_source_sum() {
        let m = load_tiny("pdc_observer");
        let ids = vec![1, 2, 3, 4];
        let (hidden, residual, sources) = m.hidden_pdc(&ids, true);
        let snapshots = m.hiddens(&ids);
        assert_eq!(hidden.iter().cloned().collect::<Vec<_>>(), *snapshots.last().unwrap());
        assert_eq!(sources.len(), 1 + 2 * m.n_layer);
        let mut sum = vec![0.0f32; m.d];
        for (index, source) in sources.iter().enumerate() {
            if index == 0 { assert_eq!(source["id"], "embed"); }
            for (j, value) in source["write"].as_array().unwrap().iter().enumerate() {
                sum[j] += value.as_f64().unwrap() as f32;
            }
        }
        assert!(sum.iter().zip(residual.iter()).all(|(a,b)|(a-b).abs()<1e-5));
        let dump = m.pdc_source_dump(&serde_json::json!({"contexts":[ids],"candidate_tokens":[0,1,2]})).unwrap();
        assert_eq!(dump["cases"][0]["observer_hidden_max_error"], 0.0);
        assert_eq!(dump["cases"][0]["native_output_row_calls"], m.b.config[5]);
        assert!(m.pdc_source_dump(&serde_json::json!({"contexts":[[]],"candidate_tokens":[0]})).is_err());
        assert!(m.pdc_source_dump(&serde_json::json!({"contexts":[[1]],"candidate_tokens":[0,0]})).is_err());
    }

}
