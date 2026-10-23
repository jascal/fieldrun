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
//!   - **attention_k_eq_v**: GLOBAL (full) layers carry no v_proj — V is the k_proj output (value-normed, NOT RoPE'd) —
//!     and use `num_global_key_value_heads` KV heads; sliding layers are unaffected (per-layer `nkv_of(l)`).
//!   - **KV-sharing** (`num_kv_shared_layers`): the last N layers carry no k_proj/v_proj/k_norm — they keep their own
//!     q_proj/q_norm but borrow the assembled K/V of the LAST non-shared layer of their own type (`kv_src[l]`).
//! Validated to FULL parity (per-layer residual + per-position logits, not just top-1) against a tiny random-init
//! `Gemma4ForCausalLM`, dense AND MoE, via `experiments/gemma4/` (make_tiny → convert → `--hidden-dump`/`--logits-dump`
//! → compare): per-layer ≤ ~1e-5, logits ≤ ~6e-6, argmax 100%. Incremental KV-cache `generate`/`generate_stream`
//! (f32 + int8-KV; per-layer GQA width, since local/global head_dim differ) and `explain` are wired. The big embedding
//! tables (main embed + the multi-GB PLE table) and the packed experts are converted STREAMED (see `convert.rs`), so
//! `fieldrun convert --arch gemma4` stays within RAM on a small box even for the 26B+/31B checkpoints.

use std::collections::HashMap;

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Gemma4 {
    b: Bundle,
    n_layer: usize,
    h: usize,
    nkv: usize,
    nkv_g: usize,     // KV heads on GLOBAL layers under attention_k_eq_v (else == nkv)
    k_eq_v: bool,     // attention_k_eq_v: global layers reuse K as V (no v_proj), with nkv_g KV heads
    kv_src: Vec<usize>,   // per layer: which layer's K/V to use (== l normally; an earlier same-type layer if KV-shared)
    stores_kv: Vec<bool>, // per layer: is this the K/V source a later shared layer reuses (the uncached path stashes it)
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
        let (n_layer, h, nkv, nkv_g) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
        let (hd_local, hd_global, d) = (c[4] as usize, c[5] as usize, c[6] as usize);
        let tied = c[9] != 0;
        let window = c[10] as usize;
        let ple = c[11] as usize;
        let (moe, n_exp, topk, moe_inter) = (c[12] != 0, c[13] as usize, c[14] as usize, c[15] as usize);
        let sliding: Vec<bool> = (0..n_layer).map(|l| c[16 + l] != 0).collect();
        // attention_k_eq_v flag packed right after the sliding flags (c[16+nl]); absent (older bundles) → off.
        let k_eq_v = c.len() > 16 + n_layer && c[16 + n_layer] != 0;
        // KV-sharing: the last `n_kv_shared` layers borrow the assembled K/V of the LAST non-shared layer of their own
        // type (sliding vs full). Packed at c[17+nl]; absent → 0 (off). kv_src[l] points at that source (== l if not
        // shared); stores_kv[l] marks a source layer (the uncached forward stashes it for the shared layers to reuse).
        let n_kv_shared = if c.len() > 17 + n_layer { c[17 + n_layer] as usize } else { 0 };
        let first_shared = n_layer.saturating_sub(n_kv_shared);
        let (mut src_sliding, mut src_global): (Option<usize>, Option<usize>) = (None, None);
        for l in 0..first_shared {
            if sliding[l] { src_sliding = Some(l); } else { src_global = Some(l); }
        }
        let kv_src: Vec<usize> = (0..n_layer)
            .map(|l| if l >= first_shared {
                if sliding[l] { src_sliding.unwrap_or(l) } else { src_global.unwrap_or(l) }
            } else { l })
            .collect();
        let stores_kv: Vec<bool> = (0..n_layer).map(|l| n_kv_shared > 0 && (Some(l) == src_sliding || Some(l) == src_global)).collect();
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
            b, n_layer, h, nkv, nkv_g, k_eq_v, kv_src, stores_kv, hd_local, hd_global, d, ple, eps,
            escale: (d as f32).sqrt(), ple_escale: (ple as f32).sqrt(), proj_scale: (d as f32).powf(-0.5),
            inv_local, inv_global, window, sliding, tied, moe, n_exp, topk, moe_inter, layer_scalar, kv_int8,
        }
    }

    /// The MoE-FFN expert branch for one layer: route each token to its top-k experts, then for each *active* expert
    /// dequantise its weights once (paged in from the mmap) and run its assigned tokens. `x_pre` is the pre-FFN hidden
    /// (= the residual the router and experts both read). Returns the (already pre/post-normed) expert contribution.
    /// `scan = Some((expert, neuron))` additionally returns the per-row value of that ONE routed unit —
    /// `w_router · gelu_tanh(gate_n) · up_n` — recorded in the same pass so the expert's weights are read
    /// once. Rows whose top-k does not contain `expert` stay `NaN`: the unit is not computed there at all,
    /// so it has no value rather than a value of zero (see `Model::blind_moe_*`).
    ///
    /// NB the Gemma-4 router has three terms a plain softmax-router does not: a weightless RMSNorm, a
    /// per-channel `router.scale`, and a per-EXPERT `router.per_expert_scale` that is part of the weight.
    /// The activation is GEGLU (`gelu_tanh(gate) · up`), not SwiGLU. Both are load-bearing here.
    fn moe_branch(&self, l: usize, x_pre: &Array2<f32>, scan: Option<(usize, usize)>) -> (Array2<f32>, Vec<f32>) {
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
        let mut scan_out: Vec<f32> = scan.map(|_| vec![f32::NAN; seq]).unwrap_or_default();
        let mi = self.moe_inter;
        // Prefetch every active expert's weights up front (MADV_WILLNEED) so the OS pages experts 2..k from the mmap
        // while expert 1 is computed — overlapping the per-token page-in stalls that bound MoE decode under offload.
        for &e in assign.keys() {
            self.b.prefetch(&format!("{p}experts.{e}.gate_up"));
            self.b.prefetch(&format!("{p}experts.{e}.down"));
        }
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
                if let Some((se, sn)) = scan {
                    if *e == se && sn < hh.ncols() {
                        scan_out[t] = w * hh[[i, sn]];
                    }
                }
                for c in 0..self.d { h2[[t, c]] += w * outp[[i, c]]; }
            }
        }
        (self.norm(&h2, &format!("{p}post_feedforward_layernorm_2")), scan_out)
    }

    fn unembed(&self) -> &str {
        if self.tied { "embed" } else { "lm_head" }
    }

    fn hd_of(&self, l: usize) -> usize {
        if self.sliding[l] { self.hd_local } else { self.hd_global }
    }

    /// KV-head count for layer `l`: global (full) layers use `nkv_g` under attention_k_eq_v, everything else `nkv`.
    fn nkv_of(&self, l: usize) -> usize {
        if self.k_eq_v && !self.sliding[l] { self.nkv_g } else { self.nkv }
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
        if self.ple == 0 {
            return Vec::new(); // PLE off (ple=0, e.g. 26B-A4B): no per-layer inputs; the gated-residual block is skipped
        }
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

    /// Project + q-norm + RoPE the queries for layer `l` at absolute position `pos`. Always present — KV-shared layers
    /// keep their own q_proj/q_norm and only borrow K/V from an earlier same-type layer. q has `h` heads of width hd_of(l).
    fn derive_q(&self, l: usize, a: &Array2<f32>, pos: usize) -> Array2<f32> {
        let p = format!("l{l}.");
        let hd = self.hd_of(l);
        let mut q = self.b.mm(a, &format!("{p}self_attn.q_proj"));
        self.head_norm(&mut q, Some(&format!("{p}self_attn.q_norm")), self.h, hd);
        self.rope(&mut q, self.h, hd, pos, self.inv_for(l));
        q
    }

    /// Project + per-head-norm + RoPE the K/V for layer `l` at absolute position `pos`. Skipped on KV-shared layers
    /// (they reuse an earlier same-type layer's assembled K/V). Returns post-norm, post-RoPE (k, v) with `nkv_of(l)`
    /// heads of width hd_of(l). Under attention_k_eq_v a global layer has NO v_proj — V is the k_proj OUTPUT
    /// (value-normed, NOT RoPE'd), captured before k_norm/RoPE mutate `k`. v-norm is RMS-only (no weight).
    fn derive_kv(&self, l: usize, a: &Array2<f32>, pos: usize) -> (Array2<f32>, Array2<f32>) {
        let p = format!("l{l}.");
        let (hd, nkv) = (self.hd_of(l), self.nkv_of(l));
        let inv = self.inv_for(l);
        let mut k = self.b.mm(a, &format!("{p}self_attn.k_proj"));
        let mut v = if self.k_eq_v && !self.sliding[l] {
            k.clone()
        } else {
            self.b.mm(a, &format!("{p}self_attn.v_proj"))
        };
        self.head_norm(&mut k, Some(&format!("{p}self_attn.k_norm")), nkv, hd);
        self.head_norm(&mut v, None, nkv, hd);
        self.rope(&mut k, nkv, hd, pos, inv);
        (k, v)
    }

    /// Is layer `l` a KV-shared layer (no own k/v/norms — reuses `kv_src[l]`'s assembled K/V)? `kv_src[l] == l` for a
    /// normal layer, and points at an earlier same-type layer for a shared one.
    fn is_shared(&self, l: usize) -> bool {
        self.kv_src[l] != l
    }

    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let h = self.h;
        let emb = self.b.rows_f32("embed", ids);
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale));
        }
        let pli = self.per_layer_inputs(ids, &x);

            let mut kv_store: Vec<Option<(Array2<f32>, Array2<f32>)>> = vec![None; self.n_layer];
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            // --- attention ---
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, 0);
            let (k, v) = if self.is_shared(l) {
                kv_store[self.kv_src[l]].clone().expect("KV-shared source computed in an earlier layer")
            } else {
                let kv = self.derive_kv(l, &a, 0);
                if self.stores_kv[l] {
                    kv_store[l] = Some(kv.clone());
                }
                kv
            };
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
                &h1 + &{ let (mb, _) = self.moe_branch(l, &x, None); mb }
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));

            // --- PLE gated-residual block (skipped when ple=0, e.g. 26B-A4B) ---
            if self.ple > 0 {
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate")); // (seq, ple)
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l]; // gate by the per-layer embedding
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection")); // (seq, d)
                x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            }
            x *= self.layer_scalar[l]; // per-layer output scalar — the last op of the layer (Gemma4ForCausalLM)
        }
        self.norm(&x, "norm")
    }

    /// Forward capturing the recursion substrate (mirrors `hidden` for Gemma 4): per-layer post-block residual (for the
    /// per-layer logit-lens) + the element-wise max over late-layer heads of the attention matrix (the binding signal).
    /// Sliding-window layers mask distant keys (their distant scores are -inf → 0 after softmax), so a *distant* fold
    /// can only register on the GLOBAL (full-attention) layers — those are what the late-third maxback picks up.
    fn recursion_capture(&self, ids: &[i64]) -> (Vec<Array2<f32>>, Array2<f32>) {
        let seq = ids.len();
        let h = self.h;
        let late0 = 2 * self.n_layer / 3;
        let emb = self.b.rows_f32("embed", ids);
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale));
        }
        let pli = self.per_layer_inputs(ids, &x);
        let mut kv_store: Vec<Option<(Array2<f32>, Array2<f32>)>> = vec![None; self.n_layer];
        let mut resids: Vec<Array2<f32>> = Vec::with_capacity(self.n_layer);
        let mut maxback = Array2::<f32>::zeros((seq, seq));
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, 0);
            let (k, v) = if self.is_shared(l) {
                kv_store[self.kv_src[l]].clone().expect("KV-shared source computed in an earlier layer")
            } else {
                let kv = self.derive_kv(l, &a, 0);
                if self.stores_kv[l] { kv_store[l] = Some(kv.clone()); }
                kv
            };
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kvh = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kvh * hd..(kvh + 1) * hd]);
                let vh = v.slice(s![.., kvh * hd..(kvh + 1) * hd]);
                let mut scores = qh.dot(&kh.t()); // scaling = 1.0 in Gemma 4
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i || (sliding && j + self.window <= i) { scores[[i, j]] = -1e30; }
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
                &h1 + &{ let (mb, _) = self.moe_branch(l, &x, None); mb }
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            if self.ple > 0 {
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l];
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
                x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            }
            x *= self.layer_scalar[l];
            resids.push(x.clone());
        }
        (resids, maxback)
    }

    /// Debug: residual-stream snapshots in transformers `output_hidden_states` convention — [inputs_embeds (=√d·embed),
    /// out-L0, …, out-L{n-2}, POST-final-norm] (each flat seq·d, row-major), nl+1 entries. Gemma4's last hidden_state
    /// IS the post-final-norm output (verified hs[-1] == last_hidden_state), so the final slot is normed, matching HF.
    /// Mirrors `hidden()` exactly — kept as a separate walk (like the qwen35moe harness) so the hot path stays clean.
    pub fn hiddens(&self, ids: &[i64]) -> Vec<Vec<f32>> {
        let seq = ids.len();
        let h = self.h;
        let emb = self.b.rows_f32("embed", ids);
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale));
        }
        let mut snaps = Vec::with_capacity(self.n_layer + 1);
        snaps.push(x.iter().cloned().collect()); // hs[0] = inputs_embeds
        let pli = self.per_layer_inputs(ids, &x);
        let mut kv_store: Vec<Option<(Array2<f32>, Array2<f32>)>> = vec![None; self.n_layer];
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, 0);
            let (k, v) = if self.is_shared(l) {
                kv_store[self.kv_src[l]].clone().expect("KV-shared source computed in an earlier layer")
            } else {
                let kv = self.derive_kv(l, &a, 0);
                if self.stores_kv[l] {
                    kv_store[l] = Some(kv.clone());
                }
                kv
            };
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t());
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
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = gelu_tanh(*hv) * uv;
            }
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            let combined = if self.moe {
                let h1 = self.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &{ let (mb, _) = self.moe_branch(l, &x, None); mb }
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            if self.ple > 0 { // PLE gated-residual block — skipped when ple=0 (e.g. 26B-A4B)
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l];
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
                x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            }
            x *= self.layer_scalar[l];
            if l + 1 < self.n_layer {
                snaps.push(x.iter().cloned().collect()); // hs[l+1] = out-L{l}
            }
        }
        snaps.push(self.norm(&x, "norm").iter().cloned().collect()); // hs[N] = post-final-norm
        snaps
    }

    fn head_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let logits = self.b.rowdot_f32(self.unembed(), &xfn.row(xfn.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn explanation(&self, ids: &[i64]) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let h = self.h;
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
            let mut kv_store: Vec<Option<(Array2<f32>, Array2<f32>)>> = vec![None; self.n_layer];
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, 0);
            let (k, v) = if self.is_shared(l) {
                kv_store[self.kv_src[l]].clone().expect("KV-shared source computed in an earlier layer")
            } else {
                let kv = self.derive_kv(l, &a, 0);
                if self.stores_kv[l] {
                    kv_store[l] = Some(kv.clone());
                }
                kv
            };
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
                &h1 + &{ let (mb, _) = self.moe_branch(l, &x, None); mb }
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            if self.ple > 0 { // PLE gated-residual block — skipped when ple=0 (e.g. 26B-A4B)
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l];
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
                x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            }
            x *= self.layer_scalar[l]; // per-layer output scalar — the last op of the layer (Gemma4ForCausalLM)
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
            |l, n| self.b.weight_row(&format!("l{l}.mlp.down_proj"), n),
            |l, head| head_raw_contrib(&self.b, &format!("l{l}.self_attn.o_proj"), &head_act[l], head, head_hd[l]),
            |c| self.b.rowdot_f32(un, c),
        )
    }

    /// Run `m` new positions through the layers, caching K/V (post value-norm / RoPE; per-layer GQA width nkv*hd_of(l),
    /// which differs between local and global layers) and attending over the whole cache (causal + per-layer sliding
    /// window). The PLE gated-residual block is per-token, so it is recomputed for the new rows only. cur = absolute
    /// position of the first new row.
    /// `forward_block` that also returns the per-row value of ONE routed unit at the target layer — the
    /// blind-max scoring kernel. Identical walk and arithmetic (same PLE inputs, same KV-sharing source,
    /// same sliding pattern, same dense+MoE sum and norms), so its value is bit-identical to a naive
    /// per-position forward; it only ADVANCES the cache, which is why the scorer snapshots before probing.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_scan(&self, ids: &[i64], emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>],
                          vc: &mut [Array2<f32>], scan: Option<(usize, usize, usize)>) -> Vec<f32> {
        let h = self.h;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let pli = self.per_layer_inputs(ids, &x);
        let mut scan_out: Vec<f32> = Vec::new();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, cur);
            if !self.is_shared(l) {
                let (k, v) = self.derive_kv(l, &a, cur);
                kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
                vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            }
            let src = self.kv_src[l];
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = kc[src].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let vh = vc[src].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
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
                let sc = scan.filter(|(sl, ..)| *sl == l).map(|(_, e, n)| (e, n));
                let (mb, vals) = self.moe_branch(l, &x, sc);
                if !vals.is_empty() { scan_out = vals; }
                let h1 = self.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &mb
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            if self.ple > 0 {
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l];
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
                x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            }
            x *= self.layer_scalar[l];
        }
        scan_out
    }

    fn forward_block(&self, ids: &[i64], emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let h = self.h;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let pli = self.per_layer_inputs(ids, &x);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, cur);
            if !self.is_shared(l) {
                let (k, v) = self.derive_kv(l, &a, cur);
                kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
                vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            }
            let src = self.kv_src[l]; // KV-shared layers attend over an earlier same-type layer's cache
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = kc[src].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let vh = vc[src].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
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
                &h1 + &{ let (mb, _) = self.moe_branch(l, &x, None); mb }
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            if self.ple > 0 { // PLE gated-residual block — skipped when ple=0 (e.g. 26B-A4B)
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l];
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
                x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            }
            x *= self.layer_scalar[l]; // per-layer output scalar — the last op of the layer (Gemma4ForCausalLM)
        }
        self.norm(&x, "norm")
    }

    /// `forward_block` with an int8 KV cache (per-layer GQA width, per-kv-head scale): ~4x smaller cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_q(&self, ids: &[i64], emb: &Array2<f32>, cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>],
                       vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]) -> Array2<f32> {
        let h = self.h;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let pli = self.per_layer_inputs(ids, &x);
        let q8 = |v: f32, sc: f32| (v / sc).round().clamp(-127.0, 127.0) as i8;
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            let kvdim = nkv * hd;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, cur);
            let src = self.kv_src[l]; // KV-shared layers attend over an earlier same-type layer's cache
            if !self.is_shared(l) {
                let (k, v) = self.derive_kv(l, &a, cur);
                for i in 0..m {
                    let pos = cur + i;
                    for kh in 0..nkv {
                        let base = kh * hd;
                        let sck = ((0..hd).fold(0f32, |mx, c| mx.max(k[[i, base + c]].abs())) / 127.0).max(1e-8);
                        let scv = ((0..hd).fold(0f32, |mx, c| mx.max(v[[i, base + c]].abs())) / 127.0).max(1e-8);
                        ks[src][pos * nkv + kh] = sck;
                        vs[src][pos * nkv + kh] = scv;
                        for c in 0..hd {
                            kc[src][pos * kvdim + base + c] = q8(k[[i, base + c]], sck);
                            vc[src][pos * kvdim + base + c] = q8(v[[i, base + c]], scv);
                        }
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
                    let (sck, scv) = (ks[src][pos * nkv + kv], vs[src][pos * nkv + kv]);
                    for c in 0..hd {
                        kh_a[[pos, c]] = kc[src][pos * kvdim + kv * hd + c] as f32 * sck;
                        vh_a[[pos, c]] = vc[src][pos * kvdim + kv * hd + c] as f32 * scv;
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
                &h1 + &{ let (mb, _) = self.moe_branch(l, &x, None); mb }
            } else {
                mlp
            };
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            if self.ple > 0 { // PLE gated-residual block — skipped when ple=0 (e.g. 26B-A4B)
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l];
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
                x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            }
            x *= self.layer_scalar[l]; // per-layer output scalar — the last op of the layer (Gemma4ForCausalLM)
        }
        self.norm(&x, "norm")
    }

    fn generate_kv_int8(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let total = prompt.len() + n_new;
        let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|l| vec![0i8; total * self.nkv_of(l) * self.hd_of(l)]).collect();
        let mut vc = kc.clone();
        let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|l| vec![0f32; total * self.nkv_of(l)]).collect();
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

/// The blind-max scorer for this arch: a KV-cache twin of [`Gemma4::forward_block_scan`] held at position
/// `pos`, with a snapshot so candidate tokens never see each other's writes. The cache is per-LAYER-WIDTH here
/// (`nkv_of(l) · hd_of(l)` — sliding and global layers differ, and KV-shared layers point at an earlier
/// layer's cache), so allocation goes through the same per-layer expression `generate` uses.
struct Gemma4Scorer {
    kc: Vec<Array2<f32>>,
    vc: Vec<Array2<f32>>,
    snap_kc: Vec<Array2<f32>>,
    snap_vc: Vec<Array2<f32>>,
    pos: usize,
}

impl Gemma4Scorer {
    fn alloc(m: &Gemma4, total: usize) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let kc: Vec<Array2<f32>> = (0..m.n_layer)
            .map(|l| Array2::zeros((total, m.nkv_of(l) * m.hd_of(l)))).collect();
        let vc = kc.clone();
        (kc, vc)
    }

    /// Prefill `prefix`, then score candidates at the next position.
    fn new(m: &Gemma4, prefix: &[i64], total: usize) -> Option<Gemma4Scorer> {
        if prefix.is_empty() || prefix.len() >= total {
            return None;
        }
        let (mut kc, mut vc) = Gemma4Scorer::alloc(m, total);
        let emb = m.b.rows_f32("embed", prefix);
        m.forward_block(prefix, &emb, 0, &mut kc, &mut vc);
        let snap_kc = kc.clone();
        let snap_vc = vc.clone();
        Some(Gemma4Scorer { kc, vc, snap_kc, snap_vc, pos: prefix.len() })
    }

    /// An empty scorer at position 0 — no seed context, for the corpus scan, which places every token.
    fn empty(m: &Gemma4, total: usize) -> Gemma4Scorer {
        let (kc, vc) = Gemma4Scorer::alloc(m, total);
        let snap_kc = kc.clone();
        let snap_vc = vc.clone();
        Gemma4Scorer { kc, vc, snap_kc, snap_vc, pos: 0 }
    }

    /// Score `token` at `pos` by the ROUTED unit's `w_router · act`, WITHOUT committing. `None` for a bad
    /// unit; `Some(NaN)` when the router did not select that expert here — no value, not a value of zero.
    fn try_token_moe(&mut self, m: &Gemma4, token: i64, layer: usize, expert: usize, neuron: usize) -> Option<f32> {
        if layer >= m.n_layer || expert >= m.n_exp { return None; }
        self.kc.clone_from(&self.snap_kc); // candidates must not see each other's writes
        self.vc.clone_from(&self.snap_vc);
        let emb = m.b.rows_f32("embed", &[token]);
        let vals = m.forward_block_scan(&[token], &emb, self.pos, &mut self.kc, &mut self.vc,
                                        Some((layer, expert, neuron)));
        vals.first().copied()
    }

    /// Advance the cache by `token` (the winner's value was already read by the probe).
    fn commit(&mut self, m: &Gemma4, token: i64) {
        let emb = m.b.rows_f32("embed", &[token]);
        m.forward_block(&[token], &emb, self.pos, &mut self.kc, &mut self.vc);
        self.pos += 1;
        self.snap_kc.clone_from(&self.kc);
        self.snap_vc.clone_from(&self.vc);
    }
}

impl Model for Gemma4 {
    /// On a Gemma-4 MoE bundle the scorer IS implemented, so the honest answer is `RoutedLayers` (the dense
    /// per-layer unit is declined; `--blind-max L,E,N` names a routed one). A DENSE Gemma-4 bundle reports
    /// `NoScorer` honestly: only the routed arm exists for this arch, so claiming `Supported` would be the
    /// very incoherence this enum exists to prevent.
    fn blind_support(&self) -> crate::model::BlindSupport {
        if self.moe && self.n_exp != 0 {
            crate::model::BlindSupport::RoutedLayers
        } else {
            crate::model::BlindSupport::NoScorer
        }
    }

    /// One ROUTED unit's self-description: expert `e` neuron `n`'s own output row. Gemma-4 experts are
    /// PACKED, so `down` is the second half of nothing — it is its own (d, moe_inter)-shaped projection read
    /// as (in, out): row `n` is one contiguous d-wide run. Paged, so it goes through `expert_f32`.
    fn blind_moe_proposals(&self, layer: usize, expert: usize, neuron: usize, k: usize, polarity: f32) -> Option<Vec<i64>> {
        if layer >= self.n_layer || !self.moe || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
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
        if cands.is_empty() || layer >= self.n_layer || !self.moe || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let mut sc = Gemma4Scorer::new(self, prefix, prefix.len() + steps + 1)?;
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
    /// `expert`. Chunked exactly like the dense arm, advancing the same per-layer-width cache.
    fn blind_moe_corpus_best(&self, layer: usize, expert: usize, neuron: usize, ids: &[i64],
                             polarity: f32) -> Option<crate::model::BlindArm> {
        if ids.len() < 2 || layer >= self.n_layer || !self.moe || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let mut sc = Gemma4Scorer::empty(self, ids.len());
        let (mut best_obj, mut best_idx, mut routed) = (f32::NEG_INFINITY, 0usize, 0usize);
        for chunk in ids.chunks(crate::model::CORPUS_CHUNK) {
            let start = sc.pos;
            let emb = self.b.rows_f32("embed", chunk);
            let vals = self.forward_block_scan(chunk, &emb, start, &mut sc.kc, &mut sc.vc,
                                                Some((layer, expert, neuron)));
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


    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32(self.unembed(), &last);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn logits(&self, ids: &[i64]) -> Option<Vec<f32>> {
        let xf = self.hidden(ids);
        Some(self.b.rowdot_f32(self.unembed(), &xf.row(ids.len() - 1).to_vec()))
    }

    fn recursion_trace(&self, ids: &[i64]) -> Option<Vec<crate::model::RecPos>> {
        if ids.len() < 3 { return Some(vec![]); }
        let (resids, maxback) = self.recursion_capture(ids);
        let un = self.unembed();
        // arch-specific lens only: final RMSNorm ("norm") then unembed argmax (gemma's final logit softcap is monotone
        // so it does not change the argmax → omitted). The rest of the assembly is shared (build_rec_trace).
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

    fn dims(&self) -> Option<(usize, usize)> {
        Some((self.n_layer, self.h))
    }

    /// CAUSAL ablation forward (mirrors `hidden`): zero a listed head / whole-attn-block / mlp-neuron / whole-mlp-block
    /// write before it enters the residual, then predict. By gemma RMSNorm, `norm(0) = 0`, so zeroing a block's
    /// pre-norm write removes its residual contribution exactly. PLE + `layer_scalar` are kept (we ablate the attn/mlp
    /// block, not the per-layer-embedding term). Enables `--causal-dump` (load-bearing-block discovery) on Gemma 4.
    fn predict_ablated_blocks(&self, ids: &[i64], heads: &[(usize, usize)], neurons: &[(usize, usize)], attn_layers: &[usize], mlp_layers: &[usize]) -> Option<i64> {
        let seq = ids.len();
        let h = self.h;
        let emb = self.b.rows_f32("embed", ids);
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale));
        }
        let pli = self.per_layer_inputs(ids, &x);
        let mut kv_store: Vec<Option<(Array2<f32>, Array2<f32>)>> = vec![None; self.n_layer];
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, 0);
            let (k, v) = if self.is_shared(l) {
                kv_store[self.kv_src[l]].clone().expect("KV-shared source computed in an earlier layer")
            } else {
                let kv = self.derive_kv(l, &a, 0);
                if self.stores_kv[l] { kv_store[l] = Some(kv.clone()); }
                kv
            };
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kvh = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kvh * hd..(kvh + 1) * hd]);
                let vh = v.slice(s![.., kvh * hd..(kvh + 1) * hd]);
                let mut scores = qh.dot(&kh.t());
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i || (sliding && j + self.window <= i) { scores[[i, j]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            for &(al, ah) in heads {
                if al == l && ah < h { attn_out.slice_mut(s![.., ah * hd..(ah + 1) * hd]).fill(0.0); }
            }
            if attn_layers.contains(&l) { attn_out.fill(0.0); }
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            x = &x + &self.norm(&o, &format!("{p}post_attention_layernorm"));

            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) { *hv = gelu_tanh(*hv) * uv; }
            for &(nl, nn) in neurons {
                if nl == l && nn < hidden.ncols() { for r in 0..seq { hidden[[r, nn]] = 0.0; } }
            }
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            let mut combined = if self.moe {
                let h1 = self.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &{ let (mb, _) = self.moe_branch(l, &x, None); mb }
            } else {
                mlp
            };
            if mlp_layers.contains(&l) { combined.fill(0.0); }
            x = &x + &self.norm(&combined, &format!("{p}post_feedforward_layernorm"));

            if self.ple > 0 {
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l];
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
                x = &x + &self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
            }
            x *= self.layer_scalar[l];
        }
        let xf = self.norm(&x, "norm");
        let logits = self.b.rowdot_f32(self.unembed(), &xf.row(seq - 1).to_vec());
        Some(logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
    }

    /// Per-block residual-stream WRITES at the last position, folded through the final RMSNorm so that
    /// ⟨write_b, U_v⟩ is block `b`'s EXACT additive logit contribution to token `v` (this is the substrate
    /// `--export-logic` / Density-Minimization consume; see `residual_decomp`). Gemma 4's residual is richer than
    /// the rope backbone, so three extra factors are folded in vs `rope::residual_normed_writes`:
    ///   • the √d **embed scale** (the embed write is `√d·embed`);
    ///   • each block write is **post-block-normed** (`post_attention_layernorm` / `post_feedforward_layernorm` /
    ///     `post_per_layer_input_norm`) before it enters the residual — so the captured write IS that normed vector;
    ///   • the per-layer **`layer_scalar`** rescales the WHOLE residual at each layer's end, so a write created in
    ///     layer `l` carries the cumulative factor `Π_{j≥l} ls_j`. Folding that in keeps the decomposition exact:
    ///       x_pre_final_norm = (Π ls)·(√d·embed) + Σ_l (Π_{j≥l} ls_j)·(attn_l + ffn_l + ple_l).
    /// PLE-off models (ple=0) simply emit no `L{l}.ple` write. The FFN write is one block (dense + MoE combined,
    /// like qwen3moe). Mirrors `hidden()` exactly as a separate walk (the precedent set by the other archs).
    fn residual_normed_writes(&self, ids: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        let seq = ids.len();
        let h = self.h;
        let last = seq - 1;
        let emb = self.b.rows_f32("embed", ids);
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale));
        }
        let pli = self.per_layer_inputs(ids, &x);
        let mut labels: Vec<String> = vec!["embed".into()];
        let mut writes: Vec<Vec<f32>> = vec![x.row(last).to_vec()];
        let mut wlayer: Vec<usize> = vec![0]; // the layer each write belongs to (embed → 0, i.e. cumll over all layers)
        let mut kv_store: Vec<Option<(Array2<f32>, Array2<f32>)>> = vec![None; self.n_layer];
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, 0);
            let (k, v) = if self.is_shared(l) {
                kv_store[self.kv_src[l]].clone().expect("KV-shared source computed in an earlier layer")
            } else {
                let kv = self.derive_kv(l, &a, 0);
                if self.stores_kv[l] {
                    kv_store[l] = Some(kv.clone());
                }
                kv
            };
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t());
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
            let aw = self.norm(&o, &format!("{p}post_attention_layernorm"));
            writes.push(aw.row(last).to_vec());
            labels.push(format!("L{l}.attn"));
            wlayer.push(l);
            x = &x + &aw;
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = gelu_tanh(*hv) * uv;
            }
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            let combined = if self.moe {
                let h1 = self.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &{ let (mb, _) = self.moe_branch(l, &x, None); mb }
            } else {
                mlp
            };
            let fw = self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            writes.push(fw.row(last).to_vec());
            labels.push(format!("L{l}.ffn"));
            wlayer.push(l);
            x = &x + &fw;
            if self.ple > 0 {
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l];
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
                let pw = self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
                writes.push(pw.row(last).to_vec());
                labels.push(format!("L{l}.ple"));
                wlayer.push(l);
                x = &x + &pw;
            }
            x *= self.layer_scalar[l];
        }
        // cumulative layer_scalar from layer l onward: cumll[l] = Π_{j=l}^{nl-1} ls_j (cumll[nl] = 1).
        let mut cumll = vec![1.0f32; self.n_layer + 1];
        for l in (0..self.n_layer).rev() {
            cumll[l] = cumll[l + 1] * self.layer_scalar[l];
        }
        for (wi, w) in writes.iter_mut().enumerate() {
            let factor = cumll[wlayer[wi]];
            if factor != 1.0 {
                for v in w.iter_mut() {
                    *v *= factor;
                }
            }
        }
        // final RMSNorm geometry at the last position (Gemma-4 norm = weight-direct, no center/bias): with the cumll
        // factors applied, Σ_b write_b == the pre-final-norm residual `x.row(last)`, so the folded contribution vector
        // is d̃_b = inv_rms · gain ⊙ write_b and ⟨d̃_b, U_v⟩ is block b's exact logit contribution to token v.
        let xpre = x.row(last).to_vec();
        let dlen = xpre.len();
        let inv_rms = 1.0 / (xpre.iter().map(|v| v * v).sum::<f32>() / dlen as f32 + self.eps).sqrt();
        let gain = self.b.arr1("norm");
        let dvec: Vec<Vec<f32>> = writes
            .iter()
            .map(|w| w.iter().zip(gain.iter()).map(|(wd, gd)| inv_rms * wd * gd).collect())
            .collect();
        Some((labels, dvec))
    }

    fn residual_decomp(&self, ids: &[i64], toks: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        // The per-block folded contribution vectors projected onto the requested token rows: contrib[b][i] =
        // ⟨d̃_b, U_{toks[i]}⟩. By construction Σ_b contrib[b][i] == logit(toks[i]). Enables `--export-logic` for gemma4.
        let (labels, dvec) = self.residual_normed_writes(ids)?;
        let un = self.unembed();
        let urows: Vec<Vec<f32>> = toks.iter().map(|&t| self.b.weight_row(un, t as usize)).collect();
        let contrib: Vec<Vec<f32>> = dvec
            .iter()
            .map(|d| urows.iter().map(|u| d.iter().zip(u.iter()).map(|(dd, ud)| dd * ud).sum::<f32>()).collect())
            .collect();
        Some((labels, contrib))
    }

    fn decision_decomp(&self, ids: &[i64], extra: &[i64], cap: usize)
        -> Option<(i64, i64, f32, f32, Vec<i64>, Vec<(String, Vec<f32>)>)> {
        // ONE forward (no explain capture): folded per-block writes d̃_b at the predicting position.
        let (labels, dvec) = self.residual_normed_writes(ids)?;
        // Σ_b d̃_b is the folded residual; U·(Σ d̃_b) is the exact full-vocab logit vector → argmax + runner-up.
        let d = dvec[0].len();
        let mut full = vec![0f32; d];
        for v in &dvec {
            for (i, &x) in v.iter().enumerate() { full[i] += x; }
        }
        let logits = self.b.rowdot_f32(self.unembed(), &full);
        let (mut pred, mut p1, mut runner, mut p2) = (0usize, f32::MIN, 0usize, f32::MIN);
        for (i, &l) in logits.iter().enumerate() {
            if l > p1 { p2 = p1; runner = pred; p1 = l; pred = i; }
            else if l > p2 { p2 = l; runner = i; }
        }
        // candidate set = {predicted, runner-up} ∪ extra (store/n-gram proposed), deduped, capped.
        let mut cands = vec![pred as i64];
        if runner != pred { cands.push(runner as i64); }
        for &x in extra {
            if cands.len() >= cap { break; }
            if !cands.contains(&x) { cands.push(x); }
        }
        // per-block contribution to each candidate's logit: ⟨d̃_b, U_cand⟩ (candidate-only unembed → cheap).
        let un = self.unembed();
        let urows: Vec<Vec<f32>> = cands.iter().map(|&c| self.b.weight_row(un, c as usize)).collect();
        let blocks: Vec<(String, Vec<f32>)> = labels
            .iter()
            .zip(dvec.iter())
            .map(|(lab, dv)| (lab.clone(), urows.iter().map(|u| dv.iter().zip(u).map(|(a, b)| a * b).sum::<f32>()).collect()))
            .collect();
        Some((pred as i64, runner as i64, p1, p2, cands, blocks))
    }

    fn decomp_all(&self, ids: &[i64], cap: usize)
        -> Option<Vec<(i64, i64, f32, f32, Vec<i64>, Vec<(String, Vec<f32>)>)>> {
        // ONE forward over the whole sentence, keeping each block's write at ALL positions (seq×d matrices). Mirrors
        // `residual_normed_writes` but captures full matrices instead of the last row — so every position's decision
        // comes out of this single (fixed-cost) forward.
        let seq = ids.len();
        let h = self.h;
        let emb = self.b.rows_f32("embed", ids);
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale));
        }
        let pli = self.per_layer_inputs(ids, &x);
        let mut labels: Vec<String> = vec!["embed".into()];
        let mut wmats: Vec<Array2<f32>> = vec![x.clone()];
        let mut wlayer: Vec<usize> = vec![0];
        let mut kv_store: Vec<Option<(Array2<f32>, Array2<f32>)>> = vec![None; self.n_layer];
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let hd = self.hd_of(l);
            let nkv = self.nkv_of(l);
            let rep = h / nkv;
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let q = self.derive_q(l, &a, 0);
            let (k, v) = if self.is_shared(l) {
                kv_store[self.kv_src[l]].clone().expect("KV-shared source computed in an earlier layer")
            } else {
                let kv = self.derive_kv(l, &a, 0);
                if self.stores_kv[l] { kv_store[l] = Some(kv.clone()); }
                kv
            };
            let sliding = self.sliding[l];
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t());
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i || (sliding && j + self.window <= i) { scores[[i, j]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let aw = self.norm(&o, &format!("{p}post_attention_layernorm"));
            x = &x + &aw;
            wmats.push(aw); labels.push(format!("L{l}.attn")); wlayer.push(l);
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) { *hv = gelu_tanh(*hv) * uv; }
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            let combined = if self.moe {
                let h1 = self.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &{ let (mb, _) = self.moe_branch(l, &x, None); mb }
            } else { mlp };
            let fw = self.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            x = &x + &fw;
            wmats.push(fw); labels.push(format!("L{l}.ffn")); wlayer.push(l);
            if self.ple > 0 {
                let mut g = self.b.mm(&x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[l];
                let proj = self.b.mm(&g, &format!("{p}per_layer_projection"));
                let pw = self.norm(&proj, &format!("{p}post_per_layer_input_norm"));
                x = &x + &pw;
                wmats.push(pw); labels.push(format!("L{l}.ple")); wlayer.push(l);
            }
            x *= self.layer_scalar[l];
        }
        // cumulative layer_scalar from layer l onward, folded into each write (so Σ_b wmats[b] == x at every position).
        let mut cumll = vec![1.0f32; self.n_layer + 1];
        for l in (0..self.n_layer).rev() { cumll[l] = cumll[l + 1] * self.layer_scalar[l]; }
        for (wi, w) in wmats.iter_mut().enumerate() {
            let f = cumll[wlayer[wi]];
            if f != 1.0 { w.mapv_inplace(|v| v * f); }
        }
        // x is now the pre-final-norm residual at all positions. Per position: fold the final norm, full unembed for the
        // argmax + top-`cap` candidates, then project each block onto those candidates.
        let gain = self.b.arr1("norm");
        let gain = gain.as_slice().unwrap();
        let un = self.unembed();
        let dlen = self.d;
        let mut out = Vec::with_capacity(seq);
        for pos in 0..seq {
            let xp = x.row(pos);
            let inv_rms = 1.0 / (xp.iter().map(|v| v * v).sum::<f32>() / dlen as f32 + self.eps).sqrt();
            let normed: Vec<f32> = xp.iter().zip(gain).map(|(v, g)| inv_rms * v * g).collect();
            let logits = self.b.rowdot_f32(un, &normed);
            let vocab = logits.len();
            let k = cap.min(vocab).max(1);
            let mut order: Vec<usize> = (0..vocab).collect();
            order.select_nth_unstable_by(k - 1, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            order.truncate(k);
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let pred = order[0] as i64;
            let runner = if order.len() > 1 { order[1] as i64 } else { pred };
            let (pl, rl) = (logits[order[0]], logits[*order.get(1).unwrap_or(&order[0])]);
            let cands: Vec<i64> = order.iter().map(|&i| i as i64).collect();
            let urows: Vec<Vec<f32>> = cands.iter().map(|&c| self.b.weight_row(un, c as usize)).collect();
            let blocks: Vec<(String, Vec<f32>)> = labels
                .iter()
                .zip(wmats.iter())
                .map(|(lab, w)| {
                    let wp = w.row(pos);
                    // d̃_b at this position = inv_rms · gain ⊙ write_b[pos]; contrib = ⟨d̃_b, U_cand⟩
                    let dv: Vec<f32> = wp.iter().zip(gain).map(|(v, g)| inv_rms * v * g).collect();
                    (lab.clone(), urows.iter().map(|u| dv.iter().zip(u).map(|(a, b)| a * b).sum::<f32>()).collect())
                })
                .collect();
            out.push((pred, runner, pl, rl, cands, blocks));
        }
        Some(out)
    }

    fn explain(&self, ids: &[i64]) -> Option<crate::explain::Explanation> {
        Some(self.explanation(ids))
    }

    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        if self.kv_int8 {
            return self.generate_kv_int8(prompt, n_new);
        }
        let total = prompt.len() + n_new;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|l| Array2::zeros((total, self.nkv_of(l) * self.hd_of(l)))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|l| Array2::zeros((total, self.nkv_of(l) * self.hd_of(l)))).collect();
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
        if self.kv_int8 {
            let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|l| vec![0i8; total * self.nkv_of(l) * self.hd_of(l)]).collect();
            let mut vc = kc.clone();
            let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|l| vec![0f32; total * self.nkv_of(l)]).collect();
            let mut vs = ks.clone();
            let mut emb = self.b.rows_f32("embed", prompt) * self.escale;
            let xb = self.forward_block_q(prompt, &emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
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
                let xb = self.forward_block_q(&[next], &emb, pos, &mut kc, &mut ks, &mut vc, &mut vs);
                next = self.head_argmax(&xb);
                pos += 1;
            }
            return out;
        }
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|l| Array2::zeros((total, self.nkv_of(l) * self.hd_of(l)))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|l| Array2::zeros((total, self.nkv_of(l) * self.hd_of(l)))).collect();
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

    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool, cache: &mut crate::model::PrefixKv) -> Vec<i64> {
        if self.kv_int8 {
            let n_layer = self.n_layer;
            let alloc = |total: usize| {
                let kc: Vec<Vec<i8>> = (0..n_layer).map(|l| vec![0i8; total * self.nkv_of(l) * self.hd_of(l)]).collect();
                let vc = kc.clone();
                let ks: Vec<Vec<f32>> = (0..n_layer).map(|l| vec![0f32; total * self.nkv_of(l)]).collect();
                let vs = ks.clone();
                (kc, vc, ks, vs)
            };
            let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>], vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]| {
                let emb = self.b.rows_f32("embed", ids) * self.escale;
                self.forward_block_q(ids, &emb, cur, kc, ks, vc, vs)
            };
            return crate::model::prefix_generate_q(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, _ctx| self.head_argmax(xb));
        }
        let n_layer = self.n_layer;
        // per-layer GQA width: local and global layers have different head_dim, so kc[l] width = nkv * hd_of(l)
        let alloc = |total: usize| {
            let kc: Vec<Array2<f32>> = (0..n_layer).map(|l| Array2::zeros((total, self.nkv_of(l) * self.hd_of(l)))).collect();
            let vc = kc.clone();
            (kc, vc)
        };
        let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]| {
            let emb = self.b.rows_f32("embed", ids) * self.escale;
            self.forward_block(ids, &emb, cur, kc, vc)
        };
        crate::model::prefix_generate(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, _ctx| self.head_argmax(xb))
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

    /// A tiny Gemma-4 bundle, `moe = true` swapping the dense MLP for the routed branch. No PLE and a uniform
    /// head_dim keep the fixture minimal; the router's `scale` / `per_expert_scale` are real (not identity)
    /// because they are load-bearing. Each test gets its OWN path — `cargo test` runs these concurrently.
    fn build_tiny(tag: &str, moe: bool) -> Gemma4 {
        let (nl, nh, nkv, hd, d, vocab, ffn) = (4usize, 4, 2, 8, 32, 24, 16);
        let (n_exp, topk, mi) = (4usize, 2usize, 32usize);
        let window = 0usize; // full attention — the sliding pattern is not what this test module is about
        let dir = std::env::temp_dir().join(format!("fr_bm_g4_{tag}_{}", std::process::id()));
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
        if moe {
            // tied unembed: the fixture omits lm_head, so `unembed()` resolves to `embed`
        } else {
            add("lm_head".into(), vec![vocab, d], v(3, vocab * d));
        }
        for l in 0..nl {
            let p = format!("l{l}.");
            add(format!("{p}input_layernorm"), vec![d], v(10 + l, d));
            add(format!("{p}post_attention_layernorm"), vec![d], v(20 + l, d));
            add(format!("{p}post_feedforward_layernorm"), vec![d], v(25 + l, d));
            add(format!("{p}self_attn.q_proj"), vec![d, nh * hd], v(300 + l, d * nh * hd));
            add(format!("{p}self_attn.k_proj"), vec![d, nkv * hd], v(310 + l, d * nkv * hd));
            add(format!("{p}self_attn.v_proj"), vec![d, nkv * hd], v(320 + l, d * nkv * hd));
            add(format!("{p}self_attn.o_proj"), vec![nh * hd, d], v(330 + l, nh * hd * d));
            add(format!("{p}self_attn.q_norm"), vec![hd], v(340 + l, hd));
            add(format!("{p}self_attn.k_norm"), vec![hd], v(350 + l, hd));
            add(format!("{p}layer_scalar"), vec![1], v(600 + l, 1));
            add(format!("{p}pre_feedforward_layernorm"), vec![d], v(40 + l, d));
            add(format!("{p}mlp.gate_proj"), vec![d, ffn], v(70 + l, d * ffn));
            add(format!("{p}mlp.up_proj"), vec![d, ffn], v(80 + l, d * ffn));
            add(format!("{p}mlp.down_proj"), vec![ffn, d], v(90 + l, ffn * d));
            if moe {
                add(format!("{p}post_feedforward_layernorm_1"), vec![d], v(50 + l, d));
                add(format!("{p}post_feedforward_layernorm_2"), vec![d], v(51 + l, d));
                add(format!("{p}pre_feedforward_layernorm_2"), vec![d], v(52 + l, d));
                add(format!("{p}router.proj"), vec![d, n_exp], v(100 + l, d * n_exp));
                add(format!("{p}router.scale"), vec![d], v(400 + l, d));
                add(format!("{p}router.per_expert_scale"), vec![n_exp], v(410 + l, n_exp));
                for e in 0..n_exp {
                    // packed: `.experts.` puts these in the PAGED map `expert_f32` reads
                    add(format!("{p}experts.{e}.gate_up"), vec![d, 2 * mi], v(110 + e, d * 2 * mi));
                    add(format!("{p}experts.{e}.down"), vec![mi, d], v(130 + e, mi * d));
                }
            }
        }
        let mut cfg: Vec<i64> = vec![nl as i64, nh as i64, nkv as i64, nkv as i64, hd as i64, hd as i64,
                                    d as i64, ffn as i64, vocab as i64, if moe { 1 } else { 0 },
                                    window as i64, 0 /* ple */, moe as i64, n_exp as i64,
                                    topk as i64, mi as i64];
        for _ in 0..nl { cfg.push(0); } // no sliding layers
        let manifest = serde_json::json!({ "format": crate::bundle::FORMAT, "version": crate::bundle::VERSION,
            "arch": "gemma4", "config": cfg, "config_f": [10000.0, 1e-6, 1.0, 1.0], "arrays": specs });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
        Gemma4::new(Bundle::load(&stem).unwrap(), 0.0, false)
    }

    /// Naive re-walk: the residual entering layer `l`'s MoE branch (`x_pre`), with NO cache. `moe_branch`
    /// takes the residual BEFORE the layer's FFN norms, so this walks layers `0..l` whole and then stops
    /// right after layer l's attention write + `post_attention_layernorm`. An independent copy of the walk,
    /// so the cached scorer has something honest to be compared against.
    fn naive_x_pre(m: &Gemma4, ids: &[i64], l: usize) -> Array2<f32> {
        let seq = ids.len();
        let h = m.h;
        let pli = m.per_layer_inputs(ids, &m.b.rows_f32("embed", ids));
        let mut x = m.b.rows_f32("embed", ids);
        let attend = |j: usize, x: &mut Array2<f32>| {
            let p = format!("l{j}.");
            let hd = m.hd_of(j);
            let nkv = m.nkv_of(j);
            let rep = h / nkv;
            let a = m.norm(x, &format!("{p}input_layernorm"));
            let q = m.derive_q(j, &a, 0);
            let (k, v) = m.derive_kv(j, &a, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t());
                for i in 0..seq {
                    for j2 in 0..seq {
                        if j2 > i { scores[[i, j2]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            let o = m.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let n = m.norm(&o, &format!("{p}post_attention_layernorm"));
            *x = &*x + &n;
        };
        let ffn = |j: usize, x: &mut Array2<f32>| {
            let p = format!("l{j}.");
            let a2 = m.norm(x, &format!("{p}pre_feedforward_layernorm"));
            let gate = m.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = m.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) { *hv = gelu_tanh(*hv) * uv; }
            let mlp = m.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            let combined = if m.moe {
                let (mb, _) = m.moe_branch(j, x, None);
                let h1 = m.norm(&mlp, &format!("{p}post_feedforward_layernorm_1"));
                &h1 + &mb
            } else {
                mlp
            };
            let n = m.norm(&combined, &format!("{p}post_feedforward_layernorm"));
            *x = &*x + &n;
            if m.ple > 0 {
                let mut g = m.b.mm(x, &format!("{p}per_layer_input_gate"));
                g.mapv_inplace(gelu_tanh);
                g = &g * &pli[j];
                let proj = m.b.mm(&g, &format!("{p}per_layer_projection"));
                let pn = m.norm(&proj, &format!("{p}post_per_layer_input_norm"));
                *x = &*x + &pn;
            }
            *x *= m.layer_scalar[j];
        };
        for j in 0..l {
            attend(j, &mut x);
            ffn(j, &mut x);
        }
        attend(l, &mut x); // layer l's FFN must NOT run — x_pre is its input
        x
    }

    /// The routed probe rests on this: one cached scorer step over the KV state must equal a naive
    /// from-scratch forward over the same ids, bit for bit.
    #[test]
    fn gemma4_scorer_step_matches_naive_bitwise() {
        let m = build_tiny("step", true);
        let ids: Vec<i64> = vec![1, 9, 4, 7, 2, 5];
        let total = ids.len() + 6;
        for layer in [0usize, 2, 3] {
            let mut sc = Gemma4Scorer::new(&m, &ids, total).unwrap();
            assert_eq!(sc.pos, ids.len());
            for (i, &tok) in [7i64, 3, 11].iter().enumerate() {
                let (e, n) = ((i + layer) % m.n_exp, (i * 5) % 32);
                let got = sc.try_token_moe(&m, tok, layer, e, n).unwrap();
                let mut want_ids = ids.clone();
                want_ids.push(tok);
                let xp = naive_x_pre(&m, &want_ids, layer);
                let (_, want) = m.moe_branch(layer, &xp, Some((e, n)));
                let want = want[want.len() - 1];
                if want.is_nan() {
                    assert!(got.is_nan(), "layer {layer}: expert not selected, so both must be NaN");
                } else {
                    assert_eq!(got.to_bits(), want.to_bits(),
                        "layer {layer} tok {tok}: scorer {:e} != naive {:e}", got, want);
                }
            }
        }
    }

    /// Candidates must be INDEPENDENT: probing B after A must equal probing B on a fresh scorer.
    #[test]
    fn gemma4_candidates_are_independent() {
        let m = build_tiny("indep", true);
        let ids: Vec<i64> = vec![2, 4, 6];
        let (l, e, n) = (1usize, 2usize, 3usize);
        let mut sc = Gemma4Scorer::new(&m, &ids, 32).unwrap();
        let _ = sc.try_token_moe(&m, 11, l, e, n);
        let after_a = sc.try_token_moe(&m, 7, l, e, n).unwrap();
        let mut fresh = Gemma4Scorer::new(&m, &ids, 32).unwrap();
        let on_fresh = fresh.try_token_moe(&m, 7, l, e, n).unwrap();
        assert_eq!(after_a.to_bits(), on_fresh.to_bits(),
            "probing A then B differs from probing B alone: candidates leak into each other");
    }

    /// The routed objective must equal `w_router · gelu_tanh(gate_n) · up_n` written out by hand, with the
    /// FULL Gemma-4 router: weightless RMSNorm, per-channel `router.scale`, 1/√d, softmax, top-k, renorm,
    /// and the per-EXPERT `per_expert_scale` — which is part of the weight, not a separate term. GEGLU, not
    /// SwiGLU. Dropping any of those still "works", just wrongly, which is the point of this test.
    #[test]
    fn gemma4_routed_objective_matches_reference() {
        let m = build_tiny("ref", true);
        let (l, n) = (1usize, 7usize);
        // Long enough that a top-k of 2 out of 4 routes each expert somewhere; the guard below fails the test
        // rather than letting a never-routed expert make it vacuous.
        let stream: Vec<i64> = (0..16).map(|i: i64| i % 7).collect();
        let xp = naive_x_pre(&m, &stream, l);
        let p = format!("l{l}.");
        let d = m.d;
        let dscale = (d as f32).powf(-0.5);
        // the pre-norm'd input every expert consumes, exactly as the branch computes it
        let h2 = m.norm(&xp, &format!("{p}pre_feedforward_layernorm_2"));
        let mut n_checked = 0;
        for e in 0..m.n_exp {
            let (_out, got) = m.moe_branch(l, &xp, Some((e, n)));
            // rebuild this expert's scores from scratch
            let hn = m.rmsnorm(&xp, None);
            let rscale = m.b.arr1o(&format!("{p}router.scale"));
            let mut hs = hn;
            for mut row in hs.rows_mut() {
                for (i, val) in row.iter_mut().enumerate() { *val = *val * rscale[i] * dscale; }
            }
            let scores = m.b.mm(&hs, &format!("{p}router.proj"));
            let pes = m.b.arr1o(&format!("{p}router.per_expert_scale"));
            let mi = m.moe_inter;
            assert_eq!(got.len(), stream.len(), "one value per position");
            let mut n_routed = 0;
            for t in 0..stream.len() {
                let row = scores.row(t);
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
                let denom: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / denom).collect();
                let mut idx: Vec<usize> = (0..m.n_exp).collect();
                idx.sort_by(|&x, &y| probs[y].partial_cmp(&probs[x]).unwrap());
                idx.truncate(m.topk);
                if !idx.contains(&e) {
                    assert!(got[t].is_nan(), "e={e} t={t}: expert not selected, so the unit has NO value there");
                    continue;
                }
                n_routed += 1;
                let tksum: f32 = idx.iter().map(|&x| probs[x]).sum();
                let w = probs[e] / tksum * pes[e];
                // The activation is read through the SAME `expert_mm` the kernel uses. This test is
                // about the ROUTER — the weightless norm, the per-channel `router.scale`, the 1/√d,
                // softmax, top-k, renorm and the per-EXPERT `per_expert_scale` — which is the part that
                // differs between archs and the part that is silently wrong if you drop a term. The
                // packed (d, 2*mi) GEMM layout is not re-derived here: it is covered by the 60/60 torch
                // gate, and re-deriving it by hand just duplicates a convention.
                let rows = h2.slice(s![t..t + 1, ..]).to_owned();
                let gate_up = m.b.expert_mm(&rows, &format!("{p}experts.{e}.gate_up")); // (1, 2*mi)
                let act = gelu_tanh(gate_up[[0, n]]) * gate_up[[0, n + mi]]; // GEGLU, not SwiGLU
                let want = w * act;
                assert_eq!(got[t].to_bits(), want.to_bits(),
                    "e={e} t={t}: branch {:e} != reference {:e}", got[t], want);
            }
            if n_routed > 0 { n_checked += 1; }
        }
        assert!(n_checked > 0, "no expert was ever selected — the reference proves nothing");
    }

    /// Coherence: a MoE bundle must report `RoutedLayers` (the scorer IS implemented, so `NoScorer` would be
    /// a lie), and a DENSE bundle must report `NoScorer` — only the routed arm exists for this arch, so
    /// claiming `Supported` would promise a dense probe that does not exist.
    #[test]
    fn gemma4_blind_support_is_coherent() {
        assert_eq!(build_tiny("decl", true).blind_support(), BlindSupport::RoutedLayers);
        assert_eq!(build_tiny("decl2", false).blind_support(), BlindSupport::NoScorer);
    }
}
