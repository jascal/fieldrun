//! Tier B — composition, DeepSeek-V3 / Kimi-K2: MLA (multi-head latent attention) + DeepSeek MoE.
//!
//! MLA compresses attention through low-rank latents: q goes d → q_lora → (nh · qk_head_dim) via q_a/q_b (with an
//! RMSNorm on the latent); kv goes d → (kv_lora + qk_rope) via kv_a, the kv_lora part is RMSNorm'd then expanded by
//! kv_b to per-head (qk_nope + v_head). Each head's key/query is [no-RoPE part (qk_nope) ‖ RoPE part (qk_rope)], where
//! the RoPE part of the key is a SINGLE shared vector (MQA-style) broadcast to all heads. v_head_dim ≠ qk_head_dim.
//! The MoE has a shared always-on expert plus group-limited sigmoid routing with a learned bias correction; the first
//! `first_k_dense_replace` layers are dense. Experts read on demand from the mmap (offload). YaRN long-context RoPE
//! scaling (the real DeepSeek-V3/R1/Kimi-K2 configs all ship it) is supported: the inv_freq ramp blend, the
//! mscale/mscale_all_dim attention factor on cos/sin, and the mscale² softmax-scale correction. Interleaved rotary
//! weights (`rope_interleave`, the DeepSeek default) are de-interleaved at convert time, so this kernel's split-half
//! rope is exact either way. A faithful port of `DeepseekV3ForCausalLM`, validated top-1 against a tiny random-init
//! instance. Incremental KV-cache `generate`/`generate_stream` (f32 + int8-KV) caches the *assembled* per-head K/V
//! (byte-identical to the naive recompute) and `explain` reads the live circuits + FFN features.

use std::collections::HashMap;

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Mla {
    b: Bundle,
    nl: usize,
    nh: usize,
    d: usize,
    q_lora: usize,
    kv_lora: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head: usize,
    qkh: usize, // qk_head_dim = qk_nope + qk_rope
    eps: f32,
    scale: f32,      // qk_head_dim^-0.5, ×mscale² under YaRN
    att_factor: f32, // YaRN attention factor on cos/sin (1.0 without YaRN)
    inv: Vec<f32>,
    n_routed: usize,
    n_group: usize,
    topk_group: usize,
    topk: usize,
    norm_topk: bool,
    routed_scaling: f32,
    first_k: usize,
    tied: bool,
    kv_int8: bool, // store the assembled per-head K/V cache as int8 with a per-head scale during generate
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softmax_rows(a: &mut Array2<f32>) {
    for mut row in a.rows_mut() {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in row.iter_mut() { *v = (*v - m).exp(); s += *v; }
        row.mapv_inplace(|v| v / s);
    }
}

/// YaRN's mscale helper (`yarn_get_mscale`): 1 below scale 1, else `0.1·m·ln(scale) + 1`.
fn yarn_mscale(scale: f64, m: f64) -> f64 {
    if scale <= 1.0 { 1.0 } else { 0.1 * m * scale.ln() + 1.0 }
}

/// YaRN inverse frequencies (transformers `_compute_yarn_parameters`): blend the interpolated (`1/(factor·θ^{2j/dim})`)
/// and extrapolated (`1/θ^{2j/dim}`) frequencies with a linear ramp between the beta_fast/beta_slow correction dims,
/// computed over the pre-scaling (`original_max_position_embeddings`) context.
fn yarn_inv_freq(theta: f64, dim: usize, factor: f64, beta_fast: f64, beta_slow: f64,
                 orig_max: f64, truncate: bool) -> Vec<f32> {
    let corr_dim = |rot: f64| (dim as f64 * (orig_max / (rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * theta.ln());
    let (mut low, mut high) = (corr_dim(beta_fast), corr_dim(beta_slow));
    if truncate { low = low.floor(); high = high.ceil(); }
    let (low, mut high) = (low.max(0.0), high.min(dim as f64 - 1.0));
    if high == low { high += 0.001; } // prevent singularity (as upstream)
    (0..dim / 2).map(|j| {
        let pf = theta.powf(2.0 * j as f64 / dim as f64);
        let extrap = 1.0 - ((j as f64 - low) / (high - low)).clamp(0.0, 1.0);
        ((1.0 / (factor * pf)) * (1.0 - extrap) + (1.0 / pf) * extrap) as f32
    }).collect()
}

impl Mla {
    pub fn new(b: Bundle, _route: f32, kv_int8: bool) -> Mla {
        // config: [nl, nh, d, q_lora, kv_lora, qk_nope, qk_rope, v_head, vocab, tied,
        //          n_routed, n_shared, topk, moe_inter, n_group, topk_group, norm_topk, first_k, ffn_dense]
        let c = &b.config;
        let (nl, nh, d) = (c[0] as usize, c[1] as usize, c[2] as usize);
        let (q_lora, kv_lora) = (c[3] as usize, c[4] as usize);
        let (qk_nope, qk_rope, v_head) = (c[5] as usize, c[6] as usize, c[7] as usize);
        let tied = c[9] != 0;
        let (n_routed, topk) = (c[10] as usize, c[12] as usize);
        let (n_group, topk_group, norm_topk, first_k) = (c[14] as usize, c[15] as usize, c[16] != 0, c[17] as usize);
        let qkh = qk_nope + qk_rope;
        // config_f: [theta, eps, routed_scaling] + optional YaRN block
        //           [yarn(0/1), factor, beta_fast, beta_slow, mscale, mscale_all_dim, original_max_pos, truncate(0/1),
        //            attention_factor (0 = derive from mscale/mscale_all_dim)]
        let cf = &b.config_f;
        let (theta, eps, routed_scaling) = (cf[0] as f32, cf[1] as f32, cf[2] as f32);
        let mut scale = (qkh as f32).powf(-0.5);
        let (inv, att_factor) = if cf.len() > 3 && cf[3] != 0.0 {
            let (factor, beta_fast, beta_slow) = (cf[4], cf[5], cf[6]);
            let (mscale, mscale_all_dim, orig_max) = (cf[7], cf[8], cf[9]);
            let (truncate, att_explicit) = (cf[10] != 0.0, cf[11]);
            let inv = yarn_inv_freq(cf[0], qk_rope, factor, beta_fast, beta_slow, orig_max, truncate);
            let att = if att_explicit != 0.0 { att_explicit }
                      else if mscale != 0.0 && mscale_all_dim != 0.0 {
                          yarn_mscale(factor, mscale) / yarn_mscale(factor, mscale_all_dim)
                      } else { yarn_mscale(factor, 1.0) };
            if mscale_all_dim != 0.0 {
                let m = yarn_mscale(factor, mscale_all_dim); // softmax-scale correction (DeepseekV3Attention)
                scale *= (m * m) as f32;
            }
            (inv, att as f32)
        } else {
            ((0..qk_rope / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / qk_rope as f32)).collect(), 1.0)
        };
        Mla { b, nl, nh, d, q_lora, kv_lora, qk_nope, qk_rope, v_head, qkh, eps,
              scale, att_factor, inv, n_routed, n_group, topk_group, topk, norm_topk, routed_scaling,
              first_k, tied, kv_int8 }
    }

    fn unembed(&self) -> &str { if self.tied { "embed" } else { "lm_head" } }

    fn norm(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let w = self.b.arr1o(name);
        let mut out = x.clone();
        for mut row in out.rows_mut() {
            let n = row.len() as f32;
            let ms = row.iter().map(|v| v * v).sum::<f32>() / n;
            let rinv = 1.0 / (ms + self.eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() { *v = *v * rinv * w[i]; }
        }
        out
    }

    /// split-half RoPE on a single qk_rope-length slice at position `pos`. The YaRN attention factor scales cos/sin
    /// upstream (i.e. the whole rotated slice); it is 1.0 without YaRN.
    fn rope_one(&self, slice: &mut [f32], pos: usize) {
        let half = self.qk_rope / 2;
        for j in 0..half {
            let ang = pos as f32 * self.inv[j];
            let (c, s) = (ang.cos(), ang.sin());
            let (a, b) = (slice[j], slice[j + half]);
            slice[j] = (a * c - b * s) * self.att_factor;
            slice[j + half] = (b * c + a * s) * self.att_factor;
        }
    }

    fn dense_mlp(&self, l: usize, a2: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.");
        let gate = self.b.mm(a2, &format!("{p}mlp.gate_proj"));
        let up = self.b.mm(a2, &format!("{p}mlp.up_proj"));
        let mut hh = gate;
        for (h, u) in hh.iter_mut().zip(up.iter()) { *h = silu(*h) * u; }
        self.b.mm(&hh, &format!("{p}mlp.down_proj"))
    }

    fn swiglu_shared(&self, l: usize, a2: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.");
        let gate = self.b.mm(a2, &format!("{p}shared.gate"));
        let up = self.b.mm(a2, &format!("{p}shared.up"));
        let mut hh = gate;
        for (h, u) in hh.iter_mut().zip(up.iter()) { *h = silu(*h) * u; }
        self.b.mm(&hh, &format!("{p}shared.down"))
    }

    /// DeepSeek MoE: group-limited sigmoid routing (with bias correction for the *choice*, sigmoid scores for the
    /// *weight*) over routed experts + an always-on shared expert. Experts paged from the mmap on demand.
    fn moe(&self, l: usize, a2: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.");
        let logits = self.b.mm(a2, &format!("{p}gate")); // (seq, n_routed)
        let bias = self.b.arr1o(&format!("{p}gate_bias"));
        let seq = a2.nrows();
        let gsz = self.n_routed / self.n_group;
        let mut assign: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for t in 0..seq {
            let scores: Vec<f32> = logits.row(t).iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(); // sigmoid
            let choice: Vec<f32> = scores.iter().zip(bias.iter()).map(|(s, b)| s + b).collect();
            // group scores = sum of top-2 choice scores in each group; keep the top `topk_group` groups
            let mut gscore: Vec<(usize, f32)> = (0..self.n_group).map(|g| {
                let mut grp: Vec<f32> = choice[g * gsz..(g + 1) * gsz].to_vec();
                grp.sort_by(|a, b| b.partial_cmp(a).unwrap());
                (g, grp.iter().take(2).sum())
            }).collect();
            gscore.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let keep: std::collections::HashSet<usize> = gscore.iter().take(self.topk_group).map(|&(g, _)| g).collect();
            // top-k experts among the kept groups, by choice score
            let mut cand: Vec<usize> = (0..self.n_routed).filter(|&e| keep.contains(&(e / gsz))).collect();
            cand.sort_by(|&a, &b| choice[b].partial_cmp(&choice[a]).unwrap());
            cand.truncate(self.topk);
            // weights = sigmoid score (NO bias), renormed over the top-k, scaled
            let denom: f32 = if self.norm_topk { cand.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20 } else { 1.0 };
            for &e in &cand {
                assign.entry(e).or_default().push((t, scores[e] / denom * self.routed_scaling));
            }
        }
        let mut out = Array2::<f32>::zeros((seq, self.d));
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
            let gate = self.b.expert_mm(&rows, &format!("{p}experts.{e}.gate"));
            let up = self.b.expert_mm(&rows, &format!("{p}experts.{e}.up"));
            let mut hh = gate;
            for (h, u) in hh.iter_mut().zip(up.iter()) { *h = silu(*h) * u; }
            let down = self.b.expert_mm(&hh, &format!("{p}experts.{e}.down"));
            for (i, &(t, w)) in toks.iter().enumerate() {
                for c in 0..self.d { out[[t, c]] += w * down[[i, c]]; }
            }
        }
        out + self.swiglu_shared(l, a2) // routed + shared
    }

    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let (nh, qkh, qk_nope, qk_rope, vh) = (self.nh, self.qkh, self.qk_nope, self.qk_rope, self.v_head);
        let kpv_hd = qk_nope + vh; // kv_b per-head width
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            // --- q: latent down/up (or direct), then RoPE on the rope sub-slice of each head ---
            let mut q = if self.q_lora > 0 {
                let qa = self.norm(&self.b.mm(&a, &format!("{p}q_a")), &format!("{p}q_a_ln"));
                self.b.mm(&qa, &format!("{p}q_b"))
            } else {
                self.b.mm(&a, &format!("{p}q"))
            }; // (seq, nh*qkh)
            // --- kv: compressed; split latent (kv_lora) + shared rope key (qk_rope) ---
            let ckv = self.b.mm(&a, &format!("{p}kv_a")); // (seq, kv_lora + qk_rope)
            let mut krot = ckv.slice(s![.., self.kv_lora..self.kv_lora + qk_rope]).to_owned(); // (seq, qk_rope) shared
            let klat = self.norm(&ckv.slice(s![.., 0..self.kv_lora]).to_owned(), &format!("{p}kv_a_ln"));
            let kpv = self.b.mm(&klat, &format!("{p}kv_b")); // (seq, nh*(qk_nope+v_head))
            // RoPE: q_rot per head, k_rot shared (per position)
            for t in 0..seq {
                for h in 0..nh {
                    let base = h * qkh + qk_nope;
                    self.rope_one(&mut q.as_slice_mut().unwrap()[t * nh * qkh + base..t * nh * qkh + base + qk_rope], t);
                }
                self.rope_one(&mut krot.as_slice_mut().unwrap()[t * qk_rope..(t + 1) * qk_rope], t);
            }
            // assemble per-head K (qk_nope from kpv ‖ shared krot) and V (v_head from kpv)
            let mut attn_out = Array2::<f32>::zeros((seq, nh * vh));
            for h in 0..nh {
                let mut kh = Array2::<f32>::zeros((seq, qkh));
                let mut vhead = Array2::<f32>::zeros((seq, vh));
                for t in 0..seq {
                    for c in 0..qk_nope { kh[[t, c]] = kpv[[t, h * kpv_hd + c]]; }
                    for c in 0..qk_rope { kh[[t, qk_nope + c]] = krot[[t, c]]; }
                    for c in 0..vh { vhead[[t, c]] = kpv[[t, h * kpv_hd + qk_nope + c]]; }
                }
                let qh = q.slice(s![.., h * qkh..(h + 1) * qkh]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..seq {
                    for j in (i + 1)..seq { scores[[i, j]] = -1e30; } // causal
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., h * vh..(h + 1) * vh]).assign(&scores.dot(&vhead));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}o_proj"));

            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let mlp = if l < self.first_k { self.dense_mlp(l, &a2) } else { self.moe(l, &a2) };
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    fn head_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let logits = self.b.rowdot_f32(self.unembed(), &xfn.row(xfn.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    /// SwiGLU hidden of a layer's FFN feature source for explain: the dense MLP on a dense layer, else the always-on
    /// shared expert (both are dense arrays, so `weight_row` names the neuron). `row` is the (1, d) post-LN hidden.
    fn feat_hidden(&self, l: usize, row: &Array2<f32>) -> Vec<f32> {
        let p = format!("l{l}.");
        let (g, u) = if l < self.first_k {
            (format!("{p}mlp.gate_proj"), format!("{p}mlp.up_proj"))
        } else {
            (format!("{p}shared.gate"), format!("{p}shared.up"))
        };
        let gate = self.b.mm(row, &g);
        let up = self.b.mm(row, &u);
        let mut hh = gate;
        for (h, uu) in hh.iter_mut().zip(up.iter()) { *h = silu(*h) * uu; }
        hh.row(0).to_vec()
    }

    fn explanation(&self, ids: &[i64]) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let (nh, qkh, qk_nope, qk_rope, vh) = (self.nh, self.qkh, self.qk_nope, self.qk_rope, self.v_head);
        let kpv_hd = qk_nope + vh;
        let mut x = self.b.rows_f32("embed", ids);
        let mut att_last: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut head_act: Vec<Vec<f32>> = Vec::new(); // attn_out's last row (nh*vh) — for head direct-logit attribution
        let mut mlp_h: Vec<Vec<f32>> = Vec::new();
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = if self.q_lora > 0 {
                let qa = self.norm(&self.b.mm(&a, &format!("{p}q_a")), &format!("{p}q_a_ln"));
                self.b.mm(&qa, &format!("{p}q_b"))
            } else {
                self.b.mm(&a, &format!("{p}q"))
            };
            let ckv = self.b.mm(&a, &format!("{p}kv_a"));
            let mut krot = ckv.slice(s![.., self.kv_lora..self.kv_lora + qk_rope]).to_owned();
            let klat = self.norm(&ckv.slice(s![.., 0..self.kv_lora]).to_owned(), &format!("{p}kv_a_ln"));
            let kpv = self.b.mm(&klat, &format!("{p}kv_b"));
            for t in 0..seq {
                for hh in 0..nh {
                    let base = hh * qkh + qk_nope;
                    self.rope_one(&mut q.as_slice_mut().unwrap()[t * nh * qkh + base..t * nh * qkh + base + qk_rope], t);
                }
                self.rope_one(&mut krot.as_slice_mut().unwrap()[t * qk_rope..(t + 1) * qk_rope], t);
            }
            let mut attn_out = Array2::<f32>::zeros((seq, nh * vh));
            let mut layer_att = Vec::with_capacity(nh);
            for hh in 0..nh {
                let mut kh = Array2::<f32>::zeros((seq, qkh));
                let mut vhead = Array2::<f32>::zeros((seq, vh));
                for t in 0..seq {
                    for c in 0..qk_nope { kh[[t, c]] = kpv[[t, hh * kpv_hd + c]]; }
                    for c in 0..qk_rope { kh[[t, qk_nope + c]] = krot[[t, c]]; }
                    for c in 0..vh { vhead[[t, c]] = kpv[[t, hh * kpv_hd + qk_nope + c]]; }
                }
                let qh = q.slice(s![.., hh * qkh..(hh + 1) * qkh]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..seq {
                    for j in (i + 1)..seq { scores[[i, j]] = -1e30; }
                }
                softmax_rows(&mut scores);
                layer_att.push(scores.row(seq - 1).to_vec());
                attn_out.slice_mut(s![.., hh * vh..(hh + 1) * vh]).assign(&scores.dot(&vhead));
            }
            att_last.push(layer_att);
            head_act.push(attn_out.row(seq - 1).to_vec());
            x = &x + &self.b.mm(&attn_out, &format!("{p}o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let last = a2.slice(s![seq - 1..seq, ..]).to_owned();
            mlp_h.push(self.feat_hidden(l, &last));
            let mlp = if l < self.first_k { self.dense_mlp(l, &a2) } else { self.moe(l, &a2) };
            x = &x + &mlp;
        }
        let xf = self.norm(&x, "norm");
        let un = self.unembed();
        let lg = self.b.rowdot_f32(un, &xf.row(seq - 1).to_vec());
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let gain = self.b.arr1("norm").to_vec();
        let (first_k, nl) = (self.first_k, self.nl);
        let _ = nl;
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
            &u_pred,
            |l, n| {
                let name = if l < first_k { format!("l{l}.mlp.down_proj") } else { format!("l{l}.shared.down") };
                self.b.weight_row(&name, n)
            },
            |l, head| head_raw_contrib(&self.b, &format!("l{l}.o_proj"), &head_act[l], head, vh),
            |c| self.b.rowdot_f32(un, c),
        )
    }

    /// Run `m` new positions through the layers, caching the *assembled* per-head K (nh*qkh, = [kv_b no-RoPE ‖ shared
    /// RoPE key]) and V (nh*vh, the kv_b value part) and attending over the whole cache. The MLA latent compression is
    /// recomputed for the new rows only; the cache holds the expanded per-head K/V so attention is byte-identical to the
    /// naive recompute. cur = absolute position of the first new row.
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let (nh, qkh, qk_nope, qk_rope, vh) = (self.nh, self.qkh, self.qk_nope, self.qk_rope, self.v_head);
        let kpv_hd = qk_nope + vh;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = if self.q_lora > 0 {
                let qa = self.norm(&self.b.mm(&a, &format!("{p}q_a")), &format!("{p}q_a_ln"));
                self.b.mm(&qa, &format!("{p}q_b"))
            } else {
                self.b.mm(&a, &format!("{p}q"))
            };
            let ckv = self.b.mm(&a, &format!("{p}kv_a"));
            let mut krot = ckv.slice(s![.., self.kv_lora..self.kv_lora + qk_rope]).to_owned();
            let klat = self.norm(&ckv.slice(s![.., 0..self.kv_lora]).to_owned(), &format!("{p}kv_a_ln"));
            let kpv = self.b.mm(&klat, &format!("{p}kv_b"));
            for i in 0..m {
                let pos = cur + i;
                for hh in 0..nh {
                    let base = hh * qkh + qk_nope;
                    self.rope_one(&mut q.as_slice_mut().unwrap()[i * nh * qkh + base..i * nh * qkh + base + qk_rope], pos);
                }
                self.rope_one(&mut krot.as_slice_mut().unwrap()[i * qk_rope..(i + 1) * qk_rope], pos);
            }
            for i in 0..m {
                let pos = cur + i;
                for hh in 0..nh {
                    for c in 0..qk_nope { kc[l][[pos, hh * qkh + c]] = kpv[[i, hh * kpv_hd + c]]; }
                    for c in 0..qk_rope { kc[l][[pos, hh * qkh + qk_nope + c]] = krot[[i, c]]; }
                    for c in 0..vh { vc[l][[pos, hh * vh + c]] = kpv[[i, hh * kpv_hd + qk_nope + c]]; }
                }
            }
            let mut attn_out = Array2::<f32>::zeros((m, nh * vh));
            for hh in 0..nh {
                let qh = q.slice(s![.., hh * qkh..(hh + 1) * qkh]);
                let kh = kc[l].slice(s![0..klen, hh * qkh..(hh + 1) * qkh]);
                let vhead = vc[l].slice(s![0..klen, hh * vh..(hh + 1) * vh]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..m {
                    let abs = cur + i;
                    for j in (abs + 1)..klen { scores[[i, j]] = -1e30; }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., hh * vh..(hh + 1) * vh]).assign(&scores.dot(&vhead));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let mlp = if l < self.first_k { self.dense_mlp(l, &a2) } else { self.moe(l, &a2) };
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    /// `forward_block` with an int8 KV cache: the assembled per-head K (qkh) and V (vh) are quantised per head with a
    /// per-head scale. ~4x smaller cache; per-head quant keeps tokens ~identical.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_q(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>],
                       vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]) -> Array2<f32> {
        let (nh, qkh, qk_nope, qk_rope, vh) = (self.nh, self.qkh, self.qk_nope, self.qk_rope, self.v_head);
        let kpv_hd = qk_nope + vh;
        let (kdim, vdim) = (nh * qkh, nh * vh);
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let q8 = |v: f32, sc: f32| (v / sc).round().clamp(-127.0, 127.0) as i8;
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = if self.q_lora > 0 {
                let qa = self.norm(&self.b.mm(&a, &format!("{p}q_a")), &format!("{p}q_a_ln"));
                self.b.mm(&qa, &format!("{p}q_b"))
            } else {
                self.b.mm(&a, &format!("{p}q"))
            };
            let ckv = self.b.mm(&a, &format!("{p}kv_a"));
            let mut krot = ckv.slice(s![.., self.kv_lora..self.kv_lora + qk_rope]).to_owned();
            let klat = self.norm(&ckv.slice(s![.., 0..self.kv_lora]).to_owned(), &format!("{p}kv_a_ln"));
            let kpv = self.b.mm(&klat, &format!("{p}kv_b"));
            for i in 0..m {
                let pos = cur + i;
                for hh in 0..nh {
                    let base = hh * qkh + qk_nope;
                    self.rope_one(&mut q.as_slice_mut().unwrap()[i * nh * qkh + base..i * nh * qkh + base + qk_rope], pos);
                }
                self.rope_one(&mut krot.as_slice_mut().unwrap()[i * qk_rope..(i + 1) * qk_rope], pos);
            }
            // assemble per-head K/V into f32 then quantise per head (one scale per head over its qkh / vh values)
            for i in 0..m {
                let pos = cur + i;
                for hh in 0..nh {
                    let mut krow = vec![0f32; qkh];
                    for c in 0..qk_nope { krow[c] = kpv[[i, hh * kpv_hd + c]]; }
                    for c in 0..qk_rope { krow[qk_nope + c] = krot[[i, c]]; }
                    let sck = (krow.iter().fold(0f32, |mx, &v| mx.max(v.abs())) / 127.0).max(1e-8);
                    ks[l][pos * nh + hh] = sck;
                    for c in 0..qkh { kc[l][pos * kdim + hh * qkh + c] = q8(krow[c], sck); }
                    let scv = ((0..vh).fold(0f32, |mx, c| mx.max(kpv[[i, hh * kpv_hd + qk_nope + c]].abs())) / 127.0).max(1e-8);
                    vs[l][pos * nh + hh] = scv;
                    for c in 0..vh { vc[l][pos * vdim + hh * vh + c] = q8(kpv[[i, hh * kpv_hd + qk_nope + c]], scv); }
                }
            }
            let mut attn_out = Array2::<f32>::zeros((m, nh * vh));
            for hh in 0..nh {
                let mut kh = Array2::<f32>::zeros((klen, qkh));
                let mut vhead = Array2::<f32>::zeros((klen, vh));
                for pos in 0..klen {
                    let sck = ks[l][pos * nh + hh];
                    let scv = vs[l][pos * nh + hh];
                    for c in 0..qkh { kh[[pos, c]] = kc[l][pos * kdim + hh * qkh + c] as f32 * sck; }
                    for c in 0..vh { vhead[[pos, c]] = vc[l][pos * vdim + hh * vh + c] as f32 * scv; }
                }
                let qh = q.slice(s![.., hh * qkh..(hh + 1) * qkh]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..m {
                    let abs = cur + i;
                    for j in (abs + 1)..klen { scores[[i, j]] = -1e30; }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., hh * vh..(hh + 1) * vh]).assign(&scores.dot(&vhead));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let mlp = if l < self.first_k { self.dense_mlp(l, &a2) } else { self.moe(l, &a2) };
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    fn generate_kv_int8(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let total = prompt.len() + n_new;
        let (kdim, vdim) = (self.nh * self.qkh, self.nh * self.v_head);
        let mut kc: Vec<Vec<i8>> = (0..self.nl).map(|_| vec![0i8; total * kdim]).collect();
        let mut vc: Vec<Vec<i8>> = (0..self.nl).map(|_| vec![0i8; total * vdim]).collect();
        let mut ks: Vec<Vec<f32>> = (0..self.nl).map(|_| vec![0f32; total * self.nh]).collect();
        let mut vs = ks.clone();
        let emb = self.b.rows_f32("embed", prompt);
        let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }
}

impl Model for Mla {
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
        let (kdim, vdim) = (self.nh * self.qkh, self.nh * self.v_head);
        let mut kc: Vec<Array2<f32>> = (0..self.nl).map(|_| Array2::zeros((total, kdim))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.nl).map(|_| Array2::zeros((total, vdim))).collect();
        let emb = self.b.rows_f32("embed", prompt);
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }

    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let total = prompt.len() + max_tokens;
        let (kdim, vdim) = (self.nh * self.qkh, self.nh * self.v_head);
        if self.kv_int8 {
            let mut kc: Vec<Vec<i8>> = (0..self.nl).map(|_| vec![0i8; total * kdim]).collect();
            let mut vc: Vec<Vec<i8>> = (0..self.nl).map(|_| vec![0i8; total * vdim]).collect();
            let mut ks: Vec<Vec<f32>> = (0..self.nl).map(|_| vec![0f32; total * self.nh]).collect();
            let mut vs = ks.clone();
            let emb = self.b.rows_f32("embed", prompt);
            let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
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
                let e = self.b.rows_f32("embed", &[next]);
                let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
                next = self.head_argmax(&xb);
                pos += 1;
            }
            return out;
        }
        let mut kc: Vec<Array2<f32>> = (0..self.nl).map(|_| Array2::zeros((total, kdim))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.nl).map(|_| Array2::zeros((total, vdim))).collect();
        let emb = self.b.rows_f32("embed", prompt);
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
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
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
        out
    }

    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool, cache: &mut crate::model::PrefixKv) -> Vec<i64> {
        if self.kv_int8 {
            let (kdim, vdim, nh, n_layer) = (self.nh * self.qkh, self.nh * self.v_head, self.nh, self.nl);
            let alloc = |total: usize| {
                let kc: Vec<Vec<i8>> = (0..n_layer).map(|_| vec![0i8; total * kdim]).collect();
                let vc: Vec<Vec<i8>> = (0..n_layer).map(|_| vec![0i8; total * vdim]).collect();
                let ks: Vec<Vec<f32>> = (0..n_layer).map(|_| vec![0f32; total * nh]).collect();
                let vs = ks.clone();
                (kc, vc, ks, vs)
            };
            let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>], vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]| {
                let emb = self.b.rows_f32("embed", ids);
                self.forward_block_q(&emb, cur, kc, ks, vc, vs)
            };
            return crate::model::prefix_generate_q(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb| self.head_argmax(xb));
        }
        let (kdim, vdim, n_layer) = (self.nh * self.qkh, self.nh * self.v_head, self.nl);
        let alloc = |total: usize| {
            let kc: Vec<Array2<f32>> = (0..n_layer).map(|_| Array2::zeros((total, kdim))).collect();
            let vc: Vec<Array2<f32>> = (0..n_layer).map(|_| Array2::zeros((total, vdim))).collect();
            (kc, vc)
        };
        let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]| {
            let emb = self.b.rows_f32("embed", ids);
            self.forward_block(&emb, cur, kc, vc)
        };
        crate::model::prefix_generate(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb| self.head_argmax(xb))
    }
}
