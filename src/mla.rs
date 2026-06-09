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
//! instance. predict only (generate/explain TBD).

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
    pub fn new(b: Bundle, _route: f32, _kv_int8: bool) -> Mla {
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
              first_k, tied }
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
}

impl Model for Mla {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32(self.unembed(), &last);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}
