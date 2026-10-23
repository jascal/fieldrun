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
pub(crate) fn yarn_inv_freq(theta: f64, dim: usize, factor: f64, beta_fast: f64, beta_slow: f64,
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
    /// `scan = Some((expert, neuron))` additionally returns the per-row value of that ONE routed unit —
    /// `w_router · silu(gate_n) · up_n` — recorded in the same pass so the expert's weights are read once.
    /// Rows whose top-k does not contain `expert` stay `NaN`: the unit is not computed there at all, so it has
    /// no value rather than a value of zero (see `Model::blind_moe_*`). The always-on shared expert is NOT
    /// part of the value: it is a separate additive term, not part of the routed unit's own description.
    ///
    /// NB the DeepSeek router has three stages that must not be conflated: sigmoid SCORES, a bias that moves
    /// only the CHOICE (including a group score built from the top-2 biased scores per group), and a weight
    /// that is the *unbiased* sigmoid renormed over the top-k and scaled by `routed_scaling`.
    fn moe(&self, l: usize, a2: &Array2<f32>, scan: Option<(usize, usize)>) -> (Array2<f32>, Vec<f32>) {
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
            let gate = self.b.expert_mm(&rows, &format!("{p}experts.{e}.gate"));
            let up = self.b.expert_mm(&rows, &format!("{p}experts.{e}.up"));
            let mut hh = gate;
            for (h, u) in hh.iter_mut().zip(up.iter()) { *h = silu(*h) * u; }
            let down = self.b.expert_mm(&hh, &format!("{p}experts.{e}.down"));
            for (i, &(t, w)) in toks.iter().enumerate() {
                if let Some((se, sn)) = scan {
                    if *e == se && sn < hh.ncols() {
                        scan_out[t] = w * hh[[i, sn]];
                    }
                }
                for c in 0..self.d { out[[t, c]] += w * down[[i, c]]; }
            }
        }
        (out + self.swiglu_shared(l, a2), scan_out) // routed + shared
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
            let mlp = if l < self.first_k { self.dense_mlp(l, &a2) } else { self.moe(l, &a2, None).0 };
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
            let mlp = if l < self.first_k { self.dense_mlp(l, &a2) } else { self.moe(l, &a2, None).0 };
            x = &x + &mlp;
        }
        let xf = self.norm(&x, "norm");
        let x_last = x.row(seq - 1).to_vec(); // residual either side of the final norm — recovers its frozen scale
        let xf_last = xf.row(seq - 1).to_vec();
        let un = self.unembed();
        let lg = self.b.rowdot_f32(un, &xf_last);
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
            &[],
            &x_last,
            &xf_last,
            &u_pred,
            0,
            &|_v| Vec::new(),
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
            let mlp = if l < self.first_k { self.dense_mlp(l, &a2) } else { self.moe(l, &a2, None).0 };
            x = &x + &mlp;
        }
        self.norm(&x, "norm")
    }

    /// `forward_block` that also returns the per-row value of ONE routed unit at the target layer — the
    /// blind-max scoring kernel. The body below is a COPY of `forward_block` with exactly two changes (the
    /// scan plumbing and the return type). That is deliberate: MLA's walk assembles the latent K/V from
    /// q-LoRA plus a no-RoPE/shared-RoPE split, re-absorbs the value half and applies the YaRN ramp — far too
    /// much machinery to re-derive from memory, and a hand-re-written twin is exactly how the cached scorer
    /// and the forward would drift apart. The twin's bit-exactness against a naive per-position forward is
    /// asserted in this file's tests, so the copy cannot rot unnoticed.
    fn forward_block_scan(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>],
                          scan: Option<(usize, usize, usize)>) -> Vec<f32> {
        let (nh, qkh, qk_nope, qk_rope, vh) = (self.nh, self.qkh, self.qk_nope, self.qk_rope, self.v_head);
        let kpv_hd = qk_nope + vh;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let mut scan_out: Vec<f32> = Vec::new();
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
            let (mlp, vals) = if l < self.first_k {
                (self.dense_mlp(l, &a2), Vec::new())
            } else {
                let sc = scan.filter(|(sl, ..)| *sl == l).map(|(_, e, n)| (e, n));
                self.moe(l, &a2, sc)
            };
            if !vals.is_empty() { scan_out = vals; }
            x = &x + &mlp;
        }
        scan_out
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
            let mlp = if l < self.first_k { self.dense_mlp(l, &a2) } else { self.moe(l, &a2, None).0 };
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

/// The blind-max scorer for this arch: a KV-cache twin of [`Mla::forward_block_scan`] held at position `pos`,
/// with a snapshot so candidate tokens never see each other's writes. MLA's cache holds the ASSEMBLED per-head
/// K/V, so its two widths differ (K is the compressed latent head, V the value head) — allocation matches
/// `generate` exactly.
struct MlaScorer {
    kc: Vec<Array2<f32>>,
    vc: Vec<Array2<f32>>,
    snap_kc: Vec<Array2<f32>>,
    snap_vc: Vec<Array2<f32>>,
    pos: usize,
}

impl MlaScorer {
    fn alloc(m: &Mla, total: usize) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let (kdim, vdim) = (m.nh * m.qkh, m.nh * m.v_head);
        let kc: Vec<Array2<f32>> = (0..m.nl).map(|_| Array2::zeros((total, kdim))).collect();
        let vc: Vec<Array2<f32>> = (0..m.nl).map(|_| Array2::zeros((total, vdim))).collect();
        (kc, vc)
    }

    /// Prefill `prefix`, then score candidates at the next position.
    fn new(m: &Mla, prefix: &[i64], total: usize) -> Option<MlaScorer> {
        if prefix.is_empty() || prefix.len() >= total {
            return None;
        }
        let (mut kc, mut vc) = MlaScorer::alloc(m, total);
        let emb = m.b.rows_f32("embed", prefix);
        m.forward_block(&emb, 0, &mut kc, &mut vc);
        let snap_kc = kc.clone();
        let snap_vc = vc.clone();
        Some(MlaScorer { kc, vc, snap_kc, snap_vc, pos: prefix.len() })
    }

    /// An empty scorer at position 0 — no seed context, for the corpus scan, which places every token.
    fn empty(m: &Mla, total: usize) -> MlaScorer {
        let (kc, vc) = MlaScorer::alloc(m, total);
        let snap_kc = kc.clone();
        let snap_vc = vc.clone();
        MlaScorer { kc, vc, snap_kc, snap_vc, pos: 0 }
    }

    /// Score `token` at `pos` by the ROUTED unit's `w_router · act`, WITHOUT committing. `None` for a bad
    /// unit; `Some(NaN)` when the router did not select that expert here — no value, not a value of zero.
    fn try_token_moe(&mut self, m: &Mla, token: i64, layer: usize, expert: usize, neuron: usize) -> Option<f32> {
        // A DeepSeek-style model keeps its first `first_k` layers DENSE: there is no expert there at all,
        // so a routed unit on such a layer is not a weak unit, it is a nonexistent one.
        if layer >= m.nl || layer < m.first_k || expert >= m.n_routed { return None; }
        self.kc.clone_from(&self.snap_kc); // candidates must not see each other's writes
        self.vc.clone_from(&self.snap_vc);
        let emb = m.b.rows_f32("embed", &[token]);
        let vals = m.forward_block_scan(&emb, self.pos, &mut self.kc, &mut self.vc, Some((layer, expert, neuron)));
        vals.first().copied()
    }

    /// Advance the cache by `token` (the winner's value was already read by the probe).
    fn commit(&mut self, m: &Mla, token: i64) {
        let emb = m.b.rows_f32("embed", &[token]);
        m.forward_block(&emb, self.pos, &mut self.kc, &mut self.vc);
        self.pos += 1;
        self.snap_kc.clone_from(&self.kc);
        self.snap_vc.clone_from(&self.vc);
    }
}

impl Model for Mla {
    /// The scorer IS implemented here, so a routed DeepSeek bundle must not report `NoScorer` — it is
    /// `RoutedLayers`: the dense per-layer unit is declined, `--blind-max L,E,N` names a routed one.
    fn blind_support(&self) -> crate::model::BlindSupport {
        if self.n_routed != 0 && self.nl > self.first_k {
            crate::model::BlindSupport::RoutedLayers
        } else {
            crate::model::BlindSupport::NoScorer
        }
    }

    /// One ROUTED unit's self-description: expert `e` neuron `n`'s own output row (`experts.{e}.down`),
    /// scaled by the final-norm gain and projected through the unembed — the same construction
    /// `blind_proposals` uses for a dense `mlp.down_proj`, so the two are comparable.
    fn blind_moe_proposals(&self, layer: usize, expert: usize, neuron: usize, k: usize, polarity: f32) -> Option<Vec<i64>> {
        if layer >= self.nl || layer < self.first_k || self.n_routed == 0 || expert >= self.n_routed {
            return None;
        }
        // Expert weights are PAGED, not resident: `weight_row` cannot see them, so go through
        // `expert_f32`, which returns (shape, values) as (in, out) row-major. down is (moe_inter, d), so
        // neuron `n`'s output direction is one contiguous d-wide row of it.
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
        if cands.is_empty() || layer >= self.nl || layer < self.first_k || self.n_routed == 0 || expert >= self.n_routed {
            return None;
        }
        let mut sc = MlaScorer::new(self, prefix, prefix.len() + steps + 1)?;
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
    /// `expert`. Chunked exactly like the dense arm, advancing the same assembled-K/V cache.
    fn blind_moe_corpus_best(&self, layer: usize, expert: usize, neuron: usize, ids: &[i64],
                             polarity: f32) -> Option<crate::model::BlindArm> {
        if ids.len() < 2 || layer >= self.nl || layer < self.first_k || self.n_routed == 0 || expert >= self.n_routed {
            return None;
        }
        let mut sc = MlaScorer::empty(self, ids.len());
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
        Some((self.nl, self.nh))
    }

    /// Per-block DLA decomposition (the `--pil-dump` seam for the MLA / DeepSeek family — exotic attention).
    /// MLA's latent-compressed attention only changes how `attn_out` is computed; the residual write is still
    /// `o_proj(attn_out)` per layer plus the FFN (dense or routed-expert MoE), summing to the pre-final-norm
    /// residual. RMSNorm geometry (no center/bias), so contribution = `inv_rms · Σ_d gain_d write_b_d U_v_d`.
    /// Mirrors `hidden` exactly, recording each layer's attn and FFN write at the last position.
    fn residual_decomp(&self, ids: &[i64], toks: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        let seq = ids.len();
        let last = seq - 1;
        let (nh, qkh, qk_nope, qk_rope, vh) = (self.nh, self.qkh, self.qk_nope, self.qk_rope, self.v_head);
        let kpv_hd = qk_nope + vh;
        let mut x = self.b.rows_f32("embed", ids);
        let mut labels: Vec<String> = vec!["embed".into()];
        let mut writes: Vec<Vec<f32>> = vec![x.row(last).to_vec()];
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
            for tpos in 0..seq {
                for hh in 0..nh {
                    let base = hh * qkh + qk_nope;
                    self.rope_one(&mut q.as_slice_mut().unwrap()[tpos * nh * qkh + base..tpos * nh * qkh + base + qk_rope], tpos);
                }
                self.rope_one(&mut krot.as_slice_mut().unwrap()[tpos * qk_rope..(tpos + 1) * qk_rope], tpos);
            }
            let mut attn_out = Array2::<f32>::zeros((seq, nh * vh));
            for hh in 0..nh {
                let mut kh = Array2::<f32>::zeros((seq, qkh));
                let mut vhead = Array2::<f32>::zeros((seq, vh));
                for tpos in 0..seq {
                    for c in 0..qk_nope { kh[[tpos, c]] = kpv[[tpos, hh * kpv_hd + c]]; }
                    for c in 0..qk_rope { kh[[tpos, qk_nope + c]] = krot[[tpos, c]]; }
                    for c in 0..vh { vhead[[tpos, c]] = kpv[[tpos, hh * kpv_hd + qk_nope + c]]; }
                }
                let qh = q.slice(s![.., hh * qkh..(hh + 1) * qkh]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..seq {
                    for j in (i + 1)..seq { scores[[i, j]] = -1e30; }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., hh * vh..(hh + 1) * vh]).assign(&scores.dot(&vhead));
            }
            let aw = self.b.mm(&attn_out, &format!("{p}o_proj"));
            writes.push(aw.row(last).to_vec());
            labels.push(format!("L{l}.attn"));
            x = &x + &aw;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let mw = if l < self.first_k { self.dense_mlp(l, &a2) } else { self.moe(l, &a2, None).0 };
            writes.push(mw.row(last).to_vec());
            labels.push(format!("L{l}.{}", if l < self.first_k { "mlp" } else { "moe" }));
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
            return crate::model::prefix_generate_q(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, _ctx| self.head_argmax(xb));
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

    /// A tiny `mla` bundle written straight to disk: compressed-latent attention + `first_k = 2` DENSE
    /// layers, so the "a routed unit on a dense layer does not exist" guard is exercised, and group-limited
    /// sigmoid routing (2 groups of 4 experts, top-2 of the groups) with a choice bias. Each test gets its
    /// OWN path — `cargo test` runs these concurrently.
    fn build_tiny(tag: &str) -> Mla {
        let (nl, nh, d, vocab) = (4usize, 4, 32, 24);
        let (qkh, qk_nope, qk_rope, v_head) = (12usize, 8usize, 4usize, 8usize);
        let (inter, n_routed, n_group, topk, topk_group, first_k) = (16usize, 8, 2usize, 2usize, 2usize, 2usize);
        let kv_lora = 8usize; // also the q-LoRA rank; > 0 so the q_a/q_b path runs
        let dir = std::env::temp_dir().join(format!("fr_bm_mla_{tag}_{}", std::process::id()));
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
        let kpv_hd = qk_nope + v_head;
        for l in 0..nl {
            let p = format!("l{l}.");
            add(format!("{p}in_ln"), vec![d], v(10 + l, d));
            add(format!("{p}post_ln"), vec![d], v(20 + l, d));
            add(format!("{p}q_a"), vec![d, kv_lora], v(300 + l, d * kv_lora));
            add(format!("{p}q_a_ln"), vec![kv_lora], v(310 + l, kv_lora));
            add(format!("{p}q_b"), vec![kv_lora, nh * qkh], v(320 + l, kv_lora * nh * qkh));
            add(format!("{p}kv_a"), vec![d, kv_lora + qk_rope], v(340 + l, d * (kv_lora + qk_rope)));
            add(format!("{p}kv_a_ln"), vec![kv_lora], v(350 + l, kv_lora + qk_rope));
            add(format!("{p}kv_b"), vec![kv_lora, nh * kpv_hd], v(360 + l, kv_lora * nh * kpv_hd));
            add(format!("{p}o_proj"), vec![nh * v_head, d], v(370 + l, nh * v_head * d));
            if l < first_k {
                add(format!("{p}mlp.gate_proj"), vec![d, inter], v(70 + l, d * inter));
                add(format!("{p}mlp.up_proj"), vec![d, inter], v(80 + l, d * inter));
                add(format!("{p}mlp.down_proj"), vec![inter, d], v(90 + l, inter * d));
            } else {
                add(format!("{p}gate"), vec![d, n_routed], v(100 + l, d * n_routed));
                add(format!("{p}gate_bias"), vec![n_routed], v(400 + l, n_routed));
                for e in 0..n_routed {
                    // `.experts.` in the name is what puts these in the PAGED map `expert_f32` reads.
                    add(format!("{p}experts.{e}.gate"), vec![d, inter], v(110 + e, d * inter));
                    add(format!("{p}experts.{e}.up"), vec![d, inter], v(120 + e, d * inter));
                    add(format!("{p}experts.{e}.down"), vec![inter, d], v(130 + e, inter * d));
                }
                // gate/up project d -> inter; down projects inter -> d (the mm input widths must line up)
                add(format!("{p}shared.gate"), vec![d, inter], v(140 + l, d * inter));
                add(format!("{p}shared.up"), vec![d, inter], v(150 + l, d * inter));
                add(format!("{p}shared.down"), vec![inter, d], v(160 + l, inter * d));
            }
        }
        // [nl, nh, d, q_lora, kv_lora, qk_nope, qk_rope, v_head, vocab, tied,
        //  n_routed, n_shared, topk, moe_inter, n_group, topk_group, norm_topk, first_k, ffn_dense]
        let cfg: Vec<i64> = vec![nl as i64, nh as i64, d as i64, kv_lora as i64, kv_lora as i64,
                                qk_nope as i64, qk_rope as i64, v_head as i64, vocab as i64, 0 /* tied */,
                                n_routed as i64, 1 /* n_shared */, topk as i64, inter as i64,
                                n_group as i64, topk_group as i64, 1 /* norm_topk */, first_k as i64,
                                inter as i64 /* ffn_dense */];
        // config_f carries the float scalars: rope_theta, rms_eps, routed_scaling
        let manifest = serde_json::json!({ "format": crate::bundle::FORMAT, "version": crate::bundle::VERSION,
            "arch": "mla", "config": cfg, "config_f": [10000.0, 1e-5, 1.0], "arrays": specs });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
        Mla::new(Bundle::load(&stem).unwrap(), 0.0, false)
    }

    /// Naive re-walk: the post-attention-normed hidden entering layer `l`'s FFN, with NO cache. The
    /// attention body below is lifted from `forward_block` rather than re-derived from memory — MLA assembles
    /// the latent K/V from q-LoRA plus a no-RoPE/shared-RoPE split and re-absorbs the value half, and a
    /// hand-written twin is precisely how a reference could silently stop being a reference. What this
    /// helper DOES re-implement independently is the *caching*: it recomputes every position from scratch on
    /// every call, which is exactly what the cached scorer must agree with.
    fn naive_a2(m: &Mla, ids: &[i64], l: usize) -> Array2<f32> {
        let (nh, qkh, qk_nope, qk_rope, vh) = (m.nh, m.qkh, m.qk_nope, m.qk_rope, m.v_head);
        let kpv_hd = qk_nope + vh;
        let seq = ids.len();
        let klen = seq; // no cache: every position is recomputed
        let mut x = m.b.rows_f32("embed", ids);
        let mut kc: Vec<Array2<f32>> = (0..m.nl).map(|_| Array2::zeros((klen, nh * qkh))).collect();
        let mut vc: Vec<Array2<f32>> = (0..m.nl).map(|_| Array2::zeros((klen, nh * vh))).collect();
        for ll in 0..=l {
            let p = format!("l{ll}.");
            let a = m.norm(&x, &format!("{p}in_ln"));
            let mut q = if m.q_lora > 0 {
                let qa = m.norm(&m.b.mm(&a, &format!("{p}q_a")), &format!("{p}q_a_ln"));
                m.b.mm(&qa, &format!("{p}q_b"))
            } else {
                m.b.mm(&a, &format!("{p}q"))
            };
            let ckv = m.b.mm(&a, &format!("{p}kv_a"));
            let mut krot = ckv.slice(s![.., m.kv_lora..m.kv_lora + qk_rope]).to_owned();
            let klat = m.norm(&ckv.slice(s![.., 0..m.kv_lora]).to_owned(), &format!("{p}kv_a_ln"));
            let kpv = m.b.mm(&klat, &format!("{p}kv_b"));
            for i in 0..seq {
                let pos = i; // positions restart at 0: the caller passes the full prefix
                for hh in 0..nh {
                    let base = hh * qkh + qk_nope;
                    m.rope_one(&mut q.as_slice_mut().unwrap()[i * nh * qkh + base..i * nh * qkh + base + qk_rope], pos);
                }
                m.rope_one(&mut krot.as_slice_mut().unwrap()[i * qk_rope..(i + 1) * qk_rope], pos);
            }
            for i in 0..seq {
                let pos = i;
                for hh in 0..nh {
                    for c in 0..qk_nope { kc[ll][[pos, hh * qkh + c]] = kpv[[i, hh * kpv_hd + c]]; }
                    for c in 0..qk_rope { kc[ll][[pos, hh * qkh + qk_nope + c]] = krot[[i, c]]; }
                    for c in 0..vh { vc[ll][[pos, hh * vh + c]] = kpv[[i, hh * kpv_hd + qk_nope + c]]; }
                }
            }
            let mut attn_out = Array2::<f32>::zeros((seq, nh * vh));
            for hh in 0..nh {
                let qh = q.slice(s![.., hh * qkh..(hh + 1) * qkh]);
                let kh = kc[ll].slice(s![0..klen, hh * qkh..(hh + 1) * qkh]);
                let vhead = vc[ll].slice(s![0..klen, hh * vh..(hh + 1) * vh]);
                let mut scores = qh.dot(&kh.t()) * m.scale;
                for i in 0..seq {
                    let abs = i;
                    for j in (abs + 1)..klen { scores[[i, j]] = -1e30; }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., hh * vh..(hh + 1) * vh]).assign(&scores.dot(&vhead));
            }
            x = &x + &m.b.mm(&attn_out, &format!("{p}o_proj"));
            let a2 = m.norm(&x, &format!("{p}post_ln"));
            if ll == l {
                return a2; // layer l's FFN must NOT run — a2 is its input
            }
            let mlp = if ll < m.first_k { m.dense_mlp(ll, &a2) } else { m.moe(ll, &a2, None).0 };
            x = &x + &mlp;
        }
        unreachable!("l is within range")
    }

    /// The routed probe rests on this: one cached scorer step over the assembled-K/V state must equal a naive
    /// from-scratch forward over the same ids, bit for bit.
    #[test]
    fn mla_scorer_step_matches_naive_bitwise() {
        let m = build_tiny("step");
        let ids: Vec<i64> = vec![1, 9, 4, 7, 2, 5];
        let total = ids.len() + 6;
        for layer in [m.first_k, m.nl - 1] {
            let mut sc = MlaScorer::new(&m, &ids, total).unwrap();
            assert_eq!(sc.pos, ids.len());
            for (i, &tok) in [7i64, 3, 11].iter().enumerate() {
                let (e, n) = ((i * 3 + layer) % m.n_routed, (i * 5) % 16);
                let got = sc.try_token_moe(&m, tok, layer, e, n).unwrap();
                let mut want_ids = ids.clone();
                want_ids.push(tok);
                let a2 = naive_a2(&m, &want_ids, layer);
                let (_, want) = m.moe(layer, &a2, Some((e, n)));
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
    fn mla_candidates_are_independent() {
        let m = build_tiny("indep");
        let ids: Vec<i64> = vec![2, 4, 6];
        let (l, e, n) = (m.first_k, 2usize, 3usize);
        let mut sc = MlaScorer::new(&m, &ids, 32).unwrap();
        let _ = sc.try_token_moe(&m, 11, l, e, n);
        let after_a = sc.try_token_moe(&m, 7, l, e, n).unwrap();
        let mut fresh = MlaScorer::new(&m, &ids, 32).unwrap();
        let on_fresh = fresh.try_token_moe(&m, 7, l, e, n).unwrap();
        assert_eq!(after_a.to_bits(), on_fresh.to_bits(),
            "probing A then B differs from probing B alone: candidates leak into each other");
    }

    /// The routed objective must equal `w_router · silu(gate_n) · up_n` written out by hand, with the FULL
    /// three-stage DeepSeek router: sigmoid scores, a bias that moves only the CHOICE (and builds the group
    /// score from the top-2 BIASED scores per group), and a weight that is the UNBIASED sigmoid renormed over
    /// the top-k and scaled by `routed_scaling`. Conflating choice and weight still "works", just wrongly.
    #[test]
    fn mla_routed_objective_matches_reference() {
        let m = build_tiny("ref");
        let (l, n) = (m.first_k, 5usize);
        // Long enough that the top-2-of-2-groups routing actually selects each expert somewhere; the guard
        // below fails the test rather than letting a never-routed expert make it vacuous.
        let stream: Vec<i64> = (0..16).map(|i: i64| i % 7).collect();
        let a2 = naive_a2(&m, &stream, l);
        let p = format!("l{l}.");
        let logits = m.b.mm(&a2, &format!("{p}gate"));
        let bias = m.b.arr1o(&format!("{p}gate_bias"));
        let gsz = m.n_routed / m.n_group;
        let mut n_checked = 0;
        for e in 0..m.n_routed {
            let (_out, got) = m.moe(l, &a2, Some((e, n)));
            assert_eq!(got.len(), stream.len(), "one value per position");
            let mut n_routed = 0;
            for t in 0..stream.len() {
                let scores: Vec<f32> = logits.row(t).iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                let choice: Vec<f32> = scores.iter().zip(bias.iter()).map(|(s, b)| s + b).collect();
                let mut gscore: Vec<(usize, f32)> = (0..m.n_group).map(|g| {
                    let mut grp: Vec<f32> = choice[g * gsz..(g + 1) * gsz].to_vec();
                    grp.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    (g, grp.iter().take(2).sum())
                }).collect();
                gscore.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let keep: std::collections::HashSet<usize> =
                    gscore.iter().take(m.topk_group).map(|&(g, _)| g).collect();
                let mut cand: Vec<usize> = (0..m.n_routed).filter(|&x| keep.contains(&(x / gsz))).collect();
                cand.sort_by(|&a, &b| choice[b].partial_cmp(&choice[a]).unwrap());
                cand.truncate(m.topk);
                if !cand.contains(&e) {
                    assert!(got[t].is_nan(), "e={e} t={t}: expert not selected, so the unit has NO value there");
                    continue;
                }
                n_routed += 1;
                let denom: f32 = if m.norm_topk { cand.iter().map(|&x| scores[x]).sum::<f32>() + 1e-20 } else { 1.0 };
                let w = scores[e] / denom * m.routed_scaling;
                // activation read through the SAME expert_mm the kernel uses: what this pins is the ROUTER,
                // not the GEMM layout (covered by the 60/60 torch gate).
                let rows = a2.slice(s![t..t + 1, ..]).to_owned();
                let gate = m.b.expert_mm(&rows, &format!("{p}experts.{e}.gate"));
                let up = m.b.expert_mm(&rows, &format!("{p}experts.{e}.up"));
                let act = silu(gate[[0, n]]) * up[[0, n]];
                let want = w * act;
                assert_eq!(got[t].to_bits(), want.to_bits(),
                    "e={e} t={t}: branch {:e} != reference {:e}", got[t], want);
            }
            if n_routed > 0 { n_checked += 1; }
        }
        assert!(n_checked > 0, "no expert was ever selected — the reference proves nothing");
    }

    /// Coherence + the dense-layer guard: a routed bundle reports `RoutedLayers`, and a routed unit on one of
    /// the first `first_k` DENSE layers does not exist (there is no expert there at all).
    #[test]
    fn mla_blind_support_and_dense_layer_guard() {
        let m = build_tiny("decl");
        assert_eq!(m.blind_support(), BlindSupport::RoutedLayers);
        for e in [0usize, 3] {
            assert!(m.blind_moe_proposals(0, e, 1, 4, 1.0).is_none(),
                "layer 0 is dense (first_k = {}) — a routed unit there must not answer", m.first_k);
        }
        assert!(m.blind_moe_proposals(m.first_k, 0, 1, 4, 1.0).is_some(),
            "a genuinely routed layer must answer");
    }
}
