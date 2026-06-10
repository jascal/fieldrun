//! Tier B — composition, DeepSeek-V4 (the new HCA/CSA attention family). A faithful Rust port of
//! `DeepseekV4ForCausalLM`. V4 is NOT MLA — it throws away the V3 latent-KV machinery and replaces it with a stack of
//! genuinely new mechanisms. This file implements **Stage 1: the sliding-only backbone** (every layer
//! `sliding_attention`, no compressor) — the hard novel parts:
//!   - **Shared-KV MQA**: one KV head; `kv_proj → head_dim`; the same tensor is read as both K and V.
//!   - **q-LoRA queries**: `q_a_proj → q_a_norm (weighted RMSNorm) → q_b_proj → per-head q_b_norm (UNWEIGHTED RMSNorm)`.
//!   - **Partial INTERLEAVED RoPE** on the trailing `rope_head_dim` of each head (adjacent-pair rotation, not half-split).
//!   - **Attention sink** (gpt-oss): one learnable logit per head appended to the softmax, then dropped from the output.
//!   - **Undo-RoPE on the output**: because K==V, the value carries RoPE on its rope slice; rotate the attention output
//!     by the conjugate (`-sin`) at the query position so each contribution depends only on the relative distance.
//!   - **Grouped low-rank o_proj**: block-diagonal `o_a_proj` (per group) → flatten → `o_b_proj`.
//!   - **mHC residual**: the residual is `hc_mult` parallel streams; two HyperConnection modules per layer collapse the
//!     streams in (a `pre`-weighted sum) and mix them out (a Sinkhorn-projected doubly-stochastic `comb` + a `post`
//!     placement gate). `HyperHead` collapses the streams before the final norm.
//!   - **MoE**: a `sqrtsoftplus` router (bias-corrected top-k SELECTION, raw-score WEIGHTS, renorm × routed_scaling) +
//!     gpt-oss SwiGLU clamps on routed AND shared experts; one always-on shared expert. (No group-limited routing, no
//!     first_k_dense.) Hash routing (the first few layers on a real checkpoint) is a later stage.
//! Validated top-1 against a tiny random-init `DeepseekV4ForCausalLM` (the faithfulness gate). KV-cache generate/explain
//! and the CSA/HCA compressors + Lightning Indexer are follow-on stages.

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Dsv4 {
    b: Bundle,
    nl: usize,
    nh: usize,      // attention heads
    hd: usize,      // head_dim
    q_lora: usize,  // q LoRA latent rank
    rd: usize,      // rope_head_dim (trailing dims of each head that rotate)
    d: usize,       // hidden_size
    n_exp: usize,
    top_k: usize,
    o_groups: usize,
    o_lora: usize,
    moe_inter: usize,
    hc: usize,         // hc_mult — parallel residual streams
    sinkhorn: usize,   // hc_sinkhorn_iters
    window: usize,     // sliding_window
    tied: bool,
    inv: Vec<f32>,     // rope inv_freq, rd/2 entries (the "main" theta)
    eps: f32,
    hc_eps: f32,
    limit: f32,        // swiglu clamp
    rscale: f32,       // routed_scaling_factor
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    // numerically-stable log(1+exp(x))
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}
fn sqrtsoftplus(x: f32) -> f32 {
    softplus(x).sqrt()
}

/// Weighted RMSNorm over each row (DeepSeek uses the weight directly — no (1+w) bake). `w.len() == x.ncols()`.
fn rmsnorm_w(x: &Array2<f32>, w: &[f32], eps: f32) -> Array2<f32> {
    let n = x.ncols();
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for (j, v) in row.iter_mut().enumerate() {
            *v = *v * r * w[j];
        }
    }
    out
}

/// Unweighted RMSNorm over each row (no affine). Used for q_b_norm (per head) and the HyperConnection input norm.
fn rmsnorm_u(x: &Array2<f32>, eps: f32) -> Array2<f32> {
    let n = x.ncols();
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for v in row.iter_mut() {
            *v *= r;
        }
    }
    out
}

impl Dsv4 {
    pub fn new(b: Bundle, _route: f32, _kv_int8: bool) -> Dsv4 {
        // config: [nl, nh, hd, q_lora, rd, d, n_exp, top_k, o_groups, o_lora, moe_inter, vocab, hc_mult, sinkhorn, window, tied]
        let c = &b.config;
        let (nl, nh, hd, q_lora, rd, d) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize, c[4] as usize, c[5] as usize);
        let (n_exp, top_k, o_groups, o_lora, moe_inter) = (c[6] as usize, c[7] as usize, c[8] as usize, c[9] as usize, c[10] as usize);
        let (hc, sinkhorn, window) = (c[12] as usize, c[13] as usize, c[14] as usize);
        let tied = c.len() > 15 && c[15] != 0;
        // config_f: [rope_theta, eps, swiglu_limit, routed_scaling, hc_eps]
        let (theta, eps, limit, rscale, hc_eps) =
            (b.config_f[0] as f32, b.config_f[1] as f32, b.config_f[2] as f32, b.config_f[3] as f32, b.config_f[4] as f32);
        let inv = (0..rd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / rd as f32)).collect();
        Dsv4 { b, nl, nh, hd, q_lora, rd, d, n_exp, top_k, o_groups, o_lora, moe_inter, hc, sinkhorn, window, tied, inv, eps, hc_eps, limit, rscale }
    }

    fn unembed(&self) -> &str {
        if self.tied { "embed" } else { "lm_head" }
    }

    /// cos/sin for every position, `rd/2` angles each: `cos[s][i] = cos(pos_s · inv_freq[i])`.
    fn rope_tables(&self, seq: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let half = self.rd / 2;
        let mut cos = vec![vec![0f32; half]; seq];
        let mut sin = vec![vec![0f32; half]; seq];
        for (s, (cr, sr)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
            for i in 0..half {
                let ang = s as f32 * self.inv[i];
                cr[i] = ang.cos();
                sr[i] = ang.sin();
            }
        }
        (cos, sin)
    }

    /// Rotate the trailing `rd` dims of one head vector in place: interleaved adjacent-pair rotation by the position's
    /// angle. `sign = +1.0` applies RoPE, `-1.0` applies the conjugate (the undo-RoPE on the attention output).
    fn rope_head(&self, v: &mut [f32], cos_s: &[f32], sin_s: &[f32], sign: f32) {
        let base = v.len() - self.rd;
        for i in 0..self.rd / 2 {
            let (c, sn) = (cos_s[i], sign * sin_s[i]);
            let (a, b) = (v[base + 2 * i], v[base + 2 * i + 1]);
            v[base + 2 * i] = a * c - b * sn;
            v[base + 2 * i + 1] = b * c + a * sn;
        }
    }

    /// The mHC HyperConnection: collapse `hc` streams into one sequence (returns `collapsed`), plus the `post` placement
    /// gate `[S, hc]` and the Sinkhorn-projected doubly-stochastic mixer `comb` `[S, hc, hc]` for the residual update.
    /// `streams[k]` is stream k, shape `[S, D]`.
    fn hyper_conn(&self, p: &str, streams: &[Array2<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, Array2<f32>) {
        let seq = streams[0].nrows();
        let (hc, d) = (self.hc, self.d);
        // flat[s, j*d + c] = streams[j][s,c], then unweighted-RMSNorm over the H*D row.
        let mut flat = Array2::<f32>::zeros((seq, hc * d));
        for s in 0..seq {
            for j in 0..hc {
                for c in 0..d {
                    flat[[s, j * d + c]] = streams[j][[s, c]];
                }
            }
        }
        let flat = rmsnorm_u(&flat, self.eps);
        let mix = self.b.mm(&flat, &format!("{p}.fn")); // [S, (2+hc)*hc]
        let base = self.b.arr1o(&format!("{p}.base")); // [(2+hc)*hc]
        let scale = self.b.arr1o(&format!("{p}.scale")); // [3]
        let (sc0, sc1, sc2) = (scale[0], scale[1], scale[2]);

        let mut post = vec![vec![0f32; hc]; seq];
        let mut comb = vec![vec![vec![0f32; hc]; hc]; seq];
        let mut pre = vec![vec![0f32; hc]; seq];
        for s in 0..seq {
            let row = mix.row(s);
            for k in 0..hc {
                pre[s][k] = sigmoid(row[k] * sc0 + base[k]) + self.hc_eps;
                post[s][k] = 2.0 * sigmoid(row[hc + k] * sc1 + base[hc + k]);
            }
            // comb_logits[a][b] = comb_w[a*hc+b] * sc2 + comb_b[a*hc+b]; softmax over b; + eps
            for a in 0..hc {
                let off = 2 * hc + a * hc;
                let mut mx = f32::NEG_INFINITY;
                for b in 0..hc {
                    let l = row[off + b] * sc2 + base[off + b];
                    comb[s][a][b] = l;
                    if l > mx { mx = l; }
                }
                let mut sum = 0.0;
                for b in 0..hc {
                    let e = (comb[s][a][b] - mx).exp();
                    comb[s][a][b] = e;
                    sum += e;
                }
                for b in 0..hc {
                    comb[s][a][b] = comb[s][a][b] / sum + self.hc_eps;
                }
            }
            // Sinkhorn: first a column-normalise (sum over the FIRST axis a), then iters-1 of (row, col).
            self.col_norm(&mut comb[s]);
            for _ in 0..self.sinkhorn.saturating_sub(1) {
                self.row_norm(&mut comb[s]);
                self.col_norm(&mut comb[s]);
            }
        }
        // collapsed[s,c] = sum_k pre[s,k] * streams[k][s,c]
        let mut collapsed = Array2::<f32>::zeros((seq, d));
        for s in 0..seq {
            for k in 0..hc {
                for c in 0..d {
                    collapsed[[s, c]] += pre[s][k] * streams[k][[s, c]];
                }
            }
        }
        (post, comb, collapsed)
    }

    fn row_norm(&self, m: &mut [Vec<f32>]) {
        let hc = self.hc;
        for a in 0..hc {
            let sum: f32 = m[a].iter().sum::<f32>() + self.hc_eps;
            for b in 0..hc { m[a][b] /= sum; }
        }
    }
    fn col_norm(&self, m: &mut [Vec<f32>]) {
        let hc = self.hc;
        for b in 0..hc {
            let mut sum = self.hc_eps;
            for a in 0..hc { sum += m[a][b]; }
            for a in 0..hc { m[a][b] /= sum; }
        }
    }

    /// Apply the mHC residual update in place: `streams[k] = post[:,k] ⊙ sub + Σ_j comb[:,j,k] · streams[j]`.
    fn hyper_update(&self, streams: &mut Vec<Array2<f32>>, sub: &Array2<f32>, post: &[Vec<f32>], comb: &[Vec<Vec<f32>>]) {
        let seq = sub.nrows();
        let (hc, d) = (self.hc, self.d);
        let old = streams.clone();
        for (k, stream) in streams.iter_mut().enumerate() {
            for s in 0..seq {
                for c in 0..d {
                    let mut v = post[s][k] * sub[[s, c]];
                    for j in 0..hc {
                        v += comb[s][j][k] * old[j][[s, c]];
                    }
                    stream[[s, c]] = v;
                }
            }
        }
    }

    /// The sliding-window self-attention block (q-LoRA, shared-KV MQA, sink, undo-RoPE, grouped o_proj). `a` is the
    /// input-layernorm'd collapsed hidden `[S, D]`; returns the attention output `[S, D]`.
    fn attention(&self, l: usize, a: &Array2<f32>, cos: &[Vec<f32>], sin: &[Vec<f32>]) -> Array2<f32> {
        let p = format!("l{l}.self_attn.");
        let seq = a.nrows();
        let (nh, hd) = (self.nh, self.hd);
        let scaling = (hd as f32).powf(-0.5);
        // q path: q_a → q_a_norm → q_b → per-head q_b_norm → rope
        let qr = rmsnorm_w(&self.b.mm(a, &format!("{p}q_a_proj")), &self.b.arr1(&format!("{p}q_a_norm")).to_vec(), self.eps);
        let qb = self.b.mm(&qr, &format!("{p}q_b_proj")); // [S, nh*hd]
        // q_b_norm is an unweighted RMSNorm over head_dim, per head; then partial interleaved rope.
        let mut q = qb.clone();
        for s in 0..seq {
            for h in 0..nh {
                let off = h * hd;
                let ms: f32 = (0..hd).map(|c| { let v = q[[s, off + c]]; v * v }).sum::<f32>() / hd as f32;
                let r = 1.0 / (ms + self.eps).sqrt();
                let mut head: Vec<f32> = (0..hd).map(|c| q[[s, off + c]] * r).collect();
                self.rope_head(&mut head, &cos[s], &sin[s], 1.0);
                for c in 0..hd { q[[s, off + c]] = head[c]; }
            }
        }
        // kv path (single head): kv_proj → kv_norm → rope
        let kv0 = rmsnorm_w(&self.b.mm(a, &format!("{p}kv_proj")), &self.b.arr1(&format!("{p}kv_norm")).to_vec(), self.eps);
        let mut kv = kv0.clone(); // [S, hd]
        for s in 0..seq {
            let mut head: Vec<f32> = (0..hd).map(|c| kv[[s, c]]).collect();
            self.rope_head(&mut head, &cos[s], &sin[s], 1.0);
            for c in 0..hd { kv[[s, c]] = head[c]; }
        }
        let sinks = self.b.arr1o(&format!("{p}sinks")); // [nh]

        // attention output, per head, K==V==kv, sliding-window causal, sink in the softmax.
        let mut attn = Array2::<f32>::zeros((seq, nh * hd));
        for hh in 0..nh {
            for s in 0..seq {
                let lo = (s + 1).saturating_sub(self.window); // valid j in [lo, s]
                let mut logits: Vec<f32> = Vec::with_capacity(s - lo + 2);
                for j in lo..=s {
                    let dot: f32 = (0..hd).map(|c| q[[s, hh * hd + c]] * kv[[j, c]]).sum();
                    logits.push(dot * scaling);
                }
                logits.push(sinks[hh]); // the sink logit
                let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0.0;
                for v in logits.iter_mut() { *v = (*v - mx).exp(); z += *v; }
                // weighted sum over j (the sink column is dropped from the output)
                for (idx, j) in (lo..=s).enumerate() {
                    let w = logits[idx] / z;
                    for c in 0..hd { attn[[s, hh * hd + c]] += w * kv[[j, c]]; }
                }
            }
        }
        // undo-rope on the output's rope slice (conjugate, query position)
        for s in 0..seq {
            for hh in 0..nh {
                let off = hh * hd;
                let mut head: Vec<f32> = (0..hd).map(|c| attn[[s, off + c]]).collect();
                self.rope_head(&mut head, &cos[s], &sin[s], -1.0);
                for c in 0..hd { attn[[s, off + c]] = head[c]; }
            }
        }
        // grouped o_proj: per group block-diagonal o_a, concat, then o_b.
        let gin = nh * hd / self.o_groups;
        let mut oa = Array2::<f32>::zeros((seq, self.o_groups * self.o_lora));
        for g in 0..self.o_groups {
            let slice = attn.slice(s![.., g * gin..(g + 1) * gin]).to_owned();
            let y = self.b.mm(&slice, &format!("{p}o_a_proj.{g}")); // [S, o_lora]
            oa.slice_mut(s![.., g * self.o_lora..(g + 1) * self.o_lora]).assign(&y);
        }
        self.b.mm(&oa, &format!("{p}o_b_proj"))
    }

    /// The SwiGLU shared expert (gpt-oss clamps): `down(silu(clamp(gate)) · clamp(up))`.
    fn shared_expert(&self, l: usize, m: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.mlp.shared_experts.");
        let mut gate = self.b.mm(m, &format!("{p}gate_proj"));
        let up = self.b.mm(m, &format!("{p}up_proj"));
        let mut hid = Array2::<f32>::zeros(gate.raw_dim());
        for ((g, u), h) in gate.iter_mut().zip(up.iter()).zip(hid.iter_mut()) {
            *h = silu(g.min(self.limit)) * u.clamp(-self.limit, self.limit);
        }
        self.b.mm(&hid, &format!("{p}down_proj"))
    }

    /// One routed expert (gpt-oss clamps) for a set of token rows: `down(silu(clamp(gate)) · clamp(up))`.
    fn routed_expert(&self, l: usize, e: usize, rows: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.experts.{e}.");
        let mut gate = self.b.expert_mm(rows, &format!("{p}gate"));
        let up = self.b.expert_mm(rows, &format!("{p}up"));
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g = silu(g.min(self.limit)) * u.clamp(-self.limit, self.limit);
        }
        self.b.expert_mm(&gate, &format!("{p}down"))
    }

    /// The sparse MoE block: sqrtsoftplus router (bias-corrected top-k selection, raw-score weights, renorm × scale) +
    /// the always-on shared expert.
    fn moe(&self, l: usize, m: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.mlp.");
        let seq = m.nrows();
        let logits = self.b.mm(m, &format!("{p}gate")); // [S, n_exp]
        let bias = self.b.arr1o(&format!("{p}e_score_correction_bias")); // [n_exp]
        // per-token top-k expert selection + weights
        let mut assign: std::collections::HashMap<usize, Vec<(usize, f32)>> = std::collections::HashMap::new();
        for s in 0..seq {
            let scores: Vec<f32> = (0..self.n_exp).map(|e| sqrtsoftplus(logits[[s, e]])).collect();
            let mut idx: Vec<usize> = (0..self.n_exp).collect();
            idx.sort_by(|&x, &y| (scores[y] + bias[y]).partial_cmp(&(scores[x] + bias[x])).unwrap());
            idx.truncate(self.top_k);
            let denom: f32 = idx.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
            for &e in &idx {
                let w = scores[e] / denom * self.rscale;
                assign.entry(e).or_default().push((s, w));
            }
        }
        let mut routed = Array2::<f32>::zeros((seq, self.d));
        for (e, toks) in &assign {
            let mut rows = Array2::<f32>::zeros((toks.len(), self.d));
            for (i, &(s, _)) in toks.iter().enumerate() { rows.row_mut(i).assign(&m.row(s)); }
            let out = self.routed_expert(l, *e, &rows);
            for (i, &(s, w)) in toks.iter().enumerate() {
                for c in 0..self.d { routed[[s, c]] += w * out[[i, c]]; }
            }
        }
        &routed + &self.shared_expert(l, m)
    }

    /// The full forward over `ids`, returning the final hidden `[S, D]` (post hc_head collapse + final norm).
    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let (cos, sin) = self.rope_tables(seq);
        let emb = self.b.rows_f32("embed", ids); // [S, D] — no embedding scale
        // residual = hc parallel streams, each a copy of the embedding
        let mut streams: Vec<Array2<f32>> = (0..self.hc).map(|_| emb.clone()).collect();
        for l in 0..self.nl {
            // attention site
            let (post, comb, collapsed) = self.hyper_conn(&format!("l{l}.attn_hc"), &streams);
            let a = rmsnorm_w(&collapsed, &self.b.arr1(&format!("l{l}.input_layernorm")).to_vec(), self.eps);
            let attn = self.attention(l, &a, &cos, &sin);
            self.hyper_update(&mut streams, &attn, &post, &comb);
            // ffn site
            let (post, comb, collapsed) = self.hyper_conn(&format!("l{l}.ffn_hc"), &streams);
            let mm = rmsnorm_w(&collapsed, &self.b.arr1(&format!("l{l}.post_attention_layernorm")).to_vec(), self.eps);
            let mlp = self.moe(l, &mm);
            self.hyper_update(&mut streams, &mlp, &post, &comb);
        }
        // hc_head: collapse the streams with a single `pre`-weighted sum, then the final weighted RMSNorm.
        let mut flat = Array2::<f32>::zeros((seq, self.hc * self.d));
        for s in 0..seq {
            for j in 0..self.hc {
                for c in 0..self.d { flat[[s, j * self.d + c]] = streams[j][[s, c]]; }
            }
        }
        let flat = rmsnorm_u(&flat, self.eps);
        let mix = self.b.mm(&flat, "hc_head.hc_fn"); // [S, hc]
        let hbase = self.b.arr1o("hc_head.hc_base");
        let hscale = self.b.arr1o("hc_head.hc_scale")[0];
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for s in 0..seq {
            for j in 0..self.hc {
                let pre = sigmoid(mix[[s, j]] * hscale + hbase[j]) + self.hc_eps;
                for c in 0..self.d { x[[s, c]] += pre * streams[j][[s, c]]; }
            }
        }
        rmsnorm_w(&x, &self.b.arr1("norm").to_vec(), self.eps)
    }
}

impl Model for Dsv4 {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32(self.unembed(), &last);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}
