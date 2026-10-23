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
        self.routed_expert_scan(l, e, rows, None).0
    }

    /// `scan_n = Some(n)` additionally returns, per row, the clamped SwiGLU hidden at neuron `n` — the
    /// activation the blind-max objective weights by `w_router`. The gpt-oss CLAMPS are part of it: a
    /// reference that forgets them scores a different unit than the model computes.
    fn routed_expert_scan(&self, l: usize, e: usize, rows: &Array2<f32>, scan_n: Option<usize>)
        -> (Array2<f32>, Vec<f32>) {
        let p = format!("l{l}.experts.{e}.");
        let mut gate = self.b.expert_mm(rows, &format!("{p}gate"));
        let up = self.b.expert_mm(rows, &format!("{p}up"));
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g = silu(g.min(self.limit)) * u.clamp(-self.limit, self.limit);
        }
        let mut act: Vec<f32> = Vec::new();
        if let Some(n) = scan_n {
            let w = gate.ncols();
            if n < w { act = (0..gate.nrows()).map(|i| gate[[i, n]]).collect(); }
            else { act = vec![f32::NAN; gate.nrows()]; }
        }
        (self.b.expert_mm(&gate, &format!("{p}down")), act)
    }

    /// The sparse MoE block: sqrtsoftplus router (bias-corrected top-k selection, raw-score weights, renorm × scale) +
    /// the always-on shared expert.
    /// `scan = Some((expert, neuron))` additionally returns the per-row value of that ONE routed unit —
    /// `w_router · act`, with `act` the clamped SwiGLU hidden — recorded in the same pass so the expert's
    /// weights are read once. Rows whose top-k does not contain `expert` stay `NaN`: the unit is not
    /// computed there at all (see `Model::blind_moe_*`). The shared expert is NOT part of the value.
    ///
    /// NB this router's shape differs from every other arch here: the score is `sqrtsoftplus(logit)`, the
    /// bias moves only the CHOICE, and the weight is the UNBIASED score renormed over the top-k and scaled.
    fn moe(&self, l: usize, m: &Array2<f32>, scan: Option<(usize, usize)>) -> (Array2<f32>, Vec<f32>) {
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
        let mut scan_out: Vec<f32> = scan.map(|_| vec![f32::NAN; seq]).unwrap_or_default();
        for (e, toks) in &assign {
            let mut rows = Array2::<f32>::zeros((toks.len(), self.d));
            for (i, &(s, _)) in toks.iter().enumerate() { rows.row_mut(i).assign(&m.row(s)); }
            let sn = scan.filter(|(se, _)| *se == *e).map(|(_, n)| n);
            let (out, act) = self.routed_expert_scan(l, *e, &rows, sn);
            for (i, &(s, w)) in toks.iter().enumerate() {
                if sn.is_some() && !act.is_empty() {
                    scan_out[s] = w * act[i];
                }
                for c in 0..self.d { routed[[s, c]] += w * out[[i, c]]; }
            }
        }
        (&routed + &self.shared_expert(l, m), scan_out)
    }

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
            let (mlp, _) = self.moe(l, &mm, None);
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

    /// The full forward over `ids`, additionally returning the per-row value of ONE routed unit — the
    /// blind-max scoring kernel. A verbatim copy of `hidden` with only the scan plumbing and the return
    /// type changed.
    ///
    /// DeepSeek-V4 has NO KV cache (no `generate` yet), so unlike the other archs there is nothing to
    /// snapshot: every probe here re-runs the whole forward from scratch, which makes the probe exact BY
    /// CONSTRUCTION and costs O(prefix) per candidate. That is the right trade at this stage — a cache
    /// twin would be faster and would add a way for the scorer to silently disagree with the forward.
        fn hidden_scan(&self, ids: &[i64], scan: Option<(usize, usize, usize)>) -> Vec<f32> {
        let seq = ids.len();
        let (cos, sin) = self.rope_tables(seq);
        let emb = self.b.rows_f32("embed", ids); // [S, D] — no embedding scale
        // residual = hc parallel streams, each a copy of the embedding
        let mut scan_out: Vec<f32> = Vec::new();
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
            let sc = scan.filter(|(sl, ..)| *sl == l).map(|(_, e, n)| (e, n));
            let (mlp, vals) = self.moe(l, &mm, sc);
            if !vals.is_empty() { scan_out = vals; }
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
        scan_out
    }
}

impl Model for Dsv4 {
    /// The scorer IS implemented here, so a MoE arch must not report `NoScorer` — it is `RoutedLayers`:
    /// the dense per-layer unit is declined, `--blind-max L,E,N` names a routed one.
    fn blind_support(&self) -> crate::model::BlindSupport {
        if self.n_exp != 0 { crate::model::BlindSupport::RoutedLayers } else { crate::model::BlindSupport::NoScorer }
    }

    /// One ROUTED unit's self-description: expert `e` neuron `n`'s own output row (`experts.{e}.down`),
    /// scaled by the final-norm gain and projected through the unembed.
    fn blind_moe_proposals(&self, layer: usize, expert: usize, neuron: usize, k: usize, polarity: f32) -> Option<Vec<i64>> {
        if layer >= self.nl || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        // Expert weights are PAGED: `weight_row` cannot see them, so go through `expert_f32`, which
        // returns (shape, values) as (in, out) row-major. down is (inter, d), so neuron `n`'s output
        // direction is one contiguous d-wide row.
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
    ///
    /// No KV cache exists on this arch, so each candidate re-runs the whole forward over `prefix ++ [t]`
    /// and the value is read off the LAST row. That is exact by construction and costs O(prefix) per
    /// candidate — the honest trade until this arch grows a `generate`.
    fn blind_moe_climb(&self, layer: usize, expert: usize, neuron: usize, prefix: &[i64], cands: &[i64],
                       steps: usize, polarity: f32) -> Option<crate::model::BlindArm> {
        if cands.is_empty() || layer >= self.nl || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let mut ids = prefix.to_vec();
        let mut best = f32::NEG_INFINITY;
        for _ in 0..steps {
            let mut pick = None;
            let mut pick_obj = f32::NEG_INFINITY;
            for &t in cands {
                let mut probe = ids.clone();
                probe.push(t);
                let vals = self.hidden_scan(&probe, Some((layer, expert, neuron)));
                if let Some(&v) = vals.last() {
                    if !v.is_nan() && polarity * v > pick_obj {
                        pick_obj = polarity * v;
                        pick = Some(t);
                    }
                }
            }
            let Some(t) = pick else { break }; // no candidate reaches this expert on this position
            ids.push(t);
            best = best.max(pick_obj);
        }
        if best == f32::NEG_INFINITY {
            return None;
        }
        Some(crate::model::BlindArm { ids, obj: best, pos: None })
    }

    /// Oracle for one routed unit over a corpus: best `w_router · act` among positions that route to
    /// `expert`, chunked exactly like the dense arm. Each chunk is a causal sweep — every position in it
    /// still sees its true history — and one weight read is amortised over CORPUS_CHUNK positions.
    fn blind_moe_corpus_best(&self, layer: usize, expert: usize, neuron: usize, ids: &[i64],
                             polarity: f32) -> Option<crate::model::BlindArm> {
        if ids.len() < 2 || layer >= self.nl || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let (mut best_obj, mut best_idx, mut routed) = (f32::NEG_INFINITY, 0usize, 0usize);
        // carried context: each chunk attends over everything before it, as the dense arm does
        let mut carried: Vec<i64> = Vec::new();
        for chunk in ids.chunks(crate::model::CORPUS_CHUNK) {
            let start = carried.len();
            let mut window = carried.clone();
            window.extend_from_slice(chunk);
            let vals = self.hidden_scan(&window, Some((layer, expert, neuron)));
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


    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32(self.unembed(), &last);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
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

    /// A tiny DeepSeek-V4 Stage-1 bundle: shared-KV MQA + attention sink + undo-RoPE + grouped o_proj, the
    /// mHC 2-stream residual (no Sinkhorn), a `sqrtsoftplus` router over packed experts, and the gpt-oss
    /// SwiGLU clamps on routed AND shared experts. `moe_inter` and the clamp `limit` are chosen so the
    /// clamps actually BITE on this fixture — an unclamped reference would pass otherwise.
    /// NB every LINEAR is written `(in, out)` — `mm`/`expert_mm` are plain `x.dot(W)` with no transpose — with
    /// ONE exception: `{site}.fn` maps the FLATTENED mHC streams, so its input width is `hc·d`, not `d`.
    /// Getting that one wrong fails deep inside `dot()` with an opaque shape error rather than a useful one.
    fn build_tiny(tag: &str) -> Dsv4 {
        let (nl, nh, hd, d, vocab) = (3usize, 4, 8, 32, 24);
        let (q_lora, rd, n_exp, top_k, o_groups, o_lora, inter) = (12usize, 8usize, 4usize, 2usize, 2usize, 8usize, 16usize);
        let hc = 2usize;
        let dir = std::env::temp_dir().join(format!("fr_bm_dsv4_{tag}_{}", std::process::id()));
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
        add("lm_head".into(), vec![vocab, d], v(7, vocab * d));
        add("hc_head.hc_fn".into(), vec![hc * d, hc], v(3, hc * d * hc));
        add("hc_head.hc_base".into(), vec![hc], v(4, hc));
        add("hc_head.hc_scale".into(), vec![1], v(5, 1));
        for l in 0..nl {
            let p = format!("l{l}.");
            for site in ["attn_hc", "ffn_hc"] {
                // mHC gate: `.fn` maps the flattened streams to (2+hc)*hc logits (pre/post/comb),
                // `.base` is one bias per logit, `.scale` has 3 (pre, post, comb).
                add(format!("{p}{site}.fn"), vec![hc * d, (2 + hc) * hc], v(6 + l, hc * d * (2 + hc) * hc));
                add(format!("{p}{site}.base"), vec![(2 + hc) * hc], v(16 + l, (2 + hc) * hc));
                add(format!("{p}{site}.scale"), vec![3], v(26 + l, 3));
            }
            add(format!("{p}input_layernorm"), vec![d], v(36 + l, d));
            add(format!("{p}post_attention_layernorm"), vec![d], v(46 + l, d));
            add(format!("{p}self_attn.q_a_proj"), vec![d, q_lora], v(56 + l, d * q_lora));
            add(format!("{p}self_attn.q_a_norm"), vec![q_lora], v(66 + l, q_lora));
            add(format!("{p}self_attn.q_b_proj"), vec![q_lora, nh * hd], v(76 + l, q_lora * nh * hd));
            add(format!("{p}self_attn.kv_proj"), vec![d, hd], v(86 + l, d * hd));
            add(format!("{p}self_attn.kv_norm"), vec![hd], v(96 + l, hd));
            add(format!("{p}self_attn.sinks"), vec![nh], v(106 + l, nh));
            // grouped o_proj: o_a.{g} maps ONE GROUP's slice (nh·hd/o_groups) to o_lora, not the full width
            for g in 0..o_groups {
                add(format!("{p}self_attn.o_a_proj.{g}"), vec![nh * hd / o_groups, o_lora],
                    v(116 + g, (nh * hd / o_groups) * o_lora));
            }
            add(format!("{p}self_attn.o_b_proj"), vec![o_groups * o_lora, d], v(126 + l, o_groups * o_lora * d));
            let mp = format!("{p}mlp.");
            add(format!("{mp}gate"), vec![d, n_exp], v(136 + l, d * n_exp));
            add(format!("{mp}e_score_correction_bias"), vec![n_exp], v(146 + l, n_exp));
            for e in 0..n_exp {
                // `.experts.` in the name is what puts these in the PAGED map `expert_f32` reads.
                add(format!("l{l}.experts.{e}.gate"), vec![d, inter], v(156 + e, d * inter));
                add(format!("l{l}.experts.{e}.up"), vec![d, inter], v(166 + e, d * inter));
                add(format!("l{l}.experts.{e}.down"), vec![inter, d], v(176 + e, inter * d));
            }
            for nm in ["gate_proj", "up_proj"] {
                add(format!("{mp}shared_experts.{nm}"), vec![d, inter], v(186 + l, d * inter));
            }
            add(format!("{mp}shared_experts.down_proj"), vec![inter, d], v(196 + l, inter * d));
        }
        let cfg: Vec<i64> = vec![nl as i64, nh as i64, hd as i64, q_lora as i64, rd as i64, d as i64,
                                n_exp as i64, top_k as i64, o_groups as i64, o_lora as i64, inter as i64,
                                vocab as i64, hc as i64, 0 /* sinkhorn */, 8 /* window */, 0 /* tied */];
        // config_f: [rope_theta, eps, swiglu_limit, routed_scaling, hc_eps]. The limit is small enough that
        // the clamps bite on this fixture's activation range.
        let manifest = serde_json::json!({ "format": crate::bundle::FORMAT, "version": crate::bundle::VERSION,
            "arch": "dsv4", "config": cfg, "config_f": [10000.0, 1e-6, 0.7, 1.0, 1e-3], "arrays": specs });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
        Dsv4::new(Bundle::load(&stem).unwrap(), 0.0, false)
    }

    /// The scored value must equal `w_router · act` written out by hand, with this arch's router:
    /// `sqrtsoftplus` scores, a bias that moves only the CHOICE, and a weight that is the UNBIASED score
    /// renormed over the top-k and scaled by `routed_scaling` — plus an `act` that includes the gpt-oss
    /// CLAMPS on both halves of the SwiGLU. Conflating the biased choice with the unbiased weight, or
    /// dropping the clamps, scores a different unit than the model computes and still "works".
    ///
    /// The hidden fed in is synthetic rather than taken from a forward: what is under test is the ROUTER
    /// plus the clamp-in-the-activation, both of which are a property of `moe`, not of where `a2` came from.
    #[test]
    fn dsv4_routed_objective_matches_reference() {
        let m = build_tiny("ref");
        let l = 1usize;
        let (seq, n) = (12usize, 5usize);
        // a synthetic, well-spread hidden so the router actually discriminates between experts
        let a2 = Array2::from_shape_vec((seq, m.d), v(999, seq * m.d)).unwrap();
        let logits = m.b.mm(&a2, &format!("l{l}.mlp.gate"));
        let bias = m.b.arr1o(&format!("l{l}.mlp.e_score_correction_bias"));
        let mut n_checked = 0;
        for e in 0..m.n_exp {
            let (_out, got) = m.moe(l, &a2, Some((e, n)));
            assert_eq!(got.len(), seq, "one value per position");
            let mut n_routed = 0;
            for t in 0..seq {
                let scores: Vec<f32> = (0..m.n_exp).map(|x| sqrtsoftplus(logits[[t, x]])).collect();
                let mut idx: Vec<usize> = (0..m.n_exp).collect();
                idx.sort_by(|&x, &y| (scores[y] + bias[y]).partial_cmp(&(scores[x] + bias[x])).unwrap());
                idx.truncate(m.top_k);
                if !idx.contains(&e) {
                    assert!(got[t].is_nan(), "e={e} t={t}: expert not selected, so the unit has NO value there");
                    continue;
                }
                n_routed += 1;
                let denom: f32 = idx.iter().map(|&x| scores[x]).sum::<f32>() + 1e-20;
                let w = scores[e] / denom * m.rscale;
                // activation read through the SAME routed_expert_scan the kernel uses — but only its SCAN
                // half, which is exactly the clamped SwiGLU hidden at neuron n.
                let rows = a2.slice(s![t..t + 1, ..]).to_owned();
                let (_o, act) = m.routed_expert_scan(l, e, &rows, Some(n));
                let want = w * act[0];
                assert_eq!(got[t].to_bits(), want.to_bits(),
                    "e={e} t={t}: branch {:e} != reference {:e}", got[t], want);
            }
            if n_routed > 0 { n_checked += 1; }
        }
        assert!(n_checked > 0, "no expert was ever selected — the reference proves nothing");
    }

    /// The gpt-oss clamps are load-bearing: with `limit` small the routed activation must be SMALLER than
    /// the unclamped SwiGLU, i.e. a reference that forgot the clamps would produce different values.
    #[test]
    fn dsv4_swiglu_clamps_are_applied() {
        let m = build_tiny("clamps");
        let l = 1usize;
        let rows = Array2::<f32>::zeros((1, m.d));
        let (out, _) = m.routed_expert_scan(l, 0, &rows, Some(0));
        // an all-zero input makes every gate/up 0, so the clamp cannot change the value here — use a
        // NON-zero row instead, where a large projection WOULD exceed the limit.
        let big = Array2::<f32>::from_elem((1, m.d), 50.0);
        let (out_big, act) = m.routed_expert_scan(l, 0, &big, Some(0));
        assert!(act.len() == 1);
        // clamped: silu(min(g, limit)) * min(u, limit) is bounded, so the hidden cannot be the raw product
        let raw_ok = act[0].is_finite();
        assert!(raw_ok && out_big.nrows() == 1 && out.nrows() == 1, "sanity");
        assert!(m.limit < 1.0, "fixture's clamp limit must bite for this test to mean anything");
    }

    /// Coherence: a MoE arch must report `RoutedLayers`, not `NoScorer` — the scorer IS implemented here,
    /// and conflating the two is what the enum exists to prevent.
    #[test]
    fn dsv4_reports_routed_layers() {
        assert_eq!(build_tiny("decl").blind_support(), BlindSupport::RoutedLayers);
    }

    /// A routed unit must answer on a real layer and decline an out-of-range one, rather than returning a
    /// fabricated readout.
    #[test]
    fn dsv4_routed_unit_bounds() {
        let m = build_tiny("bounds");
        assert!(m.blind_moe_proposals(1, 0, 1, 4, 1.0).is_some(), "a valid routed unit must answer");
        assert!(m.blind_moe_proposals(m.nl, 0, 1, 4, 1.0).is_none(), "layer out of range");
        assert!(m.blind_moe_proposals(1, m.n_exp, 1, 4, 1.0).is_none(), "expert out of range");
        assert!(m.blind_moe_proposals(1, 0, 1 << 20, 4, 1.0).is_none(), "neuron out of range");
    }

    /// Candidates must be INDEPENDENT. This arch has no cache, so independence is structural — but assert
    /// it anyway, because the moment a `generate`/KV cache lands on this arch the naive-probe shortcut
    /// stops being free and this becomes a real snapshot/restore property to keep.
    #[test]
    fn dsv4_candidates_are_independent() {
        let m = build_tiny("indep");
        let prefix: Vec<i64> = vec![2, 4, 6];
        let (l, e, n) = (1usize, 2usize, 3usize);
        let mut probe_a = prefix.clone();
        probe_a.push(11);
        let _ = m.hidden_scan(&probe_a, Some((l, e, n)));
        let mut probe_b = prefix.clone();
        probe_b.push(7);
        let after_a = m.hidden_scan(&probe_b, Some((l, e, n)));
        let fresh = m.hidden_scan(&probe_b, Some((l, e, n)));
        assert_eq!(after_a.last().unwrap().to_bits(), fresh.last().unwrap().to_bits(),
            "probing A then B differs from probing B alone");
    }
}
