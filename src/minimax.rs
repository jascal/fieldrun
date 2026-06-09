//! Tier B — composition, MiniMax-M2. The RoPE backbone (RMSNorm, single-base RoPE, GQA, standard two-norm pre-norm)
//! with two MiniMax specifics: **full-width q/k-norm** (a single RMSNorm over the whole nh·head_dim / nkv·head_dim
//! projection output, before the head reshape + RoPE — not the per-head QK-norm of Qwen3/Gemma), and an all-MoE FFN
//! with a **sigmoid router** (sigmoid scores + a bias-correction buffer pick the experts; the sigmoid scores,
//! renormalised over the top-k, are the weights — no group limiting, no shared expert). SwiGLU experts read on demand
//! from the mmap (offload). A faithful port of `MiniMaxM2ForCausalLM`. predict only (generate/explain TBD).

use std::collections::HashMap;

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct MiniMax {
    b: Bundle,
    nl: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    eps: f32,
    scale: f32,
    inv: Vec<f32>,
    n_exp: usize,
    topk: usize,
    tied: bool,
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn softmax_rows(a: &mut Array2<f32>) {
    for mut row in a.rows_mut() {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in row.iter_mut() { *v = (*v - m).exp(); s += *v; }
        row.mapv_inplace(|v| v / s);
    }
}

impl MiniMax {
    pub fn new(b: Bundle, _route: f32, _kv_int8: bool) -> MiniMax {
        // config: [nl, nh, nkv, hd, d, vocab, tied, n_exp, topk, inter]
        let c = &b.config;
        let (nl, nh, nkv, hd) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
        let tied = c[6] != 0;
        let (n_exp, topk) = (c[7] as usize, c[8] as usize);
        let (theta, eps) = (b.config_f[0] as f32, b.config_f[1] as f32);
        let inv = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        MiniMax { b, nl, nh, nkv, hd, eps, scale: (hd as f32).powf(-0.5), inv, n_exp, topk, tied }
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

    fn rope(&self, x: &mut Array2<f32>, n_heads: usize) {
        let (hd, half) = (self.hd, self.hd / 2);
        for (i, mut row) in x.rows_mut().into_iter().enumerate() {
            for head in 0..n_heads {
                let base = head * hd;
                for j in 0..half {
                    let ang = i as f32 * self.inv[j];
                    let (c, s) = (ang.cos(), ang.sin());
                    let (a, b) = (row[base + j], row[base + j + half]);
                    row[base + j] = a * c - b * s;
                    row[base + j + half] = b * c + a * s;
                }
            }
        }
    }

    /// MoE over the post-attention-normed hidden `a2`: sigmoid router (+bias for the choice, sigmoid renormed for the
    /// weight, no group limiting / shared expert), SwiGLU experts paged from the mmap.
    fn moe(&self, l: usize, a2: &Array2<f32>) -> Array2<f32> {
        let p = format!("l{l}.");
        let logits = self.b.mm(a2, &format!("{p}gate")); // (seq, n_exp)
        let bias = self.b.arr1o(&format!("{p}gate_bias"));
        let seq = a2.nrows();
        let mut assign: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for t in 0..seq {
            let sig: Vec<f32> = logits.row(t).iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
            let choice: Vec<f32> = sig.iter().zip(bias.iter()).map(|(s, b)| s + b).collect();
            let mut idx: Vec<usize> = (0..self.n_exp).collect();
            idx.sort_by(|&a, &b| choice[b].partial_cmp(&choice[a]).unwrap());
            idx.truncate(self.topk);
            let denom: f32 = idx.iter().map(|&e| sig[e]).sum::<f32>() + 0.0;
            for &e in &idx {
                assign.entry(e).or_default().push((t, sig[e] / denom));
            }
        }
        let mut out = Array2::<f32>::zeros((seq, a2.ncols()));
        for (e, toks) in &assign {
            let mut rows = Array2::<f32>::zeros((toks.len(), a2.ncols()));
            for (i, &(t, _)) in toks.iter().enumerate() { rows.row_mut(i).assign(&a2.row(t)); }
            let gate = self.b.expert_mm(&rows, &format!("{p}experts.{e}.gate"));
            let up = self.b.expert_mm(&rows, &format!("{p}experts.{e}.up"));
            let mut hh = gate;
            for (h, u) in hh.iter_mut().zip(up.iter()) { *h = silu(*h) * u; }
            let down = self.b.expert_mm(&hh, &format!("{p}experts.{e}.down"));
            for (i, &(t, w)) in toks.iter().enumerate() {
                for cc in 0..out.ncols() { out[[t, cc]] += w * down[[i, cc]]; }
            }
        }
        out
    }

    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let rep = nh / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            // full-width q/k norm on the projection output, then reshape + RoPE
            let mut q = self.norm(&self.b.mm(&a, &format!("{p}self_attn.q_proj")), &format!("{p}q_norm"));
            let mut k = self.norm(&self.b.mm(&a, &format!("{p}self_attn.k_proj")), &format!("{p}k_norm"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, nh);
            self.rope(&mut k, nkv);
            let mut attn_out = Array2::<f32>::zeros((seq, nh * hd));
            for head in 0..nh {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..seq {
                    for j in (i + 1)..seq { scores[[i, j]] = -1e30; }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            x = &x + &self.moe(l, &a2);
        }
        self.norm(&x, "norm")
    }
}

impl Model for MiniMax {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32(self.unembed(), &last);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}
