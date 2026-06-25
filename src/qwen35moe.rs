//! Qwen3.6 (`qwen3_5_moe`) — hybrid Gated-DeltaNet / full-attention MoE. Every 4th layer is full GQA
//! attention (QK-norm + RoPE, like qwen3moe); the other 3 are Gated DeltaNet *linear* attention
//! (`crate::deltanet`, verified vs transformers in #112–#114). Every layer's FFN is a SparseMoeBlock:
//! softmax top-k routed experts (qwen3moe-style) PLUS a sigmoid-gated shared expert. `layer_types` (per
//! layer, from config) selects linear vs full. Faithful port of `Qwen3_5MoeForCausalLM`'s text path;
//! validated per-layer against transformers via `experiments/qwen3next/` (make_tiny → convert → compare).

use std::collections::HashMap;

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::deltanet::{causal_conv1d_silu, delta_gate, gated_deltanet_mha, rmsnorm_gated};
use crate::model::Model;

pub struct Qwen35Moe {
    b: Bundle,
    n_layer: usize,
    h: usize,
    nkv: usize,
    hd: usize,
    d: usize,
    eps: f32,
    scale: f32,
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

impl Qwen35Moe {
    pub fn new(b: Bundle, _route: f32, _kv_int8: bool) -> Qwen35Moe {
        // config_i: [nl,nh,nkv,hd,d,vocab,tied,n_exp,topk,moe_inter,shared_inter,norm_topk,
        //            nvh,nkh,hkd,hvd,conv_k, <nl layer_types: 0=full 1=linear>]
        let c = &b.config;
        let (n_layer, h, nkv, hd, d) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize, c[4] as usize);
        let tied = c[6] != 0;
        let (n_exp, topk, shared_inter) = (c[7] as usize, c[8] as usize, c[10] as usize); // c[11]=norm_topk unused (always renorm)
        let (nvh, nkh, hkd, hvd, conv_k) =
            (c[12] as usize, c[13] as usize, c[14] as usize, c[15] as usize, c[16] as usize);
        let linear: Vec<bool> = (0..n_layer).map(|l| c[17 + l] != 0).collect();
        let (theta, eps) = (b.config_f[0] as f32, b.config_f[1] as f32);
        let inv = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        Qwen35Moe { b, n_layer, h, nkv, hd, d, eps, scale: (hd as f32).powf(-0.5), inv, n_exp, topk,
                    shared_inter, nvh, nkh, hkd, hvd, conv_k, linear, tied }
    }

    fn unembed(&self) -> &str {
        if self.tied { "embed" } else { "lm_head" }
    }

    fn norm(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let w = self.b.arr1o(name);
        let mut out = x.clone();
        for mut row in out.rows_mut() {
            let n = row.len() as f32;
            let ms = row.iter().map(|v| v * v).sum::<f32>() / n;
            let inv = 1.0 / (ms + self.eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = *v * inv * w[i];
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
                    row[base + c] = row[base + c] * inv * w[c];
                }
            }
        }
    }

    fn rope(&self, x: &mut Array2<f32>, n_heads: usize, pos0: usize) {
        let (hd, half) = (self.hd, self.hd / 2);
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
        let mut q = self.b.mm(a, &format!("{p}self_attn.q_proj"));
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

    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let attn = if self.linear[l] { self.linear_attn(l, &a) } else { self.full_attn(l, &a) };
            x = &x + &attn;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            x = &x + &self.moe_branch(l, &a2);
        }
        self.norm(&x, "norm")
    }

    /// Debug: the residual stream snapshots [embed, after-L0, …, after-L{n-1}] (each flat seq·d, row-major),
    /// matching transformers `output_hidden_states` — for per-layer parity localization via compare.py.
    pub fn hiddens(&self, ids: &[i64]) -> Vec<Vec<f32>> {
        let mut snaps = Vec::with_capacity(self.n_layer + 1);
        let mut x = self.b.rows_f32("embed", ids);
        snaps.push(x.iter().cloned().collect());
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let attn = if self.linear[l] { self.linear_attn(l, &a) } else { self.full_attn(l, &a) };
            x = &x + &attn;
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            x = &x + &self.moe_branch(l, &a2);
            snaps.push(x.iter().cloned().collect());
        }
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
}
