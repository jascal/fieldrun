//! Tier B — composition. The GPT-2 forward pass as Rust matmuls over the flat bundle, a faithful port of pylm's
//! `numpy_lm.py`: token+position embeddings, then per layer LayerNorm -> causal multi-head attention -> LayerNorm ->
//! GELU MLP, a final LayerNorm, and the tied unembed. This is the half flat retrieval cannot do (genuine dense
//! computation, the forge tax) — it runs here as plain `ndarray` matmuls, no framework. fp32 in, exact vs numpy.

use ndarray::{s, Array1, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Gpt2 {
    b: Bundle,
    n_layer: usize,
    n_head: usize,
    d: usize,
    route: f32,    // Tier C: fraction of MLP neurons to compute per token (0 = off / full)
    kv_int8: bool, // store the KV cache as int8 (per-head scale) — 4x smaller cache for long context
}

fn layernorm(x: &Array2<f32>, g: Array1<f32>, b: Array1<f32>) -> Array2<f32> {
    let eps = 1e-5f32;
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let n = row.len() as f32;
        let mu = row.sum() / n;
        let var = row.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mu) * inv * g[i] + b[i];
        }
    }
    out
}

fn gelu(x: &mut Array2<f32>) {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    x.mapv_inplace(|v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh()));
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

impl Gpt2 {
    pub fn new(b: Bundle, route: f32, kv_int8: bool) -> Gpt2 {
        let c = &b.config; // [n_layer, n_head, n_embd, n_positions, vocab]
        let (n_layer, n_head, d) = (c[0] as usize, c[1] as usize, c[2] as usize);
        Gpt2 { b, n_layer, n_head, d, route, kv_int8 }
    }

    fn down(&self, h: &Array2<f32>, name: &str) -> Array2<f32> {
        if self.route > 0.0 && self.route < 1.0 {
            self.b.mm_routed_down(h, name, self.route)
        } else {
            self.b.mm(h, name)
        }
    }

    /// Final hidden states (seq, d) after the last LayerNorm — the forward pass minus the unembed. Mirrors
    /// `numpy_lm.NumpyGPT2.logits` up to `x @ wte.T`. The unembed is split out so `predict` projects only the last row.
    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let hd = self.d / self.n_head;
        let mut x = &self.b.rows_f32("wte", ids) + &self.b.rows_f32("wpe", &(0..seq as i64).collect::<Vec<_>>());

        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = self.b.mm(&a, &format!("{p}attn.c_attn.weight")) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
            let mut attn_out = Array2::<f32>::zeros((seq, self.d));
            for h in 0..self.n_head {
                let q = qkv.slice(s![.., h * hd..(h + 1) * hd]);
                let k = qkv.slice(s![.., self.d + h * hd..self.d + (h + 1) * hd]);
                let v = qkv.slice(s![.., 2 * self.d + h * hd..2 * self.d + (h + 1) * hd]);
                let mut scores = q.dot(&k.t()) / (hd as f32).sqrt();
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[[i, j]] = -1e10; // causal mask
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., h * hd..(h + 1) * hd]).assign(&scores.dot(&v));
            }
            x = &x + &(self.b.mm(&attn_out, &format!("{p}attn.c_proj.weight"))
                + &self.b.arr1(&format!("{p}attn.c_proj.bias")));

            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut h_mlp = self.b.mm(&a2, &format!("{p}mlp.c_fc.weight")) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut h_mlp);
            x = &x + &(self.down(&h_mlp, &format!("{p}mlp.c_proj.weight"))
                + &self.b.arr1(&format!("{p}mlp.c_proj.bias")));
        }

        layernorm(&x, self.b.arr1("ln_f.weight"), self.b.arr1("ln_f.bias"))
    }

    /// Explain the prediction: the live attention-head circuits + top MLP features at the predicting position.
    fn explanation(&self, ids: &[i64]) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let hd = self.d / self.n_head;
        let mut x = &self.b.rows_f32("wte", ids) + &self.b.rows_f32("wpe", &(0..seq as i64).collect::<Vec<_>>());
        let mut att_last: Vec<Vec<Vec<f32>>> = Vec::new(); // per layer, per head, last-position attention row
        let mut mlp_h: Vec<Vec<f32>> = Vec::new();
        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = self.b.mm(&a, &format!("{p}attn.c_attn.weight")) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
            let mut attn_out = Array2::<f32>::zeros((seq, self.d));
            let mut layer_att = Vec::with_capacity(self.n_head);
            for h in 0..self.n_head {
                let q = qkv.slice(s![.., h * hd..(h + 1) * hd]);
                let k = qkv.slice(s![.., self.d + h * hd..self.d + (h + 1) * hd]);
                let v = qkv.slice(s![.., 2 * self.d + h * hd..2 * self.d + (h + 1) * hd]);
                let mut scores = q.dot(&k.t()) / (hd as f32).sqrt();
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[[i, j]] = -1e10;
                    }
                }
                softmax_rows(&mut scores);
                layer_att.push(scores.row(seq - 1).to_vec());
                attn_out.slice_mut(s![.., h * hd..(h + 1) * hd]).assign(&scores.dot(&v));
            }
            att_last.push(layer_att);
            x = &x + &(self.b.mm(&attn_out, &format!("{p}attn.c_proj.weight")) + &self.b.arr1(&format!("{p}attn.c_proj.bias")));
            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut hm = self.b.mm(&a2, &format!("{p}mlp.c_fc.weight")) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut hm);
            mlp_h.push(hm.row(seq - 1).to_vec());
            x = &x + &(self.b.mm(&hm, &format!("{p}mlp.c_proj.weight")) + &self.b.arr1(&format!("{p}mlp.c_proj.bias")));
        }
        let xf = layernorm(&x, self.b.arr1("ln_f.weight"), self.b.arr1("ln_f.bias"));
        let lg = self.b.rowdot_f32("wte", &xf.row(seq - 1).to_vec());
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        assemble(ids, &att_last, &mlp_h, model_predicts, |l, n, act| {
            let w_out = self.b.weight_row(&format!("h{l}.mlp.c_proj.weight"), n); // neuron n's write direction (any dtype)
            top_promoted(&self.b.rowdot_f32("wte", &w_out), act, 5)
        })
    }

    /// Run `m` new positions (rows of `emb`, already token+position embeddings for absolute positions `cur..cur+m`)
    /// through all layers, appending their K/V to the cache and attending against the whole cache. Returns the
    /// pre-final-LN hidden states (m, d). Used for both prefill (m = prompt len, cur = 0) and decode (m = 1).
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let d = self.d;
        let hd = d / self.n_head;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = self.b.mm(&a, &format!("{p}attn.c_attn.weight")) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
            kc[l].slice_mut(s![cur..klen, ..]).assign(&qkv.slice(s![.., d..2 * d])); // append new K/V to the cache
            vc[l].slice_mut(s![cur..klen, ..]).assign(&qkv.slice(s![.., 2 * d..3 * d]));
            let q = qkv.slice(s![.., 0..d]);
            let mut attn_out = Array2::<f32>::zeros((m, d));
            for hh in 0..self.n_head {
                let qh = q.slice(s![.., hh * hd..(hh + 1) * hd]);
                let kh = kc[l].slice(s![0..klen, hh * hd..(hh + 1) * hd]); // attend over the whole cache
                let vh = vc[l].slice(s![0..klen, hh * hd..(hh + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
                for i in 0..m {
                    for j in (cur + i + 1)..klen {
                        scores[[i, j]] = -1e10; // causal: new row i is absolute position cur+i
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., hh * hd..(hh + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &(self.b.mm(&attn_out, &format!("{p}attn.c_proj.weight")) + &self.b.arr1(&format!("{p}attn.c_proj.bias")));
            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut hm = self.b.mm(&a2, &format!("{p}mlp.c_fc.weight")) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut hm);
            x = &x + &(self.down(&hm, &format!("{p}mlp.c_proj.weight")) + &self.b.arr1(&format!("{p}mlp.c_proj.bias")));
        }
        x
    }

    /// Same as `forward_block` but with an int8 KV cache (per-position-per-head scale): quantise new K/V on write,
    /// dequantise on read. 4x smaller cache for long context; the small per-head quant error keeps tokens ~identical.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_q(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>],
                       vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]) -> Array2<f32> {
        let (d, h_n) = (self.d, self.n_head);
        let hd = d / h_n;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let q8 = |v: f32, sc: f32| (v / sc).round().clamp(-127.0, 127.0) as i8;
        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = self.b.mm(&a, &format!("{p}attn.c_attn.weight")) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
            for i in 0..m {
                let pos = cur + i;
                for head in 0..h_n {
                    let (kb, vb) = (d + head * hd, 2 * d + head * hd);
                    let sck = (0..hd).fold(0f32, |mx, c| mx.max(qkv[[i, kb + c]].abs())) / 127.0;
                    let scv = (0..hd).fold(0f32, |mx, c| mx.max(qkv[[i, vb + c]].abs())) / 127.0;
                    let (sck, scv) = (if sck > 0.0 { sck } else { 1.0 }, if scv > 0.0 { scv } else { 1.0 });
                    ks[l][pos * h_n + head] = sck;
                    vs[l][pos * h_n + head] = scv;
                    for c in 0..hd {
                        kc[l][pos * d + head * hd + c] = q8(qkv[[i, kb + c]], sck);
                        vc[l][pos * d + head * hd + c] = q8(qkv[[i, vb + c]], scv);
                    }
                }
            }
            let q = qkv.slice(s![.., 0..d]);
            let mut attn_out = Array2::<f32>::zeros((m, d));
            for head in 0..h_n {
                let mut kh = Array2::<f32>::zeros((klen, hd)); // dequantise the cached head
                let mut vh = Array2::<f32>::zeros((klen, hd));
                for pos in 0..klen {
                    let (sck, scv) = (ks[l][pos * h_n + head], vs[l][pos * h_n + head]);
                    for c in 0..hd {
                        kh[[pos, c]] = kc[l][pos * d + head * hd + c] as f32 * sck;
                        vh[[pos, c]] = vc[l][pos * d + head * hd + c] as f32 * scv;
                    }
                }
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
                for i in 0..m {
                    for j in (cur + i + 1)..klen {
                        scores[[i, j]] = -1e10;
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &(self.b.mm(&attn_out, &format!("{p}attn.c_proj.weight")) + &self.b.arr1(&format!("{p}attn.c_proj.bias")));
            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut hm = self.b.mm(&a2, &format!("{p}mlp.c_fc.weight")) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut hm);
            x = &x + &(self.down(&hm, &format!("{p}mlp.c_proj.weight")) + &self.b.arr1(&format!("{p}mlp.c_proj.bias")));
        }
        x
    }

    fn generate_kv_int8(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let (d, total) = (self.d, prompt.len() + n_new);
        let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|_| vec![0i8; total * d]).collect();
        let mut vc = kc.clone();
        let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|_| vec![0f32; total * self.n_head]).collect();
        let mut vs = ks.clone();
        let ppos: Vec<i64> = (0..prompt.len() as i64).collect();
        let emb = &self.b.rows_f32("wte", prompt) + &self.b.rows_f32("wpe", &ppos);
        let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let e = &self.b.rows_f32("wte", &[next]) + &self.b.rows_f32("wpe", &[pos as i64]);
            let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }

    fn head_argmax(&self, xb: &Array2<f32>) -> i64 {
        let xfn = layernorm(xb, self.b.arr1("ln_f.weight"), self.b.arr1("ln_f.bias"));
        let logits = self.b.rowdot_f32("wte", &xfn.row(xb.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}

impl Model for Gpt2 {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids); // unembed only the predicting position
        let logits = self.b.rowdot_f32("wte", &xf.row(ids.len() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn explain(&self, ids: &[i64]) -> Option<crate::explain::Explanation> {
        Some(self.explanation(ids))
    }

    /// KV-cache generation: prefill the prompt once, then each new token only forwards its own row and attends against
    /// the cached K/V — O(1) layer work per token instead of re-running the whole context. Identical greedy tokens to
    /// the naive path (the cache is exact), just without the recompute.
    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        if self.kv_int8 {
            return self.generate_kv_int8(prompt, n_new);
        }
        let d = self.d;
        let total = prompt.len() + n_new;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, d))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, d))).collect();
        let ppos: Vec<i64> = (0..prompt.len() as i64).collect();
        let emb = &self.b.rows_f32("wte", prompt) + &self.b.rows_f32("wpe", &ppos); // prefill
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let e = &self.b.rows_f32("wte", &[next]) + &self.b.rows_f32("wpe", &[pos as i64]); // decode one token
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }
}
