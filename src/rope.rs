//! Tier B — composition, RoPE family (Llama-3.2 / Qwen2.5). A faithful Rust port of pylm's `numpy_rope.py`: RMSNorm +
//! rotary position embedding + grouped-query attention + SwiGLU MLP, over a fieldrun bundle (`arch: "rope"`). Mirrors
//! the numpy kernel array-for-array, so it reproduces it (and torch) exactly. fp32 in.

use ndarray::{s, Array1, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Rope {
    b: Bundle,
    n_layer: usize,
    h: usize,
    nkv: usize,
    hd: usize,
    eps: f32,
    inv: Vec<f32>, // rotary frequencies, length hd/2
    route: f32,    // Tier C: fraction of MLP neurons to compute per token (0 = off)
}

fn rmsnorm(x: &Array2<f32>, w: Array1<f32>, eps: f32) -> Array2<f32> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let n = row.len() as f32;
        let ms = row.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = 1.0 / (ms + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = *v * inv * w[i];
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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

impl Rope {
    pub fn new(b: Bundle, route: f32) -> Rope {
        let c = &b.config; // [n_layer, H, nkv, hd, d, ffn, vocab, tied]
        let (n_layer, h, nkv, hd) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
        let theta = b.config_f[0] as f32;
        let eps = b.config_f[1] as f32;
        let inv = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        Rope { b, n_layer, h, nkv, hd, eps, inv, route }
    }

    fn down(&self, h: &Array2<f32>, name: &str) -> Array2<f32> {
        if self.route > 0.0 && self.route < 1.0 {
            self.b.mm_routed_down(h, name, self.route)
        } else {
            self.b.mm(h, name)
        }
    }

    /// Apply rotary embedding in place to a (seq, n_heads*hd) block, treating each head's hd-vector independently.
    /// Row i is absolute position `pos0 + i` (pos0 > 0 for KV-cache decode of a single token).
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

    fn proj(&self, a: &Array2<f32>, name: &str) -> Array2<f32> {
        let mut y = self.b.mm(a, name);
        let bias = format!("{name}.bias");
        if self.b.has(&bias) {
            y = y + &self.b.arr1(&bias);
        }
        y
    }

    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids); // dtype-agnostic lookup (embed stays f16 under int8)

        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep; // GQA: this query head reads kv head head/rep
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[[i, j]] = -1e30;
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));

            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            x = &x + &self.down(&hidden, &format!("{p}mlp.down_proj"));
        }

        rmsnorm(&x, self.b.arr1("norm"), self.eps)
    }

    fn unembed_name(&self) -> &'static str {
        if self.b.config[7] != 0 { "embed" } else { "lm_head" } // tied embed, else a separate (fp16) head
    }

    /// Run `m` new positions through the layers, caching K/V (post-RoPE, GQA width nkv*hd) and attending over the whole
    /// cache. cur = absolute position of the first new row. Prefill (m = prompt, cur = 0) or decode (m = 1).
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, h, cur);
            self.rope(&mut k, nkv, cur);
            kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
            vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = kc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let vh = vc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
                for i in 0..m {
                    for j in (cur + i + 1)..klen {
                        scores[[i, j]] = -1e30;
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            x = &x + &self.down(&hidden, &format!("{p}mlp.down_proj"));
        }
        rmsnorm(&x, self.b.arr1("norm"), self.eps)
    }

    fn head_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let logits = self.b.rowdot_f32(self.unembed_name(), &xfn.row(xfn.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}

impl Model for Rope {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids); // unembed only the predicting position
        let logits = self.b.rowdot_f32(self.unembed_name(), &xf.row(ids.len() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let total = prompt.len() + n_new;
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
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
}
