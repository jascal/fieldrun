//! Tier B — composition, RoPE family (Llama-3.2 / Qwen2.5). A faithful Rust port of pylm's `numpy_rope.py`: RMSNorm +
//! rotary position embedding + grouped-query attention + SwiGLU MLP, over a fieldrun bundle (`arch: "rope"`). Mirrors
//! the numpy kernel array-for-array, so it reproduces it (and torch) exactly. fp32 in.

use ndarray::{s, Array2, ArrayView1, Axis};

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
}

fn rmsnorm(x: &Array2<f32>, w: ArrayView1<f32>, eps: f32) -> Array2<f32> {
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
    pub fn new(b: Bundle) -> Rope {
        let c = &b.config; // [n_layer, H, nkv, hd, d, ffn, vocab, tied]
        let (n_layer, h, nkv, hd) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
        let theta = b.config_f[0] as f32;
        let eps = b.config_f[1] as f32;
        let inv = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        Rope { b, n_layer, h, nkv, hd, eps, inv }
    }

    /// Apply rotary embedding in place to a (seq, n_heads*hd) block, treating each head's hd-vector independently.
    fn rope(&self, x: &mut Array2<f32>, n_heads: usize) {
        let (hd, half) = (self.hd, self.hd / 2);
        for (pos, mut row) in x.rows_mut().into_iter().enumerate() {
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
        let mut y = a.dot(&self.b.arr2(name));
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
        let embed = self.b.arr2("embed");
        let mut x = Array2::<f32>::zeros((seq, embed.ncols()));
        for (t, &id) in ids.iter().enumerate() {
            x.row_mut(t).assign(&embed.row(id as usize));
        }

        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, h);
            self.rope(&mut k, nkv);
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
            x = &x + &attn_out.dot(&self.b.arr2(&format!("{p}self_attn.o_proj")));

            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = a2.dot(&self.b.arr2(&format!("{p}mlp.gate_proj")));
            let up = a2.dot(&self.b.arr2(&format!("{p}mlp.up_proj")));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            x = &x + &hidden.dot(&self.b.arr2(&format!("{p}mlp.down_proj")));
        }

        rmsnorm(&x, self.b.arr1("norm"), self.eps)
    }

    fn unembed(&self) -> ndarray::ArrayView2<f32> {
        if self.b.config[7] != 0 { self.b.arr2("embed") } else { self.b.arr2("lm_head") }
    }
}

impl Model for Rope {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.index_axis(Axis(0), ids.len() - 1);   // unembed only the predicting position
        let logits = last.dot(&self.unembed().t());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}
