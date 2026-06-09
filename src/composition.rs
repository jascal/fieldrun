//! Tier B — composition. The GPT-2 forward pass as Rust matmuls over the flat bundle, a faithful port of pylm's
//! `numpy_lm.py`: token+position embeddings, then per layer LayerNorm -> causal multi-head attention -> LayerNorm ->
//! GELU MLP, a final LayerNorm, and the tied unembed. This is the half flat retrieval cannot do (genuine dense
//! computation, the forge tax) — it runs here as plain `ndarray` matmuls, no framework. fp32 in, exact vs numpy.

use ndarray::{s, Array2, ArrayView1, Axis};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Gpt2 {
    b: Bundle,
    n_layer: usize,
    n_head: usize,
    d: usize,
}

fn layernorm(x: &Array2<f32>, g: ArrayView1<f32>, b: ArrayView1<f32>) -> Array2<f32> {
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
    pub fn new(b: Bundle) -> Gpt2 {
        let c = &b.config; // [n_layer, n_head, n_embd, n_positions, vocab]
        let (n_layer, n_head, d) = (c[0] as usize, c[1] as usize, c[2] as usize);
        Gpt2 { b, n_layer, n_head, d }
    }

    /// Final hidden states (seq, d) after the last LayerNorm — the forward pass minus the unembed. Mirrors
    /// `numpy_lm.NumpyGPT2.logits` up to `x @ wte.T`. The unembed is split out so `predict` projects only the last row.
    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let hd = self.d / self.n_head;
        let wte = self.b.arr2("wte");
        let wpe = self.b.arr2("wpe");
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for (t, &id) in ids.iter().enumerate() {
            x.row_mut(t).assign(&(&wte.row(id as usize) + &wpe.row(t)));
        }

        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = a.dot(&self.b.arr2(&format!("{p}attn.c_attn.weight"))) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
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
            x = &x + &(attn_out.dot(&self.b.arr2(&format!("{p}attn.c_proj.weight")))
                + &self.b.arr1(&format!("{p}attn.c_proj.bias")));

            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut h_mlp = a2.dot(&self.b.arr2(&format!("{p}mlp.c_fc.weight"))) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut h_mlp);
            x = &x + &(h_mlp.dot(&self.b.arr2(&format!("{p}mlp.c_proj.weight")))
                + &self.b.arr1(&format!("{p}mlp.c_proj.bias")));
        }

        layernorm(&x, self.b.arr1("ln_f.weight"), self.b.arr1("ln_f.bias"))
    }

    /// Full logits (seq, vocab) — for explain / scoring every position. `predict` uses the cheaper last-row path.
    pub fn logits(&self, ids: &[i64]) -> Array2<f32> {
        self.hidden(ids).dot(&self.b.arr2("wte").t())
    }
}

impl Model for Gpt2 {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.index_axis(Axis(0), ids.len() - 1);   // unembed only the predicting position
        let logits = last.dot(&self.b.arr2("wte").t());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}
