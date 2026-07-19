//! Encoder-only BERT (deepset/gbert-base class): token+position+token-type embeddings -> embeddings LayerNorm,
//! then per layer post-LN bidirectional multi-head attention and an exact-erf GELU MLP. No causal mask, no KV
//! cache, no LM head — the product is the per-token hidden states (HF `output_hidden_states` convention), the
//! substrate a downstream head (e.g. satzklar-model's biaffine dependency parser) consumes via `--encode-dump`.
//! config: [n_layer, n_head, d, d_ff, vocab, max_pos, type_vocab]; config_f: [layer_norm_eps].

use ndarray::{s, Array2};

use crate::bundle::Bundle;

pub struct Bert {
    b: Bundle,
    n_layer: usize,
    n_head: usize,
    d: usize,
    eps: f32,
}

fn layernorm(x: &Array2<f32>, g: &[f32], bi: &[f32], eps: f32) -> Array2<f32> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let n = row.len() as f32;
        let mu = row.sum() / n;
        let var = row.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mu) * inv * g[i] + bi[i];
        }
    }
    out
}

/// Exact erf GELU (torch/HF `gelu`), not the tanh approximation the decoder kernels use — BERT is trained with
/// the erf form and the downstream biaffine head is sensitive to the difference accumulating over 12 layers.
fn gelu_erf(x: &mut Array2<f32>) {
    x.mapv_inplace(|v| {
        let vd = v as f64;
        (0.5 * vd * (1.0 + libm::erf(vd / std::f64::consts::SQRT_2))) as f32
    });
}

fn softmax_rows(a: &mut Array2<f32>) {
    for mut row in a.rows_mut() {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        row.mapv_inplace(|v| v / sum);
    }
}

impl Bert {
    pub fn new(b: Bundle) -> Bert {
        let c = &b.config; // [n_layer, n_head, d, d_ff, vocab, max_pos, type_vocab]
        let (n_layer, n_head, d) = (c[0] as usize, c[1] as usize, c[2] as usize);
        let eps = *b.config_f.first().unwrap_or(&1e-12) as f32;
        Bert { b, n_layer, n_head, d, eps }
    }

    fn ln(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let g = self.b.arr1(&format!("{name}.weight"));
        let bi = self.b.arr1(&format!("{name}.bias"));
        layernorm(x, g.as_slice().unwrap(), bi.as_slice().unwrap(), self.eps)
    }

    fn lin(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        self.b.mm(x, &format!("{name}.weight")) + &self.b.arr1(&format!("{name}.bias"))
    }

    /// Per-token hidden states in HF `output_hidden_states` convention: 13 snapshots for a 12-layer model —
    /// [embeddings output (post embeddings-LayerNorm), layer 1 output, …, layer N output]. Each snapshot is the
    /// flat seq·d row-major f32 states; the last one is `last_hidden_state`. All positions attend to all
    /// positions (single unpadded sequence — batch padding masks are the caller's concern, and a lone sequence
    /// needs none). token_type is all-zeros (segment A), matching how supar drives the encoder.
    pub fn hiddens(&self, ids: &[i64]) -> Vec<Vec<f32>> {
        let seq = ids.len();
        let hd = self.d / self.n_head;
        let mut x = self.b.rows_f32("wte", ids)
            + &self.b.rows_f32("wpe", &(0..seq as i64).collect::<Vec<_>>())
            + &self.b.rows_f32("wtt", &vec![0i64; seq]);
        x = self.ln(&x, "emb_ln");

        let mut snaps = Vec::with_capacity(self.n_layer + 1);
        snaps.push(x.iter().cloned().collect());

        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let q = self.lin(&x, &format!("{p}q"));
            let k = self.lin(&x, &format!("{p}k"));
            let v = self.lin(&x, &format!("{p}v"));
            let mut attn_out = Array2::<f32>::zeros((seq, self.d));
            for h in 0..self.n_head {
                let qh = q.slice(s![.., h * hd..(h + 1) * hd]);
                let kh = k.slice(s![.., h * hd..(h + 1) * hd]);
                let vh = v.slice(s![.., h * hd..(h + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt(); // bidirectional: no causal mask
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., h * hd..(h + 1) * hd]).assign(&scores.dot(&vh));
            }
            // post-LN residual blocks (BERT ordering): LN(x + sublayer(x))
            x = self.ln(&(&x + &self.lin(&attn_out, &format!("{p}ao"))), &format!("{p}ln1"));
            let mut hm = self.lin(&x, &format!("{p}fc"));
            gelu_erf(&mut hm);
            x = self.ln(&(&x + &self.lin(&hm, &format!("{p}out"))), &format!("{p}ln2"));
            snaps.push(x.iter().cloned().collect());
        }
        snaps
    }
}
