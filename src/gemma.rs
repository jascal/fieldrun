//! Tier B — composition, Gemma-2. A faithful Rust port of pylm's `numpy_gemma.py`, the hardest architecture: √d
//! embedding scale, a four-norm sandwich per layer (input / post-attention / pre-feedforward / post-feedforward, the
//! post-norms on the sub-layer output before the residual), attention-logit and final-logit soft-capping (tanh), GeGLU,
//! grouped-query attention with head_dim ≠ d/H, alternating sliding-window/full attention, and RMSNorm as x·(1+w) (the
//! +1 baked into the exported weights). Weights stay f16 in RAM and upcast per matmul (`arr2o`) — the in-RAM-precision
//! path, so Gemma-2-2b's 256k-vocab model fits ~half its f32 footprint.

use ndarray::{s, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Gemma {
    b: Bundle,
    n_layer: usize,
    h: usize,
    nkv: usize,
    hd: usize,
    d: usize,
    eps: f32,
    attn_cap: f32,
    final_cap: f32,
    scale: f32,
    escale: f32,
    inv: Vec<f32>,
    window: usize,
    route: f32,    // Tier C: fraction of MLP neurons per token (0 = off; int8 down falls back to dense)
    kv_int8: bool, // store the KV cache (GQA width) as int8 with a per-kv-head scale
}

fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
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

impl Gemma {
    pub fn new(b: Bundle, route: f32, kv_int8: bool) -> Gemma {
        // config: [n_layer, H, nkv, hd, d, ffn, vocab, tied]; config_f: [theta, eps, attn_cap, final_cap, qscalar, escale]
        let (n_layer, h, nkv, hd, d) = (b.config[0] as usize, b.config[1] as usize, b.config[2] as usize,
                                        b.config[3] as usize, b.config[4] as usize);
        let (theta, eps, attn_cap, final_cap, qscalar, escale) = (b.config_f[0] as f32, b.config_f[1] as f32,
            b.config_f[2] as f32, b.config_f[3] as f32, b.config_f[4] as f32, b.config_f[5] as f32);
        let inv = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        Gemma { b, n_layer, h, nkv, hd, d, eps, attn_cap, final_cap, scale: qscalar.powf(-0.5), escale, inv, window: 4096, route, kv_int8 }
    }

    fn down(&self, h: &Array2<f32>, name: &str) -> Array2<f32> {
        if self.route > 0.0 && self.route < 1.0 {
            self.b.mm_routed_down(h, name, self.route)
        } else {
            self.b.mm(h, name)
        }
    }

    fn norm(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let w = self.b.arr1o(name); // (1+w) baked at export
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

    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let emb = self.b.rows_f32("embed", ids); // upcast only the looked-up rows, not the whole 256k table
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale)); // Gemma scales the input embedding by √d
        }

        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                if self.attn_cap > 0.0 {
                    scores.mapv_inplace(|s| self.attn_cap * (s / self.attn_cap).tanh()); // attn-logit soft-cap
                }
                let sliding = l % 2 == 0; // even layers use the sliding window; odd are full attention
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i || (sliding && i >= self.window && j <= i - self.window) {
                            scores[[i, j]] = -1e30;
                        }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            x = &x + &self.norm(&o, &format!("{p}post_attention_layernorm")); // post-norm before the residual

            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = gelu_tanh(*hv) * uv;
            }
            let mlp = self.down(&hidden, &format!("{p}mlp.down_proj"));
            x = &x + &self.norm(&mlp, &format!("{p}post_feedforward_layernorm"));
        }
        self.norm(&x, "norm")
    }

    fn explanation(&self, ids: &[i64]) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let emb = self.b.rows_f32("embed", ids);
        let mut x = Array2::<f32>::zeros((seq, self.d));
        for t in 0..seq {
            x.row_mut(t).assign(&(&emb.row(t) * self.escale));
        }
        let mut att_last: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut mlp_h: Vec<Vec<f32>> = Vec::new();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let sliding = l % 2 == 0;
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            let mut layer_att = Vec::with_capacity(h);
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                if self.attn_cap > 0.0 {
                    scores.mapv_inplace(|s| self.attn_cap * (s / self.attn_cap).tanh());
                }
                for i in 0..seq {
                    for j in 0..seq {
                        if j > i || (sliding && i >= self.window && j <= i - self.window) {
                            scores[[i, j]] = -1e30;
                        }
                    }
                }
                softmax_rows(&mut scores);
                layer_att.push(scores.row(seq - 1).to_vec());
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            att_last.push(layer_att);
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            x = &x + &self.norm(&o, &format!("{p}post_attention_layernorm"));
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = gelu_tanh(*hv) * uv;
            }
            mlp_h.push(hidden.row(seq - 1).to_vec());
            let mlp = self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
            x = &x + &self.norm(&mlp, &format!("{p}post_feedforward_layernorm"));
        }
        let xf = self.norm(&x, "norm");
        let lg = self.b.rowdot_f32("embed", &xf.row(seq - 1).to_vec());
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        assemble(ids, &att_last, &mlp_h, model_predicts, |l, n, act| {
            let w_out = self.b.weight_row(&format!("l{l}.mlp.down_proj"), n);
            top_promoted(&self.b.rowdot_f32("embed", &w_out), act, 5)
        })
    }

    /// Run `m` new positions (rows of `emb`, already √d-scaled) through the layers, caching K/V (post-RoPE, GQA width
    /// nkv*hd) and attending over the cache with Gemma's soft-cap + sliding window. cur = absolute first-row position.
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, h, cur);
            self.rope(&mut k, nkv, cur);
            kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
            vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            let sliding = l % 2 == 0;
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = kc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let vh = vc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                if self.attn_cap > 0.0 {
                    scores.mapv_inplace(|s| self.attn_cap * (s / self.attn_cap).tanh());
                }
                for i in 0..m {
                    let abs = cur + i; // absolute position of this query row
                    for j in 0..klen {
                        if j > abs || (sliding && abs >= self.window && j <= abs - self.window) {
                            scores[[i, j]] = -1e30;
                        }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            x = &x + &self.norm(&o, &format!("{p}post_attention_layernorm"));
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = gelu_tanh(*hv) * uv;
            }
            let mlp = self.down(&hidden, &format!("{p}mlp.down_proj"));
            x = &x + &self.norm(&mlp, &format!("{p}post_feedforward_layernorm"));
        }
        self.norm(&x, "norm")
    }

    /// `forward_block` with an int8 KV cache (GQA width, per-kv-head scale), keeping Gemma's soft-cap + sliding window.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_q(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>],
                       vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]) -> Array2<f32> {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let kvdim = nkv * hd;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let q8 = |v: f32, sc: f32| (v / sc).round().clamp(-127.0, 127.0) as i8;
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}input_layernorm"));
            let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, h, cur);
            self.rope(&mut k, nkv, cur);
            for i in 0..m {
                let pos = cur + i;
                for kh in 0..nkv {
                    let base = kh * hd;
                    let sck = ((0..hd).fold(0f32, |mx, c| mx.max(k[[i, base + c]].abs())) / 127.0).max(1e-8);
                    let scv = ((0..hd).fold(0f32, |mx, c| mx.max(v[[i, base + c]].abs())) / 127.0).max(1e-8);
                    ks[l][pos * nkv + kh] = sck;
                    vs[l][pos * nkv + kh] = scv;
                    for c in 0..hd {
                        kc[l][pos * kvdim + base + c] = q8(k[[i, base + c]], sck);
                        vc[l][pos * kvdim + base + c] = q8(v[[i, base + c]], scv);
                    }
                }
            }
            let sliding = l % 2 == 0;
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let mut kh_a = Array2::<f32>::zeros((klen, hd));
                let mut vh_a = Array2::<f32>::zeros((klen, hd));
                for pos in 0..klen {
                    let (sck, scv) = (ks[l][pos * nkv + kv], vs[l][pos * nkv + kv]);
                    for c in 0..hd {
                        kh_a[[pos, c]] = kc[l][pos * kvdim + kv * hd + c] as f32 * sck;
                        vh_a[[pos, c]] = vc[l][pos * kvdim + kv * hd + c] as f32 * scv;
                    }
                }
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let mut scores = qh.dot(&kh_a.t()) * self.scale;
                if self.attn_cap > 0.0 {
                    scores.mapv_inplace(|s| self.attn_cap * (s / self.attn_cap).tanh());
                }
                for i in 0..m {
                    let abs = cur + i;
                    for j in 0..klen {
                        if j > abs || (sliding && abs >= self.window && j <= abs - self.window) {
                            scores[[i, j]] = -1e30;
                        }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh_a));
            }
            let o = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            x = &x + &self.norm(&o, &format!("{p}post_attention_layernorm"));
            let a2 = self.norm(&x, &format!("{p}pre_feedforward_layernorm"));
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = gelu_tanh(*hv) * uv;
            }
            let mlp = self.down(&hidden, &format!("{p}mlp.down_proj"));
            x = &x + &self.norm(&mlp, &format!("{p}post_feedforward_layernorm"));
        }
        self.norm(&x, "norm")
    }

    fn generate_kv_int8(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let total = prompt.len() + n_new;
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|_| vec![0i8; total * kvdim]).collect();
        let mut vc = kc.clone();
        let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|_| vec![0f32; total * self.nkv]).collect();
        let mut vs = ks.clone();
        let mut emb = self.b.rows_f32("embed", prompt) * self.escale;
        let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            emb = self.b.rows_f32("embed", &[next]) * self.escale;
            let xb = self.forward_block_q(&emb, pos, &mut kc, &mut ks, &mut vc, &mut vs);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }

    fn head_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let last = xfn.row(xfn.nrows() - 1).to_vec();
        let logits = self.b.rowdot_f32("embed", &last); // tied unembed, streamed f16; softcap is monotone → skip
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}

impl Model for Gemma {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32("embed", &last); // tied unembed, streamed f16 (no (vocab, d) f32 alloc)
        // final-logit soft-cap is a monotone tanh → argmax unchanged, so skip it for predict
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn explain(&self, ids: &[i64]) -> Option<crate::explain::Explanation> {
        Some(self.explanation(ids))
    }

    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        if self.kv_int8 {
            return self.generate_kv_int8(prompt, n_new);
        }
        let total = prompt.len() + n_new;
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let pe = self.b.rows_f32("embed", prompt); // √d-scaled prompt embeddings
        let mut emb = pe * self.escale;
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            emb = self.b.rows_f32("embed", &[next]) * self.escale;
            let xb = self.forward_block(&emb, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }

    // KV-cached streaming (early-stop at eos + per-token emit) for chat/serve; mirrors `generate`'s f32 loop.
    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let total = prompt.len() + max_tokens;
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc = kc.clone();
        let mut emb = self.b.rows_f32("embed", prompt) * self.escale;
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
            emb = self.b.rows_f32("embed", &[next]) * self.escale;
            let xb = self.forward_block(&emb, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
        out
    }
}
