//! Tier B — composition, MiniMax-M2. The RoPE backbone (RMSNorm, single-base RoPE, GQA, standard two-norm pre-norm)
//! with two MiniMax specifics: **full-width q/k-norm** (a single RMSNorm over the whole nh·head_dim / nkv·head_dim
//! projection output, before the head reshape + RoPE — not the per-head QK-norm of Qwen3/Gemma), and an all-MoE FFN
//! with a **sigmoid router** (sigmoid scores + a bias-correction buffer pick the experts; the sigmoid scores,
//! renormalised over the top-k, are the weights — no group limiting, no shared expert). SwiGLU experts read on demand
//! from the mmap (offload). A faithful port of `MiniMaxM2ForCausalLM`. Incremental KV-cache `generate`/`generate_stream`
//! (f32 + int8-KV) and `explain` (live head circuits + the dominant expert's promoted features) are wired.

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
    kv_int8: bool, // store the KV cache (GQA width) as int8 with a per-kv-head scale during generate
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
    pub fn new(b: Bundle, _route: f32, kv_int8: bool) -> MiniMax {
        // config: [nl, nh, nkv, hd, d, vocab, tied, n_exp, topk, inter]
        let c = &b.config;
        let (nl, nh, nkv, hd) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
        let tied = c[6] != 0;
        let (n_exp, topk) = (c[7] as usize, c[8] as usize);
        let (theta, eps) = (b.config_f[0] as f32, b.config_f[1] as f32);
        let inv = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        MiniMax { b, nl, nh, nkv, hd, eps, scale: (hd as f32).powf(-0.5), inv, n_exp, topk, tied, kv_int8 }
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

    /// Rotary embedding on a (m, n_heads*hd) block; row i is absolute position `pos0 + i` (pos0 > 0 for KV-cache decode).
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
        // Prefetch every active expert's weights up front (MADV_WILLNEED) so the OS pages experts 2..k from the mmap
        // while expert 1 is computed — overlapping the per-token page-in stalls that bound MoE decode under offload.
        for &e in assign.keys() {
            self.b.prefetch(&format!("{p}experts.{e}.gate"));
            self.b.prefetch(&format!("{p}experts.{e}.up"));
            self.b.prefetch(&format!("{p}experts.{e}.down"));
        }
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
            self.rope(&mut q, nh, 0);
            self.rope(&mut k, nkv, 0);
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

    /// For explain: the dominant expert (highest routed choice) for one token's post-attention-normed hidden `row`
    /// (a (1, d) array), and that expert's SwiGLU hidden activations — names the "MLP feature" of an all-MoE layer.
    fn top_expert_feature(&self, l: usize, row: &Array2<f32>) -> (usize, Vec<f32>) {
        let p = format!("l{l}.");
        let logits = self.b.mm(row, &format!("{p}gate"));
        let bias = self.b.arr1o(&format!("{p}gate_bias"));
        let sig: Vec<f32> = logits.row(0).iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        let choice: Vec<f32> = sig.iter().zip(bias.iter()).map(|(s, b)| s + b).collect();
        let e = (0..self.n_exp).max_by(|&a, &b| choice[a].partial_cmp(&choice[b]).unwrap()).unwrap();
        let gate = self.b.expert_mm(row, &format!("{p}experts.{e}.gate"));
        let up = self.b.expert_mm(row, &format!("{p}experts.{e}.up"));
        let mut hh = gate;
        for (h, u) in hh.iter_mut().zip(up.iter()) { *h = silu(*h) * u; }
        (e, hh.row(0).to_vec())
    }

    fn explanation(&self, ids: &[i64]) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let rep = nh / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        let mut att_last: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut head_act: Vec<Vec<f32>> = Vec::new();
        let mut mlp_h: Vec<Vec<f32>> = Vec::new();
        let mut top_expert: Vec<usize> = Vec::new();
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.norm(&self.b.mm(&a, &format!("{p}self_attn.q_proj")), &format!("{p}q_norm"));
            let mut k = self.norm(&self.b.mm(&a, &format!("{p}self_attn.k_proj")), &format!("{p}k_norm"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, nh, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, nh * hd));
            let mut layer_att = Vec::with_capacity(nh);
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
                layer_att.push(scores.row(seq - 1).to_vec());
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            att_last.push(layer_att);
            head_act.push(attn_out.row(seq - 1).to_vec());
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            let last = a2.slice(s![seq - 1..seq, ..]).to_owned();
            let (e, hidden) = self.top_expert_feature(l, &last);
            top_expert.push(e);
            mlp_h.push(hidden);
            x = &x + &self.moe(l, &a2);
        }
        let xf = self.norm(&x, "norm");
        let un = self.unembed();
        let lg = self.b.rowdot_f32(un, &xf.row(seq - 1).to_vec());
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let gain = self.b.arr1("norm").to_vec();
        assemble(
            ids,
            &att_last,
            &mlp_h,
            model_predicts,
            |l, n, act| {
                let w_out = self.b.expert_row(&format!("l{l}.experts.{}.down", top_expert[l]), n);
                top_promoted(&self.b.rowdot_f32(un, &w_out), act, 5)
            },
            |l, head| head_dla(&self.b, &format!("l{l}.self_attn.o_proj"), un, &head_act[l], head, hd, &gain, false, 5),
        )
    }

    /// Run `m` new positions through the layers, caching K/V (post-RoPE, GQA width nkv*hd) and attending over the whole
    /// cache. cur = absolute position of the first new row. Prefill (m = prompt, cur = 0) or decode (m = 1). The MoE FFN
    /// runs only on the new rows; its experts page in from the mmap per token (the cache holds no expert state).
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let rep = nh / nkv;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.norm(&self.b.mm(&a, &format!("{p}self_attn.q_proj")), &format!("{p}q_norm"));
            let mut k = self.norm(&self.b.mm(&a, &format!("{p}self_attn.k_proj")), &format!("{p}k_norm"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, nh, cur);
            self.rope(&mut k, nkv, cur);
            kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
            vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            let mut attn_out = Array2::<f32>::zeros((m, nh * hd));
            for head in 0..nh {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = kc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let vh = vc[l].slice(s![0..klen, kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * self.scale;
                for i in 0..m {
                    for j in (cur + i + 1)..klen { scores[[i, j]] = -1e30; }
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

    /// `forward_block` with an int8 KV cache (GQA width, per-kv-head scale): ~4x smaller cache, ~identical tokens.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_q(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>],
                       vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]) -> Array2<f32> {
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let rep = nh / nkv;
        let kvdim = nkv * hd;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let q8 = |v: f32, sc: f32| (v / sc).round().clamp(-127.0, 127.0) as i8;
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = self.norm(&x, &format!("{p}in_ln"));
            let mut q = self.norm(&self.b.mm(&a, &format!("{p}self_attn.q_proj")), &format!("{p}q_norm"));
            let mut k = self.norm(&self.b.mm(&a, &format!("{p}self_attn.k_proj")), &format!("{p}k_norm"));
            let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
            self.rope(&mut q, nh, cur);
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
            let mut attn_out = Array2::<f32>::zeros((m, nh * hd));
            for head in 0..nh {
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
                for i in 0..m {
                    for j in (cur + i + 1)..klen { scores[[i, j]] = -1e30; }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh_a));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = self.norm(&x, &format!("{p}post_ln"));
            x = &x + &self.moe(l, &a2);
        }
        self.norm(&x, "norm")
    }

    fn generate_kv_int8(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let total = prompt.len() + n_new;
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Vec<i8>> = (0..self.nl).map(|_| vec![0i8; total * kvdim]).collect();
        let mut vc = kc.clone();
        let mut ks: Vec<Vec<f32>> = (0..self.nl).map(|_| vec![0f32; total * self.nkv]).collect();
        let mut vs = ks.clone();
        let emb = self.b.rows_f32("embed", prompt);
        let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }

    fn head_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let logits = self.b.rowdot_f32(self.unembed(), &xfn.row(xfn.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}

impl Model for MiniMax {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let last = xf.row(ids.len() - 1).to_vec();
        let logits = self.b.rowdot_f32(self.unembed(), &last);
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
        let mut kc: Vec<Array2<f32>> = (0..self.nl).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.nl).map(|_| Array2::zeros((total, kvdim))).collect();
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

    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let total = prompt.len() + max_tokens;
        let kvdim = self.nkv * self.hd;
        if self.kv_int8 {
            // int8 KV cache for chat/serve — 4x smaller cache (longer context in the same budget); lossy by design.
            let mut kc: Vec<Vec<i8>> = (0..self.nl).map(|_| vec![0i8; total * kvdim]).collect();
            let mut vc = kc.clone();
            let mut ks: Vec<Vec<f32>> = (0..self.nl).map(|_| vec![0f32; total * self.nkv]).collect();
            let mut vs = ks.clone();
            let emb = self.b.rows_f32("embed", prompt);
            let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
            let mut next = self.head_argmax(&xb);
            let mut out = Vec::new();
            let mut pos = prompt.len();
            loop {
                if eos.contains(&next) { break; }
                out.push(next);
                if !emit(next) || out.len() == max_tokens { break; }
                let e = self.b.rows_f32("embed", &[next]);
                let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
                next = self.head_argmax(&xb);
                pos += 1;
            }
            return out;
        }
        let mut kc: Vec<Array2<f32>> = (0..self.nl).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc = kc.clone();
        let emb = self.b.rows_f32("embed", prompt);
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
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
        out
    }

    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool, cache: &mut crate::model::PrefixKv) -> Vec<i64> {
        if self.kv_int8 {
            let (kvdim, nkv, n_layer) = (self.nkv * self.hd, self.nkv, self.nl);
            let alloc = |total: usize| {
                let kc: Vec<Vec<i8>> = (0..n_layer).map(|_| vec![0i8; total * kvdim]).collect();
                let vc = kc.clone();
                let ks: Vec<Vec<f32>> = (0..n_layer).map(|_| vec![0f32; total * nkv]).collect();
                let vs = ks.clone();
                (kc, vc, ks, vs)
            };
            let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>], vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]| {
                let emb = self.b.rows_f32("embed", ids);
                self.forward_block_q(&emb, cur, kc, ks, vc, vs)
            };
            return crate::model::prefix_generate_q(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb| self.head_argmax(xb));
        }
        let (kvdim, n_layer) = (self.nkv * self.hd, self.nl);
        let alloc = |total: usize| {
            let kc: Vec<Array2<f32>> = (0..n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
            let vc = kc.clone();
            (kc, vc)
        };
        let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]| {
            let emb = self.b.rows_f32("embed", ids);
            self.forward_block(&emb, cur, kc, vc)
        };
        crate::model::prefix_generate(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb| self.head_argmax(xb))
    }
}
