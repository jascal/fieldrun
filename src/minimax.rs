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
    /// `scan = Some((expert, neuron))` additionally returns the per-row value of that ONE routed unit —
    /// `w_router · silu(gate_n) · up_n` — recorded in the same pass so the expert's weights are read once.
    /// Rows whose top-k does not contain `expert` stay `NaN`: the unit is not computed there at all, so it has
    /// no value rather than a value of zero (see `Model::blind_moe_*`). NB the router here is the MiniMax one:
    /// sigmoid scores, a bias that moves only the top-k CHOICE, and a weight of `sig_e / sum_topk(sig)`.
    fn moe(&self, l: usize, a2: &Array2<f32>, scan: Option<(usize, usize)>) -> (Array2<f32>, Vec<f32>) {
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
        let mut scan_out: Vec<f32> = scan.map(|_| vec![f32::NAN; seq]).unwrap_or_default();
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
                if let Some((se, sn)) = scan {
                    if *e == se && sn < hh.ncols() {
                        scan_out[t] = w * hh[[i, sn]];
                    }
                }
                for cc in 0..out.ncols() { out[[t, cc]] += w * down[[i, cc]]; }
            }
        }
        (out, scan_out)
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
            let (mo, _) = self.moe(l, &a2, None);
            x = &x + &mo;
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
            let (mo, _) = self.moe(l, &a2, None);
            x = &x + &mo;
        }
        let xf = self.norm(&x, "norm");
        let x_last = x.row(seq - 1).to_vec(); // residual either side of the final norm — recovers its frozen scale
        let xf_last = xf.row(seq - 1).to_vec();
        let un = self.unembed();
        let lg = self.b.rowdot_f32(un, &xf_last);
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let gain = self.b.arr1("norm").to_vec();
        let u_pred = self.b.weight_row(un, model_predicts as usize);
        assemble(
            ids,
            &att_last,
            &head_act,
            &mlp_h,
            &lg,
            model_predicts,
            &gain,
            false,
            &[],
            &x_last,
            &xf_last,
            &u_pred,
            0,
            &|_v| Vec::new(),
            |l, n| self.b.expert_row(&format!("l{l}.experts.{}.down", top_expert[l]), n),
            |l, head| head_raw_contrib(&self.b, &format!("l{l}.self_attn.o_proj"), &head_act[l], head, hd),
            |c| self.b.rowdot_f32(un, c),
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
            let (mo, _) = self.moe(l, &a2, None);
            x = &x + &mo;
        }
        self.norm(&x, "norm")
    }

    /// `forward_block` that also returns the per-row value of ONE routed unit at the target layer — the
    /// blind-max scoring kernel. Identical walk and arithmetic to `forward_block` (same full-width q/k norm,
    /// RoPE positions, causal mask and routed FFN), so its value is bit-identical to a naive per-position
    /// forward; the only difference is that it ADVANCES the KV cache, which is why the scorer snapshots
    /// before probing a candidate.
    fn forward_block_scan(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>],
                          scan: Option<(usize, usize, usize)>) -> Vec<f32> {
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let rep = nh / nkv;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let mut scan_out: Vec<f32> = Vec::new();
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
            let (mo, vals) = self.moe(l, &a2, scan.filter(|(sl, ..)| *sl == l).map(|(_, e, n)| (e, n)));
            if !vals.is_empty() { scan_out = vals; }
            x = &x + &mo;
        }
        scan_out
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
            let (mo, _) = self.moe(l, &a2, None);
            x = &x + &mo;
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

/// The blind-max scorer for this arch: a KV-cache twin of [`MiniMax::forward_block_scan`] held at position `pos`,
/// with a snapshot so candidate tokens never see each other's writes.
struct MiniMaxScorer {
    kc: Vec<Array2<f32>>,
    vc: Vec<Array2<f32>>,
    snap_kc: Vec<Array2<f32>>,
    snap_vc: Vec<Array2<f32>>,
    pos: usize,
}

impl MiniMaxScorer {
    /// Prefill `prefix`, then score candidates at the next position.
    fn new(m: &MiniMax, prefix: &[i64], total: usize) -> Option<MiniMaxScorer> {
        if prefix.is_empty() || prefix.len() >= total {
            return None;
        }
        let (kc, vc) = MiniMaxScorer::alloc(m, total);
        let mut kc = kc;
        let mut vc = vc;
        let emb = m.b.rows_f32("embed", prefix);
        m.forward_block(&emb, 0, &mut kc, &mut vc);
        let snap_kc = kc.clone();
        let snap_vc = vc.clone();
        Some(MiniMaxScorer { kc, vc, snap_kc, snap_vc, pos: prefix.len() })
    }

    /// An empty scorer at position 0 — no seed context, for the corpus scan, which places every token.
    fn empty(m: &MiniMax, total: usize) -> MiniMaxScorer {
        let (kc, vc) = MiniMaxScorer::alloc(m, total);
        let snap_kc = kc.clone();
        let snap_vc = vc.clone();
        MiniMaxScorer { kc, vc, snap_kc, snap_vc, pos: 0 }
    }

    fn alloc(m: &MiniMax, total: usize) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let kvdim = m.nkv * m.hd;
        let kc: Vec<Array2<f32>> = (0..m.nl).map(|_| Array2::zeros((total, kvdim))).collect();
        let vc = kc.clone();
        (kc, vc)
    }

    /// Score `token` at `pos` by the ROUTED unit's `w_router · act`, WITHOUT committing. `None` for a bad
    /// unit; `Some(NaN)` when the router did not select that expert here — no value, not a value of zero.
    fn try_token_moe(&mut self, m: &MiniMax, token: i64, layer: usize, expert: usize, neuron: usize) -> Option<f32> {
        if layer >= m.nl || expert >= m.n_exp {
            return None;
        }
        self.kc.clone_from(&self.snap_kc); // candidates must not see each other's writes
        self.vc.clone_from(&self.snap_vc);
        let emb = m.b.rows_f32("embed", &[token]);
        let vals = m.forward_block_scan(&emb, self.pos, &mut self.kc, &mut self.vc, Some((layer, expert, neuron)));
        vals.first().copied()
    }

    /// Advance the cache by `token` (the winner's value was already read by the probe).
    fn commit(&mut self, m: &MiniMax, token: i64) {
        let emb = m.b.rows_f32("embed", &[token]);
        m.forward_block(&emb, self.pos, &mut self.kc, &mut self.vc);
        self.pos += 1;
        self.snap_kc.clone_from(&self.kc);
        self.snap_vc.clone_from(&self.vc);
    }
}

impl Model for MiniMax {
    /// The scorer IS implemented here, so an all-MoE arch must not report `NoScorer` — it is `RoutedLayers`:
    /// a token's router picks which experts run, so "neuron n" of a layer has no fixed meaning across
    /// positions, while expert `e`'s neuron `n` DOES (`--blind-max L,E,N`). Naming which of the two
    /// problems it is, is the whole point of this method.
    fn blind_support(&self) -> crate::model::BlindSupport {
        if self.n_exp != 0 { crate::model::BlindSupport::RoutedLayers } else { crate::model::BlindSupport::Supported }
    }

    /// One ROUTED unit's self-description: expert `e` neuron `n`'s own output row (`experts.{e}.down`),
    /// scaled by the final-norm gain and projected through the unembed — the same construction
    /// `blind_proposals` uses for a dense `mlp.down_proj`, so the two are comparable.
    fn blind_moe_proposals(&self, layer: usize, expert: usize, neuron: usize, k: usize, polarity: f32) -> Option<Vec<i64>> {
        if layer >= self.nl || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        // Expert weights are PAGED, not resident: `weight_row` cannot see them, so go through
        // `expert_f32`, which returns (shape, values) as (in, out) row-major. down is (inter, d), so
        // neuron `n`'s output direction is one contiguous d-wide row of it.
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
    fn blind_moe_climb(&self, layer: usize, expert: usize, neuron: usize, prefix: &[i64], cands: &[i64],
                       steps: usize, polarity: f32) -> Option<crate::model::BlindArm> {
        if cands.is_empty() || layer >= self.nl || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let mut sc = MiniMaxScorer::new(self, prefix, prefix.len() + steps + 1)?;
        let mut ids = prefix.to_vec();
        let mut best = f32::NEG_INFINITY;
        for _ in 0..steps {
            let mut pick = None;
            let mut pick_obj = f32::NEG_INFINITY;
            for &t in cands {
                match sc.try_token_moe(self, t, layer, expert, neuron) {
                    Some(v) if !v.is_nan() && polarity * v > pick_obj => {
                        pick_obj = polarity * v;
                        pick = Some(t);
                    }
                    _ => {}
                }
            }
            let Some(t) = pick else { break }; // no candidate reaches this expert on this position
            sc.commit(self, t);
            ids.push(t);
            best = best.max(pick_obj);
        }
        if best == f32::NEG_INFINITY {
            return None;
        }
        Some(crate::model::BlindArm { ids, obj: best, pos: None })
    }

    /// Oracle for one routed unit over a corpus: best `w_router · act` among positions that route to
    /// `expert`. Chunked exactly like the dense arm (one weight read amortised over CORPUS_CHUNK
    /// positions, each still scored with true causal history) and advancing the same KV cache.
    fn blind_moe_corpus_best(&self, layer: usize, expert: usize, neuron: usize, ids: &[i64],
                             polarity: f32) -> Option<crate::model::BlindArm> {
        if ids.len() < 2 || layer >= self.nl || self.n_exp == 0 || expert >= self.n_exp {
            return None;
        }
        let mut sc = MiniMaxScorer::empty(self, ids.len());
        let (mut best_obj, mut best_idx, mut routed) = (f32::NEG_INFINITY, 0usize, 0usize);
        for chunk in ids.chunks(crate::model::CORPUS_CHUNK) {
            let start = sc.pos;
            let emb = self.b.rows_f32("embed", chunk);
            let vals = self.forward_block_scan(&emb, start, &mut sc.kc, &mut sc.vc, Some((layer, expert, neuron)));
            sc.pos += chunk.len();
            if vals.len() != chunk.len() {
                continue;
            }
            for (j, &v) in vals.iter().enumerate() {
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
            return crate::model::prefix_generate_q(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, _ctx| self.head_argmax(xb));
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
        crate::model::prefix_generate(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, _ctx| self.head_argmax(xb))
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

    /// A tiny all-MoE `minimax` bundle written straight to disk. Each test gets its OWN path — `cargo test`
    /// runs these concurrently, so a shared path would let one test load another's half-written file.
    fn build_tiny(tag: &str) -> MiniMax {
        let (nl, nh, nkv, hd, d, vocab, inter) = (4usize, 4, 2, 8, 32, 24, 32);
        let n_exp = 4usize;
        let dir = std::env::temp_dir().join(format!("fr_bm_mm_{tag}_{}", std::process::id()));
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
        add("lm_head".into(), vec![vocab, d], v(3, vocab * d));
        for l in 0..nl {
            let p = format!("l{l}.");
            add(format!("{p}in_ln"), vec![d], v(10 + l, d));
            add(format!("{p}post_ln"), vec![d], v(20 + l, d));
            add(format!("{p}self_attn.q_proj"), vec![d, nh * hd], v(300 + l, d * nh * hd));
            add(format!("{p}self_attn.k_proj"), vec![d, nkv * hd], v(310 + l, d * nkv * hd));
            add(format!("{p}self_attn.v_proj"), vec![d, nkv * hd], v(320 + l, d * nkv * hd));
            add(format!("{p}self_attn.o_proj"), vec![nh * hd, d], v(330 + l, nh * hd * d));
            // MiniMax norms q/k over the FULL projection width, not per head
            add(format!("{p}q_norm"), vec![nh * hd], v(340 + l, nh * hd));
            add(format!("{p}k_norm"), vec![nkv * hd], v(350 + l, nkv * hd));
            add(format!("{p}gate"), vec![d, n_exp], v(100 + l, d * n_exp));
            add(format!("{p}gate_bias"), vec![n_exp], v(400 + l, n_exp));
            for e in 0..n_exp {
                // `.experts.` in the name is what puts these in the PAGED map `expert_f32` reads.
                add(format!("{p}experts.{e}.gate"), vec![d, inter], v(110 + e, d * inter));
                add(format!("{p}experts.{e}.up"), vec![d, inter], v(120 + e, d * inter));
                add(format!("{p}experts.{e}.down"), vec![inter, d], v(130 + e, inter * d));
            }
        }
        let cfg: Vec<i64> = vec![nl as i64, nh as i64, nkv as i64, hd as i64, d as i64, vocab as i64,
                                    0, n_exp as i64, 2, inter as i64];
        let manifest = serde_json::json!({ "format": crate::bundle::FORMAT, "version": crate::bundle::VERSION,
            "arch": "minimax", "config": cfg, "config_f": [10000.0, 1e-5], "arrays": specs });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
        MiniMax::new(Bundle::load(&stem).unwrap(), 0.0, false)
    }

    /// Naive re-walk: the post-attention-normed hidden entering layer `l`'s MoE, with NO cache. An
    /// independent copy of the attention walk, so the cached scorer has something honest to be compared to.
    fn naive_a2(m: &MiniMax, ids: &[i64], l: usize) -> Array2<f32> {
        let seq = ids.len();
        let (nh, nkv, hd) = (m.nh, m.nkv, m.hd);
        let rep = nh / nkv;
        let mut x = m.b.rows_f32("embed", ids);
        // Layers strictly BEFORE l, each run whole (attention + routed FFN).
        for j in 0..l {
            let p = format!("l{j}.");
            let a = m.norm(&x, &format!("{p}in_ln"));
            let mut q = m.norm(&m.b.mm(&a, &format!("{p}self_attn.q_proj")), &format!("{p}q_norm"));
            let mut k = m.norm(&m.b.mm(&a, &format!("{p}self_attn.k_proj")), &format!("{p}k_norm"));
            let v = m.b.mm(&a, &format!("{p}self_attn.v_proj"));
            m.rope(&mut q, nh, 0);
            m.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, nh * hd));
            for head in 0..nh {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) * m.scale;
                for i in 0..seq {
                    for j2 in 0..seq {
                        if j2 > i { scores[[i, j2]] = -1e30; }
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &m.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = m.norm(&x, &format!("{p}post_ln"));
            let (mo, _) = m.moe(j, &a2, None);
            x = &x + &mo;
        }
        // Layer l: attention only — its FFN input `a2` is what we return, so the FFN must NOT run.
        let p = format!("l{l}.");
        let a = m.norm(&x, &format!("{p}in_ln"));
        let mut q = m.norm(&m.b.mm(&a, &format!("{p}self_attn.q_proj")), &format!("{p}q_norm"));
        let mut k = m.norm(&m.b.mm(&a, &format!("{p}self_attn.k_proj")), &format!("{p}k_norm"));
        let v = m.b.mm(&a, &format!("{p}self_attn.v_proj"));
        m.rope(&mut q, nh, 0);
        m.rope(&mut k, nkv, 0);
        let mut attn_out = Array2::<f32>::zeros((seq, nh * hd));
        for head in 0..nh {
            let kv = head / rep;
            let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
            let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
            let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
            let mut scores = qh.dot(&kh.t()) * m.scale;
            for i in 0..seq {
                for j2 in 0..seq {
                    if j2 > i { scores[[i, j2]] = -1e30; }
                }
            }
            softmax_rows(&mut scores);
            attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
        }
        x = &x + &m.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
        m.norm(&x, &format!("{p}post_ln"))
    }

    /// The routed probe rests on this: one cached scorer step over the KV state must equal a naive
    /// from-scratch forward over the same ids, bit for bit. If the snapshot/restore path drifted by even one
    /// ULP, every blind-max ratio on this arch would be measured against a fiction.
    #[test]
    fn minimax_scorer_step_matches_naive_bitwise() {
        let m = build_tiny("step");
        let ids: Vec<i64> = vec![1, 9, 4, 7, 2, 5];
        let total = ids.len() + 6;
        for layer in [0usize, 2, 3] {
            let mut sc = MiniMaxScorer::new(&m, &ids, total).unwrap();
            assert_eq!(sc.pos, ids.len());
            for (i, &tok) in [7i64, 3, 11].iter().enumerate() {
                let (e, n) = ((i + layer) % m.n_exp, (i * 5) % 32);
                let got = sc.try_token_moe(&m, tok, layer, e, n).unwrap();
                let mut want_ids = ids.clone();
                want_ids.push(tok);
                let a2 = naive_a2(&m, &want_ids, layer);
                let (_, want) = m.moe(layer, &a2, Some((e, n)));
                let want = want[want.len() - 1];
                if want.is_nan() {
                    assert!(got.is_nan(), "layer {layer}: expert not selected, so both must be NaN");
                } else {
                    assert_eq!(got.to_bits(), want.to_bits(),
                        "layer {layer} tok {tok}: scorer {:e} != naive {:e}", got, want);
                }
            }
        }
    }

    /// Candidates must be INDEPENDENT: probing B after A must equal probing B on a fresh scorer.
    #[test]
    fn minimax_candidates_are_independent() {
        let m = build_tiny("indep");
        let ids: Vec<i64> = vec![2, 4, 6];
        let (l, e, n) = (1usize, 2usize, 3usize);
        let mut sc = MiniMaxScorer::new(&m, &ids, 32).unwrap();
        let _ = sc.try_token_moe(&m, 11, l, e, n);
        let after_a = sc.try_token_moe(&m, 7, l, e, n).unwrap();
        let mut fresh = MiniMaxScorer::new(&m, &ids, 32).unwrap();
        let on_fresh = fresh.try_token_moe(&m, 7, l, e, n).unwrap();
        assert_eq!(after_a.to_bits(), on_fresh.to_bits(),
            "probing A then B differs from probing B alone: candidates leak into each other");
    }

    /// The routed objective must equal `w_router · silu(gate_n) · up_n` written out by hand — with THIS
    /// arch's router: sigmoid scores, a bias that moves only the top-k CHOICE, and weights `sig_e / sum_topk(sig)`.
    /// A top-k chosen on the bias but weighted on the sigmoid is the bug this catches.
    #[test]
    fn minimax_routed_objective_matches_reference() {
        let m = build_tiny("ref");
        let (l, n) = (1usize, 7usize);
        // Long enough (and every expert checked) that a top-k of 2 out of 4 actually routes each expert
        // somewhere — a 6-token stream can leave one expert never selected, which would make the
        // reference below vacuous rather than wrong.
        let stream: Vec<i64> = (0..16).map(|i: i64| i % 7).collect();
        let a2 = naive_a2(&m, &stream, l);
        let logits = m.b.mm(&a2, &format!("l{l}.gate"));
        let bias = m.b.arr1o(&format!("l{l}.gate_bias"));
        let mut n_checked = 0;
        for e in 0..m.n_exp {
            let (_out, got) = m.moe(l, &a2, Some((e, n)));
            let (_, gv) = m.b.expert_f32(&format!("l{l}.experts.{e}.gate"));
            let (_, uv) = m.b.expert_f32(&format!("l{l}.experts.{e}.up"));
            let (dsh, _) = m.b.expert_f32(&format!("l{l}.experts.{e}.down"));
            let (d, inter) = (dsh[1], dsh[0]);
            assert_eq!(got.len(), stream.len(), "one value per position");
            let mut n_routed = 0;
            for t in 0..stream.len() {
                let sig: Vec<f32> = logits.row(t).iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                let choice: Vec<f32> = sig.iter().zip(bias.iter()).map(|(s, b)| s + b).collect();
                let mut idx: Vec<usize> = (0..m.n_exp).collect();
                idx.sort_by(|&x, &y| choice[y].partial_cmp(&choice[x]).unwrap());
                idx.truncate(m.topk);
                if !idx.contains(&e) {
                    assert!(got[t].is_nan(), "e={e} t={t}: expert not selected, so the unit has NO value there");
                    continue;
                }
                n_routed += 1;
                let denom: f32 = idx.iter().map(|&x| sig[x]).sum::<f32>();
                let w = sig[e] / denom;
                let (mut g, mut u) = (0f32, 0f32);
                for i in 0..d {
                    g += a2[[t, i]] * gv[i * inter + n];
                    u += a2[[t, i]] * uv[i * inter + n];
                }
                let want = w * silu(g) * u;
                // Tolerance, not bits: this reference re-does the dot product in its own summation order
                // and `expert_mm` sums it in the GEMM's. Bit-exactness is the scorer-vs-naive test above.
                assert!((got[t] - want).abs() <= 1e-5 * want.abs().max(1e-3),
                    "e={e} t={t}: branch {:e} != reference {:e}", got[t], want);
            }
            if n_routed > 0 {
                n_checked += 1;
            }
        }
        assert!(n_checked > 0, "no expert was ever selected — the reference proves nothing");
    }

    /// An all-MoE arch must report `RoutedLayers`, NOT `NoScorer`: the scorer IS implemented here, and
    /// calling it unsupported is the exact conflation the enum exists to prevent.
    #[test]
    fn minimax_reports_routed_layers() {
        assert_eq!(build_tiny("decl").blind_support(), BlindSupport::RoutedLayers);
    }
}
