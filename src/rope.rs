//! Tier B — composition, RoPE family (Llama-3.2 / Qwen2.5 / Qwen3-dense / Mistral / Phi). A faithful Rust port of pylm's
//! `numpy_rope.py`: RMSNorm + rotary position embedding + grouped-query attention + SwiGLU MLP, over a fieldrun bundle
//! (`arch: "rope"`). Optional per-head **QK-norm** (Qwen3-dense) is applied when the bundle carries `q_norm`/`k_norm`;
//! optional q/k/v bias (Qwen2.5) too. Mirrors the numpy kernel array-for-array, so it reproduces it (and torch)
//! exactly. fp32 in.

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
    kv_int8: bool, // store the KV cache (GQA width) as int8 with a per-kv-head scale
    qk_norm: bool, // Qwen3-dense: per-head RMSNorm on q/k after projection, before RoPE (absent on Llama/Qwen2.5)
    gate: Option<std::sync::Arc<crate::headgate::HeadGate>>, // --pruned-head: margin-gated pruned unembed on decode
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
    pub fn new(b: Bundle, route: f32, kv_int8: bool) -> Rope {
        let c = &b.config; // [n_layer, H, nkv, hd, d, ffn, vocab, tied]
        let (n_layer, h, nkv, hd) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
        let theta = b.config_f[0] as f32;
        let eps = b.config_f[1] as f32;
        let inv = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        let qk_norm = b.has("l0.self_attn.q_norm"); // Qwen3-dense ships it; Llama/Qwen2.5/Mistral/Phi don't
        Rope { b, n_layer, h, nkv, hd, eps, inv, route, kv_int8, qk_norm, gate: None }
    }

    /// Per-head RMSNorm over head_dim (Qwen3 QK-norm), weight applied directly — on the q/k projection output
    /// (n_heads heads of width hd packed per row) before RoPE. A no-op unless the bundle carries q_norm/k_norm.
    fn head_norm(&self, x: &mut Array2<f32>, name: &str, n_heads: usize) {
        let w = self.b.arr1o(name);
        let hd = self.hd;
        for mut row in x.rows_mut() {
            for head in 0..n_heads {
                let base = head * hd;
                let ms = (0..hd).map(|c| { let v = row[base + c]; v * v }).sum::<f32>() / hd as f32;
                let inv = 1.0 / (ms + self.eps).sqrt();
                for c in 0..hd {
                    row[base + c] = row[base + c] * inv * w[c];
                }
            }
        }
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
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
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

    /// Copy of `hidden` with the K/V cache round-tripped through a quantizer per (position, kv-head) — the
    /// `--probe-kv-quant` fidelity sweep. `turbo_bits=None` → the int8 per-head max-scale scheme (== `forward_block_q`'s
    /// runtime cache); `Some(b)` → TurboQuant (SRHT rotation + Lloyd–Max levels, `Codec::roundtrip` per head-vector).
    /// The round-trip is applied to K and V right after RoPE, before attention, so this measures the DISTORTION the
    /// cache quant injects into the decision — the cheap test of whether TurboQuant's isotropy buys a lower-bit KV
    /// cache (→ longer context in fixed RAM) than per-head int8 can. (The persistent bit-packed cache wired into the
    /// streaming decode is the runtime mode, a follow-up; here k/v ARE the whole cache since this is a cur=0 prefill.)
    fn hidden_kvq(&self, ids: &[i64], turbo_bits: Option<u8>) -> Array2<f32> {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let codec = turbo_bits.map(|b| crate::turboquant::Codec::new(b, 0x5EED_4B0B, hd));
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let mut v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            // round-trip K and V per (row, kv-head) through the chosen quantizer (the cache-quant distortion).
            for i in 0..seq {
                for kh in 0..nkv {
                    let base = kh * hd;
                    for arr in [&mut k, &mut v] {
                        let vec: Vec<f32> = (0..hd).map(|c| arr[[i, base + c]]).collect();
                        let rt = match &codec {
                            Some(cd) => cd.roundtrip(&vec),
                            None => {
                                let sc = (vec.iter().fold(0f32, |mx, &val| mx.max(val.abs())) / 127.0).max(1e-8);
                                vec.iter().map(|&val| (val / sc).round().clamp(-127.0, 127.0) * sc).collect()
                            }
                        };
                        for c in 0..hd {
                            arr[[i, base + c]] = rt[c];
                        }
                    }
                }
            }
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
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

    /// Causal-ablation copy of `hidden`: re-run the forward with the given attention heads `ah` (layer, head) and MLP
    /// neurons `an` (layer, neuron) ZEROED out of the residual stream (so downstream layers recompute without them).
    /// A separate copy keeps the faithfulness-gated `hidden` pristine. Research tool (`--probe-ablate`), not gated.
    /// `ablk`/`mblk` zero a *whole* attention / MLP block of the listed layers (for the rescue-localization block sweep),
    /// on top of the per-head (`ah`) / per-neuron (`an`) ablations.
    fn hidden_ab(&self, ids: &[i64], ah: &[(usize, usize)], an: &[(usize, usize)], ablk: &[usize], mblk: &[usize]) -> Array2<f32> {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
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
            for &(al, hh) in ah {
                if al == l {
                    attn_out.slice_mut(s![.., hh * hd..(hh + 1) * hd]).fill(0.0); // ablate head: zero its value-output
                }
            }
            if ablk.contains(&l) {
                attn_out.fill(0.0); // ablate the whole attention block of this layer
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));

            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hid = gate;
            for (hv, uv) in hid.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            for &(al, nn) in an {
                if al == l {
                    hid.slice_mut(s![.., nn..nn + 1]).fill(0.0); // ablate neuron: zero its post-SwiGLU activation
                }
            }
            if mblk.contains(&l) {
                hid.fill(0.0); // ablate the whole MLP block of this layer
            }
            x = &x + &self.down(&hid, &format!("{p}mlp.down_proj"));
        }
        rmsnorm(&x, self.b.arr1("norm"), self.eps)
    }

    fn unembed_name(&self) -> &'static str {
        if self.b.config[7] != 0 { "embed" } else { "lm_head" } // tied embed, else a separate (fp16) head
    }

    fn explanation(&self, ids: &[i64], decomp_k: usize) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        let mut att_last: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut head_act: Vec<Vec<f32>> = Vec::new(); // per layer: attn_out's last row (h*hd) — for head direct-logit attribution
        let mut mlp_h: Vec<Vec<f32>> = Vec::new();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            let mut layer_att = Vec::with_capacity(h);
            for head in 0..h {
                let kv = head / rep;
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
                layer_att.push(scores.row(seq - 1).to_vec());
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            att_last.push(layer_att);
            head_act.push(attn_out.row(seq - 1).to_vec());
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            mlp_h.push(hidden.row(seq - 1).to_vec());
            x = &x + &self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
        }
        let xf = rmsnorm(&x, self.b.arr1("norm"), self.eps);
        let x_last = x.row(seq - 1).to_vec(); // residual either side of the final norm — recovers its frozen scale
        let xf_last = xf.row(seq - 1).to_vec();
        let lg = self.b.rowdot_f32(self.unembed_name(), &xf_last);
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let un = self.unembed_name();
        let gain = self.b.arr1("norm").to_vec(); // final RMSNorm gain — for direct-logit attribution
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
            decomp_k,
            &|v: i64| self.b.weight_row(un, v as usize), // competitor unembed rows — the cone the descent intersects
            |l, n| self.b.weight_row(&format!("l{l}.mlp.down_proj"), n),
            |l, head| head_raw_contrib(&self.b, &format!("l{l}.self_attn.o_proj"), &head_act[l], head, hd),
            |c| self.b.rowdot_f32(un, c),
        )
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
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
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

    /// `forward_block` that also CAPTURES the last new row's per-layer (attention rows, attn_out, mlp hidden) + the
    /// pre/post-final-norm residual — the explain substrate, computed under the KV cache. Byte-identical to the full
    /// forward by causality (each token's K/V depends only on earlier tokens). Uses `b.mm` for down_proj to match
    /// `explanation()` exactly (== `down()` in the default non-routed case). Returns
    /// (att_last[layer][head][klen], head_act[layer][h*hd], mlp_h[layer][inter], x_last[d], xf_last[d]).
    #[allow(clippy::type_complexity)]
    fn forward_block_capture(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let m = emb.nrows();
        let klen = cur + m;
        let last = m - 1;
        let mut x = emb.clone();
        let (mut att_last, mut head_act, mut mlp_h) = (Vec::with_capacity(self.n_layer), Vec::with_capacity(self.n_layer), Vec::with_capacity(self.n_layer));
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
            self.rope(&mut q, h, cur);
            self.rope(&mut k, nkv, cur);
            kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
            vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            let mut layer_att = Vec::with_capacity(h);
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
                layer_att.push(scores.row(last).to_vec());
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            att_last.push(layer_att);
            head_act.push(attn_out.row(last).to_vec());
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            mlp_h.push(hidden.row(last).to_vec());
            x = &x + &self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
        }
        let xf = rmsnorm(&x, self.b.arr1("norm"), self.eps);
        (att_last, head_act, mlp_h, x.row(last).to_vec(), xf.row(last).to_vec())
    }

    /// Build the Explanation for the LAST token of `ids` from already-captured per-layer rows (shared by the cached
    /// stream and a byte-identity check). `att_last`/`head_act`/`mlp_h` are the last-position captures, `x_last`/`xf_last`
    /// the residual either side of the final norm.
    #[allow(clippy::too_many_arguments)]
    fn assemble_from_capture(&self, ids: &[i64], att_last: &[Vec<Vec<f32>>], head_act: &[Vec<f32>], mlp_h: &[Vec<f32>], x_last: &[f32], xf_last: &[f32], decomp_k: usize) -> crate::explain::Explanation {
        use crate::explain::*;
        let hd = self.hd;
        let lg = self.b.rowdot_f32(self.unembed_name(), xf_last);
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let un = self.unembed_name();
        let gain = self.b.arr1("norm").to_vec();
        let u_pred = self.b.weight_row(un, model_predicts as usize);
        assemble(
            ids, att_last, head_act, mlp_h, &lg, model_predicts, &gain, false, &[], x_last, xf_last, &u_pred,
            decomp_k,
            &|v: i64| self.b.weight_row(un, v as usize),
            |l, n| self.b.weight_row(&format!("l{l}.mlp.down_proj"), n),
            |l, head| head_raw_contrib(&self.b, &format!("l{l}.self_attn.o_proj"), &head_act[l], head, hd),
            |c| self.b.rowdot_f32(un, c),
        )
    }

    /// KV-CACHED explain stream: process `ids` once through a single growing KV cache (O(seq) attention work instead of
    /// one full forward per position) and call `f(pos, Explanation)` for every position `pos` in `start..ids.len()` — the
    /// decision predicting `ids[pos]` from context `ids[..pos]`. Byte-identical to looping `explanation(&ids[..pos])`.
    fn explanation_stream(&self, ids: &[i64], decomp_k: usize, start: usize, f: &mut dyn FnMut(usize, crate::explain::Explanation)) {
        let seq = ids.len();
        if seq == 0 {
            return;
        }
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((seq, kvdim))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((seq, kvdim))).collect();
        // prefill ids[..start] (cur=0, m=start) so the cache is warm, then decode one token at a time.
        if start > 1 {
            let emb = self.b.rows_f32("embed", &ids[..start - 1]);
            let _ = self.forward_block_capture(&emb, 0, &mut kc, &mut vc); // warm the cache for ids[..start-1]
        }
        // each step appends token ids[pos-1] (cur = pos-1) → the cache now covers ids[..pos]; explain that last position.
        let lo = start.max(1);
        for pos in lo..=seq {
            let emb = self.b.rows_f32("embed", &ids[pos - 1..pos]); // the single new token
            let (att_last, head_act, mlp_h, x_last, xf_last) = self.forward_block_capture(&emb, pos - 1, &mut kc, &mut vc);
            let ex = self.assemble_from_capture(&ids[..pos], &att_last, &head_act, &mlp_h, &x_last, &xf_last, decomp_k);
            f(pos, ex);
        }
    }

    /// `forward_block` with an int8 KV cache (GQA width, per-kv-head scale): quantise post-RoPE K and V on write,
    /// dequantise on read. ~4x smaller cache; per-head quant error keeps tokens ~identical.
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
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
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
                let mut scores = qh.dot(&kh_a.t()) / (hd as f32).sqrt();
                for i in 0..m {
                    for j in (cur + i + 1)..klen {
                        scores[[i, j]] = -1e30;
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh_a));
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

    fn generate_kv_int8(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let total = prompt.len() + n_new;
        let kvdim = self.nkv * self.hd;
        let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|_| vec![0i8; total * kvdim]).collect();
        let mut vc = kc.clone();
        let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|_| vec![0f32; total * self.nkv]).collect();
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
        let logits = self.b.rowdot_f32(self.unembed_name(), &xfn.row(xfn.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    /// `head_argmax` behind the margin gate (`--pruned-head`): score only the KB's candidate rows; accept the in-set
    /// argmax iff the in-set normalized margin (exact facet distance, FINDINGS §5b) clears the gate's threshold, else
    /// run the full head. `ctx` = prompt + emitted-so-far (the KB keys its candidate set on it). With no gate
    /// installed this IS `head_argmax`.
    fn head_argmax_gated(&self, xfn: &Array2<f32>, ctx: &[i64]) -> i64 {
        if let Some(g) = &self.gate {
            let r = xfn.row(xfn.nrows() - 1).to_vec();
            let un = self.unembed_name();
            if let Some(t) = g.try_pruned(ctx, &|c| self.b.rowdot_f32_subset(un, &r, c), &|v| self.b.weight_row(un, v)) {
                return t;
            }
        }
        self.head_argmax(xfn)
    }

    /// Forward (full prefill, like `hidden`) capturing the recursion substrate: the post-block residual of EVERY
    /// layer (pre-final-norm — for per-layer logit-lens) and the element-wise MAX over late-layer heads of the
    /// attention score matrix (for the binding signal; late = last third, where the return/fold attention lives).
    fn recursion_capture(&self, ids: &[i64]) -> (Vec<Array2<f32>>, Array2<f32>) {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let late0 = 2 * self.n_layer / 3;
        let mut x = self.b.rows_f32("embed", ids);
        let mut resids: Vec<Array2<f32>> = Vec::with_capacity(self.n_layer);
        let mut maxback = Array2::<f32>::zeros((seq, seq));
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
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
                if l >= late0 {
                    for i in 0..seq {
                        for j in 0..=i {
                            if scores[[i, j]] > maxback[[i, j]] {
                                maxback[[i, j]] = scores[[i, j]];
                            }
                        }
                    }
                }
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
            resids.push(x.clone());
        }
        (resids, maxback)
    }
}

impl Model for Rope {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids); // unembed only the predicting position
        let logits = self.b.rowdot_f32(self.unembed_name(), &xf.row(ids.len() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn explain(&self, ids: &[i64]) -> Option<crate::explain::Explanation> {
        Some(self.explanation(ids, 0))
    }

    fn explain_decomp(&self, ids: &[i64], k: usize) -> Option<crate::explain::Explanation> {
        Some(self.explanation(ids, k)) // Density-Minimization substrate populated (--probe-decompose)
    }

    fn explain_stream(&self, ids: &[i64], decomp_k: usize, start: usize, f: &mut dyn FnMut(usize, crate::explain::Explanation)) {
        self.explanation_stream(ids, decomp_k, start, f); // KV-cached growing-prefix explain — byte-identical, O(seq)
    }

    fn final_residual(&self, ids: &[i64]) -> Option<Vec<f32>> {
        let xf = self.hidden(ids); // post-final-norm residual; row(last) is the exact vector the unembedding dots
        Some(xf.row(ids.len() - 1).to_vec())
    }

    fn recursion_trace(&self, ids: &[i64]) -> Option<Vec<crate::model::RecPos>> {
        use crate::model::RecPos;
        let seq = ids.len();
        if seq < 3 {
            return Some(vec![]);
        }
        let (resids, mut maxback) = self.recursion_capture(ids);
        let nl = self.n_layer;
        let late0 = 2 * nl / 3;
        let un = self.unembed_name();
        // per-layer logit-lens argmax per position (apply the FINAL norm to each layer's residual, then unembed)
        let mut lens = vec![vec![0i64; seq]; nl];
        for l in 0..nl {
            let normed = rmsnorm(&resids[l], self.b.arr1("norm"), self.eps);
            for p in 0..seq {
                let lg = self.b.rowdot_f32(un, &normed.row(p).to_vec());
                lens[l][p] = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
            }
        }
        // zero the attention SINK (cols 0/1) so the binding signal is a real distant fold, not sink mass
        for i in 0..seq {
            maxback[[i, 0]] = 0.0;
            if seq > 1 {
                maxback[[i, 1]] = 0.0;
            }
        }
        let mut out = Vec::new();
        for p in 0..seq.saturating_sub(1) {
            // logit lens at p predicts token p+1; the model's prediction = the last-layer lens
            let final_top1 = lens[nl - 1][p];
            let mut resolve = nl;
            for l in 0..nl {
                if lens[l][p] == final_top1 {
                    resolve = l + 1;
                    break;
                }
            }
            let lens_late: Vec<(usize, i64)> = (late0..nl).map(|l| (l + 1, lens[l][p])).collect();
            let lens_full: Vec<(usize, i64)> = (0..nl).map(|l| (l + 1, lens[l][p])).collect();
            let (mut back, mut conc) = (p, 0f32);
            for k in 0..p {
                if maxback[[p, k]] > conc {
                    conc = maxback[[p, k]];
                    back = k;
                }
            }
            out.push(RecPos { pos: p, final_top1, resolve_layer: resolve, n_layer: nl, lens_late, lens_full, back, conc });
        }
        Some(out)
    }

    fn recursion_lens_at(&self, ids: &[i64], positions: &[usize]) -> Option<Vec<Vec<(usize, i64)>>> {
        use ndarray::s;
        let (resids, _) = self.recursion_capture(ids);
        let nl = self.n_layer;
        let late0 = 2 * nl / 3;
        let un = self.unembed_name();
        let mut out = Vec::with_capacity(positions.len());
        for &p in positions {
            let mut late = Vec::new();
            for l in late0..nl {
                if p >= resids[l].nrows() { continue; }
                let row = resids[l].slice(s![p..p + 1, ..]).to_owned(); // 1×d
                let normed = rmsnorm(&row, self.b.arr1("norm"), self.eps);
                let lg = self.b.rowdot_f32(un, &normed.row(0).to_vec());
                let am = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
                late.push((l + 1, am));
            }
            out.push(late);
        }
        Some(out)
    }

    fn predict_patched(&self, ids: &[i64], layer: usize, positions: &[usize], donors: &[Vec<f32>]) -> Option<i64> {
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., kv * hd..(kv + 1) * hd]);
                let vh = v.slice(s![.., kv * hd..(kv + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
                for i in 0..seq {
                    for j in (i + 1)..seq { scores[[i, j]] = -1e30; }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) { *hv = silu(*hv) * uv; }
            x = &x + &self.down(&hidden, &format!("{p}mlp.down_proj"));
            // CAUSAL PATCH: overwrite the residual at each (layer, positions[i]) with donors[i], then keep going
            if l == layer {
                for (pos, donor) in positions.iter().zip(donors.iter()) {
                    if *pos < seq && donor.len() == x.ncols() {
                        for (j, &val) in donor.iter().enumerate() { x[[*pos, j]] = val; }
                    }
                }
            }
        }
        let normed = rmsnorm(&x, self.b.arr1("norm"), self.eps);
        let un = self.unembed_name();
        let lg = self.b.rowdot_f32(un, &normed.row(seq - 1).to_vec());
        Some(lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
    }

    fn residuals_at(&self, ids: &[i64], positions: &[usize]) -> Option<Vec<Vec<Vec<f32>>>> {
        let (resids, _) = self.recursion_capture(ids);
        let nl = self.n_layer;
        let mut out = Vec::with_capacity(positions.len());
        for &p in positions {
            let mut layers = Vec::with_capacity(nl);
            for l in 0..nl {
                if p < resids[l].nrows() {
                    layers.push(resids[l].row(p).to_vec());
                } else {
                    layers.push(Vec::new());
                }
            }
            out.push(layers);
        }
        Some(out)
    }

    fn predict_ablated(&self, ids: &[i64], heads: &[(usize, usize)], neurons: &[(usize, usize)]) -> Option<i64> {
        let xf = self.hidden_ab(ids, heads, neurons, &[], &[]);
        let logits = self.b.rowdot_f32(self.unembed_name(), &xf.row(ids.len() - 1).to_vec());
        Some(logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
    }

    fn logits(&self, ids: &[i64]) -> Option<Vec<f32>> {
        let xf = self.hidden(ids);
        Some(self.b.rowdot_f32(self.unembed_name(), &xf.row(ids.len() - 1).to_vec()))
    }

    fn logits_ablated(&self, ids: &[i64], heads: &[(usize, usize)], neurons: &[(usize, usize)]) -> Option<Vec<f32>> {
        let xf = self.hidden_ab(ids, heads, neurons, &[], &[]);
        Some(self.b.rowdot_f32(self.unembed_name(), &xf.row(ids.len() - 1).to_vec()))
    }

    fn logits_kvq(&self, ids: &[i64], turbo_bits: Option<u8>) -> Option<Vec<f32>> {
        let xf = self.hidden_kvq(ids, turbo_bits);
        Some(self.b.rowdot_f32(self.unembed_name(), &xf.row(ids.len() - 1).to_vec()))
    }

    fn set_head_gate(&mut self, gate: std::sync::Arc<crate::headgate::HeadGate>) -> bool {
        self.gate = Some(gate);
        true
    }

    fn head_gate_stats(&self) -> Option<(u64, u64)> {
        self.gate.as_ref().map(|g| g.stats())
    }

    fn clear_head_gate(&mut self) {
        self.gate = None;
    }

    fn predict_ablated_blocks(&self, ids: &[i64], heads: &[(usize, usize)], neurons: &[(usize, usize)], attn_layers: &[usize], mlp_layers: &[usize]) -> Option<i64> {
        let xf = self.hidden_ab(ids, heads, neurons, attn_layers, mlp_layers);
        let logits = self.b.rowdot_f32(self.unembed_name(), &xf.row(ids.len() - 1).to_vec());
        Some(logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
    }

    fn dims(&self) -> Option<(usize, usize)> {
        Some((self.b.config[0] as usize, self.b.config[1] as usize)) // [n_layer, H, nkv, hd, d, ffn, vocab, tied]
    }

    fn predict_block_quant(&self, ids: &[i64], block: usize, bits: u8) -> Option<i64> {
        // Mirror `hidden`, but quantize the `block`-th residual write (per-row symmetric round-trip to `bits`) before
        // it is added to the stream — so downstream layers recompute over the quantized contribution (the real
        // sensitivity, including the diffuse cushion). block 0 = embed; layer l → 2l+1 attn, 2l+2 mlp.
        let q = ((1i32 << (bits - 1)) - 1) as f32; // 127 (int8) / 7 (int4)
        let qz = |a: &mut Array2<f32>| {
            for mut row in a.rows_mut() {
                let mx = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
                if mx > 0.0 {
                    let s = mx / q;
                    row.iter_mut().for_each(|v| *v = (*v / s).round() * s);
                }
            }
        };
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        if block == 0 {
            qz(&mut x);
        }
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut qp = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            if self.qk_norm {
                self.head_norm(&mut qp, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
            self.rope(&mut qp, h, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
                let qh = qp.slice(s![.., head * hd..(head + 1) * hd]);
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
            let mut aw = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            if block == 2 * l + 1 {
                qz(&mut aw);
            }
            x = &x + &aw;
            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidn = gate;
            for (hv, uv) in hidn.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            let mut mw = self.down(&hidn, &format!("{p}mlp.down_proj"));
            if block == 2 * l + 2 {
                qz(&mut mw);
            }
            x = &x + &mw;
        }
        let xf = rmsnorm(&x, self.b.arr1("norm"), self.eps);
        let logits = self.b.rowdot_f32(self.unembed_name(), &xf.row(seq - 1).to_vec());
        Some(logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
    }

    fn residual_decomp(&self, ids: &[i64], toks: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        // `residual_decomp` = the per-block normed contribution vectors (`residual_normed_writes`) projected onto the
        // requested token rows: contrib[b][i] = ⟨d̃_b, U_{toks[i]}⟩.
        let (labels, dvec) = self.residual_normed_writes(ids)?;
        let un = self.unembed_name();
        let urows: Vec<Vec<f32>> = toks.iter().map(|&t| self.b.weight_row(un, t as usize)).collect();
        let contrib: Vec<Vec<f32>> = dvec
            .iter()
            .map(|d| urows.iter().map(|u| d.iter().zip(u.iter()).map(|(dd, ud)| dd * ud).sum::<f32>()).collect())
            .collect();
        Some((labels, contrib))
    }

    fn residual_normed_writes(&self, ids: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        // Re-run the forward, capturing each residual-stream WRITE at the last position: the embedding, then per layer
        // the attention block's o_proj output and the MLP block's down_proj output. These sum (linearly) to the
        // pre-final-norm residual; folding the final RMSNorm into each write gives d̃_b = inv_rms · gain ⊙ write_b, the
        // contribution vector in unembed space (⟨d̃_b, U_v⟩ = block b's exact logit contribution to token v).
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let last = seq - 1;
        let mut x = self.b.rows_f32("embed", ids);
        let mut labels: Vec<String> = vec!["embed".into()];
        let mut writes: Vec<Vec<f32>> = vec![x.row(last).to_vec()];
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, self.b.arr1(&format!("{p}in_ln")), self.eps);
            let mut q = self.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = self.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = self.proj(&a, &format!("{p}self_attn.v_proj"));
            if self.qk_norm {
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), h);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), nkv);
            }
            self.rope(&mut q, h, 0);
            self.rope(&mut k, nkv, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let kv = head / rep;
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
            let aw = self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            writes.push(aw.row(last).to_vec());
            labels.push(format!("L{l}.attn"));
            x = &x + &aw;
            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidn = gate;
            for (hv, uv) in hidn.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            let mw = self.down(&hidn, &format!("{p}mlp.down_proj"));
            writes.push(mw.row(last).to_vec());
            labels.push(format!("L{l}.mlp"));
            x = &x + &mw;
        }
        // final RMSNorm geometry at the last position (no center/bias for rope): logit_v = inv_rms · Σ_d gain_d x_d U_v_d,
        // and x = Σ_b write_b, so the folded contribution vector is d̃_b = inv_rms · gain ⊙ write_b (⟨d̃_b, U_v⟩ = logit).
        let xpre = x.row(last).to_vec();
        let d = xpre.len();
        let inv_rms = 1.0 / (xpre.iter().map(|v| v * v).sum::<f32>() / d as f32 + self.eps).sqrt();
        let gain = self.b.arr1("norm");
        let dvec: Vec<Vec<f32>> = writes
            .iter()
            .map(|w| w.iter().zip(gain.iter()).map(|(wd, gd)| inv_rms * wd * gd).collect())
            .collect();
        Some((labels, dvec))
    }

    fn unembed_cos(&self, a: usize, b: usize) -> Option<f32> {
        let un = self.unembed_name();
        let (ua, ub) = (self.b.weight_row(un, a), self.b.weight_row(un, b));
        let dot: f32 = ua.iter().zip(&ub).map(|(x, y)| x * y).sum();
        let (na, nb) = (ua.iter().map(|x| x * x).sum::<f32>().sqrt(), ub.iter().map(|x| x * x).sum::<f32>().sqrt());
        if na > 0.0 && nb > 0.0 { Some(dot / (na * nb)) } else { None }
    }

    fn unembed_row(&self, id: usize) -> Option<Vec<f32>> {
        let r = self.b.weight_row(self.unembed_name(), id);
        if r.is_empty() { None } else { Some(r) }
    }

    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        if self.kv_int8 {
            return self.generate_kv_int8(prompt, n_new);
        }
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

    // KV-cached streaming generation (early-stop at eos + per-token emit) — so chat/serve don't recompute the whole
    // context per token. Mirrors `generate`'s f32 KV-cache loop; uses f32 cache even under --kv-int8 (a memory knob).
    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let total = prompt.len() + max_tokens;
        let kvdim = self.nkv * self.hd;
        let mut ctx: Vec<i64> = prompt.to_vec(); // running context for the gated head's KB lookup
        if self.kv_int8 {
            // int8 KV cache for chat/serve — 4x smaller cache (longer context in the same budget); lossy by design.
            let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|_| vec![0i8; total * kvdim]).collect();
            let mut vc = kc.clone();
            let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|_| vec![0f32; total * self.nkv]).collect();
            let mut vs = ks.clone();
            let emb = self.b.rows_f32("embed", prompt);
            let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
            let mut next = self.head_argmax_gated(&xb, &ctx);
            let mut out = Vec::new();
            let mut pos = prompt.len();
            loop {
                if eos.contains(&next) { break; }
                out.push(next);
                if !emit(next) || out.len() == max_tokens { break; }
                ctx.push(next);
                let e = self.b.rows_f32("embed", &[next]);
                let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
                next = self.head_argmax_gated(&xb, &ctx);
                pos += 1;
            }
            return out;
        }
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc = kc.clone();
        let emb = self.b.rows_f32("embed", prompt);
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = self.head_argmax_gated(&xb, &ctx);
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
            ctx.push(next);
            let e = self.b.rows_f32("embed", &[next]);
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.head_argmax_gated(&xb, &ctx);
            pos += 1;
        }
        out
    }

    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool, cache: &mut crate::model::PrefixKv) -> Vec<i64> {
        if self.kv_int8 {
            let (kvdim, nkv, n_layer) = (self.nkv * self.hd, self.nkv, self.n_layer);
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
            return crate::model::prefix_generate_q(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, ctx| self.head_argmax_gated(xb, ctx));
        }
        let (kvdim, n_layer) = (self.nkv * self.hd, self.n_layer);
        let alloc = |total: usize| {
            let kc: Vec<Array2<f32>> = (0..n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
            let vc = kc.clone();
            (kc, vc)
        };
        let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]| {
            let emb = self.b.rows_f32("embed", ids);
            self.forward_block(&emb, cur, kc, vc)
        };
        crate::model::prefix_generate(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, ctx| self.head_argmax_gated(xb, ctx))
    }
}
