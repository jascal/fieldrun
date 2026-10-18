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
    fn forward_block_capture(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>], scan: Option<(usize, usize)>) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let m = emb.nrows();
        let klen = cur + m;
        let last = m - 1;
        let mut x = emb.clone();
        let (mut att_last, mut head_act, mut mlp_h) = (Vec::with_capacity(self.n_layer), Vec::with_capacity(self.n_layer), Vec::with_capacity(self.n_layer));
        // per-ROW activations of one unit, for the corpus scan: the FFN already computes every row of
        // `hidden`, so keeping one column costs O(m) and turns a batched block into m scored positions.
        // Single-row scoring cannot do this — see `Model::blind_corpus_best`.
        let mut scan_out: Vec<f32> = Vec::new();
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
            if let Some((sl, sn)) = scan {
                // `column` panics out of range, and a bad --neuron must not take the process down
                if l == sl && sn < hidden.ncols() {
                    scan_out = hidden.column(sn).to_vec();
                }
            }
            x = &x + &self.b.mm(&hidden, &format!("{p}mlp.down_proj"));
        }
        let xf = rmsnorm(&x, self.b.arr1("norm"), self.eps);
        (att_last, head_act, mlp_h, x.row(last).to_vec(), xf.row(last).to_vec(), scan_out)
    }

    /// KV-cached twin of [`Model::mlp_act_last`]: prefill `ids` through `forward_block_capture`, which already
    /// captures the same per-layer `mlp_h` vector the explain path uses. Exists so the cached and naive readouts
    /// can be asserted bit-identical — the same bar every other cache in this repo is held to.
    pub(crate) fn mlp_act_last_kv(&self, ids: &[i64], layer: usize) -> Option<Vec<f32>> {
        if layer >= self.n_layer || ids.is_empty() {
            return None;
        }
        let total = ids.len();
        let kvdim = self.nkv * self.hd;
        let mut kc = vec![Array2::<f32>::zeros((total, kvdim)); self.n_layer];
        let mut vc = vec![Array2::<f32>::zeros((total, kvdim)); self.n_layer];
        let emb = self.b.rows_f32("embed", ids);
        let (_att_last, _head_act, mlp_h, _x, _xf, _) = self.forward_block_capture(&emb, 0, &mut kc, &mut vc, None);
        mlp_h.get(layer).cloned()
    }
}

// close `impl Rope`

/// A persistent K/V cache for incremental blind-max scoring: the prefix is prefilled ONCE, then candidate
/// tokens are scored one at a time at successive positions. Scoring is O(prefix) per candidate instead of a
/// full re-forward, which is what makes a hill-climb over even a few hundred candidates affordable.
///
/// Every score WRITES the cache row at `cur`, so trying C candidates at one position is C single-row steps
/// against the same slots — the prefix rows are never disturbed. Call [`ScoreCache::commit`] to accept the
/// winning token and step to the next position; it re-scores the winner rather than assuming the cache
/// still holds it, so a caller can't corrupt the cache by scoring out of order.
pub struct ScoreCache {
    kc: Vec<Array2<f32>>,
    vc: Vec<Array2<f32>>,
    cur: usize,
}

impl Rope {
    /// Prefill `prefix` into a cache sized for `total` positions; returns None if the prefix is empty or
    /// longer than `total`.
    pub fn score_cache(&self, prefix: &[i64], total: usize) -> Option<ScoreCache> {
        if prefix.is_empty() || prefix.len() >= total {
            return None;
        }
        let kvdim = self.nkv * self.hd;
        let mut kc = vec![Array2::<f32>::zeros((total, kvdim)); self.n_layer];
        let mut vc = vec![Array2::<f32>::zeros((total, kvdim)); self.n_layer];
        let emb = self.b.rows_f32("embed", prefix);
        let (_a, _h, _mlp_h, _x, _xf, _) = self.forward_block_capture(&emb, 0, &mut kc, &mut vc, None);
        Some(ScoreCache { kc, vc, cur: prefix.len() })
    }

    /// An EMPTY cache — nothing prefilled, `cur = 0`. For the corpus scan, which places every token
    /// itself and needs no seed context.
    fn scan_cache(&self, total: usize) -> ScoreCache {
        let kvdim = self.nkv * self.hd;
        ScoreCache {
            kc: vec![Array2::<f32>::zeros((total, kvdim)); self.n_layer],
            vc: vec![Array2::<f32>::zeros((total, kvdim)); self.n_layer],
            cur: 0,
        }
    }

    /// Write `token` at `cur` and return the activation of `(layer, neuron)` at that new position. Does
    /// NOT advance — call [`ScoreCache::commit`] for that, so several candidates can be compared at one
    /// position. Uses the same `forward_block_capture` path as the cached explain stream, so the value is
    /// bit-identical to `mlp_act_last(prefix ++ [token], layer)[neuron]`.
    pub fn score_step(&self, c: &mut ScoreCache, token: i64, layer: usize, neuron: usize) -> Option<f32> {
        if layer >= self.n_layer {
            return None;
        }
        let emb = self.b.rows_f32("embed", &[token]);
        let (_a, _h, mlp_h, _x, _xf, _) = self.forward_block_capture(&emb, c.cur, &mut c.kc, &mut c.vc, None);
        mlp_h.get(layer)?.get(neuron).copied()
    }

    /// Accept `token` at the current position (re-scoring it so the cache slot definitely holds the
    /// winner) and advance to the next position.
    pub fn score_commit(&self, c: &mut ScoreCache, token: i64, layer: usize, neuron: usize) -> Option<f32> {
        let a = self.score_step(c, token, layer, neuron);
        c.cur += 1;
        a
    }

    /// The position the next `score_step` will write at.
    pub fn score_pos(&self, c: &ScoreCache) -> usize {
        c.cur
    }
}

impl Rope {
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
            let _ = self.forward_block_capture(&emb, 0, &mut kc, &mut vc, None); // warm the cache for ids[..start-1]
        }
        // each step appends token ids[pos-1] (cur = pos-1) → the cache now covers ids[..pos]; explain that last position.
        let lo = start.max(1);
        for pos in lo..=seq {
            let emb = self.b.rows_f32("embed", &ids[pos - 1..pos]); // the single new token
            let (att_last, head_act, mlp_h, x_last, xf_last, _) = self.forward_block_capture(&emb, pos - 1, &mut kc, &mut vc, None);
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

    /// The unit's top-`k` proposed tokens, ranked by `polarity` × (its down-projection · final-norm gain)
    /// dotted into the unembed — the SAME readout `explain` displays as `promotes`, signed so the negative
    /// pole is reachable. Deliberately derived from the unit's OUTPUT projection, not from its activations
    /// on any corpus: this is the unit's opinion about itself, which is what the test then tries to falsify.
    fn blind_proposals(&self, layer: usize, neuron: usize, k: usize, polarity: f32) -> Option<Vec<i64>> {
        if layer >= self.n_layer {
            return None;
        }
        let gain = self.b.arr1("norm").to_vec();
        let mut w = self.b.weight_row(&format!("l{layer}.mlp.down_proj"), neuron);
        w.iter_mut().zip(gain.iter()).for_each(|(v, g)| *v *= g);
        let proj = self.b.rowdot_f32(self.unembed_name(), &w);
        let mut idx: Vec<usize> = (0..proj.len()).collect();
        idx.sort_by(|&a, &b| {
            (polarity * proj[b]).partial_cmp(&(polarity * proj[a])).unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(idx.into_iter().take(k).map(|i| i as i64).collect())
    }

    /// Greedy hill-climb: at each step try every candidate at the current position, keep the best by
    /// `polarity × activation`, commit it, repeat. Returns the winning sequence (prefix included).
    fn blind_climb(
        &self,
        layer: usize,
        neuron: usize,
        prefix: &[i64],
        cands: &[i64],
        steps: usize,
        polarity: f32,
    ) -> Option<crate::model::BlindArm> {
        if cands.is_empty() || layer >= self.n_layer {
            return None;
        }
        let total = prefix.len() + steps + 1;
        let mut c = self.score_cache(prefix, total)?;
        let mut ids = prefix.to_vec();
        let mut best = f32::NEG_INFINITY;
        for _ in 0..steps {
            let mut pick = cands[0];
            let mut pick_obj = f32::NEG_INFINITY;
            for &t in cands {
                if let Some(a) = self.score_step(&mut c, t, layer, neuron) {
                    let o = polarity * a;
                    if o > pick_obj {
                        pick_obj = o;
                        pick = t;
                    }
                }
            }
            self.score_commit(&mut c, pick, layer, neuron);
            ids.push(pick);
            best = pick_obj;
        }
        Some(crate::model::BlindArm { ids, obj: best, pos: None })
    }

    /// Oracle arm: the best-activating position over a real corpus, as ONE causal sweep.
    ///
    /// Chunked, not single-token, and that is the whole design. Two obvious alternatives are both wrong:
    ///
    /// * **Overlapping windows** — N independent forwards, each re-computing the ~32 rows it shares with
    ///   its neighbours. Measured 0.43 s per 32-row window, so a 11.8k-token corpus costs ~84 min, and every
    ///   position is scored with an artificial, freshly-restarted context.
    /// * **One token at a time** — true causal history, but at batch 1 every step re-reads all 6 GB of
    ///   weights, so cost is bandwidth-bound: measured 0.9 s/token, i.e. ~3 h for the same corpus. Worse
    ///   than the windows it replaced.
    ///
    /// A chunk of `CORPUS_CHUNK` rows against a carried cache gets both: one weight read amortised over 32
    /// positions (the windowed throughput) AND each position's true causal history (the streaming
    /// correctness). Every position in the stream is scored exactly once.
    ///
    /// Uses the same cache the search uses, not the naive path — see `Model::blind_corpus_best` for why
    /// that costs nothing in fidelity (the values are bit-identical to the naive forward, asserted in tests).
    fn blind_corpus_best(
        &self,
        layer: usize,
        neuron: usize,
        ids: &[i64],
        polarity: f32,
    ) -> Option<crate::model::BlindArm> {
        if ids.len() < 2 || layer >= self.n_layer {
            return None;
        }
        let mut c = self.scan_cache(ids.len());
        let (mut best_obj, mut best_idx) = (f32::NEG_INFINITY, 0usize);
        for chunk in ids.chunks(crate::model::CORPUS_CHUNK) {
            let emb = self.b.rows_f32("embed", chunk);
            let start = c.cur;
            let (_a, _h, _mlp_h, _x, _xf, rows) =
                self.forward_block_capture(&emb, start, &mut c.kc, &mut c.vc, Some((layer, neuron)));
            c.cur += chunk.len();
            if rows.len() != chunk.len() {
                continue; // unit out of range for this layer's FFN width
            }
            for (j, &a) in rows.iter().enumerate() {
                let o = polarity * a;
                if o > best_obj {
                    best_obj = o;
                    best_idx = start + j;
                }
            }
        }
        if best_obj == f32::NEG_INFINITY {
            return None;
        }
        let lo = best_idx.saturating_sub(crate::model::CORPUS_CTX - 1);
        Some(crate::model::BlindArm {
            ids: ids[lo..=best_idx].to_vec(),
            obj: best_obj,
            pos: Some(best_idx),
        })
    }

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

    fn mlp_act_last(&self, ids: &[i64], layer: usize) -> Option<Vec<f32>> {
        if layer >= self.n_layer || ids.is_empty() {
            return None;
        }
        // Same arithmetic as `hidden`, in the same order, stopping at `layer` — later layers cannot
        // influence this one, so the early exit is exact rather than an approximation.
        let seq = ids.len();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..=layer {
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
            x = &x + &self.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = rmsnorm(&x, self.b.arr1(&format!("{p}post_ln")), self.eps);
            let gate = self.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = self.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            if l == layer {
                return Some(hidden.row(seq - 1).to_vec());
            }
            x = &x + &self.down(&hidden, &format!("{p}mlp.down_proj"));
        }
        None
    }

    fn final_residual(&self, ids: &[i64]) -> Option<Vec<f32>> {
        let xf = self.hidden(ids); // post-final-norm residual; row(last) is the exact vector the unembedding dots
        Some(xf.row(ids.len() - 1).to_vec())
    }

    fn recursion_trace(&self, ids: &[i64]) -> Option<Vec<crate::model::RecPos>> {
        self.recursion_trace_lens(ids, None) // plain logit-lens is the J-lens with J_l = I at every layer
    }

    fn recursion_trace_lens(&self, ids: &[i64], jmats: Option<&[ndarray::Array2<f32>]>) -> Option<Vec<crate::model::RecPos>> {
        if ids.len() < 3 {
            return Some(vec![]);
        }
        let (resids, maxback) = self.recursion_capture(ids);
        let un = self.unembed_name();
        // arch-specific lens only: (optional J_l pre-multiply) then final RMSNorm ("norm") + unembed argmax; the depth
        // sweep + resolve/binding assembly is shared (build_rec_trace). J_l maps a layer-l residual through the AVERAGED
        // downstream Jacobian before read-out, so a mid-stack activation is scored by what it is disposed to make the
        // model emit — not by the identity-path logit-lens, which the workspace note reports is noisy at mid layers.
        Some(crate::model::build_rec_trace(&resids, maxback, 2 * self.n_layer / 3, |l, resid| {
            // apply J_l row-wise: (J_l @ resid[p]) for every position p  ==  resid · J_l^T  (None ⇒ plain logit-lens)
            let read = match jmats {
                Some(js) if l < js.len() => resid.dot(&js[l].t()),
                _ => resid.clone(),
            };
            let normed = rmsnorm(&read, self.b.arr1("norm"), self.eps);
            (0..normed.nrows())
                .map(|p| {
                    let lg = self.b.rowdot_f32(un, &normed.row(p).to_vec());
                    lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
                })
                .collect()
        }))
    }

    fn jlens_capture(&self, ids: &[i64]) -> Option<Vec<ndarray::Array2<f32>>> {
        Some(self.recursion_capture(ids).0) // post-block residual of every layer (pre-final-norm); drop the maxback
    }

    /// Forward from just after `layer`: push a (possibly perturbed) post-block-`layer` residual through blocks
    /// `layer+1..n_layer` and return the pre-final-norm final residual. Same block body as `recursion_capture` /
    /// `predict_patched` (full-prefill, rope offset 0) — the JVP primitive the J-lens estimator perturbs one row of.
    fn jlens_forward_from(&self, layer: usize, x0: &ndarray::Array2<f32>) -> Option<ndarray::Array2<f32>> {
        let seq = x0.nrows();
        let (h, nkv, hd) = (self.h, self.nkv, self.hd);
        let rep = h / nkv;
        let mut x = x0.clone();
        for l in (layer + 1)..self.n_layer {
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
        }
        Some(x)
    }

    fn recursion_lens_at(&self, ids: &[i64], positions: &[usize]) -> Option<Vec<Vec<(usize, i64)>>> {
        let (resids, _) = self.recursion_capture(ids);
        let un = self.unembed_name();
        Some(crate::model::build_rec_lens_at(&resids, positions, 2 * self.n_layer / 3, |resid, p| {
            let normed = rmsnorm(&resid.slice(s![p..p + 1, ..]).to_owned(), self.b.arr1("norm"), self.eps);
            let lg = self.b.rowdot_f32(un, &normed.row(0).to_vec());
            lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
        }))
    }

    fn predict_patched(&self, ids: &[i64], layer: usize, positions: &[usize], donors: &[Vec<f32>]) -> Option<i64> {
        let lg = self.logits_patched(ids, layer, positions, donors)?;
        Some(lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
    }

    fn logits_patched(&self, ids: &[i64], layer: usize, positions: &[usize], donors: &[Vec<f32>]) -> Option<Vec<f32>> {
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
        Some(self.b.rowdot_f32(un, &normed.row(seq - 1).to_vec()))
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

    fn logits_ablated_blocks(&self, ids: &[i64], heads: &[(usize, usize)], neurons: &[(usize, usize)], attn_layers: &[usize], mlp_layers: &[usize]) -> Option<Vec<f32>> {
        let xf = self.hidden_ab(ids, heads, neurons, attn_layers, mlp_layers);
        Some(self.b.rowdot_f32(self.unembed_name(), &xf.row(ids.len() - 1).to_vec()))
    }

    fn predict_ablated_blocks(&self, ids: &[i64], heads: &[(usize, usize)], neurons: &[(usize, usize)], attn_layers: &[usize], mlp_layers: &[usize]) -> Option<i64> {
        let logits = self.logits_ablated_blocks(ids, heads, neurons, attn_layers, mlp_layers)?;
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

    fn unembed_project(&self, v: &[f32]) -> Option<Vec<f32>> {
        Some(self.b.rowdot_f32(self.unembed_name(), v)) // logit-lens of any residual vector
    }

    fn unembed_row(&self, id: usize) -> Option<Vec<f32>> {
        let r = self.b.weight_row(self.unembed_name(), id);
        if r.is_empty() { None } else { Some(r) }
    }

    fn export_unembed(&self) -> Option<crate::jlens::UnembedExport> {
        // U = the unembedding rows (tied `embed`, else `lm_head`); gamma = the final RMSNorm gain `norm`.
        // RMSNorm folds as a pure diagonal, so pil's diag(gamma) J diag(1/gamma) conjugation is EXACT here.
        // `rows_f32` DEQUANTISES f32/f16/RowI8 rows → the same (vocab, d) f32 on a quantised bundle as on an f32 one;
        // `arr2o` would hit the quantised-upcast guard on an int8/RowI8 embed (the deep bundles are all int8-quantised).
        let un = self.unembed_name();
        let vocab = self.b.config[6] as usize; // config = [n_layer, H, nkv, hd, d, ffn, vocab, tied]
        Some(crate::jlens::UnembedExport {
            u: self.b.rows_f32(un, &(0..vocab as i64).collect::<Vec<_>>()),
            gamma: self.b.arr1("norm").to_vec(),
            norm_type: "rmsnorm",
            tied: self.b.config[7] != 0,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::Bundle;
    use crate::model::{Model, POLE_NEG, POLE_POS};

    /// Deterministic pseudo-random weights — no RNG dependency, so a failure is always reproducible.
    fn v(seed: usize, n: usize) -> Vec<f32> {
        (0..n).map(|i| (((seed * 37 + i * 17) % 41) as f32) / 20.0 - 1.0).collect()
    }

    /// A 3-layer rope-family bundle with no QK-norm (so `qk_norm` is off). Returns the bundle stem.
    fn tiny_rope(tag: &str) -> String {
        let (nl, hh, nkv, hd, d, ff, vocab) = (3usize, 4usize, 2usize, 4usize, 16usize, 12usize, 20usize);
        let dir = std::env::temp_dir().join(format!("fr_blindmax_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stem = dir.join("b").to_string_lossy().into_owned();
        let mut bin: Vec<u8> = Vec::new();
        let mut specs: Vec<serde_json::Value> = Vec::new();
        let mut add = |name: String, shape: Vec<usize>, data: Vec<f32>| {
            let offset = bin.len();
            for x in &data {
                bin.extend_from_slice(&x.to_le_bytes());
            }
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
            // Linear weights are stored (in, out) — `mm(a: (seq,in), w: (in,out))`.
            add(format!("{p}self_attn.q_proj"), vec![d, hh * hd], v(30 + l, d * hh * hd));
            add(format!("{p}self_attn.k_proj"), vec![d, nkv * hd], v(40 + l, d * nkv * hd));
            add(format!("{p}self_attn.v_proj"), vec![d, nkv * hd], v(50 + l, d * nkv * hd));
            add(format!("{p}self_attn.o_proj"), vec![hh * hd, d], v(60 + l, hh * hd * d));
            add(format!("{p}mlp.gate_proj"), vec![d, ff], v(70 + l, d * ff));
            add(format!("{p}mlp.up_proj"), vec![d, ff], v(80 + l, d * ff));
            add(format!("{p}mlp.down_proj"), vec![ff, d], v(90 + l, ff * d));
        }
        let manifest = serde_json::json!({
            "format": crate::bundle::FORMAT, "version": crate::bundle::VERSION, "arch": "rope",
            // [n_layer, H, nkv, hd, d, ffn, vocab, tied=0]; config_f = [rope_theta, eps]
            "config": [nl as i64, hh as i64, nkv as i64, hd as i64, d as i64, ff as i64, vocab as i64, 0],
            "config_f": [10000.0, 1e-5], "arrays": specs,
        });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
        stem
    }

    /// Every layer's last-position post-activation row, with NO early exit — i.e. what a forward that
    /// always runs to the end produces. `mlp_act_last` stops at `layer` as an optimisation; this is the
    /// reference that optimisation must not perturb.
    fn full_mlp_h(m: &Rope, ids: &[i64]) -> Vec<Vec<f32>> {
        let seq = ids.len();
        let (h, nkv, hd) = (m.h, m.nkv, m.hd);
        let rep = h / nkv;
        let mut x = m.b.rows_f32("embed", ids);
        let mut out: Vec<Vec<f32>> = Vec::new();
        for l in 0..m.n_layer {
            let p = format!("l{l}.");
            let a = rmsnorm(&x, m.b.arr1(&format!("{p}in_ln")), m.eps);
            let mut q = m.proj(&a, &format!("{p}self_attn.q_proj"));
            let mut k = m.proj(&a, &format!("{p}self_attn.k_proj"));
            let v = m.proj(&a, &format!("{p}self_attn.v_proj"));
            m.rope(&mut q, h, 0);
            m.rope(&mut k, nkv, 0);
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
            x = &x + &m.b.mm(&attn_out, &format!("{p}self_attn.o_proj"));
            let a2 = rmsnorm(&x, m.b.arr1(&format!("{p}post_ln")), m.eps);
            let gate = m.b.mm(&a2, &format!("{p}mlp.gate_proj"));
            let up = m.b.mm(&a2, &format!("{p}mlp.up_proj"));
            let mut hidden = gate;
            for (hv, uv) in hidden.iter_mut().zip(up.iter()) {
                *hv = silu(*hv) * uv;
            }
            out.push(hidden.row(seq - 1).to_vec());
            x = &x + &m.down(&hidden, &format!("{p}mlp.down_proj"));
        }
        out
    }

    const IDS: [i64; 7] = [1, 5, 3, 9, 2, 7, 4];

    /// (a) The early exit at `layer` is EXACT: stopping there must not perturb the value, so every layer's
    /// readout must match the same layer taken from a forward that ran to the end. Bit-for-bit.
    #[test]
    fn mlp_act_last_early_exit_is_exact() {
        let stem = tiny_rope("early");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        let full = full_mlp_h(&m, &IDS);
        for l in 0..m.n_layer {
            let got = m.mlp_act_last(&IDS, l).unwrap();
            assert_eq!(got.len(), full[l].len(), "layer {l}: wrong width");
            for i in 0..got.len() {
                assert_eq!(
                    got[i].to_bits(),
                    full[l][i].to_bits(),
                    "layer {l} neuron {i}: early exit diverged — {:e} vs {:e}",
                    got[i],
                    full[l][i]
                );
            }
        }
    }

    /// (b) The KV-cached readout must be BYTE-identical to the naive one — the same bar every other cache
    /// in this repo is held to. A token equality check would not catch a small per-layer drift here.
    #[test]
    fn mlp_act_last_cached_matches_naive_bitwise() {
        let stem = tiny_rope("cached");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        for l in 0..m.n_layer {
            let naive = m.mlp_act_last(&IDS, l).unwrap();
            let cached = m.mlp_act_last_kv(&IDS, l).unwrap();
            assert_eq!(naive.len(), cached.len(), "layer {l}: width mismatch");
            for i in 0..naive.len() {
                assert_eq!(
                    naive[i].to_bits(),
                    cached[i].to_bits(),
                    "layer {l} neuron {i}: KV-cached readout diverged — {:e} vs {:e}",
                    naive[i],
                    cached[i]
                );
            }
        }
    }

    /// Guards the edges the blind-max harness will hit: no layer, a negative layer, and an empty sequence.
    #[test]
    fn mlp_act_last_rejects_bad_inputs() {
        let stem = tiny_rope("edges");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        assert!(m.mlp_act_last(&IDS, m.n_layer).is_none(), "out-of-range layer must be None");
        assert!(m.mlp_act_last(&IDS, 999).is_none(), "far out-of-range layer must be None");
        assert!(m.mlp_act_last(&[], 0).is_none(), "empty sequence must be None");
    }

    /// The cached scoring step must equal the NAIVE forward bit-for-bit. This is the property the whole
    /// blind-max search rests on: it scores candidates through `ScoreCache` and compares them to a
    /// corpus scored through `mlp_act_last`, and a per-token drift between the two paths would silently
    /// corrupt every ratio the harness reports.
    #[test]
    fn score_step_is_bit_identical_to_naive() {
        let stem = tiny_rope("step");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        let total = IDS.len() + 8;
        for l in 0..m.n_layer {
            // score each token appended to the prefix, one cached step per token
            let mut c = m.score_cache(&IDS, total).unwrap();
            for (i, &tok) in [7i64, 2, 11, 0].iter().enumerate() {
                let got = m.score_step(&mut c, tok, l, (i * 3) % 12).unwrap();
                let mut want_ids = IDS.to_vec();
                want_ids.push(tok);
                let want = m.mlp_act_last(&want_ids, l).unwrap()[(i * 3) % 12];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "layer {l} tok {tok}: cached step {got:e} != naive {want:e}"
                );
            }
        }
    }

    /// Committing must leave the cache holding the ACCEPTED token at the previous position — the next
    /// step reads the prefix through those rows, so a wrong row here would poison everything after it.
    #[test]
    fn score_commit_advances_and_persists_winner() {
        let stem = tiny_rope("commit");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        let l = 1;
        let mut c = m.score_cache(&IDS, IDS.len() + 8).unwrap();
        assert_eq!(m.score_pos(&c), IDS.len());
        // try a few candidates, commit a MIDDLE one, then score a fresh appended token
        let cands = [4i64, 9, 1];
        let mut scores = Vec::new();
        for &t in &cands {
            scores.push(m.score_step(&mut c, t, l, 2).unwrap());
        }
        let best = (0..cands.len()).max_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap()).unwrap();
        m.score_commit(&mut c, cands[best], l, 2);
        assert_eq!(m.score_pos(&c), IDS.len() + 1);
        let next_tok = 6i64;
        let got = m.score_step(&mut c, next_tok, l, 2).unwrap();
        let mut want_ids = IDS.to_vec();
        want_ids.push(cands[best]);
        want_ids.push(next_tok);
        let want = m.mlp_act_last(&want_ids, l).unwrap()[2];
        assert_eq!(got.to_bits(), want.to_bits(), "after commit: {got:e} != {want:e}");
    }

    /// An empty or over-long prefix can't back a cache — return None rather than allocating a bogus one.
    #[test]
    fn score_cache_rejects_bad_prefix() {
        let stem = tiny_rope("badprefix");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        assert!(m.score_cache(&[], 32).is_none(), "empty prefix must be None");
        assert!(m.score_cache(&IDS, IDS.len()).is_none(), "prefix >= total must be None");
    }

    /// The hill-climb reports an objective it obtained through the search cache; that number must equal the
    /// NAIVE forward over the sequence it claims to have found. If it didn't, every ratio the blind-max
    /// report shows is measured against a fiction.
    #[test]
    fn blind_climb_objective_matches_naive_recompute() {
        let stem = tiny_rope("climb");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        let (l, n) = (1usize, 4usize);
        for pol in [POLE_POS, POLE_NEG] {
            let cands = m.blind_proposals(l, n, 8, pol).unwrap();
            assert!(!cands.is_empty(), "proposals must be non-empty");
            let arm = m.blind_climb(l, n, &IDS, &cands, 3, pol).unwrap();
            let act = m.mlp_act_last(&arm.ids, l).unwrap()[n];
            assert_eq!(
                (pol * act).to_bits(),
                arm.obj.to_bits(),
                "pole {pol}: climb reported {:e} but naive recompute gives {:e}",
                arm.obj,
                pol * act
            );
            // the climb must actually be greedily improving, not just returning something
            if pol == POLE_POS {
                assert!(arm.obj > m.mlp_act_last(&IDS, l).unwrap()[n], "climb must beat the bare prefix");
            }
        }
    }

    /// The corpus arm walks the stream with the cache; its answer must equal the brute-force max over EVERY
    /// position of a naive `mlp_act_last(&stream[..=i])`. Streaming may be cheaper than brute force — it
    /// must never be a DIFFERENT number, or the oracle is measuring something the search can't see and
    /// the ratio is fiction. Compared bit-for-bit, per this repo's one rule.
    /// deliberately longer than `CORPUS_CHUNK`, so chunk boundaries — where a carried cache could drift from
/// a full recompute — are actually crossed.
    #[test]
    fn blind_corpus_best_matches_naive_per_position_max() {
        let stem = tiny_rope("corpus");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        let (l, n) = (0usize, 2usize);
        let stream: Vec<i64> = (0..75).map(|i: i64| (i * 7 + 3) % 20).collect();
        for pol in [POLE_POS, POLE_NEG] {
            let arm = m.blind_corpus_best(l, n, &stream, pol).unwrap();
            let brute: Vec<f32> =
                (0..stream.len()).map(|i| pol * m.mlp_act_last(&stream[..=i], l).unwrap()[n]).collect();
            let want = brute.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(arm.obj.to_bits(), want.to_bits(), "pole {pol}: streamed {:e} vs naive {:e}", arm.obj, want);
            // the reported index must be where that max actually is
            let at = brute.iter().position(|&x| x == want).expect("max must occur");
            assert_eq!(arm.pos, Some(at), "reported position must be the argmax position");
            // and the reported ids must be the stream ending there, tail-bounded
            assert_eq!(arm.ids.last().copied(), Some(stream[at]), "ids must end at the best position");
            assert_eq!(arm.ids, stream[at.saturating_sub(crate::model::CORPUS_CTX - 1)..=at], "ids must be the bounded tail");
        }
    }

    /// The oracle must cover EVERY position — including position 0, which a longer prefill would silently
    /// skip (a block forward only captures its last row). If the argmax were position 0 and the scan
    /// started after the prefill, this arm would report a smaller number than brute force: the one
    /// failure mode that makes the ratio look better than it is.
    #[test]
    fn blind_corpus_best_scores_position_zero() {
        let stem = tiny_rope("pos0");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        let (l, n) = (1usize, 7usize);
        let at = |st: &[i64]| -> Vec<f32> {
            (0..st.len()).map(|i| m.mlp_act_last(&st[..=i], l).unwrap()[n]).collect()
        };
        // Search for a stream whose global argmax really is position 0. Deterministic, exhaustive over
        // 2- and 3-token streams: the point is to build the case where skipping position 0 is DETECTABLE,
        // and hand-picking a "clever" stream is how that silently stops being true.
        let mut found: Option<Vec<i64>> = None;
        'search: for len in [2usize, 3] {
            for a in 0..20i64 {
                for b in 0..20i64 {
                    for c in 0..(if len == 3 { 20 } else { 0 }) {
                        let st = if len == 2 { vec![a, b] } else { vec![a, b, c] };
                        let v = at(&st);
                        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        if v[0] == mx && v[1..].iter().all(|&x| x < mx) {
                            found = Some(st);
                            break 'search;
                        }
                    }
                }
            }
        }
        let stream = found.expect("tiny model must have a stream peaking at position 0");
        let arm = m.blind_corpus_best(l, n, &stream, POLE_POS).unwrap();
        let want = at(&stream).into_iter().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(arm.pos, Some(0), "the fixture's argmax is position 0, so that is what must be reported");
        assert_eq!(arm.obj.to_bits(), want.to_bits(), "streamed {:e} vs naive {:e}", arm.obj, want);
        assert_eq!(arm.ids, vec![stream[0]], "tail of a position-0 best is just that token");
    }

    /// Polarity must flip which end of the range is sought: the negative-pole search must not return the
    /// same winner as the positive-pole search when the two differ.
    #[test]
    fn blind_climb_polarity_selects_opposite_extremes() {
        let stem = tiny_rope("pol");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        let (l, n) = (2usize, 9usize);
        let pos = m.blind_climb(l, n, &IDS, &m.blind_proposals(l, n, 10, POLE_POS).unwrap(), 2, POLE_POS).unwrap();
        let neg = m.blind_climb(l, n, &IDS, &m.blind_proposals(l, n, 10, POLE_NEG).unwrap(), 2, POLE_NEG).unwrap();
        // neg must be the negation of the smallest (most negative) achievable activation it found
        let act = m.mlp_act_last(&neg.ids, l).unwrap()[n];
        assert_eq!((-act).to_bits(), neg.obj.to_bits(), "negative-pole arm objective must be -act");
        // and the two arms should generally disagree on which token wins
        assert_ne!(pos.obj, neg.obj, "poles should not coincide for a real unit");
    }

    /// A single-token sequence still returns that token's activations (length 1, no mask needed) —
    /// blind-max evaluates short probes, so a 1-token case must not silently return None.
    #[test]
    fn mlp_act_last_single_token() {
        let stem = tiny_rope("one");
        let m = Rope::new(Bundle::load(&stem).unwrap(), 0.0, false);
        let got = m.mlp_act_last(&[3], 1).unwrap();
        assert_eq!(got.len(), 12); // ffn width
        let full = full_mlp_h(&m, &[3]);
        assert_eq!(got[0].to_bits(), full[1][0].to_bits());
    }
}
