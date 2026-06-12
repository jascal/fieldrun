//! Tier B — composition, GPT-NeoX family (Pythia / GPT-NeoX-20B). The NeoX block: LayerNorm (with bias) +
//! **partial rotary** (RoPE on the first `rotary_ndims` of each head, the rest pass through) + full-width multi-head
//! attention (no GQA) + GELU MLP, with the **parallel residual** `x = x + attn(ln1(x)) + mlp(ln2(x))` (ln2 reads the
//! SAME pre-attention x — Pythia ships `use_parallel_residual: true`; the sequential form is kept for completeness).
//! Untied unembed (`embed_out`). The fused `query_key_value` is de-interleaved into q/k/v at convert time, so the
//! kernel sees plain per-projection linears. Faithfulness gate: top-1 vs the pure-numpy reference
//! (`scripts/neox_ref.py`), exact at f32.
//!
//! Implements the full probe surface (`final_residual`, `explain`, `predict_ablated`) so the FINDINGS battery
//! (--attribute/--probe/--probe-dla/--probe-facet/--probe-ablate) runs on the Pythia ladder — the cross-architecture
//! replication FINDINGS §5 calls for. No int8-KV cache yet (--kv-int8 falls back to the f32 cache with a warning).

use ndarray::{s, Array1, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

pub struct Neox {
    b: Bundle,
    n_layer: usize,
    h: usize,
    hd: usize,
    rot: usize,     // rotary_ndims: rotary applies to dims [0, rot) of each head, [rot, hd) pass through
    parallel: bool, // use_parallel_residual (true for all Pythia)
    eps: f32,
    inv: Vec<f32>, // rotary frequencies, length rot/2 (dim base = rot, not hd)
    route: f32,    // Tier C: fraction of MLP neurons to compute per token (0 = off)
}

fn layernorm(x: &Array2<f32>, g: Array1<f32>, b: Array1<f32>, eps: f32) -> Array2<f32> {
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

/// erf via the alternating Maclaurin series in f64 — converges to f64 roundoff for the |x| ≤ 5 range that matters
/// (beyond it erf is ±1 to <1e-11), so the f32 result is exact. No libm dependency.
fn erf(xf: f32) -> f32 {
    let x = xf as f64;
    if x.abs() > 5.0 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let (mut term, mut sum) = (x, x); // term_n = (-1)^n x^(2n+1)/n!, summed as term_n/(2n+1)
    for n in 1..200u32 {
        term *= -x * x / n as f64;
        let add = term / (2 * n + 1) as f64;
        sum += add;
        if add.abs() < 1e-17 * sum.abs().max(1e-300) {
            break;
        }
    }
    (sum * 2.0 / std::f64::consts::PI.sqrt()) as f32
}

/// EXACT (erf) GELU — GPT-NeoX ships `hidden_act: "gelu"` (erf form), unlike GPT-2's `gelu_new` tanh approximation.
fn gelu(x: &mut Array2<f32>) {
    x.mapv_inplace(|v| 0.5 * v * (1.0 + erf(v / std::f32::consts::SQRT_2)));
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

impl Neox {
    pub fn new(b: Bundle, route: f32, kv_int8: bool) -> Neox {
        let c = &b.config; // [n_layer, H, hd, d, ffn, vocab, rot_ndims, parallel]
        let (n_layer, h, hd, rot, parallel) = (c[0] as usize, c[1] as usize, c[2] as usize, c[6] as usize, c[7] != 0);
        let theta = b.config_f[0] as f32;
        let eps = b.config_f[1] as f32;
        // NeoX inv_freq uses rotary_ndims as the dimension base: 1/theta^(2j/rot)
        let inv = (0..rot / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / rot as f32)).collect();
        if kv_int8 {
            eprintln!("[fieldrun] neox: --kv-int8 not wired for this arch yet — using the f32 KV cache");
        }
        Neox { b, n_layer, h, hd, rot, parallel, eps, inv, route }
    }

    /// Rotary on the first `rot` dims of each head (split-half within the rotary block, like HF `rotate_half`),
    /// dims [rot, hd) untouched. Row i is absolute position `pos0 + i`.
    fn rope(&self, x: &mut Array2<f32>, n_heads: usize, pos0: usize) {
        let (hd, half) = (self.hd, self.rot / 2);
        for (i, mut row) in x.rows_mut().into_iter().enumerate() {
            let pos = (pos0 + i) as f32;
            for head in 0..n_heads {
                let base = head * hd;
                for j in 0..half {
                    let ang = pos * self.inv[j];
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

    fn down(&self, h: &Array2<f32>, name: &str) -> Array2<f32> {
        if self.route > 0.0 && self.route < 1.0 {
            self.b.mm_routed_down(h, name, self.route)
        } else {
            self.b.mm(h, name)
        }
    }

    fn ln(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        layernorm(x, self.b.arr1(&format!("{name}.weight")), self.b.arr1(&format!("{name}.bias")), self.eps)
    }

    /// Causal attention over q/k/v blocks (seq, h*hd) at absolute position 0 — the full-recompute path.
    fn attend(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let (h, hd) = (self.h, self.hd);
        let seq = q.nrows();
        let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
        for head in 0..h {
            let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
            let kh = k.slice(s![.., head * hd..(head + 1) * hd]);
            let vh = v.slice(s![.., head * hd..(head + 1) * hd]);
            let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
            for i in 0..seq {
                for j in (i + 1)..seq {
                    scores[[i, j]] = -1e30;
                }
            }
            softmax_rows(&mut scores);
            attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
        }
        attn_out
    }

    /// One layer's attention block (ln1 → q/k/v → rope → causal attention → dense), full-recompute form.
    fn attn_block(&self, x: &Array2<f32>, l: usize) -> Array2<f32> {
        let p = format!("l{l}.");
        let a = self.ln(x, &format!("{p}ln1"));
        let mut q = self.proj(&a, &format!("{p}q_proj"));
        let mut k = self.proj(&a, &format!("{p}k_proj"));
        let v = self.proj(&a, &format!("{p}v_proj"));
        self.rope(&mut q, self.h, 0);
        self.rope(&mut k, self.h, 0);
        let attn_out = self.attend(&q, &k, &v);
        self.proj(&attn_out, &format!("{p}dense"))
    }

    /// One layer's MLP block (ln2 → fc_in → GELU → fc_out) from the given residual input.
    fn mlp_block(&self, x: &Array2<f32>, l: usize) -> Array2<f32> {
        let p = format!("l{l}.");
        let a2 = self.ln(x, &format!("{p}ln2"));
        let mut hm = self.proj(&a2, &format!("{p}fc_in"));
        gelu(&mut hm);
        let mut y = self.down(&hm, &format!("{p}fc_out"));
        y = y + &self.b.arr1(&format!("{p}fc_out.bias"));
        y
    }

    /// Final post-ln_f hidden states (seq, d) — the forward pass minus the unembed.
    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.n_layer {
            let attn = self.attn_block(&x, l);
            if self.parallel {
                let mlp = self.mlp_block(&x, l); // ln2 reads the SAME pre-attention x
                x = &(&x + &attn) + &mlp;
            } else {
                x = &x + &attn;
                let mlp = self.mlp_block(&x, l);
                x = &x + &mlp;
            }
        }
        self.ln(&x, "ln_f")
    }

    /// Causal-ablation copy of `hidden`: the given attention heads `ah` (layer, head) are zeroed at the pre-dense
    /// value-output and MLP neurons `an` (layer, neuron) at the post-GELU activation, so downstream layers recompute
    /// without them. Research tool (`--probe-ablate`), not gated.
    fn hidden_ab(&self, ids: &[i64], ah: &[(usize, usize)], an: &[(usize, usize)]) -> Array2<f32> {
        let (h, hd) = (self.h, self.hd);
        let mut x = self.b.rows_f32("embed", ids);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.ln(&x, &format!("{p}ln1"));
            let mut q = self.proj(&a, &format!("{p}q_proj"));
            let mut k = self.proj(&a, &format!("{p}k_proj"));
            let v = self.proj(&a, &format!("{p}v_proj"));
            self.rope(&mut q, h, 0);
            self.rope(&mut k, h, 0);
            let mut attn_out = self.attend(&q, &k, &v);
            for &(al, hh) in ah {
                if al == l {
                    attn_out.slice_mut(s![.., hh * hd..(hh + 1) * hd]).fill(0.0);
                }
            }
            let attn = self.proj(&attn_out, &format!("{p}dense"));
            let mlp_in = if self.parallel { x.clone() } else { &x + &attn };
            let a2 = self.ln(&mlp_in, &format!("{p}ln2"));
            let mut hm = self.proj(&a2, &format!("{p}fc_in"));
            gelu(&mut hm);
            for &(al, nn) in an {
                if al == l {
                    hm.slice_mut(s![.., nn..nn + 1]).fill(0.0);
                }
            }
            let mlp = &self.down(&hm, &format!("{p}fc_out")) + &self.b.arr1(&format!("{p}fc_out.bias"));
            // same combine either way: in the sequential form mlp was computed from (x + attn), in the parallel
            // form from x — the residual sum is x + attn + mlp in both.
            x = &(&x + &attn) + &mlp;
        }
        self.ln(&x, "ln_f")
    }

    fn explanation(&self, ids: &[i64]) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let (h, hd) = (self.h, self.hd);
        let mut x = self.b.rows_f32("embed", ids);
        let mut att_last: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut head_act: Vec<Vec<f32>> = Vec::new(); // per layer: attn_out's last row (pre-dense) — head DLA
        let mut mlp_h: Vec<Vec<f32>> = Vec::new(); // per layer: post-GELU hidden's last row — neuron DLA
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.ln(&x, &format!("{p}ln1"));
            let mut q = self.proj(&a, &format!("{p}q_proj"));
            let mut k = self.proj(&a, &format!("{p}k_proj"));
            let v = self.proj(&a, &format!("{p}v_proj"));
            self.rope(&mut q, h, 0);
            self.rope(&mut k, h, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            let mut layer_att = Vec::with_capacity(h);
            for head in 0..h {
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., head * hd..(head + 1) * hd]);
                let vh = v.slice(s![.., head * hd..(head + 1) * hd]);
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
            let attn = self.proj(&attn_out, &format!("{p}dense"));
            let mlp_in = if self.parallel { x.clone() } else { &x + &attn };
            let a2 = self.ln(&mlp_in, &format!("{p}ln2"));
            let mut hm = self.proj(&a2, &format!("{p}fc_in"));
            gelu(&mut hm);
            mlp_h.push(hm.row(seq - 1).to_vec());
            let mlp = &self.b.mm(&hm, &format!("{p}fc_out")) + &self.b.arr1(&format!("{p}fc_out.bias"));
            x = &(&x + &attn) + &mlp;
        }
        let xf = self.ln(&x, "ln_f");
        let x_last = x.row(seq - 1).to_vec(); // residual either side of the final norm — recovers its frozen scale
        let xf_last = xf.row(seq - 1).to_vec();
        let ln_bias = self.b.arr1("ln_f.bias").to_vec();
        let lg = self.b.rowdot_f32("lm_head", &xf_last);
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let gain = self.b.arr1("ln_f.weight").to_vec();
        let u_pred = self.b.weight_row("lm_head", model_predicts as usize);
        assemble(
            ids,
            &att_last,
            &head_act,
            &mlp_h,
            &lg,
            model_predicts,
            &gain,
            true,
            &ln_bias,
            &x_last,
            &xf_last,
            &u_pred,
            |l, n| self.b.weight_row(&format!("l{l}.fc_out"), n),
            |l, head| head_raw_contrib(&self.b, &format!("l{l}.dense"), &head_act[l], head, hd),
            |c| self.b.rowdot_f32("lm_head", c),
        )
    }

    /// Run `m` new positions through the layers, caching post-rope K and V (full width h*hd) and attending over the
    /// whole cache. cur = absolute position of the first new row. Prefill (m = prompt, cur = 0) or decode (m = 1).
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let (h, hd) = (self.h, self.hd);
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.ln(&x, &format!("{p}ln1"));
            let mut q = self.proj(&a, &format!("{p}q_proj"));
            let mut k = self.proj(&a, &format!("{p}k_proj"));
            let v = self.proj(&a, &format!("{p}v_proj"));
            self.rope(&mut q, h, cur);
            self.rope(&mut k, h, cur);
            kc[l].slice_mut(s![cur..klen, ..]).assign(&k);
            vc[l].slice_mut(s![cur..klen, ..]).assign(&v);
            let mut attn_out = Array2::<f32>::zeros((m, h * hd));
            for head in 0..h {
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = kc[l].slice(s![0..klen, head * hd..(head + 1) * hd]);
                let vh = vc[l].slice(s![0..klen, head * hd..(head + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
                for i in 0..m {
                    for j in (cur + i + 1)..klen {
                        scores[[i, j]] = -1e30;
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            let attn = self.proj(&attn_out, &format!("{p}dense"));
            if self.parallel {
                let mlp = self.mlp_block(&x, l);
                x = &(&x + &attn) + &mlp;
            } else {
                x = &x + &attn;
                let mlp = self.mlp_block(&x, l);
                x = &x + &mlp;
            }
        }
        self.ln(&x, "ln_f")
    }

    fn head_argmax(&self, xfn: &Array2<f32>) -> i64 {
        let logits = self.b.rowdot_f32("lm_head", &xfn.row(xfn.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}

impl Model for Neox {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids);
        let logits = self.b.rowdot_f32("lm_head", &xf.row(ids.len() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn explain(&self, ids: &[i64]) -> Option<crate::explain::Explanation> {
        Some(self.explanation(ids))
    }

    fn final_residual(&self, ids: &[i64]) -> Option<Vec<f32>> {
        let xf = self.hidden(ids); // post-ln_f residual; row(last) is the exact vector the unembedding dots
        Some(xf.row(ids.len() - 1).to_vec())
    }

    fn predict_ablated(&self, ids: &[i64], heads: &[(usize, usize)], neurons: &[(usize, usize)]) -> Option<i64> {
        let xf = self.hidden_ab(ids, heads, neurons);
        let logits = self.b.rowdot_f32("lm_head", &xf.row(ids.len() - 1).to_vec());
        Some(logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
    }

    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let total = prompt.len() + n_new;
        let kvdim = self.h * self.hd;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
        let mut vc = kc.clone();
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
        let kvdim = self.h * self.hd;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, kvdim))).collect();
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
        let (kvdim, n_layer) = (self.h * self.hd, self.n_layer);
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
