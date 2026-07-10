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
    fn hidden_ab(&self, ids: &[i64], ah: &[(usize, usize)], an: &[(usize, usize)],
                 ablk: &[usize], mblk: &[usize]) -> Array2<f32> {
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
            let mut attn = self.proj(&attn_out, &format!("{p}dense"));
            if ablk.contains(&l) { attn.fill(0.0); } // zero the whole attention block's residual write
            let mlp_in = if self.parallel { x.clone() } else { &x + &attn };
            let a2 = self.ln(&mlp_in, &format!("{p}ln2"));
            let mut hm = self.proj(&a2, &format!("{p}fc_in"));
            gelu(&mut hm);
            for &(al, nn) in an {
                if al == l {
                    hm.slice_mut(s![.., nn..nn + 1]).fill(0.0);
                }
            }
            let mut mlp = &self.down(&hm, &format!("{p}fc_out")) + &self.b.arr1(&format!("{p}fc_out.bias"));
            if mblk.contains(&l) { mlp.fill(0.0); } // zero the whole MLP block's residual write
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
            0,
            &|_v| Vec::new(),
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

    /// Forward (full prefill, like `hidden`) capturing the recursion substrate: the post-block residual of EVERY layer
    /// (pre-`ln_f` — for the per-layer lens) and the element-wise MAX over late-layer heads of the causal attention
    /// matrix (the binding signal; late = last third). The attention loop is inlined (to reach the per-head scores);
    /// the combine is the parallel residual `x + attn + mlp`, matching `hidden`. Shared by the recursion trace and the
    /// J-lens capture.
    fn recursion_capture(&self, ids: &[i64]) -> (Vec<Array2<f32>>, Array2<f32>) {
        let seq = ids.len();
        let (h, hd) = (self.h, self.hd);
        let late0 = 2 * self.n_layer / 3;
        let mut x = self.b.rows_f32("embed", ids);
        let mut resids: Vec<Array2<f32>> = Vec::with_capacity(self.n_layer);
        let mut maxback = Array2::<f32>::zeros((seq, seq));
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let a = self.ln(&x, &format!("{p}ln1"));
            let mut q = self.proj(&a, &format!("{p}q_proj"));
            let mut k = self.proj(&a, &format!("{p}k_proj"));
            let v = self.proj(&a, &format!("{p}v_proj"));
            self.rope(&mut q, h, 0);
            self.rope(&mut k, h, 0);
            let mut attn_out = Array2::<f32>::zeros((seq, h * hd));
            for head in 0..h {
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let kh = k.slice(s![.., head * hd..(head + 1) * hd]);
                let vh = v.slice(s![.., head * hd..(head + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
                for i in 0..seq {
                    for j in (i + 1)..seq { scores[[i, j]] = -1e30; }
                }
                softmax_rows(&mut scores);
                if l >= late0 {
                    for i in 0..seq {
                        for j in 0..=i {
                            if scores[[i, j]] > maxback[[i, j]] { maxback[[i, j]] = scores[[i, j]]; }
                        }
                    }
                }
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            let attn = self.proj(&attn_out, &format!("{p}dense"));
            // parallel: ln2 reads the SAME pre-attention x; sequential: ln2 reads x+attn. Residual sum is x+attn+mlp either way.
            let mlp_in = if self.parallel { x.clone() } else { &x + &attn };
            let mlp = self.mlp_block(&mlp_in, l);
            x = &(&x + &attn) + &mlp;
            resids.push(x.clone());
        }
        (resids, maxback)
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

    fn recursion_trace(&self, ids: &[i64]) -> Option<Vec<crate::model::RecPos>> {
        self.recursion_trace_lens(ids, None) // plain logit-lens is the J-lens with J_l = I at every layer
    }

    fn recursion_trace_lens(&self, ids: &[i64], jmats: Option<&[ndarray::Array2<f32>]>) -> Option<Vec<crate::model::RecPos>> {
        if ids.len() < 3 {
            return Some(vec![]);
        }
        let (resids, maxback) = self.recursion_capture(ids);
        // arch-specific lens: (optional J_l pre-multiply) then final LayerNorm ("ln_f") + `lm_head` unembed argmax.
        Some(crate::model::build_rec_trace(&resids, maxback, 2 * self.n_layer / 3, |l, resid| {
            let read = match jmats {
                Some(js) if l < js.len() => resid.dot(&js[l].t()), // (J_l @ resid[p]) per position; None ⇒ logit-lens
                _ => resid.clone(),
            };
            let normed = self.ln(&read, "ln_f");
            (0..normed.nrows())
                .map(|pp| {
                    let lg = self.b.rowdot_f32("lm_head", &normed.row(pp).to_vec());
                    lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
                })
                .collect()
        }))
    }

    fn jlens_capture(&self, ids: &[i64]) -> Option<Vec<ndarray::Array2<f32>>> {
        Some(self.recursion_capture(ids).0) // post-block residual of every layer (pre-ln_f); drop the maxback
    }

    /// Forward from just after `layer` through blocks `layer+1..n_layer`, returning the pre-`ln_f` final residual.
    /// Reuses `attn_block`/`mlp_block` with the parallel-residual combine — the JVP primitive the J-lens fit perturbs.
    fn jlens_forward_from(&self, layer: usize, x0: &ndarray::Array2<f32>) -> Option<ndarray::Array2<f32>> {
        let mut x = x0.clone();
        for l in (layer + 1)..self.n_layer {
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
        Some(x)
    }

    fn logits(&self, ids: &[i64]) -> Option<Vec<f32>> {
        let xf = self.hidden(ids);
        Some(self.b.rowdot_f32("lm_head", &xf.row(ids.len() - 1).to_vec()))
    }

    /// Per-block DLA decomposition (the `--pil-dump` / source-PR seam for the NeoX family). The residual writes
    /// — embedding, then each layer's attention `dense` and MLP `fc_out` outputs — sum to the pre-`ln_f` residual.
    /// `ln_f` is a LayerNorm (unlike rope's RMSNorm), so each block's contribution to a token's logit is
    /// `inv_std · Σ_d gain_d (write_b_d − mean_b) U_v_d`; the `ln_f` bias constant `Σ_d bias_d U_v_d` is folded into
    /// the embedding block so the per-block contributions sum to the exact logit. Reuses `attn_block`/`mlp_block`,
    /// so partial rotary / GELU / parallel residual / biases are all handled.
    fn residual_decomp(&self, ids: &[i64], toks: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        // = the per-block normed contribution vectors (`residual_normed_writes`) projected onto the token rows.
        let (labels, dvec) = self.residual_normed_writes(ids)?;
        let urows: Vec<Vec<f32>> = toks.iter().map(|&t| self.b.weight_row("lm_head", t as usize)).collect();
        let contrib: Vec<Vec<f32>> = dvec
            .iter()
            .map(|dd| urows.iter().map(|u| dd.iter().zip(u.iter()).map(|(a, b)| a * b).sum::<f32>()).collect())
            .collect();
        Some((labels, contrib))
    }

    fn residual_normed_writes(&self, ids: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        // Capture each residual write at the last position; fold the final LayerNorm into each into d̃_b living in
        // unembed space. LayerNorm centres per-block (mean(x)=Σ_b mean(write_b), so x_d−mean=Σ_b(write_b_d−mean_b)),
        // shares the scale inv_std, and the ln_f bias is folded into the embed block: d̃_b = inv·gain⊙(write_b−mean_b)
        // (+ bias on embed). Then ⟨d̃_b, U_v⟩ == block b's exact logit contribution, so recon is exact.
        let seq = ids.len();
        let last = seq - 1;
        let mut x = self.b.rows_f32("embed", ids);
        let mut labels: Vec<String> = vec!["embed".into()];
        let mut writes: Vec<Vec<f32>> = vec![x.row(last).to_vec()];
        for l in 0..self.n_layer {
            let attn = self.attn_block(&x, l);
            // parallel: ln2 reads the SAME pre-attention x; sequential: ln2 reads x+attn. Either way the two
            // writes (attn, mlp) sum into the residual: x_new = x + attn + mlp.
            let mlp = if self.parallel { self.mlp_block(&x, l) } else { self.mlp_block(&(&x + &attn), l) };
            writes.push(attn.row(last).to_vec());
            labels.push(format!("L{l}.attn"));
            writes.push(mlp.row(last).to_vec());
            labels.push(format!("L{l}.mlp"));
            x = &(&x + &attn) + &mlp;
        }
        // final LayerNorm geometry at the last position: ln_f(x)_d = gain_d (x_d − mean) / std + bias_d.
        let xpre = x.row(last).to_vec();
        let d = xpre.len();
        let mean: f32 = xpre.iter().sum::<f32>() / d as f32;
        let var: f32 = xpre.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + self.eps).sqrt();
        let gain = self.b.arr1("ln_f.weight");
        let bias = self.b.arr1("ln_f.bias");
        let dvec: Vec<Vec<f32>> = writes
            .iter()
            .enumerate()
            .map(|(bi, w)| {
                let mw: f32 = w.iter().sum::<f32>() / d as f32;
                (0..d).map(|i| inv * gain[i] * (w[i] - mw) + if bi == 0 { bias[i] } else { 0.0 }).collect()
            })
            .collect();
        Some((labels, dvec))
    }

    fn unembed_project(&self, v: &[f32]) -> Option<Vec<f32>> {
        Some(self.b.rowdot_f32("lm_head", v)) // logit-lens of any residual vector
    }

    fn unembed_row(&self, id: usize) -> Option<Vec<f32>> {
        let r = self.b.weight_row("lm_head", id);
        if r.is_empty() { None } else { Some(r) }
    }

    fn export_unembed(&self) -> Option<crate::jlens::UnembedExport> {
        // U = the untied `lm_head`; gamma = the final LayerNorm gain `ln_f.weight`. LayerNorm also mean-centers and
        // adds `ln_f.bias`, which a diagonal fold omits, so pil's gamma-conjugation is APPROXIMATE for neox.
        // `rows_f32` DEQUANTISES f32/f16/RowI8 rows (arr2o would hit the quantised-upcast guard on an int8 lm_head).
        let vocab = self.b.config[5] as usize; // config = [n_layer, H, hd, d, ffn, vocab, rot, parallel]
        Some(crate::jlens::UnembedExport {
            u: self.b.rows_f32("lm_head", &(0..vocab as i64).collect::<Vec<_>>()),
            gamma: self.b.arr1("ln_f.weight").to_vec(),
            norm_type: "layernorm",
            tied: false,
        })
    }

    fn predict_ablated(&self, ids: &[i64], heads: &[(usize, usize)], neurons: &[(usize, usize)]) -> Option<i64> {
        let xf = self.hidden_ab(ids, heads, neurons, &[], &[]);
        let logits = self.b.rowdot_f32("lm_head", &xf.row(ids.len() - 1).to_vec());
        Some(logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
    }

    /// Causal block ablation: zero whole attention/MLP blocks of the listed layers and recompute — the cross-arch
    /// `--block-ablate` sufficiency/necessity test for the decode circuit (the neox sibling of rope's method).
    fn predict_ablated_blocks(&self, ids: &[i64], heads: &[(usize, usize)], neurons: &[(usize, usize)],
                              attn_layers: &[usize], mlp_layers: &[usize]) -> Option<i64> {
        let xf = self.hidden_ab(ids, heads, neurons, attn_layers, mlp_layers);
        let logits = self.b.rowdot_f32("lm_head", &xf.row(ids.len() - 1).to_vec());
        Some(logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
    }

    fn dims(&self) -> Option<(usize, usize)> {
        Some((self.n_layer, self.h))
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
