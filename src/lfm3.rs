//! LFM3's tri-hybrid text backbone: short causal convolution plus full and sliding GQA attention.

use crate::bundle::Bundle;
use crate::model::Model;
use ndarray::{s, Array1, Array2, ArrayView1};

pub struct Lfm3 {
    b: Bundle,
    nl: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    eps: f32,
    ty: Vec<usize>,
    theta_full: f32,
    theta_slide: f32,
    partial_full: usize,
    partial_slide: usize,
    window: usize,
    tied: bool,
}

fn rms(x: &Array2<f32>, w: Array1<f32>, eps: f32) -> Array2<f32> {
    let mut y = x.clone();
    for mut r in y.rows_mut() {
        let z = 1.0 / ((r.iter().map(|v| v * v).sum::<f32>() / r.len() as f32) + eps).sqrt();
        for (i, v) in r.iter_mut().enumerate() {
            *v *= z * w[i];
        }
    }
    y
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softmax(a: &mut Array2<f32>) {
    for mut r in a.rows_mut() {
        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0.;
        for v in r.iter_mut() {
            *v = (*v - m).exp();
            z += *v;
        }
        for v in r.iter_mut() {
            *v /= z;
        }
    }
}

impl Lfm3 {
    pub fn new(b: Bundle) -> Self {
        let c = &b.config;
        Self {
            nl: c[0] as usize,
            nh: c[1] as usize,
            nkv: c[2] as usize,
            hd: c[3] as usize,
            eps: b.config_f[0] as f32,
            theta_full: b.config_f[1] as f32,
            theta_slide: b.config_f[2] as f32,
            partial_full: c[8] as usize,
            partial_slide: c[9] as usize,
            window: c[10] as usize,
            tied: c[7] != 0,
            ty: c[11..].iter().map(|&x| x as usize).collect(),
            b,
        }
    }
    fn rope(&self, x: &mut Array2<f32>, heads: usize, pos0: usize, theta: f32, partial: usize) {
        let half = self.hd / 2;
        for (i, mut r) in x.rows_mut().into_iter().enumerate() {
            for h in 0..heads {
                for j in 0..half {
                    if j >= partial / 2 {
                        continue;
                    }
                    let a = (pos0 + i) as f32 / theta.powf(2.0 * j as f32 / self.hd as f32);
                    let (c, s) = (a.cos(), a.sin());
                    let k = h * self.hd;
                    let (u, v) = (r[k + j], r[k + j + half]);
                    r[k + j] = u * c - v * s;
                    r[k + j + half] = v * c + u * s;
                }
            }
        }
    }
    fn head_norm(&self, x: &mut Array2<f32>, name: &str, heads: usize) {
        let w = self.b.arr1(name);
        for mut r in x.rows_mut() {
            for h in 0..heads {
                let o = h * self.hd;
                let z = 1.
                    / (((0..self.hd).map(|i| r[o + i] * r[o + i]).sum::<f32>() / self.hd as f32)
                        + self.eps)
                        .sqrt();
                for i in 0..self.hd {
                    r[o + i] *= z * w[i];
                }
            }
        }
    }
    fn conv(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let w = self.b.arr2o(name);
        let (n, d) = (x.nrows(), x.ncols());
        let k = w.ncols();
        let mut y = Array2::zeros((n, d));
        for i in 0..n {
            for c in 0..d {
                for j in 0..k {
                    if i >= k - 1 - j {
                        y[[i, c]] += x[[i - (k - 1 - j), c]] * w[[c, j]];
                    }
                }
            }
        }
        y
    }
    fn short_conv(&self, x: &Array2<f32>, p: &str) -> Array2<f32> {
        let d = x.ncols();
        let bc_x = self.b.mm(x, &format!("{p}conv.in_proj"));
        let mut bx = Array2::zeros((x.nrows(), d));
        for r in 0..x.nrows() {
            for c in 0..d {
                bx[[r, c]] = bc_x[[r, c]] * bc_x[[r, 2 * d + c]];
            }
        }
        let conv = self.conv(&bx, &format!("{p}conv.conv"));
        let mut y = conv;
        for r in 0..x.nrows() {
            for c in 0..d {
                y[[r, c]] *= bc_x[[r, d + c]];
            }
        }
        self.b.mm(&y, &format!("{p}conv.out_proj"))
    }
    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        self.forward(ids, Capture::default()).normed
    }

    /// Post-block residual snapshots used only for LFM3/reference parity diagnosis.
    pub fn hiddens(&self, ids: &[i64]) -> Vec<Vec<f32>> {
        self.forward(ids, Capture { snaps: true, writes: false })
            .snaps
            .into_iter()
            .map(|x| x.into_raw_vec_and_offset().0)
            .collect()
    }

    fn unembed(&self) -> &'static str {
        if self.tied { "embed" } else { "lm_head" }
    }

    /// Fold the final `embedding_norm` (weight-direct RMSNorm, no centre/bias) into a residual-stream write at one
    /// position: with the RMS frozen at the true pre-norm residual `xpre`, `d̃ = inv_rms · gain ⊙ write`, so
    /// `⟨d̃_b, U_v⟩` is block b's exact logit contribution and `Σ_b d̃_b` is the normed hidden the model unembeds.
    fn fold(&self, xpre: ArrayView1<f32>, gain: &Array1<f32>, write: ArrayView1<f32>) -> Vec<f32> {
        let d = xpre.len();
        let inv_rms = 1.0 / (xpre.iter().map(|v| v * v).sum::<f32>() / d as f32 + self.eps).sqrt();
        write.iter().zip(gain.iter()).map(|(w, g)| inv_rms * w * g).collect()
    }

    /// One forward. `writes` captures every residual-stream WRITE at all positions: the embedding, then per layer the
    /// operator block (`L{l}.conv` or `L{l}.attn`, i.e. `post_operator_norm(op)` — the sandwich norm is part of the
    /// block, it is exactly what gets added) and the FFN block (`L{l}.mlp`, `post_ffn_norm(ffn)`). By construction
    /// `Σ_b writes[b] == pre_norm` at every position. `mlp`, not `ffn`, so rosetta/pil's `L(\d+).(attn|mlp)` parse it.
    ///
    /// Memory: `writes` holds `(1 + 2·nl)` full `[seq, d]` f32 matrices — 49 × 2048 × 4 B ≈ 0.4 MB per token on
    /// LFM3-1.7B (≈130 MB at ctx 320, 1.6 GB at 4k, 13 GB at 32k). Only `decomp_all` needs every position;
    /// `residual_normed_writes` / `residual_decomp` / `decision_decomp` read just the last row, so a last-position-only
    /// capture mode is the natural fix once `explain` puts those paths on the chat/API surface.
    fn forward(&self, ids: &[i64], cap: Capture) -> Fwd {
        let n = ids.len();
        // Unlike Llama-family models, LFM3 applies `embedding_norm` to the decoder output,
        // not the token embeddings. The official checkpoints therefore have no separate final norm.
        let mut x = self.b.rows_f32("embed", ids);
        let mut snaps = Vec::with_capacity(if cap.snaps { self.nl } else { 0 });
        let mut writes: Vec<(String, Array2<f32>)> = Vec::new();
        if cap.writes {
            writes.push(("embed".into(), x.clone()));
        }
        let rep = self.nh / self.nkv;
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = rms(&x, self.b.arr1(&format!("{p}operator_norm")), self.eps);
            let op = if self.ty[l] == 0 {
                self.short_conv(&a, &p)
            } else {
                let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
                let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
                let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), self.nh);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), self.nkv);
                let theta = if self.ty[l] == 1 {
                    self.theta_full
                } else {
                    self.theta_slide
                };
                // LFM3's RoPE parameters are layer-type-specific: the reference 1.7B
                // rotates only half a head in full-attention blocks, but the whole head
                // in sliding-attention blocks.  Sharing the former width caused the
                // first meaningful residual divergence at the first sliding layer.
                let partial = if self.ty[l] == 1 { self.partial_full } else { self.partial_slide };
                self.rope(&mut q, self.nh, 0, theta, partial);
                self.rope(&mut k, self.nkv, 0, theta, partial);
                let mut out = Array2::zeros((n, self.nh * self.hd));
                for h in 0..self.nh {
                    let kv = h / rep;
                    let qh = q.slice(s![.., h * self.hd..(h + 1) * self.hd]);
                    let kh = k.slice(s![.., kv * self.hd..(kv + 1) * self.hd]);
                    let vh = v.slice(s![.., kv * self.hd..(kv + 1) * self.hd]);
                    let mut sc = qh.dot(&kh.t()) / (self.hd as f32).sqrt();
                    for i in 0..n {
                        for j in 0..n {
                            if j > i || (self.ty[l] == 2 && i - j >= self.window) {
                                sc[[i, j]] = -1e30;
                            }
                        }
                    }
                    softmax(&mut sc);
                    out.slice_mut(s![.., h * self.hd..(h + 1) * self.hd])
                        .assign(&sc.dot(&vh));
                }
                let g = self.b.mm(&a, &format!("{p}self_attn.gate_proj"));
                for (i, v) in out.iter_mut().enumerate() {
                    *v *= sigmoid(g[[i / (self.nh * self.hd), i % (self.nh * self.hd) / self.hd]]);
                }
                self.b.mm(&out, &format!("{p}self_attn.o_proj"))
            };
            let op_w = rms(&op, self.b.arr1(&format!("{p}post_operator_norm")), self.eps);
            let x1 = &x + &op_w;
            if cap.writes {
                writes.push((format!("L{l}.{}", if self.ty[l] == 0 { "conv" } else { "attn" }), op_w));
            }
            let f = rms(&x1, self.b.arr1(&format!("{p}ffn_norm")), self.eps);
            let mut h = self.b.mm(&f, &format!("{p}feed_forward.w1"));
            let w3 = self.b.mm(&f, &format!("{p}feed_forward.w3"));
            for (v, u) in h.iter_mut().zip(w3.iter()) {
                *v = silu(*v) * u;
            }
            let ff = self.b.mm(&h, &format!("{p}feed_forward.w2"));
            let ff_w = rms(&ff, self.b.arr1(&format!("{p}post_ffn_norm")), self.eps);
            x = &x1 + &ff_w;
            if cap.writes {
                writes.push((format!("L{l}.mlp"), ff_w));
            }
            if cap.snaps {
                snaps.push(x.clone());
            }
        }
        Fwd { normed: rms(&x, self.b.arr1("embedding_norm"), self.eps), pre_norm: x, snaps, writes }
    }

    /// Per-block contributions at position `pos` projected onto `cands`: `(label, ⟨d̃_b, U_c⟩ for c in cands)`.
    fn blocks_at(&self, f: &Fwd, pos: usize, cands: &[i64]) -> Vec<(String, Vec<f32>)> {
        let gain = self.b.arr1("embedding_norm");
        let urows: Vec<Vec<f32>> = cands.iter().map(|&c| self.b.weight_row(self.unembed(), c as usize)).collect();
        f.writes
            .iter()
            .map(|(lab, w)| {
                let dv = self.fold(f.pre_norm.row(pos), &gain, w.row(pos));
                (lab.clone(), urows.iter().map(|u| dv.iter().zip(u).map(|(a, b)| a * b).sum::<f32>()).collect())
            })
            .collect()
    }

    /// The decision at `pos` from the TRUE forward's logits (so `predicted` is exactly what `predict` returns, never a
    /// near-tie flipped by re-summing blocks) + the candidate set: top-`cap` by logit when `extra` is None (decomp_all),
    /// else {predicted, runner-up} ∪ extra capped at `cap` (decision_decomp).
    fn decision_at(&self, f: &Fwd, pos: usize, extra: Option<&[i64]>, cap: usize) -> Decision {
        let logits = self.b.rowdot_f32(self.unembed(), &f.normed.row(pos).to_vec());
        let mut order: Vec<usize> = (0..logits.len()).collect();
        // total order, NaN-safe: a NaN logit sorts last instead of panicking the comparator
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let (pred, runner) = (order[0], *order.get(1).unwrap_or(&order[0]));
        let cands: Vec<i64> = match extra {
            None => order.iter().take(cap.max(1)).map(|&i| i as i64).collect(),
            Some(extra) => {
                let mut c = vec![pred as i64];
                if runner != pred { c.push(runner as i64); }
                for &x in extra {
                    if c.len() >= cap { break; }
                    if !c.contains(&x) { c.push(x); }
                }
                c
            }
        };
        let blocks = self.blocks_at(f, pos, &cands);
        (pred as i64, runner as i64, logits[pred], logits[runner], cands, blocks)
    }
}

type Decision = (i64, i64, f32, f32, Vec<i64>, Vec<(String, Vec<f32>)>);

#[derive(Default, Clone, Copy)]
struct Capture {
    snaps: bool,
    writes: bool,
}

struct Fwd {
    normed: Array2<f32>,   // embedding_norm(x): what the model unembeds
    pre_norm: Array2<f32>, // x before the final norm == Σ_b writes[b]
    snaps: Vec<Array2<f32>>,
    writes: Vec<(String, Array2<f32>)>,
}
impl Model for Lfm3 {
    fn predict(&self, ids: &[i64]) -> i64 {
        let x = self.hidden(ids);
        let un = if self.tied { "embed" } else { "lm_head" };
        self.b
            .rowdot_f32(un, &x.row(x.nrows() - 1).to_vec())
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i64
    }

    fn logits(&self, ids: &[i64]) -> Option<Vec<f32>> {
        if ids.is_empty() {
            return None;
        }
        let x = self.hidden(ids);
        let un = if self.tied { "embed" } else { "lm_head" };
        Some(self.b.rowdot_f32(un, &x.row(x.nrows() - 1).to_vec()))
    }

    fn residual_normed_writes(&self, ids: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        if ids.is_empty() { return None; }
        let f = self.forward(ids, Capture { snaps: false, writes: true });
        let (last, gain) = (ids.len() - 1, self.b.arr1("embedding_norm"));
        Some(f.writes.iter().map(|(lab, w)| (lab.clone(), self.fold(f.pre_norm.row(last), &gain, w.row(last)))).unzip())
    }

    fn residual_decomp(&self, ids: &[i64], toks: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        if ids.is_empty() { return None; }
        let f = self.forward(ids, Capture { snaps: false, writes: true });
        Some(self.blocks_at(&f, ids.len() - 1, toks).into_iter().unzip())
    }

    fn decision_decomp(&self, ids: &[i64], extra: &[i64], cap: usize) -> Option<Decision> {
        if ids.is_empty() { return None; }
        let f = self.forward(ids, Capture { snaps: false, writes: true });
        Some(self.decision_at(&f, ids.len() - 1, Some(extra), cap))
    }

    fn decomp_all(&self, ids: &[i64], cap: usize) -> Option<Vec<Decision>> {
        if ids.is_empty() { return None; }
        let f = self.forward(ids, Capture { snaps: false, writes: true });
        Some((0..ids.len()).map(|p| self.decision_at(&f, p, None, cap)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::Bundle;

    fn v(seed: usize, n: usize) -> Vec<f32> {
        (0..n).map(|i| (((seed * 41 + i * 23) % 43) as f32) / 20.0 - 1.0).collect()
    }
    fn gain(seed: usize, n: usize) -> Vec<f32> {
        v(seed, n).into_iter().map(|x| 1.0 + 0.5 * x).collect()
    }

    /// A tiny LFM3 bundle written straight to disk: conv / sliding / conv / full / sliding / full, window 2 < seq,
    /// GQA 4/2, full RoPE width 4 vs sliding 8 (of hd 8). Own path per test — `cargo test` runs them concurrently.
    fn build_tiny(tag: &str, tied: bool) -> Lfm3 {
        let (nh, nkv, hd, d, vocab, ffn, k) = (4usize, 2, 8, 32, 24, 16, 3);
        let ty = [0usize, 2, 0, 1, 2, 1];
        let dir = std::env::temp_dir().join(format!("fr_lfm3_{tag}_{}", std::process::id()));
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
        add("embedding_norm".into(), vec![d], gain(2, d));
        if !tied { add("lm_head".into(), vec![vocab, d], v(3, vocab * d)); }
        for (l, &t) in ty.iter().enumerate() {
            let p = format!("l{l}.");
            for (i, n) in ["operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"].iter().enumerate() {
                add(format!("{p}{n}"), vec![d], gain(10 + 7 * l + i, d));
            }
            add(format!("{p}feed_forward.w1"), vec![d, ffn], v(60 + l, d * ffn));
            add(format!("{p}feed_forward.w3"), vec![d, ffn], v(70 + l, d * ffn));
            add(format!("{p}feed_forward.w2"), vec![ffn, d], v(80 + l, ffn * d));
            if t == 0 {
                add(format!("{p}conv.conv"), vec![d, k], v(90 + l, d * k));
                add(format!("{p}conv.in_proj"), vec![d, 3 * d], v(100 + l, d * 3 * d));
                add(format!("{p}conv.out_proj"), vec![d, d], v(110 + l, d * d));
            } else {
                add(format!("{p}self_attn.q_norm"), vec![hd], gain(120 + l, hd));
                add(format!("{p}self_attn.k_norm"), vec![hd], gain(130 + l, hd));
                add(format!("{p}self_attn.q_proj"), vec![d, nh * hd], v(140 + l, d * nh * hd));
                add(format!("{p}self_attn.k_proj"), vec![d, nkv * hd], v(150 + l, d * nkv * hd));
                add(format!("{p}self_attn.v_proj"), vec![d, nkv * hd], v(160 + l, d * nkv * hd));
                add(format!("{p}self_attn.o_proj"), vec![nh * hd, d], v(170 + l, nh * hd * d));
                add(format!("{p}self_attn.gate_proj"), vec![d, nh], v(180 + l, d * nh));
            }
        }
        // [nl, nh, nkv, hd, d, vocab, ffn, tied, partial_full, partial_slide, window, <layer types>]
        let mut cfg: Vec<i64> = vec![ty.len() as i64, nh as i64, nkv as i64, hd as i64, d as i64, vocab as i64,
                                     ffn as i64, tied as i64, 4, 8, 2];
        cfg.extend(ty.iter().map(|&t| t as i64));
        let manifest = serde_json::json!({ "format": crate::bundle::FORMAT, "version": crate::bundle::VERSION,
            "arch": "lfm3", "config": cfg, "config_f": [1e-5, 1000.0, 100.0], "arrays": specs });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
        Lfm3::new(Bundle::load(&stem).unwrap())
    }

    const IDS: [i64; 7] = [3, 17, 5, 22, 9, 1, 14];
    /// First 24 LFM3-tokenizer ids of fieldrun's README.md ("# fieldrun\n\nA single static **binary** …").
    const REAL_IDS: [i64; 24] = [11, 2352, 10943, 8, 41, 2936, 4486, 7575, 94405, 962, 342, 1319, 43090, 115336, 17,
                                 395, 10295, 401, 35185, 53, 487, 267, 9144, 58792];

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * b.abs().max(1.0)
    }

    /// LE-T5 on LFM3: Σ_b contrib[b][v] == logit(v) for every token, both unembed variants. The sum is checked over
    /// the WHOLE vocab, not a sample — a mislabelled or dropped block (e.g. folding the pre-sandwich op) breaks it.
    #[test]
    fn residual_decomp_sums_to_logits() {
        for tied in [true, false] {
            let m = build_tiny(if tied { "sum_t" } else { "sum_u" }, tied);
            let logits = m.logits(&IDS).unwrap();
            let toks: Vec<i64> = (0..logits.len() as i64).collect();
            let (labels, contrib) = m.residual_decomp(&IDS, &toks).unwrap();
            assert_eq!(labels.len(), contrib.len());
            for (i, &lg) in logits.iter().enumerate() {
                let s: f32 = contrib.iter().map(|c| c[i]).sum();
                assert!(close(s, lg), "tied={tied} tok {i}: Σ blocks {s} != logit {lg}");
            }
        }
    }

    /// One label per residual write, in forward order, naming the operator by layer type; `mlp` (not `ffn`) so the
    /// sibling-repo `L(\d+)\.(attn|mlp)` parsers bin the FFN correctly.
    #[test]
    fn block_labels_follow_layer_types() {
        let m = build_tiny("labels", true);
        let (labels, dvec) = m.residual_normed_writes(&IDS).unwrap();
        let want = ["embed", "L0.conv", "L0.mlp", "L1.attn", "L1.mlp", "L2.conv", "L2.mlp", "L3.attn", "L3.mlp",
                    "L4.attn", "L4.mlp", "L5.attn", "L5.mlp"];
        assert_eq!(labels, want);
        // Σ_b d̃_b is the normed hidden the model unembeds (the vector form of the logit identity).
        let normed = m.hidden(&IDS);
        let last = normed.row(IDS.len() - 1);
        for j in 0..last.len() {
            let s: f32 = dvec.iter().map(|d| d[j]).sum();
            assert!(close(s, last[j]), "dim {j}: Σ d̃ {s} != normed {}", last[j]);
        }
    }

    /// The fast paths agree with the slow one: decision_decomp's prediction IS predict's (taken from the true logits,
    /// not re-summed blocks), its blocks equal residual_decomp's, and decomp_all's entry p equals decision on ids[..=p].
    #[test]
    fn fast_paths_match_residual_decomp() {
        let m = build_tiny("fast", false);
        let (pred, runner, pl, rl, cands, blocks) = m.decision_decomp(&IDS, &[7, 7, 2], 4).unwrap();
        assert_eq!(pred, m.predict(&IDS));
        let logits = m.logits(&IDS).unwrap();
        assert!(close(pl, logits[pred as usize]) && close(rl, logits[runner as usize]) && pl >= rl);
        assert_eq!(cands.len(), 4, "{{pred, runner}} ∪ extra (deduped) capped at 4: {cands:?}");
        let (_, slow) = m.residual_decomp(&IDS, &cands).unwrap();
        for ((_, fast), slow) in blocks.iter().zip(&slow) {
            for (a, b) in fast.iter().zip(slow) { assert!(close(*a, *b)); }
        }
        let all = m.decomp_all(&IDS, 3).unwrap();
        assert_eq!(all.len(), IDS.len());
        for (p, (pp, _, ppl, _, pc, pb)) in all.iter().enumerate() {
            let ctx = &IDS[..=p];
            assert_eq!(*pp, m.predict(ctx), "pos {p}");
            let lg = m.logits(ctx).unwrap();
            assert!(close(*ppl, lg[*pp as usize]));
            assert_eq!(pc.len(), 3);
            for (i, &c) in pc.iter().enumerate() {
                let s: f32 = pb.iter().map(|(_, w)| w[i]).sum();
                assert!(close(s, lg[c as usize]), "pos {p} cand {c}: Σ {s} != {}", lg[c as usize]);
            }
        }
    }

    /// The window binds in this fixture (else the decomposition tests above would not cover sliding layers' masks).
    #[test]
    fn tiny_window_is_load_bearing() {
        let m = build_tiny("win", true);
        let mut wide = build_tiny("win_wide", true);
        wide.window = 1000;
        assert_ne!(m.logits(&IDS).unwrap(), wide.logits(&IDS).unwrap());
    }

    /// The same identity on a REAL checkpoint, over the full vocab (128k for LFM3-1.7B). Opt-in: needs a converted
    /// bundle — `FIELDRUN_LFM3_BUNDLE=<stem> cargo test --release --bin fieldrun lfm3::tests::real -- --ignored`.
    /// Context: the first 24 LFM3 tokens of fieldrun's README.md (past the 2- and 3-tap conv history, so conv taps
    /// read real tokens). The tolerance is relative to the logit scale because 49 f32 block sums are accumulated.
    #[test]
    #[ignore]
    fn real_bundle_decomp_sums_to_logits() {
        let Ok(stem) = std::env::var("FIELDRUN_LFM3_BUNDLE") else { panic!("set FIELDRUN_LFM3_BUNDLE=<bundle stem>") };
        let m = Lfm3::new(Bundle::load(&stem).unwrap());
        let ids: [i64; 24] = REAL_IDS;
        let logits = m.logits(&ids).unwrap();
        let toks: Vec<i64> = (0..logits.len() as i64).collect();
        let (labels, contrib) = m.residual_decomp(&ids, &toks).unwrap();
        assert_eq!(labels.len(), 1 + 2 * m.nl);
        let scale = logits.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let worst = logits.iter().enumerate()
            .map(|(i, &lg)| (contrib.iter().map(|c| c[i]).sum::<f32>() - lg).abs())
            .fold(0f32, f32::max);
        eprintln!("real lfm3: {} blocks x {} toks, max |Σ blocks - logit| = {worst:.3e} (max |logit| {scale:.2})",
                  labels.len(), toks.len());
        assert!(worst <= 1e-4 * scale.max(1.0), "max |Σ blocks - logit| {worst} vs scale {scale}");
        assert_eq!(m.decision_decomp(&ids, &[], 2).unwrap().0, m.predict(&ids));
    }
}
