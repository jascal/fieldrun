//! Explain a prediction — the Rust port of pylm's `explain.py` composition side. For a context, it reads the live
//! circuits off the real forward pass and ranks them by **direct logit attribution (DLA) to the token the model actually
//! predicts**: each attention head and MLP neuron is scored by how much it pushes the predicted token's logit, not by
//! raw activation magnitude. Heads still carry their `idiom_library` name (previous-token / duplicate-token / induction;
//! attention-sink heads collapsed to a NO-OP count) and where they read; neurons still carry the tokens they promote.
//! The predicting frame also reports the predicted logit and its margin over the runner-up. Token ids are decoded to
//! strings when a tokenizer is supplied. This is the surface the API serves.

use serde::Serialize;

/// How many heads (by residual-write norm) and neurons (by |activation|) we cheaply pre-filter before computing exact
/// DLA — and how many of each survive into the rendered frame. The pre-filter keeps the per-frame weight reads bounded
/// (explain is off the hot path but still re-runs a full forward pass per generated token); a component can only have a
/// large DLA if it writes a sizeable residual contribution, so a generous norm/|act| shortlist won't drop the winner.
const HEAD_CANDIDATES: usize = 64;
const NEURON_CAND_PER_LAYER: usize = 16;
const HEAD_SHOW: usize = 6;
const MLP_SHOW: usize = 6;

#[derive(Serialize)]
pub struct HeadCircuit {
    pub layer: usize,
    pub head: usize,
    pub role: String,
    pub attends_to: usize,
    pub attends_tok: i64, // the token the head READS (at the attended position)
    pub mass: f32,
    pub dla: f32,           // direct logit attribution: this head's contribution to the predicted token's logit
    pub promotes: Vec<i64>, // the tokens the head WRITES to the logits (top of its OV→unembed projection)
}

#[derive(Serialize)]
pub struct MlpFeature {
    pub layer: usize,
    pub neuron: usize,
    pub act: f32,
    pub dla: f32,           // direct logit attribution: this neuron's contribution to the predicted token's logit
    pub promotes: Vec<i64>,
}

#[derive(Serialize)]
pub struct Explanation {
    pub context_tail: Vec<i64>,
    pub model_predicts: i64,
    pub predicted_logit: f32,   // the logit of the predicted token at this position
    pub runner_up: i64,         // the second-place token
    pub runner_up_logit: f32,   // its logit — `predicted_logit - runner_up_logit` is the decision margin
    pub head_circuits: Vec<HeadCircuit>,
    pub sink_heads: usize,
    pub mlp_features: Vec<MlpFeature>,
    /// Every scored circuit's DLA to the predicted token (all ~64 head + ~384 neuron candidates, not just the shown
    /// top-6+6) — for concentration/participation-ratio analysis (`--probe-dla`). Not serialized (a probe aid).
    #[serde(skip)]
    pub all_dla: Vec<f32>,
}

/// How much of the explain trace to render — modelled on a database `EXPLAIN`'s option levels, because the deeper
/// facets cost the reader attention AND real compute (the circuit facet re-runs the faithful forward, the `ANALYZE`
/// analogue). `Route` is free (the route label is computed from the generated token + the KB, no extra forward);
/// `Circuits` adds the DLA circuit breakdown but ONLY on COMPOSED tokens (the attribution drives the verbosity — you
/// pay the explain-forward exactly where the model actually composed); `All` shows the circuits for every token.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExplainMode {
    Route,
    Circuits,
    All,
}

impl ExplainMode {
    /// Parse a mode name (case-insensitive). `on`/`true`/`1` alias `Route` (the cheap default); unknown → None.
    pub fn parse(s: &str) -> Option<ExplainMode> {
        match s.to_ascii_lowercase().as_str() {
            "route" | "routes" | "on" | "true" | "1" => Some(ExplainMode::Route),
            "circuits" | "circuit" | "dla" => Some(ExplainMode::Circuits),
            "all" | "full" | "verbose" => Some(ExplainMode::All),
            _ => None,
        }
    }
}

/// Name an attention head's behaviour at the predicting position from its attention row (length seq), using the
/// `idiom_library` signatures: where the last token attends tells us the circuit it runs.
pub fn classify_head(row: &[f32], ctx: &[i64]) -> (&'static str, usize, f32) {
    let seq = ctx.len();
    let (mut j, mut mass) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > mass {
            mass = v;
            j = i;
        }
    }
    let cur = ctx[seq - 1];
    let role = if j == 0 && seq > 1 {
        "sink"
    } else if j == seq - 2 {
        "previous-token"
    } else if j > 0 && j < seq && ctx[j - 1] == cur {
        "induction"
    } else if ctx[j] == cur {
        "duplicate-token"
    } else {
        "diffuse"
    };
    (role, j, mass)
}

/// Top-`k` token ids a contribution promotes, ranked by `sign * logit`. With `sign = 1.0` (the usual call here, where the
/// contribution already has its activation folded in) this is "the tokens this component currently pushes up".
pub fn top_promoted(logits: &[f32], sign: f32, k: usize) -> Vec<i64> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| (sign * logits[b]).total_cmp(&(sign * logits[a])));
    let mut top = idx[0..k].to_vec();
    top.sort_by(|&a, &b| (sign * logits[b]).total_cmp(&(sign * logits[a])));
    top.iter().map(|&i| i as i64).collect()
}

/// The raw residual contribution of one attention head at the predicting position: its `hd`-wide slice of `attn_last`
/// (attn_out's last row) routed through its `hd` rows of the output projection `o_proj`. The final norm + unembed are
/// applied by the caller (`apply_final_norm`, then `dla`/`top_promoted`), so this stays the arch-agnostic OV/output side
/// — the measured counterpart to "where it attends" (the QK side). For a real copy/induction head the projected result
/// contains `attends_tok`; a head doing intermediate work shows low-signal tokens and a small DLA.
pub fn head_raw_contrib(b: &crate::bundle::Bundle, o_proj: &str, attn_last: &[f32], head: usize, hd: usize) -> Vec<f32> {
    let base = head * hd;
    let mut acc: Vec<f32> = Vec::new();
    for i in 0..hd {
        let w = b.weight_row(o_proj, base + i); // o_proj row (base+i): a d-vector (the output for that value-dim)
        if acc.is_empty() {
            acc = vec![0.0f32; w.len()]; // d
        }
        let a = attn_last[base + i];
        for (o, wi) in acc.iter_mut().zip(&w) {
            *o += a * wi;
        }
    }
    acc
}

/// Apply the *linear* part of the final norm to a raw residual contribution `c` — optionally center (subtract the mean,
/// for a LayerNorm model like GPT-2) then scale by the per-dim `gain`. Both are linear, so they distribute over the
/// residual's component decomposition. The remaining 1/rms factor is a single positive scalar shared by every component
/// and token; it's deliberately left out here (it can't change a ranking) and folded back in once, as `final_scale`, so
/// the reported Δ lands in true logit units. Dotting the result with a token's unembed row × `final_scale` gives that
/// component's direct logit attribution to the token.
fn apply_final_norm(mut c: Vec<f32>, gain: &[f32], center: bool) -> Vec<f32> {
    if center && !c.is_empty() {
        let m = c.iter().sum::<f32>() / c.len() as f32;
        c.iter_mut().for_each(|v| *v -= m);
    }
    c.iter_mut().zip(gain).for_each(|(v, g)| *v *= g);
    c
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Recover the final norm's frozen 1/rms (RMSNorm) or 1/std (LayerNorm) scalar from the forward pass, so per-component Δs
/// are reported in true logit units (and sum — over *all* components — to the predicted logit minus the LN bias term).
/// The norm is affine in the residual: `xf = scale · gain ⊙ (x − mean) + bias`, so with `gnx = gain ⊙ (x − mean)` (i.e.
/// `apply_final_norm(x)`) the scale is the well-conditioned ratio `⟨xf − bias, gnx⟩ / ⟨gnx, gnx⟩`. Eps-free and exact for
/// both norms; `bias` is empty for RMSNorm. Returns 1.0 in the degenerate all-zero-residual case.
fn final_norm_scale(x_last: &[f32], xf_last: &[f32], gain: &[f32], center: bool, bias: &[f32]) -> f32 {
    let gnx = apply_final_norm(x_last.to_vec(), gain, center);
    let denom = dot(&gnx, &gnx);
    if denom <= 1e-12 {
        return 1.0;
    }
    let num: f32 = if bias.is_empty() {
        dot(xf_last, &gnx)
    } else {
        xf_last.iter().zip(bias).zip(&gnx).map(|((&xf, &b), &g)| (xf - b) * g).sum()
    };
    num / denom
}

/// Assemble an Explanation from captured per-layer attention + MLP activations (arch-agnostic), ranking heads and neurons
/// by direct logit attribution to the predicted token. The arch supplies the pieces that depend on its weight names:
/// `neuron_write(l, n)` → neuron n's raw down-projection write row (d); `head_raw(l, h)` → the head's raw OV contribution
/// (d, e.g. via `head_raw_contrib`); `project_vocab(c)` → a post-norm contribution projected through the unembed (for the
/// displayed "promotes" tokens). `gain`/`center`/`final_bias` describe the final norm (`final_bias` empty for RMSNorm);
/// `x_last`/`xf_last` are the residual at the predicting position just before / just after that norm (to recover its
/// frozen scale, so Δ is in true logit units); `u_pred` is the predicted token's unembed row; `logits` are the final
/// logits at the predicting position (for the predicted logit + runner-up margin).
#[allow(clippy::too_many_arguments)]
pub fn assemble<NW, HR, PV>(
    ids: &[i64],
    att_last: &[Vec<Vec<f32>>],
    head_act: &[Vec<f32>],
    mlp_h: &[Vec<f32>],
    logits: &[f32],
    model_predicts: i64,
    gain: &[f32],
    center: bool,
    final_bias: &[f32],
    x_last: &[f32],
    xf_last: &[f32],
    u_pred: &[f32],
    neuron_write: NW,
    head_raw: HR,
    project_vocab: PV,
) -> Explanation
where
    NW: Fn(usize, usize) -> Vec<f32>,
    HR: Fn(usize, usize) -> Vec<f32>,
    PV: Fn(&[f32]) -> Vec<f32>,
{
    let predicted_logit = logits.get(model_predicts as usize).copied().unwrap_or(0.0);
    let (mut runner_up, mut runner_up_logit) = (0i64, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if i as i64 != model_predicts && v > runner_up_logit {
            runner_up_logit = v;
            runner_up = i as i64;
        }
    }
    // The frozen final-norm scale puts each component's Δ in true logit units: ranking by `dla` is unaffected (a shared
    // positive factor), but the printed Δ now reconciles with `logit`/`margin` and is roughly additive across components.
    let final_scale = final_norm_scale(x_last, xf_last, gain, center, final_bias);

    // ---- attention heads ----------------------------------------------------------------------------------------
    // Pass 1 (no weight reads): classify every head once (kept for the sink/NO-OP count and the displayed label/read
    // position) and shortlist candidates by the L2 norm of their residual write — a head can only have large DLA if it
    // writes a sizeable contribution. NB: sink heads are no longer hard-excluded from the shown list; they're DLA
    // candidates like any other head, but write little, so they rank themselves out in practice.
    let mut sink_heads = 0usize;
    let mut cands: Vec<(usize, usize, &'static str, usize, f32, f32)> = Vec::new(); // (l, h, role, attends_to, mass, norm)
    for (l, la) in att_last.iter().enumerate() {
        let nh = la.len();
        if nh == 0 || head_act[l].is_empty() {
            continue;
        }
        // Layout invariant: the attention output the arch captured into head_act is the concatenation of nh heads, so its
        // width is exactly nh * head_dim — this derives head_dim for the per-head slice below and must agree with the
        // head_dim the arch's head_raw closure uses (matters most for gemma4's per-layer dim and MLA's value-head dim).
        debug_assert_eq!(head_act[l].len() % nh, 0, "head_act width must be nh * head_dim");
        let hd = head_act[l].len() / nh;
        for (h, row) in la.iter().enumerate() {
            let (role, j, mass) = classify_head(row, ids);
            if role == "sink" && mass >= 0.5 {
                sink_heads += 1;
            }
            let norm = head_act[l][h * hd..h * hd + hd].iter().map(|v| v * v).sum();
            cands.push((l, h, role, j, mass, norm));
        }
    }
    let kc = HEAD_CANDIDATES.min(cands.len());
    if kc > 0 && kc < cands.len() {
        cands.select_nth_unstable_by(kc - 1, |a, b| b.5.total_cmp(&a.5));
        cands.truncate(kc);
    }
    let mut head_circuits: Vec<HeadCircuit> = cands
        .iter()
        .map(|&(l, h, role, j, mass, _)| {
            let c = apply_final_norm(head_raw(l, h), gain, center);
            let dla = dot(&c, u_pred) * final_scale;
            HeadCircuit { layer: l, head: h, role: role.into(), attends_to: j, attends_tok: ids[j], mass, dla, promotes: Vec::new() }
        })
        .collect();
    head_circuits.sort_by(|a, b| b.dla.total_cmp(&a.dla));
    // capture every candidate head's DLA before truncating to the shown few (full-spectrum concentration analysis).
    let mut all_dla: Vec<f32> = head_circuits.iter().map(|h| h.dla).collect();
    head_circuits.truncate(HEAD_SHOW);
    // Recompute the post-norm contribution for the few shown heads to fill in their "promotes" (a full-vocab projection,
    // so it's done for HEAD_SHOW heads, not all candidates). The head_raw re-read is intentional: caching every
    // candidate's d-vector to save HEAD_SHOW reads would cost far more memory than it saves on this off-hot-path surface.
    for hc in head_circuits.iter_mut() {
        let c = apply_final_norm(head_raw(hc.layer, hc.head), gain, center);
        hc.promotes = top_promoted(&project_vocab(&c), 1.0, 5);
        debug_assert!(hc.attends_to < ids.len(), "attends_to out of range");
    }

    // ---- MLP neurons --------------------------------------------------------------------------------------------
    // Pre-filter each layer's top-|act| neurons (no weight reads), then score the shortlist by exact DLA to the predicted
    // token. This is the key change from ranking by raw |activation|: a high-magnitude suppression/normalisation neuron
    // that does not write toward the predicted token now drops out, and the neuron that actually wrote it surfaces.
    let mut ncands: Vec<(usize, usize, f32)> = Vec::new();
    for (l, h) in mlp_h.iter().enumerate() {
        if h.is_empty() {
            continue;
        }
        let k = NEURON_CAND_PER_LAYER.min(h.len());
        let mut idx: Vec<usize> = (0..h.len()).collect();
        if k < h.len() {
            idx.select_nth_unstable_by(k - 1, |&a, &b| h[b].abs().total_cmp(&h[a].abs()));
        }
        for &n in &idx[..k] {
            ncands.push((l, n, h[n]));
        }
    }
    let mut mlp_features: Vec<MlpFeature> = ncands
        .iter()
        .map(|&(l, n, act)| {
            let mut w = neuron_write(l, n);
            w.iter_mut().for_each(|v| *v *= act);
            let c = apply_final_norm(w, gain, center);
            MlpFeature { layer: l, neuron: n, act, dla: dot(&c, u_pred) * final_scale, promotes: Vec::new() }
        })
        .collect();
    mlp_features.sort_by(|a, b| b.dla.total_cmp(&a.dla));
    all_dla.extend(mlp_features.iter().map(|f| f.dla)); // + every candidate neuron's DLA
    mlp_features.truncate(MLP_SHOW);
    for f in mlp_features.iter_mut() {
        let mut w = neuron_write(f.layer, f.neuron);
        w.iter_mut().for_each(|v| *v *= f.act);
        let c = apply_final_norm(w, gain, center);
        f.promotes = top_promoted(&project_vocab(&c), 1.0, 5);
    }

    // store the FULL context (it's just the input ids) — render trims the printed preview, but nothing is lost: the
    // forward pass and every head's attends_to reference the whole sequence.
    Explanation {
        context_tail: ids.to_vec(),
        model_predicts,
        predicted_logit,
        runner_up,
        runner_up_logit,
        head_circuits,
        sink_heads,
        mlp_features,
        all_dla,
    }
}

/// Render an explanation as human-readable text. `dec` maps a token id to a display string; `max_ctx` is how many
/// trailing context tokens to print (0 = all). This only trims the preview — the model always saw the full context.
pub fn render(ex: &Explanation, dec: &dyn Fn(i64) -> String, max_ctx: usize) -> String {
    let n = ex.context_tail.len();
    let start = if max_ctx == 0 || max_ctx >= n { 0 } else { n - max_ctx };
    let lead = if start > 0 { "…" } else { "" };
    let ctx = ex.context_tail[start..].iter().map(|&t| dec(t)).collect::<Vec<_>>().join(" ");
    let margin = ex.predicted_logit - ex.runner_up_logit;
    let mut l = vec![
        format!("context {lead}{ctx}"),
        format!("model predicts {}  logit {:.2}  (margin {:+.2} vs runner-up {})", dec(ex.model_predicts), ex.predicted_logit, margin, dec(ex.runner_up)),
        format!("  COMPOSITION  content head circuits ({} idle on sink/NO-OP) — ranked by Δlogit→predicted (reads → writes):", ex.sink_heads),
    ];
    for h in &ex.head_circuits {
        let mut line = format!("    L{}.H{:<2} {:<15} Δ{:+.2}  reads {} (mass {:.3})", h.layer, h.head, h.role, h.dla, dec(h.attends_tok), h.mass);
        if !h.promotes.is_empty() {
            let toks = h.promotes.iter().map(|&t| dec(t)).collect::<Vec<_>>().join(", ");
            line.push_str(&format!("  ⇒ writes {{{toks}}}"));
        }
        l.push(line);
    }
    if ex.head_circuits.is_empty() {
        l.push("    (no attention contribution above threshold — carried by MLP features below)".to_string());
    }
    l.push("  COMPOSITION  top MLP features by Δlogit→predicted (neuron → tokens it promotes):".to_string());
    for f in &ex.mlp_features {
        let toks = f.promotes.iter().map(|&t| dec(t)).collect::<Vec<_>>().join(", ");
        l.push(format!("    L{} n{:<5} act {:<+8.2} Δ{:+.2} → {{{}}}", f.layer, f.neuron, f.act, f.dla, toks));
    }
    l.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explain_mode_parses_levels() {
        assert_eq!(ExplainMode::parse("route"), Some(ExplainMode::Route));
        assert_eq!(ExplainMode::parse("on"), Some(ExplainMode::Route)); // alias for the cheap default
        assert_eq!(ExplainMode::parse("CIRCUITS"), Some(ExplainMode::Circuits)); // case-insensitive
        assert_eq!(ExplainMode::parse("all"), Some(ExplainMode::All));
        assert_eq!(ExplainMode::parse("--raw"), None); // a following flag isn't a mode → caller defaults to Route
    }

    #[test]
    fn top_promoted_picks_largest_signed() {
        let logits = [0.1, 5.0, -3.0, 2.0, 0.0];
        assert_eq!(top_promoted(&logits, 1.0, 2), vec![1, 3]); // largest positive
        assert_eq!(top_promoted(&logits, -1.0, 1), vec![2]); // most negative when sign flipped
        assert_eq!(top_promoted(&logits, 1.0, 0), Vec::<i64>::new());
    }

    #[test]
    fn apply_final_norm_centers_then_gains() {
        // center subtracts the mean (mean of [1,2,3] = 2 -> [-1,0,1]), then per-dim gain scales.
        let out = apply_final_norm(vec![1.0, 2.0, 3.0], &[2.0, 2.0, 2.0], true);
        assert_eq!(out, vec![-2.0, 0.0, 2.0]);
        // no centering: just gain.
        let out = apply_final_norm(vec![1.0, 2.0, 3.0], &[1.0, 0.5, 2.0], false);
        assert_eq!(out, vec![1.0, 1.0, 6.0]);
    }

    #[test]
    fn classify_head_names_idioms() {
        // sink: peak on position 0 (seq > 1) regardless of tokens.
        let ctx = [10i64, 20, 30, 40];
        assert_eq!(classify_head(&[0.8, 0.1, 0.05, 0.05], &ctx).0, "sink");
        // previous-token: peak on seq-2.
        assert_eq!(classify_head(&[0.0, 0.0, 0.7, 0.3], &ctx).0, "previous-token");
        // induction: cur = last token (20); peak on j=2 where ctx[j-1]==cur and j is neither 0 nor seq-2.
        let ind = [10i64, 20, 77, 10, 20];
        assert_eq!(classify_head(&[0.0, 0.0, 0.9, 0.05, 0.05], &ind).0, "induction");
        // duplicate-token: peak on an earlier copy of cur (99 at idx1) where ctx[j-1] != cur.
        let dup = [55i64, 99, 66, 77, 99];
        assert_eq!(classify_head(&[0.0, 0.9, 0.0, 0.05, 0.05], &dup).0, "duplicate-token");
    }

    #[test]
    fn final_norm_scale_recovers_frozen_scale() {
        let x = [3.0f32, -1.0, 2.0, 0.5];
        let gain = [1.0f32, 2.0, 0.5, 1.5];
        // RMSNorm: xf = gain ⊙ x / rms, no centering, no bias.
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let s = 1.0 / rms;
        let xf: Vec<f32> = x.iter().zip(&gain).map(|(&xi, &g)| g * xi * s).collect();
        assert!((final_norm_scale(&x, &xf, &gain, false, &[]) - s).abs() < 1e-5);
        // LayerNorm: xf = gain ⊙ (x − mean)/std + bias, with centering.
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
        let s2 = 1.0 / var.sqrt();
        let bias = [0.1f32, -0.2, 0.3, 0.05];
        let xf2: Vec<f32> = x.iter().zip(&gain).zip(&bias).map(|((&xi, &g), &b)| g * (xi - mean) * s2 + b).collect();
        assert!((final_norm_scale(&x, &xf2, &gain, true, &bias) - s2).abs() < 1e-5);
    }

    // A tiny hand-built "model": d=2, two layers each one head and one neuron, identity-ish weights, so we can verify
    // assemble ranks by DLA to the predicted token rather than by raw activation magnitude.
    fn proj(c: &[f32], unembed: &[[f32; 2]]) -> Vec<f32> {
        unembed.iter().map(|row| dot(c, row)).collect()
    }

    #[test]
    fn assemble_ranks_neurons_by_dla_not_activation() {
        // vocab of 3 tokens, d=2. unembed rows: tok0=(1,0), tok1=(0,1), tok2=(-1,-1).
        let unembed = [[1.0f32, 0.0], [0.0, 1.0], [-1.0, -1.0]];
        let ids = [0i64, 1, 2];
        let gain = [1.0f32, 1.0];
        let u_pred = unembed[1]; // predicted token = tok1, whose unembed direction is (0,1)

        // one layer, one head that writes nothing useful.
        let att_last = vec![vec![vec![0.0f32, 0.0, 1.0]]];
        let head_act = vec![vec![0.0f32, 0.0]];

        // two neurons in layer 0. neuron 0: HUGE activation but writes along (1,0) -> 0 DLA to tok1.
        //                          neuron 1: small activation but writes along (0,1) -> positive DLA to tok1.
        let mlp_h = vec![vec![50.0f32, 1.0]];
        let writes = [[1.0f32, 0.0], [0.0, 1.0]];

        // x_last == xf_last with unit gain ⇒ frozen final scale = 1, so the Δs stay in the synthetic logit units below.
        let ex = assemble(
            &ids,
            &att_last,
            &head_act,
            &mlp_h,
            &[0.0, 9.0, 0.0],
            1,
            &gain,
            false,
            &[],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &u_pred,
            |_l, n| writes[n].to_vec(),
            |_l, _h| vec![0.0, 0.0],
            |c| proj(c, &unembed),
        );
        // the big-|act| neuron 0 contributes 0 to tok1; neuron 1 (small act) contributes most → it must rank first.
        assert_eq!(ex.mlp_features[0].neuron, 1, "neuron that writes toward the predicted token must rank first");
        assert!(ex.mlp_features[0].dla > ex.mlp_features[1].dla);
        // runner-up margin from the logits [0,9,0]: predicted tok1 logit 9, runner-up 0 → margin 9.
        assert_eq!(ex.model_predicts, 1);
        assert_eq!(ex.predicted_logit, 9.0);
        assert_eq!(ex.predicted_logit - ex.runner_up_logit, 9.0);
    }
}
