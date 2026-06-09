//! Explain a prediction — the Rust port of pylm's `explain.py` composition side. For a context, it reads the live
//! circuits off the real forward pass: attention heads named by their `idiom_library` signature (previous-token /
//! duplicate-token / induction; attention-sink heads collapsed to a NO-OP count), plus the top-activating MLP features,
//! each named by the tokens it promotes (the neuron's write weight projected to the unembed). Token ids are decoded to
//! strings when a tokenizer is supplied. This is the surface the API serves.

use serde::Serialize;

#[derive(Serialize)]
pub struct HeadCircuit {
    pub layer: usize,
    pub head: usize,
    pub role: String,
    pub attends_to: usize,
    pub attends_tok: i64, // the token the head READS (at the attended position)
    pub mass: f32,
    pub promotes: Vec<i64>, // the tokens the head WRITES to the logits (its direct logit attribution; empty if unknown)
}

#[derive(Serialize)]
pub struct MlpFeature {
    pub layer: usize,
    pub neuron: usize,
    pub act: f32,
    pub promotes: Vec<i64>,
}

#[derive(Serialize)]
pub struct Explanation {
    pub context_tail: Vec<i64>,
    pub model_predicts: i64,
    pub head_circuits: Vec<HeadCircuit>,
    pub sink_heads: usize,
    pub mlp_features: Vec<MlpFeature>,
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

/// Top-`k` token ids a neuron promotes, given its direct-logit contribution and its activation sign.
pub fn top_promoted(logits: &[f32], act: f32, k: usize) -> Vec<i64> {
    let sign = if act < 0.0 { -1.0 } else { 1.0 };
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(k, |&a, &b| (sign * logits[b]).partial_cmp(&(sign * logits[a])).unwrap());
    let mut top = idx[0..k].to_vec();
    top.sort_by(|&a, &b| (sign * logits[b]).partial_cmp(&(sign * logits[a])).unwrap());
    top.iter().map(|&i| i as i64).collect()
}

/// Direct logit attribution for one attention head: its output contribution at the predicting position — the head's
/// `hd`-wide slice of `attn_last` (the attn-weighted values, attn_out's last row) routed through its `hd` rows of the
/// output projection `o_proj`, pushed through the final norm and the `unembed` to the top-`k` tokens it WRITES to the
/// logits. This is the OV/output side, the measured counterpart to "where it attends" (the QK side): for a real
/// copy/induction head it contains `attends_tok`; a head that does intermediate work (doesn't write to the output)
/// shows low-signal tokens. `gain` is the final-norm weight (use `1+w` for Gemma; for ranking the 1/rms scale is
/// irrelevant so we skip it); `center` subtracts the mean first for a LayerNorm model (GPT-2). Mirrors the MLP
/// `top_promoted`, so heads get the same rigour as neurons.
pub fn head_dla(b: &crate::bundle::Bundle, o_proj: &str, unembed: &str, attn_last: &[f32], head: usize, hd: usize, gain: &[f32], center: bool, k: usize) -> Vec<i64> {
    let base = head * hd;
    let mut acc = vec![0.0f32; gain.len()]; // d
    for i in 0..hd {
        let w = b.weight_row(o_proj, base + i); // o_proj row (base+i): a d-vector (the output for that value-dim)
        let a = attn_last[base + i];
        for (o, wi) in acc.iter_mut().zip(&w) {
            *o += a * wi;
        }
    }
    // apply the final norm so this is real DLA: (optionally center) then scale by the per-dim gain — what survives to
    // the logits. The overall 1/rms factor is a positive scalar, so it doesn't affect the top-k ranking; skip it.
    if center {
        let m = acc.iter().sum::<f32>() / acc.len() as f32;
        acc.iter_mut().for_each(|o| *o -= m);
    }
    for (o, g) in acc.iter_mut().zip(gain) {
        *o *= g;
    }
    top_promoted(&b.rowdot_f32(unembed, &acc), 1.0, k) // top-k positive: what the head pushes the logits toward
}

/// Assemble an Explanation from captured per-layer attention + MLP activations (arch-agnostic). `promote(l, n, act)`
/// names the top neuron of each layer's MLP; `head_promote(l, h)` gives the head's direct-logit-attribution tokens (its
/// "writes"; pass `|_, _| Vec::new()` if an arch hasn't wired it). Heads are classified by `idiom_library` signature.
pub fn assemble<F, G>(
    ids: &[i64],
    att_last: &[Vec<Vec<f32>>],
    mlp_h: &[Vec<f32>],
    model_predicts: i64,
    promote: F,
    head_promote: G,
) -> Explanation
where
    F: Fn(usize, usize, f32) -> Vec<i64>,
    G: Fn(usize, usize) -> Vec<i64>,
{
    let mut head_circuits = Vec::new();
    let mut sink_heads = 0;
    for (l, la) in att_last.iter().enumerate() {
        for (h, row) in la.iter().enumerate() {
            let (role, j, mass) = classify_head(row, ids);
            if role == "sink" && mass >= 0.5 {
                sink_heads += 1;
            } else if matches!(role, "induction" | "duplicate-token" | "previous-token") && mass >= 0.15 {
                head_circuits.push(HeadCircuit { layer: l, head: h, role: role.into(), attends_to: j, attends_tok: ids[j], mass, promotes: Vec::new() });
            }
        }
    }
    let order = |r: &str| match r { "induction" => 0, "duplicate-token" => 1, _ => 2 };
    head_circuits.sort_by(|a, b| order(&a.role).cmp(&order(&b.role)).then(b.mass.partial_cmp(&a.mass).unwrap()));
    head_circuits.truncate(6);
    // fill in the OV/output side only for the heads we actually show (one unembed projection each).
    for hc in head_circuits.iter_mut() {
        hc.promotes = head_promote(hc.layer, hc.head);
    }

    let mut mlp_features: Vec<MlpFeature> = mlp_h
        .iter()
        .enumerate()
        .map(|(l, h)| {
            let (n, act) = h.iter().enumerate().fold((0, 0f32), |(bn, ba), (i, &v)| if v.abs() > ba.abs() { (i, v) } else { (bn, ba) });
            MlpFeature { layer: l, neuron: n, act, promotes: promote(l, n, act) }
        })
        .collect();
    mlp_features.sort_by(|a, b| b.act.abs().partial_cmp(&a.act.abs()).unwrap());
    mlp_features.truncate(6);

    Explanation { context_tail: ids[ids.len().saturating_sub(8)..].to_vec(), model_predicts, head_circuits, sink_heads, mlp_features }
}

/// Render an explanation as human-readable text. `dec` maps a token id to a display string.
pub fn render(ex: &Explanation, dec: &dyn Fn(i64) -> String) -> String {
    let mut l = vec![
        format!("context …{}", ex.context_tail.iter().map(|&t| dec(t)).collect::<Vec<_>>().join(" ")),
        format!("model predicts {}", dec(ex.model_predicts)),
        format!("  COMPOSITION  content head circuits ({} idle on sink/NO-OP) — reads → writes:", ex.sink_heads),
    ];
    for h in &ex.head_circuits {
        let mut line = format!("    L{}.H{:<2} {:<15} reads {} (mass {:.3})", h.layer, h.head, h.role, dec(h.attends_tok), h.mass);
        if !h.promotes.is_empty() {
            let toks = h.promotes.iter().map(|&t| dec(t)).collect::<Vec<_>>().join(", ");
            line.push_str(&format!("  ⇒ writes {{{toks}}}"));
        }
        l.push(line);
    }
    if ex.head_circuits.is_empty() {
        l.push("    (none above threshold — carried by MLP features below)".to_string());
    }
    l.push("  COMPOSITION  top MLP features (neuron → tokens it promotes):".to_string());
    for f in &ex.mlp_features {
        let toks = f.promotes.iter().map(|&t| dec(t)).collect::<Vec<_>>().join(", ");
        l.push(format!("    L{} n{:<5} act {:<7.2} → {{{}}}", f.layer, f.neuron, f.act, toks));
    }
    l.join("\n")
}
