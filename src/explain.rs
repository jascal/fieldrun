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
    pub attends_tok: i64,
    pub mass: f32,
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

/// Assemble an Explanation from captured per-layer attention + MLP activations (arch-agnostic). `promote(l, n, act)`
/// names the top neuron of each layer's MLP. Heads are classified by `idiom_library` signature; sinks are counted.
pub fn assemble<F: Fn(usize, usize, f32) -> Vec<i64>>(
    ids: &[i64],
    att_last: &[Vec<Vec<f32>>],
    mlp_h: &[Vec<f32>],
    model_predicts: i64,
    promote: F,
) -> Explanation {
    let mut head_circuits = Vec::new();
    let mut sink_heads = 0;
    for (l, la) in att_last.iter().enumerate() {
        for (h, row) in la.iter().enumerate() {
            let (role, j, mass) = classify_head(row, ids);
            if role == "sink" && mass >= 0.5 {
                sink_heads += 1;
            } else if matches!(role, "induction" | "duplicate-token" | "previous-token") && mass >= 0.15 {
                head_circuits.push(HeadCircuit { layer: l, head: h, role: role.into(), attends_to: j, attends_tok: ids[j], mass });
            }
        }
    }
    let order = |r: &str| match r { "induction" => 0, "duplicate-token" => 1, _ => 2 };
    head_circuits.sort_by(|a, b| order(&a.role).cmp(&order(&b.role)).then(b.mass.partial_cmp(&a.mass).unwrap()));
    head_circuits.truncate(6);

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
        format!("  COMPOSITION  content head circuits ({} idle on sink/NO-OP):", ex.sink_heads),
    ];
    for h in &ex.head_circuits {
        l.push(format!("    L{}.H{:<2} {:<15} → {} (mass {:.3})", h.layer, h.head, h.role, dec(h.attends_tok), h.mass));
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
