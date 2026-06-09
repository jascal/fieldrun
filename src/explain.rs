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
