//! The logic-export layer (LOGIC_EXPORT.md): build a per-decode **provenance** — the same object the CLI emitter
//! (`export --logic`) renders as a semiring-Datalog program AND the chat/serve `Logic` explain mode renders as a
//! per-token line. One builder, three consumers, so explain is faithful-by-construction (it is the provenance of the
//! actual decode; LE-T5) rather than a separate reconstruction that could drift.

use crate::model::Model;
use crate::retrieval::{induction_rule, CandCfg, RuleHit, Store};

/// One next-token decision, decomposed into the pieces a semiring-Datalog program needs: the candidate set, the
/// fired retrievable rule (Tier A), and the per-block residual contributions to every candidate (Tier B). By
/// residual-stream additivity `Σ_block blocks[b].1[i] == logit(candidates[i])` exactly.
pub struct Provenance {
    pub predicted: i64,
    pub runner_up: i64,
    pub pred_logit: f32,
    pub margin: f32,
    pub route: u8, // 0 RETRIEVED · 1 SELECTED · 2 COMPOSED · 3 unknown (no store)
    pub candidates: Vec<i64>,
    pub rule: Option<RuleHit>,
    pub induction: bool,
    pub blocks: Vec<(String, Vec<f32>)>, // (block label, contribution to each candidate's logit)
}

/// Build the provenance for the next token after context `c`. Needs `Model::explain` (for predicted/runner-up/logits)
/// and `Model::residual_decomp` (the per-block contributions) — rope-only; returns None otherwise. `cap` bounds the
/// candidate set.
pub fn build(lm: &dyn Model, c: &[i64], store: Option<&Store>, cfg: &CandCfg, cap: usize) -> Option<Provenance> {
    let ex = lm.explain(c)?;
    let (t, v) = (ex.model_predicts, ex.runner_up);
    let store_cands: Vec<i64> = store.map(|st| st.candidates(c, cfg)).unwrap_or_default();
    let route = match store {
        Some(st) => {
            let (kb, _) = st.predict(c);
            if kb == t { 0u8 } else if store_cands.contains(&t) { 1 } else { 2 }
        }
        None => 3,
    };
    let mut cand = vec![t];
    if v != t { cand.push(v); }
    for &x in &store_cands {
        if cand.len() >= cap { break; }
        if !cand.contains(&x) { cand.push(x); }
    }
    let (labels, contrib) = lm.residual_decomp(c, &cand)?;
    Some(Provenance {
        predicted: t,
        runner_up: v,
        pred_logit: ex.predicted_logit,
        margin: ex.predicted_logit - ex.runner_up_logit,
        route,
        candidates: cand,
        rule: store.and_then(|st| st.rule_for(c, t)).or_else(|| induction_rule(c, t)),
        induction: induction_rule(c, t).is_some(),
        blocks: labels.into_iter().zip(contrib).collect(),
    })
}

pub fn route_name(route: u8) -> &'static str {
    match route {
        0 => "RETRIEVED",
        1 => "SELECTED",
        2 => "COMPOSED",
        _ => "—",
    }
}

/// A compact, faithful per-token provenance line for the chat/serve `Logic` explain mode: route + fired rule + the top
/// contributing blocks + the margin. This is the explain readout of the same data the emitter renders as Datalog.
pub fn explain_line(p: &Provenance, label: &dyn Fn(i64) -> String) -> String {
    let mut bl: Vec<(&str, f32)> = p.blocks.iter().map(|(n, ws)| (n.as_str(), ws.first().copied().unwrap_or(0.0))).collect();
    bl.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    let top: Vec<String> = bl.iter().take(3).map(|(n, w)| format!("{n} {w:+.2}")).collect();
    let via = p.rule.as_ref().map(|r| {
        let key: Vec<String> = r.key.iter().map(|&k| label(k)).collect();
        format!(" via {}[{}]", r.idiom, key.join(","))
    }).unwrap_or_default();
    let ind = if p.induction && p.rule.as_ref().map(|r| !r.idiom.contains("induction")).unwrap_or(true) { " (induction-copy)" } else { "" };
    format!(
        "{} ⟵ {}{}{} | blocks {} | margin {:+.3} vs {}",
        label(p.predicted), route_name(p.route), via, ind, top.join(", "), p.margin, label(p.runner_up)
    )
}

/// Render the provenance as a runnable, Soufflé-compatible semiring-Datalog program (the `export --logic` artifact).
/// `ctx` is the context (for the header comment only); `label` maps a token id to display text (also comments only —
/// the program references tokens by id).
pub fn emit_dl(p: &Provenance, ctx: &[i64], label: &dyn Fn(i64) -> String) -> String {
    let mut o = String::new();
    o.push_str("% ============================================================\n");
    o.push_str("% fieldrun logic export — semiring-Datalog program for ONE next-token decision\n");
    o.push_str("% Greedy decode = (max,+) provenance; swap to log-semiring for the full distribution (LOGIC_EXPORT.md).\n");
    o.push_str("% The model SPECIALIZED to one context (a partial evaluation / decode trace). Tokens = ids; text in comments.\n");
    o.push_str("% Soufflé-compatible. Σ over contrib/3 == the true logit (LE-T5). Run: fieldrun eval <this>.dl --semiring max|log\n");
    o.push_str("% ============================================================\n\n");
    o.push_str(".decl candidate(t:number)\n.decl contrib(block:symbol, t:number, w:float)\n");
    o.push_str(".decl logit(t:number, s:float)\n.decl decide(t:number)\n.decl retrieved(t:number)\n\n");
    o.push_str("% context:");
    for &id in ctx.iter().rev().take(16).rev() {
        o.push_str(&format!(" {}", label(id)));
    }
    o.push_str(&format!(
        "\n% model predicts: {}  (logit {:.3}, margin {:+.3} over runner-up {})\n\n",
        label(p.predicted), p.pred_logit, p.margin, label(p.runner_up)
    ));
    o.push_str(&format!("% ---- candidate set (predicted ∪ runner-up ∪ KB-proposed), |C| = {} ----\n", p.candidates.len()));
    for &id in &p.candidates {
        o.push_str(&format!("candidate({}).   % {}\n", id, label(id)));
    }
    o.push('\n');
    o.push_str("% ---- TIER A: retrievable fragment (looked up; no composition) ----\n");
    if p.induction {
        o.push_str(&format!("% induction (in-context copy): the predicted token {} repeats an earlier token.\n", label(p.predicted)));
        o.push_str("retrieved(T) :- induction_copy(T).   % the clean recursive rule: copy the token after the matched prefix\n");
        o.push_str(&format!("induction_copy({}).\n", p.predicted));
    }
    if let Some(r) = &p.rule {
        if !r.idiom.contains("induction") {
            let key_s: Vec<String> = r.key.iter().map(|&k| label(k)).collect();
            let key_atom = r.key.iter().map(|k| k.to_string()).collect::<Vec<_>>().join("_");
            o.push_str(&format!("% {} rule: key [{}] → predicted token (rank {})\n", r.idiom, key_s.join(", "), r.rank.map(|x| x.to_string()).unwrap_or_else(|| "-".into())));
            o.push_str(&format!("ngram_succ(\"{}\", {}).   % {} proposes the predicted token\n", key_atom, p.predicted, r.idiom));
        }
    }
    o.push('\n');
    o.push_str("% ---- TIER B: composition (per-block residual contributions; the forge tax) ----\n");
    o.push_str("% contrib(Block, Token, Weight): block's exact contribution to Token's logit. Σ_Block = logit(Token).\n");
    o.push_str("% |W|>=0.1 blocks shown; the dense remainder folds into block \"rest\" (the irreducible high-PR\n");
    o.push_str("% forge-tax sum — no compact rule; LOGIC_EXPORT LE-T2/T4). 'rest' keeps the per-token sum exact.\n");
    for (ci, &tok) in p.candidates.iter().enumerate() {
        let total: f32 = p.blocks.iter().map(|(_, ws)| ws[ci]).sum();
        let mut shown = 0.0f32;
        for (name, ws) in &p.blocks {
            if ws[ci].abs() >= 0.1 {
                o.push_str(&format!("contrib(\"{}\", {}, {:.4}).\n", name, tok, ws[ci]));
                shown += ws[ci];
            }
        }
        o.push_str(&format!("contrib(\"rest\", {}, {:.4}).   % dense remainder for {}\n", tok, total - shown, label(tok)));
    }
    o.push('\n');
    o.push_str("% ---- accumulation (⊗ = +) and decision (⊕ = max) — the semiring decode ----\n");
    o.push_str("logit(T, S) :- candidate(T), S = sum W : { contrib(_, T, W) }.   % ⊗ over blocks (log-semiring +)\n");
    o.push_str("decide(T)   :- logit(T, S), S = max S2 : { logit(_, S2) }.        % ⊕ = max (max-product, T=0)\n");
    o.push_str(".output decide\n\n");
    // LE-T5 round-trip self-check: argmax over candidates from the contrib facts == the model's token.
    let am = (0..p.candidates.len()).max_by(|&a, &b| {
        let (sa, sb): (f32, f32) = (p.blocks.iter().map(|(_, ws)| ws[a]).sum(), p.blocks.iter().map(|(_, ws)| ws[b]).sum());
        sa.partial_cmp(&sb).unwrap()
    }).map(|i| p.candidates[i]).unwrap_or(p.predicted);
    o.push_str(&format!(
        "% LE-T5 round-trip: decide/1 under (max,+) == model argmax {} : {}\n",
        label(p.predicted), if am == p.predicted { "✓ FAITHFUL" } else { "✗ MISMATCH (candidate set missed the argmax)" }
    ));
    o
}
