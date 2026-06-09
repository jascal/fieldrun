//! fieldrun — run a decompiled LLM as a native binary.
//!
//! This first cut is Tier A (retrieval): the pure-Rust port of pylm's `lm.py`, scored over a held-out token-id stream
//! exactly as the Python `validate.py` does. Tiers B (numpy composition kernel -> Rust matmuls) and C (router) and the
//! `explain` surface land on top of this. The whole point: one static binary, flat-file knowledge, no framework.

mod retrieval;

use std::collections::HashMap;

use retrieval::Store;

#[derive(serde::Deserialize)]
struct Holdout {
    holdout_ids: Vec<i64>,
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let store_path = flag(&args, "--store").unwrap_or("../lm-sae/pylm/store_gpt2.json");
    let ids_path = flag(&args, "--ids").unwrap_or("../lm-sae/pylm/holdout_gpt2.json");
    let ctx_window: usize = flag(&args, "--ctx").and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_eval: usize = flag(&args, "--n-eval").and_then(|s| s.parse().ok()).unwrap_or(500);

    let store = Store::load(store_path).unwrap_or_else(|e| panic!("load store {store_path}: {e}"));
    let hold: Holdout = serde_json::from_str(&std::fs::read_to_string(ids_path).expect("read ids"))
        .expect("parse ids");
    let ids = hold.holdout_ids;

    let end = (ctx_window + n_eval).min(ids.len());
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut idioms: HashMap<String, usize> = HashMap::new();
    for i in ctx_window..end {
        let ctx = &ids[i.saturating_sub(ctx_window)..i];
        let (pred, idiom) = store.predict(ctx);
        *idioms.entry(idiom).or_default() += 1;
        correct += usize::from(pred == ids[i]);
        total += 1;
    }

    if let Some(path) = flag(&args, "--dump") {
        // per-position predictions for the faithfulness gate (diff against Python lm.py)
        let mut out = String::new();
        for i in ctx_window..end {
            let ctx = &ids[i.saturating_sub(ctx_window)..i];
            let (pred, _) = store.predict(ctx);
            out.push_str(&format!("{pred}\n"));
        }
        std::fs::write(path, out).expect("write dump");
        eprintln!("[fieldrun] wrote {} predictions to {path}", end - ctx_window);
    }

    let acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
    println!("[fieldrun] Tier A (retrieval) · store {store_path}");
    println!("[fieldrun] next-token top-1: {:.1}%  ({total} positions)", acc * 100.0);
    let mut by: Vec<_> = idioms.into_iter().collect();
    by.sort_by(|a, b| b.1.cmp(&a.1));
    let parts: Vec<String> = by.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!("[fieldrun] idioms: {}", parts.join(", "));
}
