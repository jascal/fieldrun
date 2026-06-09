//! fieldrun — run a decompiled LLM as a native binary.
//!
//! Tier A (retrieval, pure-Rust port of pylm's `lm.py`) and Tier B (composition, the real GPT-2 forward pass over a
//! fieldrun bundle), each scored over a held-out token-id stream exactly as Python `validate.py` / `numpy_lm.py` do.
//! The scoring loops fan out across cores with rayon (each next-token prediction is an independent, read-only forward).
//! Tier C (router), `explain`, and the API land on top. The whole point: one static binary, flat-file knowledge,
//! no framework.

mod bundle;
mod composition;
mod retrieval;

use std::collections::HashMap;

use rayon::prelude::*;

use composition::Gpt2;
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

    let hold: Holdout = serde_json::from_str(&std::fs::read_to_string(ids_path).expect("read ids"))
        .expect("parse ids");
    let ids = hold.holdout_ids;
    let end = (ctx_window + n_eval).min(ids.len());
    let ctx = |i: usize| &ids[i.saturating_sub(ctx_window)..i];
    let threads = rayon::current_num_threads();

    // Tier B (composition) — the real forward pass from a fieldrun bundle; positions scored in parallel.
    if let Some(stem) = flag(&args, "--bundle") {
        let lm = Gpt2::load(stem).unwrap_or_else(|e| panic!("load bundle {stem}: {e}"));
        let preds: Vec<i64> = (ctx_window..end).into_par_iter().map(|i| lm.predict(ctx(i))).collect();
        let correct = preds.iter().zip(ctx_window..end).filter(|(p, i)| **p == ids[*i]).count();
        dump_if(&args, &preds);
        report("Tier B (composition)", &format!("bundle {stem}.fieldrun · pure-Rust GPT-2"), correct, preds.len(), threads);
        return;
    }

    // Tier A (retrieval) — induction + n-gram + grammar over the flat store; positions scored in parallel.
    let store = Store::load(store_path).unwrap_or_else(|e| panic!("load store {store_path}: {e}"));
    let out: Vec<(i64, String)> = (ctx_window..end).into_par_iter().map(|i| store.predict(ctx(i))).collect();
    let correct = out.iter().zip(ctx_window..end).filter(|((p, _), i)| *p == ids[*i]).count();
    let preds: Vec<i64> = out.iter().map(|(p, _)| *p).collect();
    dump_if(&args, &preds);
    report("Tier A (retrieval)", &format!("store {store_path}"), correct, out.len(), threads);

    let mut idioms: HashMap<&str, usize> = HashMap::new();
    for (_, tag) in &out {
        *idioms.entry(tag.as_str()).or_default() += 1;
    }
    let mut by: Vec<_> = idioms.into_iter().collect();
    by.sort_by(|a, b| b.1.cmp(&a.1));
    let parts: Vec<String> = by.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!("[fieldrun] idioms: {}", parts.join(", "));
}

fn dump_if(args: &[String], preds: &[i64]) {
    if let Some(path) = flag(args, "--dump") {
        let out: String = preds.iter().map(|p| format!("{p}\n")).collect();
        std::fs::write(path, out).expect("write dump");
        eprintln!("[fieldrun] wrote {} predictions to {path}", preds.len());
    }
}

fn report(tier: &str, detail: &str, correct: usize, total: usize, threads: usize) {
    let acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
    println!("[fieldrun] {tier} · {detail}");
    println!("[fieldrun] next-token top-1: {:.1}%  ({total} positions, {threads} threads)", acc * 100.0);
}
