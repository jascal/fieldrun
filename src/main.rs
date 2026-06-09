//! fieldrun — run a decompiled LLM as a native binary.
//!
//! Tier A (retrieval, pure-Rust port of pylm's `lm.py`) and Tier B (composition, the real GPT-2 forward pass over a
//! fieldrun bundle), each scored over a held-out token-id stream exactly as Python `validate.py` / `numpy_lm.py` do.
//! The scoring loops fan out across cores with rayon (each next-token prediction is an independent, read-only forward).
//! Tier C (router), `explain`, and the API land on top. The whole point: one static binary, flat-file knowledge,
//! no framework.

mod api;
mod bundle;
mod composition;
mod explain;
mod gemma;
mod model;
mod retrieval;
mod rope;

use std::collections::HashMap;

use rayon::prelude::*;

use bundle::Bundle;
use composition::Gpt2;
use gemma::Gemma;
use model::Model;
use retrieval::Store;
use rope::Rope;

#[derive(serde::Deserialize)]
struct Holdout {
    holdout_ids: Vec<i64>,
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
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
        let bundle = Bundle::load(stem).unwrap_or_else(|e| panic!("load bundle {stem}: {e}"));
        let arch = bundle.arch.clone();
        let route: f32 = flag(&args, "--route-frac").and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let lm: Box<dyn Model> = match arch.as_str() {
            "gpt2" => Box::new(Gpt2::new(bundle, route)),
            "rope" => Box::new(Rope::new(bundle, route)),
            "gemma" => Box::new(Gemma::new(bundle, route)),
            other => panic!("unknown bundle arch {other:?} (have: gpt2, rope, gemma)"),
        };

        // --serve PORT: start the HTTP API over this loaded model (no ids needed).
        if let Some(port) = flag(&args, "--serve").and_then(|s| s.parse::<u16>().ok()) {
            api::serve(lm, &arch, port);
            return;
        }

        // --explain: explain the prediction at the end of the first --ctx tokens (composition circuits + features).
        if has_flag(&args, "--explain") {
            let ctx = &ids[..ctx_window.min(ids.len())];
            match lm.explain(ctx) {
                Some(ex) => {
                    let dec = load_decoder(flag(&args, "--vocab"));
                    println!("{}", explain::render(&ex, &dec));
                    if let Some(p) = flag(&args, "--out-json") {
                        std::fs::write(p, serde_json::to_string_pretty(&ex).unwrap()).expect("write json");
                    }
                }
                None => println!("[fieldrun] explain not implemented for arch {arch}"),
            }
            return;
        }

        // --generate N: greedy autoregressive generation from the first --ctx tokens; compares KV-cache vs naive.
        if let Some(n) = flag(&args, "--generate").and_then(|s| s.parse::<usize>().ok()) {
            let prompt = &ids[..ctx_window.min(ids.len())];
            let t0 = std::time::Instant::now();
            let kv = lm.generate(prompt, n);
            let kv_s = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let mut ctx2 = prompt.to_vec();
            let naive: Vec<i64> = (0..n).map(|_| { let t = lm.predict(&ctx2); ctx2.push(t); t }).collect();
            let naive_s = t1.elapsed().as_secs_f64();
            println!("[fieldrun] generate {n} tokens from a {}-token prompt · {arch}", prompt.len());
            println!("[fieldrun]   KV-cache: {kv_s:.2}s  ({:.1} tok/s)", n as f64 / kv_s);
            println!("[fieldrun]   naive   : {naive_s:.2}s  ({:.1} tok/s)", n as f64 / naive_s);
            println!("[fieldrun]   speedup : {:.1}x  ·  tokens identical: {}", naive_s / kv_s, kv == naive);
            return;
        }
        let t0 = std::time::Instant::now();
        let preds: Vec<i64> = (ctx_window..end).into_par_iter().map(|i| lm.predict(ctx(i))).collect();
        let secs = t0.elapsed().as_secs_f64();
        let correct = preds.iter().zip(ctx_window..end).filter(|(p, i)| **p == ids[*i]).count();
        dump_if(&args, &preds);
        report("Tier B (composition)", &format!("bundle {stem}.fieldrun · pure-Rust {arch}"), correct, preds.len(), threads);
        println!("[fieldrun] throughput: {:.1} forwards/s across {threads} threads ({:.0} ms/forward/core)",
                 preds.len() as f64 / secs, secs * 1000.0 * threads as f64 / preds.len() as f64);
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

/// Build an id→string decoder. With a GPT-2 `vocab.json` (token→id), invert it and show the raw BPE token (Ġ→space,
/// Ċ→newline for readability); without one, fall back to `[id]`.
fn load_decoder(vocab: Option<&str>) -> Box<dyn Fn(i64) -> String> {
    if let Some(path) = vocab {
        if let Ok(txt) = std::fs::read_to_string(path) {
            let map: HashMap<String, i64> = serde_json::from_str(&txt).unwrap_or_default();
            let inv: HashMap<i64, String> = map.into_iter().map(|(k, v)| (v, k)).collect();
            return Box::new(move |id| {
                inv.get(&id)
                    .map(|s| format!("{:?}", s.replace('\u{0120}', " ").replace('\u{010A}', "\n")))
                    .unwrap_or_else(|| format!("[{id}]"))
            });
        }
    }
    Box::new(|id| format!("[{id}]"))
}

fn report(tier: &str, detail: &str, correct: usize, total: usize, threads: usize) {
    let acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
    println!("[fieldrun] {tier} · {detail}");
    println!("[fieldrun] next-token top-1: {:.1}%  ({total} positions, {threads} threads)", acc * 100.0);
}
