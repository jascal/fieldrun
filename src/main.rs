//! fieldrun — run a decompiled LLM as a native binary.
//!
//! Tier A (retrieval, pure-Rust port of pylm's `lm.py`) and Tier B (composition, the real GPT-2 forward pass over a
//! fieldrun bundle), each scored over a held-out token-id stream exactly as Python `validate.py` / `numpy_lm.py` do.
//! The scoring loops fan out across cores with rayon (each next-token prediction is an independent, read-only forward).
//! Tier C (router), `explain`, and the API land on top. The whole point: one static binary, flat-file knowledge,
//! no framework.

// Some items are used only under `--features gpu` (`arr2`/`arr2o`/`f32_array`, `device` fields) or kept as
// bundle-format / config surface; allow dead_code so the default stable build is warning-free on every platform.
#![allow(dead_code)]

mod api;
mod bucketing;
mod bundle;
#[cfg(feature = "jit")]
mod jit;
mod logic;
mod logic_whole;
mod composition;
mod convert;
mod device;
mod dsv4;
mod explain;
#[cfg(feature = "gpu")]
mod gpu_gpt2;
#[cfg(feature = "gpu")]
mod gpu_mm;
#[cfg(feature = "gpu")]
mod gpu_rope;
mod gemma;
mod headgate;
mod gemma3;
mod gemma4;
#[cfg(feature = "api")]
mod mdfmt;
#[cfg(feature = "hub")]
mod hub;
mod minimax;
mod mla;
mod model;
mod neox;
mod qwen3moe;
mod retrieval;
mod rope;
#[cfg(feature = "api")]
mod tools;

use std::collections::HashMap;

use rayon::prelude::*;

// Force-link the selected BLAS backend (Accelerate/OpenBLAS). blas-src only emits its `-framework Accelerate` /
// `-lopenblas` link directives when the crate is actually referenced; without this `use`, the backend isn't linked and
// the build fails at link with "undefined symbol: cblas_sgemm" (e.g. "ld: ... for architecture arm64" on macOS).
#[cfg(feature = "blas")]
use blas_src as _;

use bundle::Bundle;
use composition::Gpt2;
use dsv4::Dsv4;
use gemma::Gemma;
use gemma3::Gemma3;
use gemma4::Gemma4;
use minimax::MiniMax;
use mla::Mla;
use neox::Neox;
// mdfmt (Markdown→ANSI for the chat REPL) is only used by the api `chat`; module declared below under cfg(api).
use model::Model;
use qwen3moe::Qwen3Moe;
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

    // User-facing CLI: turn any panic (bad bundle, malformed checkpoint, wrong --arch for the model, …) into a clean
    // one-line error rather than a Rust backtrace. RUST_BACKTRACE restores the full default for debugging.
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::panic::set_hook(Box::new(|info| {
            let msg = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unexpected error".to_string());
            eprintln!("[fieldrun] error: {msg}");
            eprintln!("[fieldrun] (set RUST_BACKTRACE=1 for a full trace, or run `fieldrun --help`)");
        }));
    }

    // help: explicit --help/-h, or a bare invocation (the dev-only default store/ids paths would just panic otherwise).
    if args.len() == 1 || has_flag(&args, "--help") || has_flag(&args, "-h") {
        print_help();
        return;
    }

    // RESEARCH SPIKE (only built with `--features jit`): bench a `(k,g)`-specialised JIT int4 dot vs the hand AVX2
    // kernel. `--jit-bench [k] [g] [iters]` (positional after the flag; defaults k=4864 g=32 iters=200000).
    #[cfg(feature = "jit")]
    if let Some(i) = args.iter().position(|a| a == "--jit-bench") {
        let p = |n: usize, d: usize| args.get(i + 1 + n).and_then(|s| s.parse().ok()).unwrap_or(d);
        jit::bench_i4_dot(p(0, 4864), p(1, 32), p(2, 200_000));
        return;
    }

    // `fieldrun convert --model <dir-or-hf-repo-id> --arch rope --dtype int8 -o <stem>` — HF safetensors -> bundle, no torch.
    // `--model` is a local checkpoint dir, OR (with the default `hub` feature) a Hugging Face repo id like `org/name`,
    // which is downloaded to the HF cache first. Token (gated models): `--hf-token` > $HF_TOKEN > `huggingface-cli login`.
    // Accept `--convert` as well as `convert` — `--convert` is a natural typo given every other arg is a `--flag`, and
    // silently falling through to Tier A (as a bare run does) would be baffling.
    if matches!(args.get(1).map(String::as_str), Some("convert") | Some("--convert")) {
        // --model is required; without it, print convert usage and exit cleanly (not a panic/backtrace).
        let model = match flag(&args, "--model") {
            Some(m) => m,
            None => {
                eprintln!(
                    "[fieldrun] convert: --model is required.\n  \
                     fieldrun convert --model <local-dir | hf-repo-id> --arch <arch> [--dtype int8|f16|f32] [-o <stem>]\n  \
                     e.g.  fieldrun convert --model Qwen/Qwen2.5-1.5B-Instruct --arch rope --dtype f16\n  \
                     archs: gpt2 | rope | gemma | gemma3 | gemma4 | qwen3moe | mla | minimax   (see `fieldrun --help`)"
                );
                std::process::exit(2);
            }
        };
        let arch = flag(&args, "--arch").unwrap_or("rope");
        let dtype = flag(&args, "--dtype").unwrap_or("int8");
        const ARCHS: &[&str] = &["gpt2", "neox", "rope", "gemma", "gemma3", "gemma4", "qwen3moe", "mla", "minimax", "dsv4"];
        if !ARCHS.contains(&arch) {
            eprintln!("[fieldrun] convert: unknown --arch {arch:?} (have: {})", ARCHS.join(", "));
            std::process::exit(2);
        }
        if !["int4", "q4a", "int8", "f16", "f32"].contains(&dtype) {
            eprintln!("[fieldrun] convert: unknown --dtype {dtype:?} (have: int4, q4a, int8, f16, f32)");
            std::process::exit(2);
        }
        // per-tensor-role policy: the embed/tied-unembed (the largest tensor for a big vocab) is quantised independently
        // of the linear --dtype. DEFAULT: int8 when the linears are quantised (int8/int4/q4a) — it's ~free quality (0
        // top-1 loss, see Phase 4b) and a big decode speedup + smaller bundle; f16/f32 keep an f16 embed (so the f32
        // gate is intact). Override with --embed-dtype {f16|int8}. All archs read embed via rows_f32 / unembed via
        // rowdot_f32, so the row-major int8 (rowi8) path applies everywhere.
        let embed_dtype = flag(&args, "--embed-dtype").unwrap_or_else(|| {
            if ["int8", "int4", "q4a"].contains(&dtype) { "int8" } else { "f16" }
        });
        if !["f16", "int8"].contains(&embed_dtype) {
            eprintln!("[fieldrun] convert: unknown --embed-dtype {embed_dtype:?} (have: f16, int8)");
            std::process::exit(2);
        }
        // -o is optional; default groups bundles in a home cache (~/.cache/fieldrun/bundles/<name>/<name>), NOT the
        // cwd — so converting from a dev checkout doesn't litter it. <name> = the model's last path segment minus @rev.
        let out: String = match flag(&args, "-o").or_else(|| flag(&args, "--out")) {
            Some(o) => o.to_string(),
            None => {
                let name = model.rsplit('/').next().unwrap_or(model).split('@').next().unwrap_or(model);
                format!("{}/{name}/{name}", bundles_dir())
            }
        };
        // skip if this bundle already exists (don't re-download/re-convert) unless --force. Checked before the HF pull.
        if std::path::Path::new(&format!("{out}.fieldrun.json")).exists() && !has_flag(&args, "--force") {
            println!("[convert] {out}.fieldrun already exists — skipping (use --force to rebuild)");
            return;
        }
        let model_dir: String = if std::path::Path::new(model).join("config.json").exists() {
            model.to_string() // a local checkpoint directory
        } else {
            #[cfg(feature = "hub")]
            {
                match hub::fetch(model, hub::token(flag(&args, "--hf-token"))) {
                    Ok(dir) => dir,
                    Err(e) => {
                        eprintln!("[fieldrun] convert: couldn't load --model {model:?} — not a local dir with \
                                   config.json, and the Hugging Face pull failed:\n  {e}");
                        std::process::exit(1);
                    }
                }
            }
            #[cfg(not(feature = "hub"))]
            {
                eprintln!("[fieldrun] convert: --model {model:?} is not a local dir with config.json. To pull it from \
                           the Hugging Face hub by repo id, use a build with the `hub` feature (on by default; you've \
                           disabled it).");
                std::process::exit(2);
            }
        };
        if let Err(e) = convert::convert(&model_dir, arch, dtype, embed_dtype, &out) {
            eprintln!("[fieldrun] convert failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    // `fieldrun eval <prog.dl> [--semiring max|log]` — run an emitted `export --logic` program with the built-in
    // semiring evaluator (no Soufflé needed). Parses candidate/contrib facts, accumulates logit(T)=Σ contrib (⊗=+),
    // then applies the cross-candidate ⊕: max-product (default) → decide(T)=argmax (the greedy decode, T=0); log-
    // semiring → the softmax distribution (T=1). ONE program, two semirings — LE-T5 + the two-temperature claim, run.
    if matches!(args.get(1).map(String::as_str), Some("eval")) {
        let path = match args.iter().skip(2).find(|a| !a.starts_with('-')).cloned().or_else(|| flag(&args, "--in").map(String::from)) {
            Some(p) => p,
            None => {
                eprintln!("[fieldrun] eval: give an emitted program — fieldrun eval prog.dl [--semiring max|log]");
                std::process::exit(2);
            }
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => { eprintln!("[fieldrun] eval: cannot read {path}: {e}"); std::process::exit(1); }
        };
        let semiring = flag(&args, "--semiring").unwrap_or("max");
        use std::collections::{BTreeMap, BTreeSet};
        // (step, token) -> Σ contrib. `multi` flips on for a step-indexed program (a stitched TRACE:
        // candidate(Step,T) / contrib(Step,Block,T,W)); a single-decision program stays at step 0 and prints exactly as
        // before. `order` keeps first-seen candidate order within a step.
        let mut logit: BTreeMap<(i64, i64), f64> = BTreeMap::new();
        let mut order: Vec<(i64, i64)> = Vec::new();
        let mut blocks: BTreeSet<String> = BTreeSet::new();
        let mut multi = false;
        for line in text.lines() {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("candidate(") {
                let f: Vec<&str> = rest.split(')').next().unwrap_or("").split(',').map(|s| s.trim()).collect();
                let (step, tok) = match f.len() {
                    1 => (0i64, f[0].parse::<i64>().ok()),
                    _ => { multi = true; (f[0].parse::<i64>().unwrap_or(0), f[1].parse::<i64>().ok()) }
                };
                if let Some(tok) = tok {
                    let key = (step, tok);
                    if !logit.contains_key(&key) { order.push(key); logit.insert(key, 0.0); }
                }
            } else if let Some(rest) = l.strip_prefix("contrib(") {
                let f: Vec<&str> = rest.split(')').next().unwrap_or("").split(',').map(|s| s.trim()).collect();
                // single-decision: (block, tok, w) ; stitched trace: (step, block, tok, w)
                let (step, blk, tok, w) = match f.len() {
                    3 => (0i64, f[0], f[1].parse::<i64>().ok(), f[2].parse::<f64>().ok()),
                    4 => { multi = true; (f[0].parse::<i64>().unwrap_or(0), f[1], f[2].parse::<i64>().ok(), f[3].parse::<f64>().ok()) }
                    _ => (0, "", None, None),
                };
                if let (Some(tok), Some(w)) = (tok, w) {
                    let key = (step, tok);
                    if !logit.contains_key(&key) { order.push(key); logit.insert(key, 0.0); }
                    *logit.get_mut(&key).unwrap() += w;
                    blocks.insert(blk.trim_matches('"').to_string());
                }
            }
        }
        if order.is_empty() {
            eprintln!("[fieldrun] eval: no candidate/contrib facts in {path} (is it an `export --logic` / stitched program?)");
            std::process::exit(1);
        }
        if multi {
            // stitched TRACE: decode each step independently (the per-step sums never mix — that's the point of the index).
            let steps: Vec<i64> = order.iter().map(|&(s, _)| s).collect::<BTreeSet<_>>().into_iter().collect();
            eprintln!("[fieldrun] eval {path}: TRACE · {} steps · {} blocks · semiring={semiring}", steps.len(), blocks.len());
            for s in steps {
                let mut scored: Vec<(i64, f64)> = order.iter().filter(|&&(st, _)| st == s).map(|&(_, t)| (t, logit[&(s, t)])).collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                if semiring == "log" {
                    let mx = scored[0].1;
                    let z: f64 = scored.iter().map(|(_, v)| (v - mx).exp()).sum();
                    println!("% step {s}: distribution (log-semiring, T=1):");
                    for (t, v) in scored.iter().take(6) {
                        println!("    P {:>6.3}   logit {:>8.3}   token {t}", (v - mx).exp() / z, v);
                    }
                } else {
                    let (t, v) = scored[0];
                    let runner = scored.get(1).map(|(rt, rv)| format!("  (margin {:+.4} vs {rt})", v - rv)).unwrap_or_default();
                    println!("decide({s}, {t}).   % logit {v:.4}{runner}");
                }
            }
        } else {
            eprintln!("[fieldrun] eval {path}: {} candidates · {} blocks · semiring={semiring}", order.len(), blocks.len());
            let mut scored: Vec<(i64, f64)> = order.iter().map(|&(_, t)| (t, logit[&(0, t)])).collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if semiring == "log" {
                let mx = scored[0].1;
                let z: f64 = scored.iter().map(|(_, s)| (s - mx).exp()).sum();
                println!("% distribution over candidates (log-semiring / sum-product, T=1):");
                for (t, s) in scored.iter().take(12) {
                    println!("  P {:>6.3}   logit {:>8.3}   token {t}", (s - mx).exp() / z, s);
                }
            } else {
                let (t, s) = scored[0];
                println!("decide({t}).   % logit {s:.4}  (max-product / argmax, T=0)");
                if scored.len() > 1 {
                    println!("% runner-up token {} logit {:.4}  margin {:+.4}", scored[1].0, scored[1].1, s - scored[1].1);
                }
            }
        }
        return;
    }

    // `fieldrun stitch <step.dl …> [-o out.dl]` — merge N per-step `export --logic` / `/export-logic` programs (each ONE
    // next-token decision) into ONE runnable, step-indexed semiring-Datalog program: decide(Step,T) over the whole decode
    // trajectory. PURE TEXT — no model: it parses the candidate/contrib FACTS out of each file and re-emits them under a
    // step index, so the per-step Σ-contrib sums never collide (the reason a naïve `cat` of the parts is wrong). Batch:
    // pass the files (shell glob `prefix.*.dl`) or a bare prefix. Runs in Soufflé AND `fieldrun eval` (step-aware).
    // This is the "single .dl for the whole query" — but it is a TRACE of THIS query's trajectory: it does NOT answer new
    // queries (that's the context-free whole-model emit, LOGIC_EXPORT LO3a, still open). Stitch documents; it doesn't generalize.
    if matches!(args.get(1).map(String::as_str), Some("stitch")) {
        let out_path = flag(&args, "-o").or_else(|| flag(&args, "--out")).map(String::from);
        // inputs = the non-flag args after `stitch`; a bare prefix (no .dl) expands to sibling <prefix>*.dl in its dir.
        let mut inputs: Vec<String> = Vec::new();
        let mut it = args.iter().skip(2).peekable();
        while let Some(a) = it.next() {
            if a == "-o" || a == "--out" { it.next(); continue; }
            if a.starts_with('-') { continue; }
            if a.ends_with(".dl") {
                inputs.push(a.clone());
            } else {
                let p = std::path::Path::new(a);
                let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or_else(|| std::path::Path::new("."));
                let base = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                if let Ok(rd) = std::fs::read_dir(dir) {
                    for e in rd.flatten() {
                        let name = e.file_name().to_string_lossy().to_string();
                        if name.starts_with(&base) && name.ends_with(".dl") {
                            inputs.push(e.path().to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
        inputs.sort(); // prefix.000.dl, prefix.001.dl, … sort into decode order; user's logic-001/003 sort numerically too
        inputs.dedup();
        if inputs.is_empty() {
            eprintln!("[fieldrun] stitch: no input .dl files — give the per-step programs, e.g. fieldrun stitch trace.*.dl -o whole.dl");
            std::process::exit(2);
        }
        let mut body = String::new();
        let mut steps = 0usize;
        for (k, f) in inputs.iter().enumerate() {
            let text = match std::fs::read_to_string(f) {
                Ok(t) => t,
                Err(e) => { eprintln!("[fieldrun] stitch: cannot read {f}: {e}"); std::process::exit(1); }
            };
            let pred = text.lines().find(|l| l.trim_start().starts_with("// model predicts:"))
                .map(|l| l.trim_start().trim_start_matches("// ").to_string()).unwrap_or_default();
            let fname = std::path::Path::new(f).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| f.clone());
            body.push_str(&format!("\n// ---- step {k:03}  ({fname})  {pred} ----\n"));
            let mut facts = 0usize;
            for line in text.lines() {
                let t = line.trim_start();
                let open = match t.find('(') { Some(i) => i, None => continue };
                let head = &t[..open];
                if head != "candidate" && head != "contrib" { continue; } // FACTS only — skip decls, rules, .output, comments
                let close = match t.find(')') { Some(c) => c, None => continue };
                let inner = &t[open + 1..close];
                let after = &t[close + 1..]; // ".   // comment" — preserved
                body.push_str(&format!("{head}({k}, {inner}){after}\n"));
                facts += 1;
            }
            if facts == 0 {
                eprintln!("[fieldrun] stitch: {f} had no candidate/contrib facts (is it an export --logic program?) — skipping");
            } else {
                steps += 1;
            }
        }
        let mut prog = String::new();
        prog.push_str("// ============================================================\n");
        prog.push_str("// fieldrun logic STITCH — N per-step decode programs merged into ONE step-indexed semiring-Datalog program.\n");
        prog.push_str("// Each input was ONE next-token decision; step S here indexes the decode trajectory. decide(S,T) = token at step S.\n");
        prog.push_str("// A single runnable file for the WHOLE query's decode — but a TRACE of THIS query, NOT the context-free\n");
        prog.push_str("// whole-model program (it does not answer new queries; that is LOGIC_EXPORT LO3a, open).\n");
        prog.push_str(&format!("// {steps} steps. Run: souffle <this>.dl -D -   |   fieldrun eval <this>.dl --semiring max|log\n"));
        prog.push_str("// ============================================================\n\n");
        prog.push_str(".decl candidate(step:number, t:number)\n");
        prog.push_str(".decl contrib(step:number, block:symbol, t:number, w:float)\n");
        prog.push_str(".decl logit(step:number, t:number, s:float)\n");
        prog.push_str(".decl decide(step:number, t:number)\n\n");
        prog.push_str("logit(Step, T, S) :- candidate(Step, T), S = sum W : { contrib(Step, _, T, W) }.   // ⊗ over blocks, scoped per step\n");
        prog.push_str("decide(Step, T)   :- logit(Step, T, S), S = max S2 : { logit(Step, _, S2) }.        // ⊕ = max within each step\n");
        prog.push_str(".output decide\n");
        prog.push_str(&body);
        match &out_path {
            Some(p) => match std::fs::write(p, &prog) {
                Ok(()) => eprintln!("[fieldrun] stitch → {p}  ({steps} steps from {} files) — run: souffle {p} -D -  |  fieldrun eval {p} --semiring max|log", inputs.len()),
                Err(e) => { eprintln!("[fieldrun] stitch: cannot write {p}: {e}"); std::process::exit(1); }
            },
            None => print!("{prog}"),
        }
        return;
    }

    let store_explicit = flag(&args, "--store");
    let store_path = store_explicit.unwrap_or("../lm-sae/pylm/store_gpt2.json");
    let ids_path = flag(&args, "--ids").unwrap_or("../lm-sae/pylm/holdout_gpt2.json");
    let ctx_window: usize = flag(&args, "--ctx").and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_eval: usize = flag(&args, "--n-eval").and_then(|s| s.parse().ok()).unwrap_or(500);

    // ids are needed for scoring / --generate / --explain / Tier A; --serve and --chat don't use them, so load
    // gracefully (empty if absent) rather than panicking when someone just wants to serve or chat.
    let ids: Vec<i64> = std::fs::read_to_string(ids_path)
        .ok()
        .and_then(|s| serde_json::from_str::<Holdout>(&s).ok())
        .map(|h| h.holdout_ids)
        .unwrap_or_default();
    let end = (ctx_window + n_eval).min(ids.len());
    let ctx = |i: usize| &ids[i.saturating_sub(ctx_window)..i];
    let threads = rayon::current_num_threads();

    // Tier B (composition) — the real forward pass from a fieldrun bundle; positions scored in parallel.
    if let Some(raw) = flag(&args, "--bundle") {
        let stem = resolve_bundle(raw); // bare name -> bundles/<name>/<name> if that's where convert put it
        // Clear "not found" up front (before device/spinner/a raw OS error): --bundle runs a LOCAL bundle and does not
        // pull from HF, so a missing name almost always means "you haven't converted it yet".
        if !std::path::Path::new(&format!("{stem}.fieldrun.json")).exists() {
            eprintln!("[fieldrun] bundle {raw:?} not found — no {stem}.fieldrun.json (looked for an explicit stem, then \
                       under the cache {}).", bundles_dir());
            let avail = available_bundles();
            if avail.is_empty() {
                eprintln!("[fieldrun] no bundles in the cache yet.");
            } else {
                eprintln!("[fieldrun] cached bundles: {}", avail.join(", "));
            }
            eprintln!("[fieldrun] --bundle runs a LOCAL bundle; it does NOT download from Hugging Face. To fetch + build:\n  \
                       fieldrun convert --model <hf-repo-id | local-dir> --arch <arch>   then  --bundle {raw}");
            std::process::exit(1);
        }
        // device selection (CPU default + reference; GPU opt-in via --features gpu). Matmul dispatch lands next; this
        // reports the choice + budget/fallback so the plumbing is in place.
        let model_bytes = std::fs::metadata(format!("{stem}.fieldrun.bin")).map(|m| m.len()).unwrap_or(0);
        // fit budget = detected system RAM (the real constraint — the CPU loads the weights into RAM), overridable
        // with --max-vram <GB>; 0 if both are unavailable (then the line just shows the model size, no RAM number).
        let ram_bytes = flag(&args, "--max-vram")
            .and_then(|s| s.parse::<u64>().ok())
            .map(|gb| gb * 1_000_000_000)
            .or_else(device::total_ram_bytes)
            .unwrap_or(0);
        let dev = device::select(flag(&args, "--device").unwrap_or("auto"), model_bytes, ram_bytes);
        eprintln!("[fieldrun] device: {}", dev.detail);
        // --gpu-check: validate the GPU-resident GPT-2 forward against the CPU forward (top-1 agreement + GPU tok/s).
        #[cfg(feature = "gpu")]
        if has_flag(&args, "--gpu-check") {
            let b1 = Bundle::load(&stem).expect("load bundle");
            let n = n_eval.min(50);
            let last = (ctx_window + n).min(ids.len());
            let ctxs: Vec<&[i64]> = (ctx_window..last).map(|i| &ids[i.saturating_sub(ctx_window)..i]).collect();
            // CPU reference predictions + the matching GPU kernel + name
            let (cp, name, t0, gp) = match b1.arch.as_str() {
                "gpt2" => {
                    let g = gpu_gpt2::GpuGpt2::new(&b1).expect("no GPU adapter");
                    let cpu = Gpt2::new(Bundle::load(&stem).expect("load"), 0.0, false);
                    let cp: Vec<i64> = ctxs.iter().map(|c| cpu.predict(c)).collect();
                    let t0 = std::time::Instant::now();
                    let gp: Vec<i64> = ctxs.iter().map(|c| g.predict(c, &b1)).collect();
                    (cp, g.name.clone(), t0, gp)
                }
                "rope" => {
                    let g = gpu_rope::GpuRope::new(&b1).expect("no GPU adapter");
                    let cpu = Rope::new(Bundle::load(&stem).expect("load"), 0.0, false);
                    let cp: Vec<i64> = ctxs.iter().map(|c| cpu.predict(c)).collect();
                    let t0 = std::time::Instant::now();
                    let gp: Vec<i64> = ctxs.iter().map(|c| g.predict(c, &b1)).collect();
                    (cp, g.name.clone(), t0, gp)
                }
                other => { println!("[fieldrun] --gpu-check: arch {other} not supported (gpt2, rope)"); return; }
            };
            let gsec = t0.elapsed().as_secs_f64();
            let agree = gp.iter().zip(&cp).filter(|(a, b)| a == b).count();
            println!("[fieldrun] GPU [{}] vs CPU forward: {}/{} top-1 agree · {:.1} GPU fwd/s",
                     name, agree, gp.len(), gp.len() as f64 / gsec);
            return;
        }

        // live spinner while the bundle loads (mmap + dequant; a multi-GB int8 model takes a few seconds), so it's
        // clearly working, not hung — then "loaded", then the mode (chat prompt / server line) appears.
        let bundle = {
            use std::io::Write;
            use std::sync::atomic::{AtomicBool, Ordering};
            let done = std::sync::Arc::new(AtomicBool::new(false));
            let d2 = done.clone();
            let sp = std::thread::spawn(move || {
                let frames = ['|', '/', '-', '\\'];
                let t0 = std::time::Instant::now();
                let mut i = 0usize;
                while !d2.load(Ordering::Relaxed) {
                    eprint!("\r[fieldrun] loading bundle {} {:.0}s …", frames[i % 4], t0.elapsed().as_secs_f64());
                    let _ = std::io::stderr().flush();
                    std::thread::sleep(std::time::Duration::from_millis(250));
                    i += 1;
                }
            });
            let r = Bundle::load(&stem);
            done.store(true, Ordering::Relaxed);
            let _ = sp.join();
            match r {
                Ok(b) => {
                    eprintln!("\r[fieldrun] loaded bundle ({} MB)                    ", model_bytes / 1_000_000);
                    b
                }
                Err(e) => {
                    eprintln!("\r[fieldrun] couldn't load bundle {stem:?}: {e}                    ");
                    eprintln!("[fieldrun] expected {stem}.fieldrun.json + .bin. Convert one first \
                               (`fieldrun convert --model … --arch …`) or pass the bundle stem/name.");
                    std::process::exit(1);
                }
            }
        };
        let arch = bundle.arch.clone();
        #[cfg(feature = "api")]
        let eos = bundle.eos.clone(); // for the text API / --chat stop condition
        let route: f32 = flag(&args, "--route-frac").and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let kv_int8 = has_flag(&args, "--kv-int8");
        let mut lm: Box<dyn Model> = match arch.as_str() {
            "gpt2" => Box::new(Gpt2::new(bundle, route, kv_int8)),
            "neox" => Box::new(Neox::new(bundle, route, kv_int8)),
            "rope" => Box::new(Rope::new(bundle, route, kv_int8)),
            "gemma" => Box::new(Gemma::new(bundle, route, kv_int8)),
            "gemma3" => Box::new(Gemma3::new(bundle, route, kv_int8)),
            "gemma4" => Box::new(Gemma4::new(bundle, route, kv_int8)),
            "qwen3moe" => Box::new(Qwen3Moe::new(bundle, route, kv_int8)),
            "mla" => Box::new(Mla::new(bundle, route, kv_int8)),
            "minimax" => Box::new(MiniMax::new(bundle, route, kv_int8)),
            "dsv4" => Box::new(Dsv4::new(bundle, route, kv_int8)),
            other => panic!("unknown bundle arch {other:?} (have: gpt2, neox, rope, gemma, gemma3, gemma4, qwen3moe, mla, minimax, dsv4)"),
        };

        // --pruned-head: margin-gated retrieval-pruned output head on the DECODE loops (serve/chat/stream). The KB
        // proposes ~540 candidates per step; the unembed scores only those rows; the pick is accepted iff the in-set
        // normalized margin (exact facet distance, FINDINGS §5b) clears --pruned-margin, else the full head runs.
        // Distinct from --prune-head (the explain-only measurement mode): this one changes the serving decode, so it
        // is opt-in, off by default, and measured by --gate-check (top-1 agreement vs the full head).
        if has_flag(&args, "--pruned-head") {
            let thr: f32 = flag(&args, "--pruned-margin").and_then(|s| s.parse().ok()).unwrap_or(2.0);
            match flag(&args, "--store").map(Store::load) {
                Some(Ok(s)) => {
                    if lm.set_head_gate(std::sync::Arc::new(headgate::HeadGate::new(s, thr))) {
                        eprintln!("[fieldrun] --pruned-head: margin-gated pruned unembed ON (accept ≥ {thr} normalized margin; below it, full-head fallback)");
                    } else {
                        eprintln!("[fieldrun] --pruned-head: arch {arch} doesn't wire the gated head (rope only) — running ungated");
                    }
                }
                Some(Err(e)) => eprintln!("[fieldrun] --pruned-head: couldn't load --store: {e} — running ungated"),
                None => eprintln!("[fieldrun] --pruned-head needs --store <store.json> (the KB proposes the candidate set) — running ungated"),
            }
        }

        // --prune-head (Phase 8b): measure the retrieval-pruned output head. The KB proposes a small candidate set per
        // position; the full-vocab unembed collapses to scoring only those. Because the pruned head scores the SAME
        // unembed rows, pruned-argmax == full-argmax exactly when the candidate set contains the full head's argmax — so
        // top-1 fidelity == candidate-set COVERAGE of the full argmax. Reports the coverage-vs-size curve (sweeping
        // candidate configs) + the head speedup (full vs subset unembed) + the unembed's share of per-token compute.
        // Context-only by default (recent + induction, needs no store); pass `--store <store.json>` to add KB n-grams.
        if has_flag(&args, "--prune-head") {
            use retrieval::{context_candidates, CandCfg};
            let store: Option<Store> = flag(&args, "--store").and_then(|p| match Store::load(p) {
                Ok(s) => Some(s),
                Err(e) => { eprintln!("[fieldrun] --store {p:?}: {e} (continuing context-only)"); None }
            });
            let b2 = Bundle::load(&stem).expect("reload bundle for unembed microbench");
            let un = if b2.has("lm_head") { "lm_head" } else { "embed" };
            let (vocab, d) = b2.dims(un);
            let positions: Vec<&[i64]> = (ctx_window..end).map(|i| ctx(i)).collect();
            if positions.is_empty() {
                eprintln!("[fieldrun] --prune-head: no eval positions (need --ids with > ctx_window tokens, matching this model's vocab)");
                return;
            }
            eprintln!("[fieldrun] --prune-head: {} positions (ctx {ctx_window}), unembed {un} ({vocab}×{d}), store: {}",
                      positions.len(), if store.is_some() { "n-gram KB" } else { "context-only" });
            // Ground truth: the FULL head's argmax per position (what the pruned head must reproduce). Parallel forwards.
            let t_truth = std::time::Instant::now();
            let truth: Vec<i64> = positions.par_iter().map(|c| lm.predict(c)).collect();
            let predict_ms = t_truth.elapsed().as_secs_f64() * 1e3 / positions.len() as f64;

            let build = |c: &[i64], cfg: &CandCfg| -> Vec<i64> {
                match store.as_ref() {
                    Some(s) => s.candidates(c, cfg),
                    None => {
                        let mut o = Vec::new();
                        context_candidates(c, cfg.recent, cfg.induction, &mut o);
                        let mut seen = std::collections::HashSet::new();
                        o.retain(|&t| seen.insert(t));
                        o
                    }
                }
            };
            let z = CandCfg { recent: 0, induction: 0, quad: 0, tri: 0, bi: 0, skel: 0, uni: 0, closed: false };
            // sweep: context-only (recent+induction; work with no store) then KB-augmented (n-gram/grammar/unigram —
            // these add tokens ONLY when a --store is loaded). Spans |C| from a few to a few hundred → the coverage knee.
            let cfgs: Vec<(&str, CandCfg)> = vec![
                ("recent8",          CandCfg { recent: 8,  ..z }),
                ("recent32+ind3",    CandCfg { recent: 32, induction: 3, ..z }),
                ("recent64+ind4",    CandCfg { recent: 64, induction: 4, ..z }),
                ("recent128+ind4",   CandCfg { recent: 128, induction: 4, ..z }),
                ("ngram16",          CandCfg { recent: 16, induction: 3, quad: 6, tri: 6, bi: 6, ..z }),
                ("ngram+grammar",    CandCfg { recent: 16, induction: 3, quad: 6, tri: 6, bi: 6, skel: 6, closed: true, ..z }),
                ("generous~256",     CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true }),
                ("generous~512",     CandCfg { recent: 128, induction: 4, quad: 16, tri: 16, bi: 16, skel: 16, uni: 256, closed: true }),
            ];
            println!("\n=== retrieval-pruned head — coverage sweep ({} positions) ===", positions.len());
            println!("{:<18} {:>9} {:>8} {:>14}", "config", "mean|C|", "cov%", "V/|C| (head×)");
            let mut best: Option<(f64, Vec<i64>)> = None; // (mean|C|, a representative candidate set) for the balanced config
            for (name, cfg) in &cfgs {
                let mut tot = 0usize;
                let mut cov = 0usize;
                let mut sample: Vec<i64> = Vec::new();
                for (c, &t) in positions.iter().zip(&truth) {
                    let cands = build(c, cfg);
                    tot += cands.len();
                    if cands.contains(&t) {
                        cov += 1;
                    }
                    if sample.is_empty() {
                        sample = cands;
                    }
                }
                let mean = tot as f64 / positions.len() as f64;
                let covp = 100.0 * cov as f64 / positions.len() as f64;
                println!("{name:<18} {mean:>9.1} {covp:>7.1}% {:>13.1}×", vocab as f64 / mean.max(1.0));
                if *name == "generous~256" {
                    best = Some((mean, sample));
                }
            }
            // Conditional analysis (needs a store): does the KB's CONFIDENCE (which idiom fired) predict coverage? If a
            // high-confidence idiom (induction/quad) covers the argmax far more often than the unigram floor, a gate that
            // prunes ONLY when that idiom fires is high-precision. The KB-top-1==argmax column is the Phase-6 signal: when
            // it's high, you can emit the KB token and skip the WHOLE forward (not just the head).
            if let Some(s) = store.as_ref() {
                let gen = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
                let mut by: std::collections::HashMap<String, [usize; 3]> = std::collections::HashMap::new(); // idiom -> [count, kb_top1==argmax, argmax∈cands]
                for (c, &t) in positions.iter().zip(&truth) {
                    let (kb, idiom) = s.predict(c);
                    let e = by.entry(idiom).or_default();
                    e[0] += 1;
                    if kb == t { e[1] += 1; }
                    if s.candidates(c, &gen).contains(&t) { e[2] += 1; }
                }
                let mut rows: Vec<(String, [usize; 3])> = by.into_iter().collect();
                rows.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
                println!("\nper-idiom (KB confidence signal) — does a fired idiom predict coverage / standalone correctness?");
                println!("{:<14} {:>6} {:>14} {:>12}", "idiom", "n", "KB top1=argmax", "cov(gen)");
                for (idiom, e) in &rows {
                    let (n, acc, cov) = (e[0], e[1], e[2]);
                    println!("{idiom:<14} {n:>6} {:>13.1}% {:>11.1}%", 100.0 * acc as f64 / n as f64, 100.0 * cov as f64 / n as f64);
                }
            }

            // Head speedup: time the full-vocab unembed vs the subset unembed for a representative candidate set.
            let (mean_c, cand) = best.unwrap_or((1.0, vec![0]));
            let mut s: u64 = 0x243F6A8885A308D3;
            let mut x: Vec<f32> = (0..d).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0 }).collect();
            x.truncate(d);
            let iters = 200usize;
            let t = std::time::Instant::now();
            let mut sink = 0.0f32;
            for _ in 0..iters { sink += b2.rowdot_f32(un, &x).iter().cloned().fold(f32::MIN, f32::max); }
            let full_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let t = std::time::Instant::now();
            for _ in 0..iters { sink += b2.rowdot_f32_subset(un, &x, &cand).iter().cloned().fold(f32::MIN, f32::max); }
            let sub_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("\nunembed head (|C|={:.0}):  full {full_us:.1} µs/tok   subset {sub_us:.1} µs/tok   head speedup {:.1}×", mean_c, full_us / sub_us);
            // End-to-end framing must use a single-token DECODE step (where pruning matters), NOT predict() on a 64-token
            // context (a prefill — there the unembed is a tiny share). Time a real KV-cached decode: generate from a short
            // prompt and per-token ≈ elapsed/N (the short-prompt prefill is negligible vs N decode steps).
            let short: Vec<i64> = positions[0].iter().take(8).copied().collect();
            let ndec = 24usize;
            let t = std::time::Instant::now();
            let g = lm.generate(&short, ndec);
            let decode_ms = t.elapsed().as_secs_f64() * 1e3 / g.len().max(1) as f64;
            let tok_pruned_ms = (decode_ms - full_us / 1e3 + sub_us / 1e3).max(0.0);
            println!("(64-ctx prefill {predict_ms:.1} ms/pos — for reference)");
            println!("decode token (forward+full-head) ≈ {decode_ms:.2} ms; unembed share of DECODE ≈ {:.0}%", 100.0 * (full_us / 1e3) / decode_ms.max(1e-6));
            println!("end-to-end pruned-head decode token ≈ {tok_pruned_ms:.2} ms  ⇒  {:.2}× decode tok/s (IF the candidate set covers the argmax)", decode_ms / tok_pruned_ms.max(1e-6));
            println!("(coverage = top-1 agreement with the full head; sink={sink:.3})");
            return;
        }

        // --attribute (the explain/attribution side of Phase 8b): route EACH token of a holdout to a KB rule or to
        // composition. Three routes — RETRIEVED (a symbolic KB rule's top-1 == the model's argmax: a pure lookup),
        // SELECTED (the argmax is in the KB candidate set but isn't the KB top-1: composition disambiguated within a
        // retrieved set), COMPOSED (no KB rule covers it: the irreducible forge tax). The per-token trace + the
        // aggregate retrieved/selected/composed split make the KB-vs-composition thesis observable token by token —
        // the retrieval half of explain (the composition half is the DLA circuit trace). Needs `--store`.
        if has_flag(&args, "--attribute") {
            use retrieval::CandCfg;
            let store = match flag(&args, "--store").and_then(|p| Store::load(p).ok()) {
                Some(s) => s,
                None => { eprintln!("[fieldrun] --attribute needs --store <store.json> (the KB rules to attribute against)"); return; }
            };
            let dec = load_decoder(flag(&args, "--vocab"));
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            let positions: Vec<&[i64]> = (ctx_window..end).map(|i| ctx(i)).collect();
            if positions.is_empty() {
                eprintln!("[fieldrun] --attribute: no eval positions (need --ids > ctx_window, matching this model's vocab)");
                return;
            }
            let trace_n = positions.len().min(30); // readable per-token trace; aggregate is over all positions
            eprintln!("[fieldrun] --attribute: routing {} tokens (ctx {ctx_window}) — RETRIEVED / SELECTED / COMPOSED", positions.len());
            // 3-way counts overall + per-idiom (idiom -> [retrieved, selected, composed]).
            let (mut retr, mut sel, mut comp) = (0usize, 0usize, 0usize);
            let mut by: std::collections::HashMap<String, [usize; 3]> = std::collections::HashMap::new();
            println!("\n=== per-token attribution (first {trace_n}) — token ← route via KB rule ===");
            for (i, c) in positions.iter().enumerate() {
                let truth = lm.predict(c);
                let (kb, idiom) = store.predict(c);
                let covered = store.candidates(c, &cfg).contains(&truth);
                let (route, slot) = if kb == truth { ("RETRIEVED", 0) } else if covered { ("SELECTED ", 1) } else { ("COMPOSED ", 2) };
                match slot { 0 => retr += 1, 1 => sel += 1, _ => comp += 1 };
                by.entry(idiom.clone()).or_default()[slot] += 1;
                if i < trace_n {
                    println!("  {route} {:<22} via {idiom}", dec(truth));
                }
            }
            let n = positions.len() as f64;
            println!("\n=== decomposition of {} next-token decisions ===", positions.len());
            println!("  RETRIEVED (KB rule alone = model)        {retr:>4}  {:>5.1}%", 100.0 * retr as f64 / n);
            println!("  SELECTED  (in KB set, composition picks) {sel:>4}  {:>5.1}%", 100.0 * sel as f64 / n);
            println!("  COMPOSED  (no KB rule — the forge tax)   {comp:>4}  {:>5.1}%", 100.0 * comp as f64 / n);
            let mut rows: Vec<(String, [usize; 3])> = by.into_iter().collect();
            rows.sort_by(|a, b| (b.1[0] + b.1[1] + b.1[2]).cmp(&(a.1[0] + a.1[1] + a.1[2])));
            println!("\n  by KB rule that fired:   idiom            n   retr%  sel%  comp%");
            for (idiom, e) in &rows {
                let tot = (e[0] + e[1] + e[2]).max(1) as f64;
                println!("    {idiom:<16} {:>4}  {:>5.0} {:>5.0} {:>5.0}", e[0] + e[1] + e[2],
                         100.0 * e[0] as f64 / tot, 100.0 * e[1] as f64 / tot, 100.0 * e[2] as f64 / tot);
            }
            return;
        }

        // --probe (the SELECTED conflict-resolution question): is the model's pick a FUNCTION of the rule-firing state?
        // Forward-chaining framing — the candidate set is the conflict set, SELECTED is conflict resolution. Two tests:
        //   (A) rank of the pick within its explaining rule — rank 1 == "max-incidence" conflict resolution reproduces
        //       it; the spread over ranks is the deviation a fixed count-ordering strategy can't capture.
        //   (B) within-bucket pick entropy when the conflict set is held FIXED (bucket by the n-gram key). H≈0 / 100%
        //       agreement ⇒ the pick is a function of the firing state (symbolic-representable); H>0 is the residue that
        //       needs a finer incidence space than the rules carry. Finer key (bi→tri) = finer incidence partition.
        if has_flag(&args, "--probe") {
            use retrieval::CandCfg;
            let store = match flag(&args, "--store").and_then(|p| Store::load(p).ok()) {
                Some(s) => s,
                None => { eprintln!("[fieldrun] --probe needs --store <store.json>"); return; }
            };
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            let positions: Vec<&[i64]> = (ctx_window..end).map(|i| ctx(i)).collect();
            if positions.is_empty() {
                eprintln!("[fieldrun] --probe: no eval positions");
                return;
            }
            eprintln!("[fieldrun] --probe: {} positions (ctx {ctx_window}) — running forwards…", positions.len());
            // record per position: (pick, route, last token, (t-2,t-1), rank-of-pick-in-its-rule | None if off-rule)
            struct Rec { pick: i64, route: u8, bik: i64, trik: (i64, i64), rank: Option<usize> }
            let recs: Vec<Rec> = positions.par_iter().map(|c| {
                let pick = lm.predict(c);
                let (kb, _) = store.predict(c);
                let covered = store.candidates(c, &cfg).contains(&pick);
                let route = if kb == pick { 0u8 } else if covered { 1 } else { 2 }; // RETRIEVED / SELECTED / COMPOSED
                let n = c.len();
                Rec { pick, route, bik: c[n - 1], trik: (if n >= 2 { c[n - 2] } else { -1 }, c[n - 1]),
                      rank: store.rule_for(c, pick).and_then(|r| r.rank) }
            }).collect();

            // (A) picked-rank distribution over SELECTED positions (does a fixed count-ordering reproduce the pick?).
            let sel: Vec<&Rec> = recs.iter().filter(|r| r.route == 1).collect();
            let (mut r1, mut r2, mut r3, mut r4p, mut offrule) = (0, 0, 0, 0, 0);
            for r in &sel {
                match r.rank {
                    Some(1) => r1 += 1,
                    Some(2) => r2 += 1,
                    Some(3) => r3 += 1,
                    Some(_) => r4p += 1,
                    None => offrule += 1, // pick covered via recent/closed/floor, not in any named rule's successors
                }
            }
            let ns = sel.len().max(1) as f64;
            println!("\n=== (A) SELECTED picked-rank within its explaining rule ({} SELECTED positions) ===", sel.len());
            println!("  rank 1 (== max-incidence) {r1:>4}  {:>5.1}%   ← a fixed 'pick the highest-count successor' strategy reproduces these", 100.0 * r1 as f64 / ns);
            println!("  rank 2                    {r2:>4}  {:>5.1}%", 100.0 * r2 as f64 / ns);
            println!("  rank 3                    {r3:>4}  {:>5.1}%", 100.0 * r3 as f64 / ns);
            println!("  rank 4+                   {r4p:>4}  {:>5.1}%", 100.0 * r4p as f64 / ns);
            println!("  off-rule (recent/floor)   {offrule:>4}  {:>5.1}%   ← not in any named rule's RHS at all", 100.0 * offrule as f64 / ns);

            // (B) within-bucket pick entropy when the conflict set is held fixed. Restrict to SELECTED.
            let h = |picks: &[i64]| -> f64 {
                let mut cnt: HashMap<i64, usize> = HashMap::new();
                for &p in picks { *cnt.entry(p).or_default() += 1; }
                let n = picks.len() as f64;
                cnt.values().map(|&c| { let p = c as f64 / n; -p * p.log2() }).sum()
            };
            let bucket_stats = |buckets: Vec<Vec<i64>>| -> (usize, usize, f64, f64) {
                let nz: Vec<Vec<i64>> = buckets.into_iter().filter(|b| b.len() >= 2).collect();
                let total: usize = nz.iter().map(|b| b.len()).sum();
                if total == 0 { return (0, 0, 0.0, 0.0); }
                let wh: f64 = nz.iter().map(|b| b.len() as f64 * h(b)).sum::<f64>() / total as f64; // weighted H(pick|bucket)
                // top-1 agreement: Σ plurality / total
                let agree: usize = nz.iter().map(|b| {
                    let mut cnt: HashMap<i64, usize> = HashMap::new();
                    for &p in b { *cnt.entry(p).or_default() += 1; }
                    *cnt.values().max().unwrap()
                }).sum();
                (nz.len(), total, wh, 100.0 * agree as f64 / total as f64)
            };
            let by_bi = { let mut m: HashMap<i64, Vec<i64>> = HashMap::new(); for r in &sel { m.entry(r.bik).or_default().push(r.pick); } bucket_stats(m.into_values().collect()) };
            let by_tri = { let mut m: HashMap<(i64, i64), Vec<i64>> = HashMap::new(); for r in &sel { m.entry(r.trik).or_default().push(r.pick); } bucket_stats(m.into_values().collect()) };
            let h0 = h(&sel.iter().map(|r| r.pick).collect::<Vec<_>>()); // baseline marginal entropy of the SELECTED pick
            println!("\n=== (B) is the SELECTED pick a function of the conflict set? bucket by the n-gram key, hold the conflict set fixed ===");
            println!("  baseline H(pick) over all SELECTED = {h0:.2} bits (no conditioning)");
            println!("  {:<26}{:>10}{:>10}{:>14}{:>13}", "signature (incidence)", "buckets≥2", "positions", "H(pick|sig)", "top-1 agree");
            println!("  {:<26}{:>10}{:>10}{:>13.2}{:>12.1}%", "bigram-key  (last token)", by_bi.0, by_bi.1, by_bi.2, by_bi.3);
            println!("  {:<26}{:>10}{:>10}{:>13.2}{:>12.1}%", "trigram-key (last 2 tok)", by_tri.0, by_tri.1, by_tri.2, by_tri.3);
            println!("  (H→0 / agree→100% as the key tightens ⇒ the pick IS a function of the firing state = conflict resolution;");
            println!("   a plateau below that is the residue needing a finer incidence space than the rules carry.)");
            return;
        }

        // --probe-margin (PROVABLE_OPT PO-T7, the grokking order parameter): per held-out position, the decode margin
        // m = predicted_logit − runner_up_logit and the full-spectrum DLA participation ratio (PR = circuit concentration,
        // low = consolidated/retrievable, high = diffuse/forge-tax). Reports the CERTIFIABLE-COMPRESSIBLE FRACTION
        // P(m > 2δ) at fixed δ — the fraction of tokens a δ-bounded compression provably preserves (the margin certificate
        // PO-T3) — plus median margin, median PR, and top-1 accuracy. Store-free, arch-agnostic (uses explain) — meant to
        // be tracked across training checkpoints (Pythia @stepN) to see if the certifiable fraction rises with circuit
        // consolidation (Grok's grokking prediction). One line of parseable output per run.
        if has_flag(&args, "--probe-margin") {
            let cap = (end - ctx_window).min(n_eval);
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            if positions.is_empty() {
                eprintln!("[fieldrun] --probe-margin: no eval positions (need --ids with > ctx_window tokens, matching this model's vocab)");
                return;
            }
            eprintln!("[fieldrun] --probe-margin: {} positions (ctx {ctx_window}) — explain forwards…", positions.len());
            // also collect each position's dominant (layer,head) DLA circuits, for the R3 circuit-IDENTITY
            // fingerprint (which circuits consolidate across training — diffed over the late event).
            let recs: Vec<(f32, f32, bool, Vec<(usize, usize, f32)>)> = positions.par_iter().enumerate().filter_map(|(k, c)| {
                let ex = lm.explain(c)?;
                let margin = ex.predicted_logit - ex.runner_up_logit;
                let d: Vec<f32> = ex.all_dla.iter().copied().filter(|&x| x > 0.0).collect();
                let pr = if !d.is_empty() {
                    let (s, sq): (f32, f32) = (d.iter().sum(), d.iter().map(|x| x * x).sum());
                    if sq > 0.0 { s * s / sq } else { 1.0 }
                } else { f32::NAN };
                let top: Vec<(usize, usize, f32)> = ex.head_circuits.iter().take(3)
                    .filter(|h| h.dla > 0.0).map(|h| (h.layer, h.head, h.dla)).collect();
                Some((margin, pr, ex.model_predicts == ids[ctx_window + k], top))
            }).collect();
            let n = recs.len().max(1);
            let frac = |d: f32| 100.0 * recs.iter().filter(|r| r.0 > 2.0 * d).count() as f32 / n as f32;
            let mut ms: Vec<f32> = recs.iter().map(|r| r.0).collect();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mmed = ms[ms.len() / 2];
            let mut prs: Vec<f32> = recs.iter().map(|r| r.1).filter(|x| x.is_finite()).collect();
            prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let prmed = if prs.is_empty() { f32::NAN } else { prs[prs.len() / 2] };
            let acc = 100.0 * recs.iter().filter(|r| r.2).count() as f32 / n as f32;
            // certifiable-compressible fraction P(m > 2δ) at δ = 0.5/1/2/4 (perturbation a small/medium quant would induce)
            println!("PROBE_MARGIN n={n} acc={acc:.2} margin_med={mmed:.4} pr_med={prmed:.3} cert_d0.5={:.2} cert_d1={:.2} cert_d2={:.2} cert_d4={:.2}",
                frac(0.5), frac(1.0), frac(2.0), frac(4.0));
            // PROBE_CIRCUITS: aggregate DLA per (layer,head) over all positions → the dominant-circuit fingerprint.
            let mut agg: std::collections::HashMap<(usize, usize), f32> = std::collections::HashMap::new();
            for r in &recs { for &(l, h, dla) in &r.3 { *agg.entry((l, h)).or_insert(0.0) += dla; } }
            let total: f32 = agg.values().copied().sum::<f32>().max(1e-9);
            let mut items: Vec<((usize, usize), f32)> = agg.into_iter().collect();
            items.sort_by(|a, b| b.1.total_cmp(&a.1));
            let fp = items.iter().take(8)
                .map(|&((l, h), v)| format!("{l}.{h}:{:.3}", v / total)).collect::<Vec<_>>().join("|");
            println!("PROBE_CIRCUITS n_circuits={} top={fp}", items.len());
            return;
        }

        // --probe-dla (combine vs select): for each pick, is the logit DOMINATED by one circuit (disguised selection)
        // or SPREAD over many (genuine superposition/combination)? Per position, take the per-circuit DLA contributions
        // to the predicted token (heads + neurons, from the faithful explain forward) and measure concentration —
        // top-1 share (max DLA / Σ captured DLA), participation ratio PR = (Σd)²/Σd² (effective # of circuits, 1 =
        // one dominates), and the top circuit's share of the TRUE predicted logit — bucketed by route. Prediction:
        // RETRIEVED concentrates (one rule writes it), COMPOSED spreads, SELECTED in between = partial superposition.
        if has_flag(&args, "--probe-dla") {
            use retrieval::CandCfg;
            let store = match flag(&args, "--store").and_then(|p| Store::load(p).ok()) {
                Some(s) => s,
                None => { eprintln!("[fieldrun] --probe-dla needs --store"); return; }
            };
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            let cap = (end - ctx_window).min(n_eval); // explain is the expensive faithful forward
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            if positions.is_empty() {
                eprintln!("[fieldrun] --probe-dla: no eval positions");
                return;
            }
            eprintln!("[fieldrun] --probe-dla: {} positions (ctx {ctx_window}) — running faithful explain forwards…", positions.len());
            let b2 = Bundle::load(&stem).expect("reload bundle for U-row norms");
            let un = if b2.has("lm_head") { "lm_head" } else { "embed" };
            struct Rec { route: u8, pr: f32, captured: usize, margin: f32, nmargin: f32, top_hit: bool, mu_t: usize }
            let recs: Vec<Rec> = positions.par_iter().filter_map(|c| {
                let ex = lm.explain(c)?;
                let pick = ex.model_predicts;
                let (kb, _) = store.predict(c);
                let covered = store.candidates(c, &cfg).contains(&pick);
                let route = if kb == pick { 0u8 } else if covered { 1 } else { 2 };
                // FULL spectrum participation ratio: every scored circuit's positive DLA (~64 heads + ~384 neurons).
                let mut d: Vec<f32> = ex.all_dla.iter().copied().filter(|&x| x > 0.0).collect();
                if d.is_empty() { return None; }
                d.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let (sum, sumsq): (f32, f32) = (d.iter().sum(), d.iter().map(|x| x * x).sum());
                let pr = if sumsq > 0.0 { sum * sum / sumsq } else { 1.0 };
                let margin = ex.predicted_logit - ex.runner_up_logit;
                // Q1 normalization: true facet distance = (L_t − L_v) / ‖U_t − U_v‖.
                let (ut, uv) = (b2.weight_row(un, pick as usize), b2.weight_row(un, ex.runner_up as usize));
                let nrm = ut.iter().zip(&uv).map(|(a, b)| { let dd = a - b; dd * dd }).sum::<f32>().sqrt();
                let nmargin = if nrm > 0.0 { margin / nrm } else { f32::NAN };
                // Q4b: μ_t = single-circuit readout MULTIPLICITY = # of (shown, highest-DLA) circuits whose ISOLATED
                // argmax (its #1 promoted token) is the model's pick. μ_t≫1 = redundant; μ_t=0 = emergent (Grok's
                // "argmax of the sum that is the argmax of no summand"). Counted over the top-6 heads + top-6 neurons by
                // DLA (a lower bound on the full-spectrum μ_t — those are the circuits most relevant to t's logit).
                let mu_t = ex.head_circuits.iter().filter_map(|h| h.promotes.first().copied())
                    .chain(ex.mlp_features.iter().filter_map(|m| m.promotes.first().copied())).filter(|&a| a == pick).count();
                let top_hit = {
                    let th = ex.head_circuits.first();
                    let tn = ex.mlp_features.first();
                    match (th, tn) {
                        (Some(h), Some(n)) => if h.dla >= n.dla { h.promotes.first() } else { n.promotes.first() },
                        (Some(h), None) => h.promotes.first(),
                        (None, Some(n)) => n.promotes.first(),
                        (None, None) => None,
                    }.copied() == Some(pick)
                };
                Some(Rec { route, pr, captured: d.len(), margin, nmargin, top_hit, mu_t })
            }).collect();

            let pct = |g: &[&Rec], f: &dyn Fn(&Rec) -> bool| if g.is_empty() { f32::NAN } else { 100.0 * g.iter().filter(|x| f(x)).count() as f32 / g.len() as f32 };
            let meanf = |g: &[&Rec], f: &dyn Fn(&Rec) -> f32| if g.is_empty() { f32::NAN } else { g.iter().map(|x| f(x)).sum::<f32>() / g.len() as f32 };
            println!("\n=== (C/Q1/Q4) full-spectrum DLA + μ_t multiplicity ({} captured circuits, unembed {un}) ===", recs.first().map(|r| r.captured).unwrap_or(0));
            println!("{:<12}{:>6}{:>10}{:>13}{:>13}{:>12}{:>14}", "route", "n", "PR (eff#)", "margin/‖ΔU‖", "μ_t (mean)", "μ_t≥1", "emergent μ_t=0");
            for (lbl, r) in [("RETRIEVED", 0u8), ("SELECTED", 1), ("COMPOSED", 2)] {
                let g: Vec<&Rec> = recs.iter().filter(|x| x.route == r).collect();
                if g.is_empty() { println!("{lbl:<12}{:>6}", 0); continue; }
                println!("{lbl:<12}{:>6}{:>10.1}{:>13.2}{:>13.2}{:>11.0}%{:>13.0}%", g.len(),
                    meanf(&g, &|x| x.pr), meanf(&g, &|x| x.nmargin), meanf(&g, &|x| x.mu_t as f32), pct(&g, &|x| x.mu_t > 0), pct(&g, &|x| x.mu_t == 0));
            }
            println!("(Q4b) μ_t = # of top-12-by-DLA circuits whose ISOLATED argmax is the model's token. μ_t≫1 = redundantly");
            println!("      multiply-realized (covered); μ_t=0 = EMERGENT (argmax of the sum that is the argmax of no summand).");

            // (Q1 disambiguation) confidence vs structure: within matched normalized-margin bins, does KB-coverage still
            // predict single-circuit redundancy? If the COVERED−COMPOSED any-circ gap persists at matched margin, the
            // retrieval/composition split carries information BEYOND confidence (margin alone).
            let mut nms: Vec<f32> = recs.iter().map(|r| r.nmargin).filter(|x| x.is_finite()).collect();
            nms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f32| if nms.is_empty() { 0.0 } else { nms[(((nms.len() - 1) as f32) * p) as usize] };
            let (t1, t2) = (q(0.333), q(0.667));
            println!("\n=== (Q1 disambig) within matched normalized-margin bins — does coverage predict redundancy beyond confidence? ===");
            println!("{:<14}{:>14}{:>22}{:>22}", "margin bin", "", "COVERED (R+S)", "COMPOSED");
            println!("{:<14}{:>10}{:>12}{:>12}{:>12}", "", "mean m/‖ΔU‖", "n  any-circ%", "n", "any-circ%");
            for (lbl, lo, hi) in [("low ", f32::MIN, t1), ("mid ", t1, t2), ("high", t2, f32::MAX)] {
                let inbin = |r: &&Rec| r.nmargin.is_finite() && r.nmargin >= lo && r.nmargin < hi;
                let cov: Vec<&Rec> = recs.iter().filter(|r| inbin(r) && r.route != 2).collect();
                let cmp: Vec<&Rec> = recs.iter().filter(|r| inbin(r) && r.route == 2).collect();
                let mm = meanf(&recs.iter().filter(inbin).collect::<Vec<_>>(), &|x| x.nmargin);
                println!("{lbl:<14}{mm:>10.2} {:>9} {:>5.0}%  {:>9} {:>5.0}%", cov.len(), pct(&cov, &|x| x.mu_t > 0), cmp.len(), pct(&cmp, &|x| x.mu_t > 0));
            }
            println!("⇒ if COVERED any-circ% ≫ COMPOSED any-circ% WITHIN a margin bin, the retrieval/composition split is NOT just confidence.");

            // Grok's margin-multiplicity prediction (publish-blocking falsifier): for COVERED, m(x) should be POSITIVELY
            // correlated with μ_t(x) (deeper cells recruit more redundant alignments). ≤0 falsifies it.
            let corr = |g: &[&Rec]| -> f32 {
                let n = g.len() as f32;
                if n < 2.0 { return f32::NAN; }
                let (mx, my) = (g.iter().map(|r| r.nmargin).sum::<f32>() / n, g.iter().map(|r| r.mu_t as f32).sum::<f32>() / n);
                let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0, 0.0);
                for r in g { let (dx, dy) = (r.nmargin - mx, r.mu_t as f32 - my); sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
                if sxx > 0.0 && syy > 0.0 { sxy / (sxx.sqrt() * syy.sqrt()) } else { f32::NAN }
            };
            let fin: Vec<&Rec> = recs.iter().filter(|r| r.nmargin.is_finite()).collect();
            let cov: Vec<&Rec> = fin.iter().filter(|r| r.route != 2).copied().collect();
            let retr: Vec<&Rec> = fin.iter().filter(|r| r.route == 0).copied().collect();
            let sel: Vec<&Rec> = fin.iter().filter(|r| r.route == 1).copied().collect();
            println!("\n=== (Grok prediction) corr(normalized margin, μ_t) — predicted >0 for COVERED; ≤0 falsifies ===");
            println!("  COVERED (R+S): {:.3}    RETRIEVED: {:.3}    SELECTED: {:.3}    all: {:.3}", corr(&cov), corr(&retr), corr(&sel), corr(&fin));
            return;
        }

        // --probe-decompose (Density-Minimization — the per-token bucketing analysis): for each prediction, descend its
        // deciding source coalition to a locally-minimal irreducible ATOM (the executable minimal_decider realizing
        // `decomposes`), the firing COUNT non-increasing along the way (Density.total_firing_mono; the density RATIO is
        // NOT monotone, so it is never the objective). Reports σ(t) = the atom size (the measured support number), the
        // |S|→|A| reduction, the positive-DLA mass retained, and the atom's margin slack — bucketed by route. Tests PIC
        // O2 (σ(t) ∼ PR). Route A multi-competitor cone (irreducible ⟹ ≥2 competitors) → --decomp-k (default 4). Needs an
        // arch exposing the substrate (rope/Qwen via explain_decomp); --store adds the route split (optional).
        if has_flag(&args, "--probe-decompose") {
            use retrieval::CandCfg;
            let kk: usize = flag(&args, "--decomp-k").and_then(|s| s.parse().ok()).unwrap_or(4);
            let store = flag(&args, "--store").and_then(|p| Store::load(p).ok());
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            let cap = (end - ctx_window).min(n_eval); // explain is the expensive faithful forward
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            if positions.is_empty() {
                eprintln!("[fieldrun] --probe-decompose: no eval positions (need --ids with > ctx_window tokens)");
                return;
            }
            // probe the arch once: does it populate the descent substrate? (rope/Qwen via explain_decomp; others default).
            if lm.explain_decomp(positions[0], kk).and_then(|e| e.decomp).is_none() {
                eprintln!("[fieldrun] --probe-decompose: this arch does not expose the descent substrate (rope/Qwen only)");
                return;
            }
            // Route-B faithful confirmation: confirm each linear atom against the REAL ablated forward (predict_ablated).
            // Available iff the arch implements predict_ablated (rope does). Each token then costs 2 extra forwards.
            let can_confirm = !has_flag(&args, "--no-confirm") && lm.predict_ablated(positions[0], &[], &[]).is_some();
            eprintln!("[fieldrun] --probe-decompose: {} positions (ctx {ctx_window}), K={kk} competitors, confirm={can_confirm} — faithful explain forwards…", positions.len());
            struct Rec { route: u8, n: usize, atom: usize, pr: f32, retained: f32, slack: f32, flip_atom: bool, flip_ctrl: bool }
            let recs: Vec<Rec> = positions.par_iter().filter_map(|c| {
                let ex = lm.explain_decomp(c, kk)?;
                let sub = ex.decomp.as_ref()?;
                let r = explain::decompose_descent(sub);
                let pick = ex.model_predicts;
                let route = match &store {
                    Some(s) => { let (kb, _) = s.predict(c); if kb == pick { 0u8 } else if s.candidates(c, &cfg).contains(&pick) { 1 } else { 2 } }
                    None => 3u8, // no store ⇒ no route split, everything in one "ALL" bucket
                };
                // full-spectrum participation ratio (mirrors --probe-dla) for the σ(t) ∼ PR comparison.
                let mut d: Vec<f32> = ex.all_dla.iter().copied().filter(|&x| x > 0.0).collect();
                d.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let (sum, sumsq): (f32, f32) = (d.iter().sum(), d.iter().map(|x| x * x).sum());
                let pr = if sumsq > 0.0 { sum * sum / sumsq } else { 1.0 };
                // Confirmation (necessity, the clean non-destructive test): ablate ONLY the atom A (|A| ≪ |S| circuits)
                // in the real forward → does the prediction flip? A flip ⇒ the irreducible core is causally load-bearing
                // (the §5c ablation methodology applied to the descent's atom). Control: ablate the |A| highest-DLA scored
                // sources — if the atom flips MORE than naive top-|A|-by-DLA, the cone descent selected a better core.
                // (Sufficiency — "keep only A" — is NOT testable this way: zeroing the other ~441 scored circuits is so
                // destructive that nothing survives, linear DLA ≠ causal; necessity is the faithful confirmation.)
                let (flip_atom, flip_ctrl) = if can_confirm {
                    let to_pairs = |idxs: &[usize]| {
                        let (mut h, mut n) = (Vec::new(), Vec::new());
                        for &i in idxs { let s = &sub.sources[i]; if s.kind == 0 { h.push((s.layer, s.idx)); } else { n.push((s.layer, s.idx)); } }
                        (h, n)
                    };
                    let (ah, an) = to_pairs(&r.atom);
                    let mut topk: Vec<usize> = (0..sub.sources.len()).collect();
                    topk.sort_by(|&a, &b| sub.sources[b].dla.total_cmp(&sub.sources[a].dla));
                    topk.truncate(r.atom.len());
                    let (ch, cn) = to_pairs(&topk);
                    (lm.predict_ablated(c, &ah, &an) != Some(pick), lm.predict_ablated(c, &ch, &cn) != Some(pick))
                } else {
                    (false, false)
                };
                Some(Rec { route, n: r.n_sources, atom: r.atom_size(), pr, retained: r.dla_retained, slack: r.min_slack, flip_atom, flip_ctrl })
            }).collect();
            if recs.is_empty() { eprintln!("[fieldrun] --probe-decompose: no positions produced a substrate"); return; }
            let meanf = |g: &[&Rec], f: &dyn Fn(&Rec) -> f32| if g.is_empty() { f32::NAN } else { g.iter().map(|x| f(x)).sum::<f32>() / g.len() as f32 };
            println!("\n=== Density-Minimization descent: per-token irreducible ATOM (σ(t)), K={kk} competitors, {} positions ===", recs.len());
            println!("  the atom is the locally-minimal deciding coalition (minimal_decider; a SOUND poly UNDER-approximation");
            println!("  of the true irreducible core). σ(t)=|atom|; reduction = 1 − |A|/|S|; retained = positive-DLA mass kept.");
            let confirm_hdr = if can_confirm { format!("{:>11}{:>10}", "necessary", "ctrl flip") } else { String::new() };
            println!("{:<12}{:>7}{:>11}{:>10}{:>11}{:>11}{:>11}{:>9}{}", "route", "n", "|S| src", "σ(t)=|A|", "reduction", "retained", "PR eff#", "slack", confirm_hdr);
            let groups: Vec<(&str, u8)> = if store.is_some() { vec![("RETRIEVED", 0), ("SELECTED", 1), ("COMPOSED", 2)] } else { vec![("ALL", 3)] };
            for (lbl, rt) in groups {
                let g: Vec<&Rec> = recs.iter().filter(|x| x.route == rt).collect();
                if g.is_empty() { println!("{lbl:<12}{:>7}", 0); continue; }
                let confirm_row = if can_confirm {
                    format!("{:>10.0}%{:>9.0}%", 100.0 * meanf(&g, &|x| if x.flip_atom { 1.0 } else { 0.0 }), 100.0 * meanf(&g, &|x| if x.flip_ctrl { 1.0 } else { 0.0 }))
                } else { String::new() };
                println!("{lbl:<12}{:>7}{:>11.1}{:>10.1}{:>10.0}%{:>10.0}%{:>11.1}{:>9.2}{}", g.len(),
                    meanf(&g, &|x| x.n as f32), meanf(&g, &|x| x.atom as f32),
                    100.0 * meanf(&g, &|x| 1.0 - x.atom as f32 / x.n.max(1) as f32),
                    100.0 * meanf(&g, &|x| x.retained), meanf(&g, &|x| x.pr), meanf(&g, &|x| x.slack), confirm_row);
            }
            // (PIC O2) is the support number σ(t) the participation ratio? Pearson corr(|atom|, PR) over all positions.
            let corr = {
                let n = recs.len() as f32;
                let (mx, my) = (recs.iter().map(|r| r.atom as f32).sum::<f32>() / n, recs.iter().map(|r| r.pr).sum::<f32>() / n);
                let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0, 0.0);
                for r in &recs { let (dx, dy) = (r.atom as f32 - mx, r.pr - my); sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
                if sxx > 0.0 && syy > 0.0 { sxy / (sxx.sqrt() * syy.sqrt()) } else { f32::NAN }
            };
            if can_confirm {
                let overall = |f: &dyn Fn(&Rec) -> bool| 100.0 * recs.iter().filter(|r| f(r)).count() as f32 / recs.len() as f32;
                println!("\n(confirm, Route B) ablate ONLY the atom A in the REAL forward → prediction flips: necessary = {:.0}%   (top-|A|-DLA control = {:.0}%).", overall(&|r| r.flip_atom), overall(&|r| r.flip_ctrl));
                println!("  necessary = the irreducible core is causally load-bearing (§5c methodology on the atom); necessary − ctrl = the cone descent's lift over naive top-|A|-by-DLA.");
            }
            println!("\n(PIC O2) σ(t) ∼ PR?  corr(|atom|, PR) = {corr:.3}   (σ(t) = the descent's measured support number)");
            println!("(theory) irreducible ⟹ ≥2 competitors (single_competitor_reducible); the atom never fires more neurons than S (total_firing_mono).");
            return;
        }

        // --query-decompose (per-QUERY aggregation, the ladder's middle rung): treat a contiguous run of positions as ONE
        // query and aggregate the per-token irreducible atoms into the query's circuit working-set W = ⋃_t A_t — entirely
        // IN-MEMORY from the descent results, with NO `export --logic` → `.dl` → `stitch` disk round-trip. This is the
        // Hub.thy decomposition of a query: the hub = circuits shared across many tokens (the disentangling core / a
        // candidate expert), private = per-token; the distinct budget obeys |W| ≥ Σ|A_t| / d (the d-bounded budget,
        // d = max reuse). Σ|A_t| is the per-token firing-count floor summed over the query. Rope/Qwen only (the substrate).
        if has_flag(&args, "--query-decompose") {
            let kk: usize = flag(&args, "--decomp-k").and_then(|s| s.parse().ok()).unwrap_or(4);
            let hub_frac: f32 = flag(&args, "--hub-frac").and_then(|s| s.parse().ok()).unwrap_or(0.5);
            let cap = (end - ctx_window).min(n_eval);
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            if positions.is_empty() {
                eprintln!("[fieldrun] --query-decompose: no eval positions (need --ids with > ctx_window tokens)");
                return;
            }
            if lm.explain_decomp(positions[0], kk).and_then(|e| e.decomp).is_none() {
                eprintln!("[fieldrun] --query-decompose: this arch does not expose the descent substrate (rope/Qwen only)");
                return;
            }
            eprintln!("[fieldrun] --query-decompose: aggregating {} positions as ONE query (ctx {ctx_window}), K={kk} — in-memory, no .dl export/stitch…", positions.len());
            // per-position atoms as circuit identities (kind, layer, idx); computed in parallel, then aggregated in-memory.
            let atoms: Vec<Vec<(u8, usize, usize)>> = positions.par_iter().filter_map(|c| {
                let ex = lm.explain_decomp(c, kk)?;
                let sub = ex.decomp.as_ref()?;
                let r = explain::decompose_descent(sub);
                Some(r.atom.iter().map(|&i| { let s = &sub.sources[i]; (s.kind, s.layer, s.idx) }).collect())
            }).collect();
            let q = atoms.len();
            let total: usize = atoms.iter().map(|a| a.len()).sum(); // Σ|A_t| — total firings (per-token floor, summed)
            let mut mult: HashMap<(u8, usize, usize), usize> = HashMap::new(); // circuit → # of tokens whose atom uses it
            for a in &atoms { for &id in a { *mult.entry(id).or_default() += 1; } }
            let distinct = mult.len(); // |W| — the query's distinct-circuit budget
            let dmax = mult.values().copied().max().unwrap_or(0); // d — the most-reused circuit's multiplicity
            let hub_thresh = ((hub_frac * q as f32).ceil() as usize).max(2); // "shared by ≥ hub_frac of the query's tokens"
            let mut hub: Vec<((u8, usize, usize), usize)> = mult.iter().filter(|(_, &m)| m >= hub_thresh).map(|(&id, &m)| (id, m)).collect();
            hub.sort_by(|a, b| b.1.cmp(&a.1));
            let hub_firings: usize = hub.iter().map(|(_, m)| *m).sum();
            let private = mult.values().filter(|&&m| m == 1).count();
            let reuse = if total > 0 { 1.0 - distinct as f32 / total as f32 } else { 0.0 };
            let avg_atom = if q > 0 { total as f32 / q as f32 } else { 0.0 };
            println!("\n=== Per-query density-minimization working set: {q} tokens aggregated as ONE query (K={kk}) ===");
            println!("  W = ⋃_t A_t (Hub.thy: hub = shared core, private = per-token) — computed in-memory, no .dl export/stitch.");
            println!("  tokens (Q)                {q}");
            println!("  Σ|A_t| total firings      {total}    (avg atom {avg_atom:.2}/token — the per-token floor summed)");
            println!("  |W| distinct circuits     {distinct}    (the query's circuit budget)");
            println!("  reuse 1 − |W|/Σ           {:.0}%    (circuit sharing across the query's tokens)", 100.0 * reuse);
            println!("  hub (≥ {hub_thresh} tokens)         {} circuits   carrying {hub_firings}/{total} firings ({:.0}%)", hub.len(), if total > 0 { 100.0 * hub_firings as f32 / total as f32 } else { 0.0 });
            println!("  private (1 token)         {private} circuits");
            println!("  max multiplicity d        {dmax}    (a circuit reused by up to d tokens; distinct |W| ≥ Σ/d = {})", if dmax > 0 { total / dmax } else { 0 });
            if !hub.is_empty() {
                println!("  top shared circuits (the query's reusable core — a candidate expert for the corpus phase):");
                let kind_name = |k: u8| if k == 0 { "head" } else { "neuron" };
                for ((k, l, i), m) in hub.iter().take(10) {
                    println!("    {:<6} L{l:<2} #{i:<6}  in {m}/{q} atoms", kind_name(*k));
                }
            }
            return;
        }

        // --corpus-decompose (per-CORPUS clustering, the ladder's endgame): cluster the per-token irreducible atoms across
        // the whole corpus into E **experts** — partition the corpus working set C into hub-anchored buckets (anchor = a
        // corpus-frequent hub circuit; each other circuit joins the anchor-expert it co-fires with most), then ask the MoE
        // --verify-cache: the byte-identity gate for the KV-cached explain stream. Compares the cached-stream atom at
        // each position against the uncached explain on the same growing prefix; 0 mismatches ⇒ caching is faithful.
        if has_flag(&args, "--verify-cache") {
            let kk: usize = flag(&args, "--decomp-k").and_then(|s| s.parse().ok()).unwrap_or(4);
            let cap = (end - ctx_window).min(n_eval).min(60);
            if cap == 0 { eprintln!("[fieldrun] --verify-cache: need --ids with > ctx_window tokens"); return; }
            let atom_of = |ex: &explain::Explanation| -> Option<Vec<bucketing::Circuit>> {
                let sub = ex.decomp.as_ref()?;
                let r = explain::decompose_descent(sub);
                Some(r.atom.iter().map(|&i| { let s = &sub.sources[i]; (s.kind, s.layer, s.idx) }).collect())
            };
            let mut cached: std::collections::BTreeMap<usize, (i64, Vec<bucketing::Circuit>)> = Default::default();
            lm.explain_stream(&ids[..ctx_window + cap], kk, ctx_window, &mut |pos, ex| {
                if let Some(a) = atom_of(&ex) { cached.insert(pos, (ex.model_predicts, a)); }
            });
            if cached.is_empty() { eprintln!("[fieldrun] --verify-cache: arch exposes no substrate (rope/Qwen only)"); return; }
            let (mut checked, mut pred_mm, mut atom_mm) = (0usize, 0usize, 0usize);
            for (&pos, (cpred, catom)) in &cached {
                if let Some(ex) = lm.explain_decomp(&ids[..pos], kk) {
                    checked += 1;
                    if ex.model_predicts != *cpred { pred_mm += 1; }
                    if atom_of(&ex).as_ref() != Some(catom) { atom_mm += 1; }
                }
            }
            println!("--verify-cache: {checked} positions · {pred_mm} prediction mismatch · {atom_mm} atom mismatch (cached vs uncached) → {}",
                if pred_mm == 0 && atom_mm == 0 { "PASS — KV-cached explain is byte-identical" } else { "FAIL" });
            return;
        }

        // question: does a token's deciding atom fit inside ONE expert (top-1 routable)? and how many circuits does routing
        // actually compute vs the monolithic working set? In-memory from the descent (no `.dl` export/stitch). Rope/Qwen.
        if has_flag(&args, "--corpus-decompose") {
            let kk: usize = flag(&args, "--decomp-k").and_then(|s| s.parse().ok()).unwrap_or(4);
            let e_req: usize = flag(&args, "--experts").and_then(|s| s.parse().ok()).unwrap_or(8);
            let cap = (end - ctx_window).min(n_eval);
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            if positions.is_empty() {
                eprintln!("[fieldrun] --corpus-decompose: no eval positions (need --ids with > ctx_window tokens)");
                return;
            }
            if lm.explain_decomp(positions[0], kk).and_then(|e| e.decomp).is_none() {
                eprintln!("[fieldrun] --corpus-decompose: this arch does not expose the descent substrate (rope/Qwen only)");
                return;
            }
            let report_every: usize = flag(&args, "--report-every").and_then(|s| s.parse().ok()).unwrap_or(0);
            eprintln!("[fieldrun] --corpus-decompose: {} tokens → up to {e_req} experts (K={kk}), chunked stream{} — in-memory, no .dl…",
                positions.len(), if report_every > 0 { format!(", report every {report_every}") } else { String::new() });
            // Stream the corpus in chunks so the per-position forward working set stays bounded; atoms accumulate into the
            // shared CorpusBuckets (tiny, ~72 bytes/token). --report-every N prints the running clustering at runtime.
            let want_dl = flag(&args, "--experts-dl").is_some(); // also collect the per-token signature + decode for the DL emit
            let want_interpret = has_flag(&args, "--interpret"); // collect decodes to show what routes to each expert
            let want_meta = want_dl || want_interpret;
            let dl_order: usize = flag(&args, "--dl-sig").and_then(|s| s.parse().ok()).unwrap_or(1).max(1); // sig = last N ctx tokens
            let lmr = lm.as_ref();
            let mut buckets = bucketing::CorpusBuckets::new();
            let (mut sigs, mut preds): (Vec<i64>, Vec<i64>) = (Vec::new(), Vec::new());
            let mut next_report = report_every;
            for chunk in positions.chunks(512) {
                // sig = the last dl_order context tokens (base-encoded; the lookup key); pred = the model's decode.
                let got: Vec<(Vec<bucketing::Circuit>, i64, i64)> = chunk.par_iter().filter_map(|c| {
                    let (a, p) = bucketing::atom_and_pred_at(lmr, c, kk)?;
                    let sg = if dl_order <= 1 { *c.last().unwrap_or(&-1) } else { c.iter().rev().take(dl_order).fold(0i64, |s, &t| s.wrapping_mul(1 << 20).wrapping_add(t + 1)) };
                    Some((a, sg, p))
                }).collect();
                for (a, s, p) in got {
                    if want_dl { sigs.push(s); }
                    if want_meta { preds.push(p); }
                    buckets.ingest(a);
                }
                if report_every > 0 && buckets.n_tokens() >= next_report {
                    println!("\n=== [progress] {} tokens — up to {e_req}-expert clustering so far (K={kk}) ===", buckets.n_tokens());
                    print!("{}", buckets.render(e_req));
                    next_report = buckets.n_tokens() + report_every;
                }
            }
            println!("\n=== Per-corpus expert clustering (final): up to {e_req} hub-anchored experts over {} tokens (K={kk}) ===", buckets.n_tokens());
            println!("  the corpus working set C is partitioned into E experts (anchor = a corpus-hub circuit; each other");
            println!("  circuit joins the anchor it co-fires with most). A token routes to the expert(s) its atom touches.");
            print!("{}", buckets.render(e_req));
            println!("  (proxy caveat: assumes an oracle router; a real saving needs a learned router + experts mapped to weight chunks.)");
            if has_flag(&args, "--residency") {
                // runtime residency: which experts are hot enough to stay resident vs the paged long tail (load distribution).
                let cov: f32 = flag(&args, "--resident-cov").and_then(|s| s.parse().ok()).unwrap_or(0.9);
                println!("\n=== Runtime residency profile (experts by token-load; hot set resident, tail paged on demand) ===");
                print!("{}", buckets.residency(e_req, cov));
            }
            // --experts-out <path>: emit the CONCRETE partition (each expert's anchor + full circuit list + token routing)
            // as JSON — the build artifact a router / weight-chunk pager consumes, not just the summary above.
            if let Some(path) = flag(&args, "--experts-out") {
                match serde_json::to_string_pretty(&buckets.partition(e_req)) {
                    Ok(j) => match std::fs::write(path, j) {
                        Ok(()) => eprintln!("[fieldrun] --corpus-decompose: wrote expert partition ({e_req} experts) → {path}"),
                        Err(err) => eprintln!("[fieldrun] --corpus-decompose: cannot write {path}: {err}"),
                    },
                    Err(err) => eprintln!("[fieldrun] --corpus-decompose: serialize failed: {err}"),
                }
            }
            // --experts-dl <path>: emit the partition as a Soufflé Datalog LOOKUP/SELECTION model (routing + decision as
            // lookup over a context signature, + per-expert pick-entropy marking lookup-exact vs computed experts).
            if let Some(path) = flag(&args, "--experts-dl") {
                let tf: f32 = flag(&args, "--dl-test-frac").and_then(|s| s.parse().ok()).unwrap_or(0.2); // held-out tail for generalization
                match std::fs::write(path, buckets.emit_datalog(e_req, &sigs, &preds, tf)) {
                    Ok(()) => eprintln!("[fieldrun] --corpus-decompose: wrote Datalog lookup/selection model → {path}  (run: souffle {path} -D-)"),
                    Err(err) => eprintln!("[fieldrun] --corpus-decompose: cannot write {path}: {err}"),
                }
            }
            // --interpret: what KIND of tokens route to each expert (its specialty) — decode the tokens routed there.
            if want_interpret {
                #[cfg(feature = "api")]
                let dec: Box<dyn Fn(i64) -> String> = match api::TextGen::load(&stem, eos.clone()) {
                    Some(tg) => Box::new(move |id| tg.token_label(id)),
                    None => load_decoder(flag(&args, "--vocab")),
                };
                #[cfg(not(feature = "api"))]
                let dec = load_decoder(flag(&args, "--vocab"));
                let (e_act, routes) = buckets.routes(e_req);
                let mut by_e: Vec<HashMap<i64, usize>> = vec![HashMap::new(); e_act + 1];
                for (i, &r) in routes.iter().enumerate() {
                    if let Some(&p) = preds.get(i) { *by_e[r].entry(p).or_default() += 1; }
                }
                println!("\n=== Per-expert interpretability: the decoded tokens routed to each expert (its 'specialty') ===");
                for (e, m) in by_e.iter().enumerate() {
                    if m.is_empty() { continue; }
                    let mut top: Vec<(i64, usize)> = m.iter().map(|(&t, &c)| (t, c)).collect();
                    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    let total: usize = m.values().sum();
                    let label = if e == e_act { "residual".to_string() } else { format!("e{e}") };
                    let toks: Vec<String> = top.iter().take(10).map(|(t, c)| format!("{:?}·{c}", dec(*t).replace('\n', "⏎"))).collect();
                    println!("  {label:<9} {total:>4} tok →  {}", toks.join("  "));
                }
            }
            // --experts-dl-contrib: emit the COMPOSITION decode (per-expert Σ contrib + catchall rest), runnable in
            // `fieldrun eval`. Faithful by construction; the catchall margin-share is the compactness / forge-tax meter.
            if let Some(path) = flag(&args, "--experts-dl-contrib") {
                let steps: usize = flag(&args, "--dl-contrib-steps").and_then(|s| s.parse().ok()).unwrap_or(12);
                let (e_act, emap) = buckets.expert_map(e_req);
                #[cfg(feature = "api")]
                let dec: Box<dyn Fn(i64) -> String> = match api::TextGen::load(&stem, eos.clone()) {
                    Some(tg) => Box::new(move |id| tg.token_label(id)),
                    None => load_decoder(flag(&args, "--vocab")),
                };
                #[cfg(not(feature = "api"))]
                let dec = load_decoder(flag(&args, "--vocab"));
                let take = positions.len().min(steps);
                let prog = emit_contrib_dl(lm.as_ref(), &positions[..take], kk, e_act, &emap, dec.as_ref());
                match std::fs::write(path, prog) {
                    Ok(()) => eprintln!("[fieldrun] --corpus-decompose: wrote contrib-over-expert model ({take} steps) → {path}  (run: fieldrun eval {path} --semiring max)"),
                    Err(err) => eprintln!("[fieldrun] --corpus-decompose: cannot write {path}: {err}"),
                }
            }
            // --recurse-depth D: recursively sub-bucket the residual into finer DOMAIN experts (the path toward the long
            // tail). The flat clustering dumps everything not co-firing with a top-E hub into one residual; recursion
            // re-clusters that residual, and its residual, D levels deep — resolving the collapsed tail.
            if let Some(d) = flag(&args, "--recurse-depth").and_then(|s| s.parse::<usize>().ok()).filter(|&d| d > 1) {
                let min_c: usize = flag(&args, "--recurse-min").and_then(|s| s.parse().ok()).unwrap_or(8);
                let (leaves, route) = buckets.recursive(e_req, d, min_c);
                let n: usize = leaves.iter().map(|l| l.tokens).sum();
                println!("\n=== Recursive sub-bucketing (depth {d}, {} leaf experts) — the residual resolved into domain experts ===", leaves.len());
                println!("  {:<14}{:>9}{:>9}{:>8}", "leaf", "circuits", "tokens", "share");
                for l in leaves.iter().filter(|l| l.tokens > 0) {
                    println!("  {:<14}{:>9}{:>9}{:>7.0}%", l.label, l.n_circuits, l.tokens, 100.0 * l.tokens as f32 / n.max(1) as f32);
                }
                if want_interpret {
                    #[cfg(feature = "api")]
                    let dec: Box<dyn Fn(i64) -> String> = match api::TextGen::load(&stem, eos.clone()) {
                        Some(tg) => Box::new(move |id| tg.token_label(id)),
                        None => load_decoder(flag(&args, "--vocab")),
                    };
                    #[cfg(not(feature = "api"))]
                    let dec = load_decoder(flag(&args, "--vocab"));
                    let mut by_l: Vec<HashMap<i64, usize>> = vec![HashMap::new(); leaves.len()];
                    for (i, &r) in route.iter().enumerate() {
                        if let Some(&p) = preds.get(i) { *by_l[r].entry(p).or_default() += 1; }
                    }
                    println!("\n  per-leaf specialty (top decoded tokens):");
                    for (li, l) in leaves.iter().enumerate() {
                        if by_l[li].is_empty() { continue; }
                        let mut top: Vec<(i64, usize)> = by_l[li].iter().map(|(&t, &c)| (t, c)).collect();
                        top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                        let toks: Vec<String> = top.iter().take(8).map(|(t, c)| format!("{:?}·{c}", dec(*t).replace('\n', "⏎"))).collect();
                        println!("    {:<14} {}", l.label, toks.join("  "));
                    }
                }
            }
            return;
        }

        // --probe-facet (tighten Q1): the token cells in r-space ARE the Laguerre power diagram of {U_v} (weights
        // ‖U_v‖²+2b_v). Compute the EXACT nearest facet argmin_{v≠t}(L_t−L_v)/‖U_t−U_v‖ (not the logit-runner-up proxy)
        // and (a) how often the nearest facet == the logit runner-up, (b) the killer check: for COMPOSED, is the token
        // across the nearest facet the KB's own top-1? If yes, composition = r(x) having crossed the facet out of the
        // KB's cell. Needs an arch exposing final_residual (rope).
        if has_flag(&args, "--probe-facet") {
            use retrieval::CandCfg;
            let store = match flag(&args, "--store").and_then(|p| Store::load(p).ok()) {
                Some(s) => s,
                None => { eprintln!("[fieldrun] --probe-facet needs --store"); return; }
            };
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            let b2 = Bundle::load(&stem).expect("reload bundle");
            let un = if b2.has("lm_head") { "lm_head" } else { "embed" };
            let (vocab, _d) = b2.dims(un);
            if lm.final_residual(&ids[..ctx_window.min(ids.len())]).is_none() {
                eprintln!("[fieldrun] --probe-facet: arch {arch} doesn't expose final_residual (rope only)");
                return;
            }
            eprintln!("[fieldrun] --probe-facet: precomputing ‖U_v‖² for {vocab} unembed rows…");
            let unorm: Vec<f32> = (0..vocab).into_par_iter().map(|v| b2.weight_row(un, v).iter().map(|x| x * x).sum()).collect();
            let cap = (end - ctx_window).min(n_eval).min(300);
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            eprintln!("[fieldrun] --probe-facet: {} positions — full logits + nearest-facet over {vocab} tokens…", positions.len());
            #[cfg(feature = "api")]
            let dec: Box<dyn Fn(i64) -> String> = match api::TextGen::load(&stem, eos.clone()) {
                Some(tg) => Box::new(move |id| tg.token_label(id)),
                None => load_decoder(flag(&args, "--vocab")),
            };
            #[cfg(not(feature = "api"))]
            let dec = load_decoder(flag(&args, "--vocab"));
            struct F { route: u8, dx: f32, vstar_is_ru: bool, vstar_is_kb: bool, pick: i64, kb: i64 }
            let recs: Vec<F> = positions.par_iter().filter_map(|c| {
                let r = lm.final_residual(c)?;
                let l = b2.rowdot_f32(un, &r); // full logits L_v = ⟨U_v, r⟩
                let t = l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?.0;
                let ut = b2.weight_row(un, t);
                let g = b2.rowdot_f32(un, &ut); // ⟨U_v, U_t⟩ for all v → ‖U_t−U_v‖² = ‖U_t‖²+‖U_v‖²−2⟨U_v,U_t⟩
                let (mut best_d, mut vstar) = (f32::INFINITY, t);
                let (mut best_l, mut ru) = (f32::NEG_INFINITY, t);
                for v in 0..vocab {
                    if v == t { continue; }
                    if l[v] > best_l { best_l = l[v]; ru = v; }
                    let dvv2 = unorm[t] + unorm[v] - 2.0 * g[v];
                    if dvv2 > 1e-4 {
                        let dv = (l[t] - l[v]) / dvv2.sqrt(); // exact Euclidean distance to the t–v bisector facet
                        if dv < best_d { best_d = dv; vstar = v; }
                    }
                }
                let kb = store.predict(c).0 as usize;
                let covered = store.candidates(c, &cfg).contains(&(t as i64));
                let route = if kb == t { 0u8 } else if covered { 1 } else { 2 };
                Some(F { route, dx: best_d, vstar_is_ru: vstar == ru, vstar_is_kb: vstar == kb, pick: t as i64, kb: kb as i64 })
            }).collect();
            let pct = |g: &[&F], f: &dyn Fn(&F) -> bool| if g.is_empty() { f32::NAN } else { 100.0 * g.iter().filter(|x| f(x)).count() as f32 / g.len() as f32 };
            let meanf = |g: &[&F], f: &dyn Fn(&F) -> f32| if g.is_empty() { f32::NAN } else { g.iter().map(|x| f(x)).sum::<f32>() / g.len() as f32 };
            println!("\n=== (Q1 tight) exact power-diagram nearest facet ({} positions, {vocab} tokens) ===", recs.len());
            println!("{:<12}{:>6}{:>18}{:>22}{:>24}", "route", "n", "nearest-facet dist", "v*==logit-runner-up", "v*==KB-top1 (killer)");
            for (lbl, r) in [("RETRIEVED", 0u8), ("SELECTED", 1), ("COMPOSED", 2)] {
                let g: Vec<&F> = recs.iter().filter(|x| x.route == r).collect();
                if g.is_empty() { println!("{lbl:<12}{:>6}", 0); continue; }
                println!("{lbl:<12}{:>6}{:>18.3}{:>21.0}%{:>23.0}%", g.len(), meanf(&g, &|x| x.dx), pct(&g, &|x| x.vstar_is_ru), pct(&g, &|x| x.vstar_is_kb));
            }
            let all: Vec<&F> = recs.iter().collect();
            println!("(nearest facet == logit runner-up overall: {:.0}%  ⇒ how often the runner-up proxy IS the nearest facet)", pct(&all, &|x| x.vstar_is_ru));
            // characterize the near-miss-of-retrieval subclass: tokens where the nearest facet IS the KB's prediction
            // (model and KB one facet apart). What ARE they — function words / near-synonyms / high-freq glue?
            for (lbl, r) in [("SELECTED", 1u8), ("COMPOSED", 2u8)] {
                let nm: Vec<&F> = recs.iter().filter(|x| x.route == r && x.vstar_is_kb).collect();
                if nm.is_empty() { continue; }
                println!("\n  {lbl} near-miss-of-retrieval ({} tokens) — model's pick  ⟂(one facet)⟂  KB's prediction:", nm.len());
                for f in nm.iter().take(30) {
                    println!("    {}   ⟂   KB {}", dec(f.pick), dec(f.kb));
                }
            }
            return;
        }

        // export --logic (LOGIC_EXPORT LO3): emit a runnable semiring-Datalog program SPECIALIZED to ONE next-token
        // decision — the retrievable fragment as readable clauses/facts (Tier A), the composition as per-block weighted
        // contrib facts (Tier B, the forge tax), and the decode as a (max,+) argmax aggregate. Tokens are referenced by
        // id (unique, runnable); text is in comments. Σ contrib == logit (LE-T5); a round-trip self-check confirms the
        // emitted program's decode == the model. Rope-only (needs residual_decomp). For a multi-step decode TRACE
        // (one .dl per generated token), see `--export-logic <prefix>` below.
        // export --logic-whole (LOGIC_EXPORT LO3a): the CONTEXT-FREE whole-model emit. Unlike `export --logic`
        // (one decision as partial-evaluation facts) this emits the forward pass ITSELF as Datalog rules over
        // weight facts, taking `token(pos,id)` as the only input — one program that computes the next token for
        // ANY context, runnable in Soufflé on inputs the exporter never saw. Rope family; small bundles (the
        // embed/unembed fact count vocab×d is the dense-Gram wall, LE-T4 — correct but not compact at scale).
        if args.iter().any(|a| a == "export") && has_flag(&args, "--logic-whole") {
            let maxpos: usize = flag(&args, "--maxpos").and_then(|s| s.parse().ok()).unwrap_or(64);
            let b = match Bundle::load(&stem) {
                Ok(b) => b,
                Err(e) => { eprintln!("[fieldrun] export --logic-whole: couldn't reload bundle: {e}"); return; }
            };
            let (vc, dd) = (b.config.get(6).copied().unwrap_or(0) as usize, b.config.get(4).copied().unwrap_or(0) as usize);
            let est = vc.saturating_mul(dd);
            if est > 4_000_000 && !has_flag(&args, "--force") {
                eprintln!("[fieldrun] export --logic-whole: vocab×d = {vc}×{dd} ≈ {est} embed facts (×2 if untied) — that is the\n\
                           dense-Gram / high-treewidth wall (LOGIC_EXPORT LE-T4): the program is correct but not COMPACT at this\n\
                           scale. Demonstrate on a small rope bundle, or re-run with --force to emit anyway.");
                return;
            }
            match logic_whole::emit_whole(&b, maxpos) {
                Ok(prog) => match flag(&args, "--out") {
                    Some(p) => {
                        if std::fs::write(p, &prog).is_ok() {
                            eprintln!("[fieldrun] export --logic-whole → {p}  (context-free; maxpos {maxpos}) — run:\n  \
                                       printf '0\\t<id0>\\n1\\t<id1>\\n…' > ctx/token.facts && souffle {p} -F ctx -D -");
                        } else {
                            eprintln!("[fieldrun] export --logic-whole: could not write {p}");
                        }
                    }
                    None => print!("{prog}"),
                },
                Err(e) => eprintln!("[fieldrun] {e}"),
            }
            return;
        }

        let export_logic = args.iter().any(|a| a == "export") && has_flag(&args, "--logic");
        if export_logic {
            use retrieval::CandCfg;
            let store = flag(&args, "--store").and_then(|p| Store::load(p).ok());
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            let cap_c: usize = flag(&args, "--candidates").and_then(|s| s.parse().ok()).unwrap_or(48);
            if ids.len() < 2 {
                eprintln!("[fieldrun] export --logic needs --ids with a context (≥2 tokens)");
                return;
            }
            let c: &[i64] = if ids.len() > ctx_window { &ids[..ctx_window] } else { &ids[..ids.len() - 1] };
            let Some(ex) = lm.explain(c) else {
                eprintln!("[fieldrun] export --logic: arch {arch} has no explain");
                return;
            };
            let _ = ex; // explain re-run inside logic::build; keep the early arch-support check above
            let Some(prov) = logic::build(lm.as_ref(), c, store.as_ref(), &cfg, cap_c) else {
                eprintln!("[fieldrun] export --logic: arch {arch} has no residual_decomp (rope only)");
                return;
            };
            // token text for the .dl comments comes from the tokenizer (api feature); without it, fall back to ids.
            #[cfg(feature = "api")]
            let o = {
                let tg = api::TextGen::load(&stem, eos.clone());
                let lbl = |id: i64| -> String { tg.as_ref().map(|g| g.token_label(id)).unwrap_or_else(|| format!("[{id}]")) };
                logic::emit_dl(&prov, c, &lbl)
            };
            #[cfg(not(feature = "api"))]
            let o = logic::emit_dl(&prov, c, &|id: i64| format!("[{id}]"));
            let faithful = o.contains("✓ FAITHFUL");
            match flag(&args, "--out") {
                Some(p) => {
                    if std::fs::write(p, &o).is_ok() {
                        eprintln!("[fieldrun] export --logic → {p}  ({} candidates, {} blocks, decode {})",
                            prov.candidates.len(), prov.blocks.len(), if faithful { "FAITHFUL ✓" } else { "MISMATCH ✗" });
                    } else {
                        eprintln!("[fieldrun] export --logic: could not write {p}");
                    }
                }
                None => print!("{o}"),
            }
            return;
        }

        // --export-logic <prefix>: emit a per-step semiring-Datalog TRACE of a greedy decode — one runnable .dl per
        // generated token (prefix.000.dl, prefix.001.dl, …), each an INDEPENDENT program (`fieldrun eval` / Soufflé).
        // Deliberately NOT one merged file: concatenating complete programs redeclares relations (Soufflé errors) and
        // makes `eval` sum contribs across different tokens (silently wrong) — so each decode is its own file. The
        // context advances by the model's own pick each step, so the .dl set is a faithful decode trajectory. Steps via
        // --steps N (default 8). Rope-only (needs residual_decomp). One full explain+residual_decomp forward per step.
        if let Some(prefix) = flag(&args, "--export-logic") {
            use retrieval::CandCfg;
            let store = flag(&args, "--store").and_then(|p| Store::load(p).ok());
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            let cap_c: usize = flag(&args, "--candidates").and_then(|s| s.parse().ok()).unwrap_or(48);
            let steps: usize = flag(&args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(8);
            if ids.len() < 2 {
                eprintln!("[fieldrun] --export-logic needs --ids with a context (≥2 tokens)");
                return;
            }
            if prefix.ends_with(".dl") {
                eprintln!("[fieldrun] --export-logic writes ONE file PER decode step (prefix.000.dl, …) — it wants a PREFIX, \
                           not a single .dl (concatenated programs aren't runnable). Got {prefix:?}; using its stem.");
            }
            let stem_pfx = prefix.strip_suffix(".dl").unwrap_or(prefix);
            let mut ctx_v: Vec<i64> = if ids.len() > ctx_window { ids[..ctx_window].to_vec() } else { ids[..ids.len() - 1].to_vec() };
            // token text for the .dl comments comes from the tokenizer (api feature); without it, fall back to ids.
            #[cfg(feature = "api")]
            let tg = api::TextGen::load(&stem, eos.clone());
            let (mut written, mut faithful) = (0usize, 0usize);
            for step in 0..steps {
                let Some(prov) = logic::build(lm.as_ref(), &ctx_v, store.as_ref(), &cfg, cap_c) else {
                    eprintln!("[fieldrun] --export-logic: arch {arch} has no residual_decomp (rope only)");
                    return;
                };
                #[cfg(feature = "api")]
                let o = {
                    let lbl = |id: i64| -> String { tg.as_ref().map(|g| g.token_label(id)).unwrap_or_else(|| format!("[{id}]")) };
                    logic::emit_dl(&prov, &ctx_v, &lbl)
                };
                #[cfg(not(feature = "api"))]
                let o = logic::emit_dl(&prov, &ctx_v, &|id: i64| format!("[{id}]"));
                let path = format!("{stem_pfx}.{step:03}.dl");
                if std::fs::write(&path, &o).is_ok() {
                    written += 1;
                    if o.contains("✓ FAITHFUL") { faithful += 1; }
                } else {
                    eprintln!("[fieldrun] --export-logic: could not write {path}");
                }
                ctx_v.push(prov.predicted); // advance by the model's greedy pick → the trace follows a real trajectory
            }
            eprintln!("[fieldrun] --export-logic → {stem_pfx}.{{000..{:03}}}.dl  ({written} steps, {faithful} FAITHFUL ✓) — run: fieldrun eval {stem_pfx}.000.dl --semiring max|log",
                      steps.saturating_sub(1));
            return;
        }

        // --probe-quant (research → speed bridge): does a block's pivotality D_b predict how much QUANTIZING it
        // perturbs the decode? Per position × block: quantize that one block's residual write (per-row int{bits}
        // round-trip), re-decode, record flip; correlate with D_b. If high-|D_b| flips more ⇒ protect high-D_b blocks,
        // quantize low-D_b hard (principled per-block bit allocation). Rope-only.
        if has_flag(&args, "--probe-quant") {
            use retrieval::CandCfg;
            let store = flag(&args, "--store").and_then(|p| Store::load(p).ok());
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            let bits: u8 = flag(&args, "--bits").and_then(|s| s.parse().ok()).unwrap_or(4);
            if lm.predict_block_quant(&ids[..ctx_window.min(ids.len())], 0, bits).is_none() {
                eprintln!("[fieldrun] --probe-quant: arch {arch} has no predict_block_quant (rope only)");
                return;
            }
            let cap = (end - ctx_window).min(n_eval).min(120); // ~nblocks forwards/position — keep n modest
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            eprintln!("[fieldrun] --probe-quant: {} positions × per-block int{bits} quant…", positions.len());
            struct Q { route: u8, dj: f32, contrib: f32, flip: bool }
            let recs: Vec<Q> = positions.par_iter().flat_map(|c| {
                let mut out: Vec<Q> = Vec::new();
                let Some(ex) = lm.explain(c) else { return out; };
                let (t, vs) = (ex.model_predicts, ex.runner_up);
                let Some((_lab, contrib)) = lm.residual_decomp(c, &[t, vs]) else { return out; };
                let route = match &store {
                    Some(st) => { let (kb, _) = st.predict(c); if kb == t { 0u8 } else if st.candidates(c, &cfg).contains(&t) { 1 } else { 2 } }
                    None => 3,
                };
                for b in 0..contrib.len() {
                    let flip = lm.predict_block_quant(c, b, bits) != Some(t);
                    out.push(Q { route, dj: contrib[b][0] - contrib[b][1], contrib: contrib[b][0], flip });
                }
                out
            }).collect();
            let n = recs.len().max(1);
            let pearson = |x: &[f32], y: &[f32]| -> f32 {
                let m = x.len() as f32;
                let (mx, my) = (x.iter().sum::<f32>() / m, y.iter().sum::<f32>() / m);
                let (mut sxy, mut sxx, mut syy) = (0f32, 0f32, 0f32);
                for (&a, &b) in x.iter().zip(y) { let (dx, dy) = (a - mx, b - my); sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
                if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { 0.0 }
            };
            let ys: Vec<f32> = recs.iter().map(|r| if r.flip { 1.0 } else { 0.0 }).collect();
            let xd: Vec<f32> = recs.iter().map(|r| r.dj.abs()).collect();
            let xc: Vec<f32> = recs.iter().map(|r| r.contrib.abs()).collect();
            let flip_rate = 100.0 * recs.iter().filter(|r| r.flip).count() as f32 / n as f32;
            println!("\n=== (D_j vs quant-sensitivity) per-block int{bits} quant — {n} (position×block) pairs ===");
            println!("mean single-block flip {flip_rate:.1}%   corr(|D_b|, flip) = {:+.3}   corr(|contrib_t|, flip) = {:+.3}",
                pearson(&xd, &ys), pearson(&xc, &ys));
            let mut da = xd.clone();
            da.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (q1, q2) = (da[n / 3], da[2 * n / 3]);
            println!("|D_b| tercile → single-block flip%  (rising ⇒ pivotality predicts quant-sensitivity ⇒ protect high-D_b, quantize low-D_b hard):");
            println!("  {:<6}{:>7}{:>11}{:>10}", "bin", "n", "mean|D_b|", "flip%");
            for (lbl, lo, hi) in [("low ", f32::MIN, q1), ("mid ", q1, q2), ("high", q2, f32::MAX)] {
                let g: Vec<&Q> = recs.iter().filter(|r| r.dj.abs() >= lo && r.dj.abs() < hi).collect();
                if g.is_empty() { continue; }
                let m = g.len() as f32;
                println!("  {lbl:<6}{:>7}{:>11.2}{:>9.1}%", g.len(), g.iter().map(|r| r.dj.abs()).sum::<f32>() / m, 100.0 * g.iter().filter(|r| r.flip).count() as f32 / m);
            }
            if store.is_some() {
                println!("by route (mean single-block flip% across its blocks):");
                for (lbl, r) in [("RETRIEVED", 0u8), ("SELECTED", 1), ("COMPOSED", 2)] {
                    let g: Vec<&Q> = recs.iter().filter(|x| x.route == r).collect();
                    if g.is_empty() { continue; }
                    println!("  {lbl:<12} n {:>5}  flip {:.1}%", g.len(), 100.0 * g.iter().filter(|x| x.flip).count() as f32 / g.len() as f32);
                }
            }
            return;
        }

        // --probe-reconstruct (LE-T5 / LOGIC_EXPORT LO2): decompose the predicted-token logit into per-block residual
        // writes (embed + each layer's attn + mlp). Σ_blocks == logit EXACTLY (residual-stream additivity) ⇒ the
        // reconstruction residual measures decompiler completeness (LE-T5 exact); the per-block concentration of the
        // t-vs-v* margin is the decision's block-level support number (PIC O2: small=retrieved, large=composed). Rope-only.
        if has_flag(&args, "--probe-reconstruct") {
            use retrieval::CandCfg;
            let store = flag(&args, "--store").and_then(|p| Store::load(p).ok());
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            if lm.residual_decomp(&ids[..ctx_window.min(ids.len())], &[0]).is_none() {
                eprintln!("[fieldrun] --probe-reconstruct: arch {arch} has no residual_decomp (rope only)");
                return;
            }
            let cap = (end - ctx_window).min(n_eval);
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            eprintln!("[fieldrun] --probe-reconstruct: {} positions — block decomposition…", positions.len());
            struct R { route: u8, err: f32, block_pr: f32, top1: f32, sigma: usize }
            let mut nblocks = 0usize;
            let recs: Vec<R> = positions.par_iter().filter_map(|c| {
                let ex = lm.explain(c)?;
                let (t, v) = (ex.model_predicts, ex.runner_up);
                let (_lab, contrib) = lm.residual_decomp(c, &[t, v])?;
                let lt: f32 = contrib.iter().map(|b| b[0]).sum();
                let err = (lt - ex.predicted_logit).abs(); // LE-T5: should be ~0 (additive reconstruction is exact)
                let mut db: Vec<f32> = contrib.iter().map(|b| b[0] - b[1]).collect(); // per-block t-vs-v* pivotality
                let margin = db.iter().sum::<f32>(); // == Δ (the decision margin)
                let pos: Vec<f32> = db.iter().copied().filter(|&x| x > 0.0).collect();
                let (s, sq): (f32, f32) = (pos.iter().sum(), pos.iter().map(|x| x * x).sum());
                let block_pr = if sq > 0.0 { s * s / sq } else { 1.0 };          // effective # of supporting blocks
                let top1 = if s > 0.0 { pos.iter().cloned().fold(0.0, f32::max) / s } else { 0.0 }; // top-block share
                db.sort_by(|a, b| b.partial_cmp(a).unwrap()); // σ = #top blocks to REMOVE to flip (Σ removed > Δ)
                let (mut acc, mut sigma) = (0.0f32, 0usize);
                for &x in &db { if acc > margin { break; } acc += x; sigma += 1; }
                let route = if let Some(st) = &store {
                    let (kb, _) = st.predict(c);
                    if kb == t { 0u8 } else if st.candidates(c, &cfg).contains(&t) { 1 } else { 2 }
                } else { 3 };
                Some(R { route, err, block_pr, top1, sigma })
            }).collect();
            if let Some((lab, _)) = lm.residual_decomp(positions[0], &[0]) { nblocks = lab.len(); }
            let n = recs.len().max(1) as f32;
            let (mean_err, max_err) = (recs.iter().map(|r| r.err).sum::<f32>() / n, recs.iter().map(|r| r.err).fold(0.0, f32::max));
            println!("\n=== (LE-T5) per-block logit reconstruction over {} blocks (embed + {}×{{attn,mlp}}) ===", nblocks, (nblocks - 1) / 2);
            println!("reconstruction |Σ_blocks − logit|: mean {mean_err:.2e}  max {max_err:.2e}  ⇒ {}",
                if max_err < 1e-2 { "EXACT (residual-stream additivity holds; the export is faithful by LE-T5)" } else { "NON-zero (missing components / numerical)" });
            println!("\n=== block-level decision support (margin t-vs-v* across {} blocks) ===", nblocks);
            println!("{:<12}{:>6}{:>12}{:>14}{:>16}", "route", "n", "block-PR", "top-block %", "σ (drop→flip)");
            let groups: &[(&str, u8)] = if store.is_some() { &[("RETRIEVED", 0), ("SELECTED", 1), ("COMPOSED", 2)] } else { &[("ALL", 3)] };
            for (lbl, r) in groups {
                let g: Vec<&R> = recs.iter().filter(|x| x.route == *r).collect();
                if g.is_empty() { println!("{lbl:<12}{:>6}", 0); continue; }
                let m = g.len() as f32;
                println!("{lbl:<12}{:>6}{:>12.1}{:>13.0}%{:>16.1}", g.len(),
                    g.iter().map(|x| x.block_pr).sum::<f32>() / m, 100.0 * g.iter().map(|x| x.top1).sum::<f32>() / m,
                    g.iter().map(|x| x.sigma as f32).sum::<f32>() / m);
            }
            println!("⇒ block-PR / σ = the decision's block-level support number (PIC O2): small ⇒ retrieved-concentrated, large ⇒ composed-distributed. This bounds how compact the emitted retrievable Datalog fragment can be (LOGIC_EXPORT LO3).");
            return;
        }

        // --probe-ablate (CAUSAL test of the μ_t redundancy claim): knock out the top-k DLA circuits in the FORWARD
        // PASS (re-run with them zeroed) and ask whether the prediction flips. Redundancy prediction: COVERED tokens
        // (many individually-sufficient circuits, μ_t≫1) survive (low flip); COMPOSED tokens (emergent, μ_t≈0) collapse
        // (high flip). Converts the readout correlation into a causal claim. Rope-only (needs predict_ablated).
        if has_flag(&args, "--probe-ablate") {
            use retrieval::CandCfg;
            let store = match flag(&args, "--store").and_then(|p| Store::load(p).ok()) {
                Some(s) => s,
                None => { eprintln!("[fieldrun] --probe-ablate needs --store"); return; }
            };
            let cfg = CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
            if lm.predict_ablated(&ids[..ctx_window.min(ids.len())], &[], &[]).is_none() {
                eprintln!("[fieldrun] --probe-ablate: arch {arch} doesn't support ablation (rope only)");
                return;
            }
            let cap = (end - ctx_window).min(n_eval); // k=1: 1 explain + 1 ablated forward / position (cheap → use all n_eval)
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            let head_sweep = has_flag(&args, "--head-sweep"); // per-module O(1/PR) lemma test (expensive: nh forwards/rescue)
            eprintln!("[fieldrun] --probe-ablate: {} positions — explain + ablated forwards…{}", positions.len(),
                if head_sweep { " (+per-head sweep)" } else { "" });
            // Grok's decisive falsifier: ablate the SINGLE top circuit (k=1), record margin + PR + μ_t per position,
            // then split flip@k1 by μ_t (high ≥2 vs low =0) WITHIN matched margin bins. The decoupling theorem predicts
            // NO μ_t gap at matched margin (robustness governed by Δ, PR — not μ_t); a large gap (high-μ_t flips less)
            // would mean redundancy IS causally protective. k=1 is cheap → more positions for the 2-way split.
            const KS: [usize; 4] = [1, 2, 3, 5]; // coalition sizes for the additivity test
            struct A { route: u8, margin: f32, pr: f32, mu_t: usize, flip: bool, talign: bool, dj: f32, rho: f32,
                       sk: [f32; 4], flipk: [bool; 4], // sk[i] = ΣD_j(top KS[i]) − Δ ; flipk[i] = forward flip ablating those
                       l_top: usize, sweep: Vec<(usize, bool, bool)>, // L_top = ablated circuit's layer; sweep =
                       // (downstream layer ℓ, un-rescue via ℓ's ATTN block?, un-rescue via ℓ's MLP block?) — k=1 rescues only
                       head_tried: usize, head_unresc: usize, pr_at: f32 } // per-MODULE (single downstream head) un-rescue
                       // counts (--head-sweep): tests Grok's lemma P(single-module un-rescue) ≈ 1/PR. pr_at = PR at this rescue
            let recs: Vec<A> = positions.par_iter().filter_map(|c| {
                let ex = lm.explain(c)?;
                let t = ex.model_predicts;
                // carry isolated argmax (promotes[0]) and dla_v (contribution to the runner-up) per circuit: the ablated
                // circuit's pivotality D_j = dla - dla_v is the LINEAR flip threshold (ablate ⇒ margin shifts by -D_j).
                let mut circ: Vec<(f32, bool, usize, usize, Option<i64>, f32)> = ex.head_circuits.iter().map(|h| (h.dla, true, h.layer, h.head, h.promotes.first().copied(), h.dla_v))
                    .chain(ex.mlp_features.iter().map(|m| (m.dla, false, m.layer, m.neuron, m.promotes.first().copied(), m.dla_v))).collect();
                circ.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let top = *circ.first()?;
                let (kb, _) = store.predict(c);
                let covered = store.candidates(c, &cfg).contains(&t);
                let route = if kb == t { 0u8 } else if covered { 1 } else { 2 };
                let d: Vec<f32> = ex.all_dla.iter().copied().filter(|&x| x > 0.0).collect();
                let (sum, sumsq): (f32, f32) = (d.iter().sum(), d.iter().map(|x| x * x).sum());
                let pr = if sumsq > 0.0 { sum * sum / sumsq } else { 1.0 };
                let mu_t = ex.head_circuits.iter().filter_map(|h| h.promotes.first().copied())
                    .chain(ex.mlp_features.iter().filter_map(|m| m.promotes.first().copied())).filter(|&a| a == t).count();
                let talign = top.4 == Some(t); // is the single circuit we ablate itself a t-supporter (isolated argmax == t)?
                let dj = top.0 - top.5;        // D_j of the ablated top circuit (logit units): linear flip ⟺ margin < D_j
                let rho = lm.unembed_cos(t as usize, ex.runner_up as usize).unwrap_or(0.0); // coherence cos(U_t, U_v*)
                let margin = ex.predicted_logit - ex.runner_up_logit;
                // coalition additivity: ablate the top-k circuits JOINTLY. ΣD_j over them is the LINEAR margin shift
                // (DLA is additive), so the coalition linear identity is flip ⟺ Δ < ΣD_j. As k grows we strip more
                // pivotality AND leave the forward pass less headroom to rescue — tests additivity + cushion-exhaustion.
                let (mut sk, mut flipk) = ([0f32; 4], [false; 4]);
                for (ki, &k) in KS.iter().enumerate() {
                    let kk = k.min(circ.len());
                    let (mut hs, mut ns, mut sumdj) = (Vec::new(), Vec::new(), 0f32);
                    for cc in &circ[..kk] {
                        sumdj += cc.0 - cc.5; // this circuit's D_j
                        if cc.1 { hs.push((cc.2, cc.3)); } else { ns.push((cc.2, cc.3)); }
                    }
                    sk[ki] = sumdj - margin;
                    flipk[ki] = lm.predict_ablated(c, &hs, &ns) != Some(t);
                }
                let flip = flipk[0]; // k=1 single-circuit flip (reused by every earlier table)
                // rescue localization: for a k=1 RESCUE (s>0 but forward kept t), find WHERE the rescue lives — ablate
                // {top circuit + a whole downstream layer ℓ's attention} for each ℓ > L_top and see which ℓ un-rescues
                // (flips). Concentration at small ℓ-L_top ⇒ local rescue just downstream; spread ⇒ diffuse/deep.
                let l_top = top.2;
                let mut sweep: Vec<(usize, bool, bool)> = Vec::new();
                let (mut head_tried, mut head_unresc) = (0usize, 0usize);
                if sk[0] > 0.0 && !flipk[0] {
                    if let Some((nl, nh)) = lm.dims() {
                        let (hh, nnn): (Vec<(usize, usize)>, Vec<(usize, usize)>) =
                            if top.1 { (vec![(top.2, top.3)], vec![]) } else { (vec![], vec![(top.2, top.3)]) };
                        for l2 in (l_top + 1)..nl {
                            // ablate {top circuit + whole ATTN block of ℓ} and {top + whole MLP block of ℓ} separately
                            let un_attn = lm.predict_ablated_blocks(c, &hh, &nnn, &[l2], &[]) != Some(t);
                            let un_mlp = lm.predict_ablated_blocks(c, &hh, &nnn, &[], &[l2]) != Some(t);
                            sweep.push((l2, un_attn, un_mlp));
                        }
                        // --head-sweep: per-MODULE test of Grok's lemma — ablate {top + a SINGLE downstream head} for
                        // every downstream head; P(single-module un-rescue) should ≈ 1/PR (≈2%) in the high-PR regime.
                        if head_sweep {
                            for l2 in (l_top + 1)..nl {
                                for h in 0..nh {
                                    let mut hs = hh.clone();
                                    hs.push((l2, h));
                                    head_tried += 1;
                                    if lm.predict_ablated(c, &hs, &nnn) != Some(t) { head_unresc += 1; }
                                }
                            }
                        }
                    }
                }
                Some(A { route, margin, pr, mu_t, flip, talign, dj, rho, sk, flipk, l_top, sweep, head_tried, head_unresc, pr_at: pr })
            }).collect();
            let prs: Vec<f32> = recs.iter().map(|x| x.pr).collect();
            let prmin = prs.iter().cloned().fold(f32::MAX, f32::min);
            let prmax = prs.iter().cloned().fold(f32::MIN, f32::max);
            println!("\n=== (causal) ablate the single top DLA circuit → flip? by route (PR range {prmin:.0}-{prmax:.0}, ~flat) ===");
            println!("{:<12}{:>6}{:>10}{:>10}{:>12}", "route", "n", "margin", "μ_t", "flip@k1");
            for (lbl, r) in [("RETRIEVED", 0u8), ("SELECTED", 1), ("COMPOSED", 2)] {
                let g: Vec<&A> = recs.iter().filter(|x| x.route == r).collect();
                if g.is_empty() { println!("{lbl:<12}{:>6}", 0); continue; }
                let n = g.len() as f32;
                println!("{lbl:<12}{:>6}{:>10.2}{:>10.2}{:>11.0}%", g.len(),
                    g.iter().map(|x| x.margin).sum::<f32>() / n, g.iter().map(|x| x.mu_t as f32).sum::<f32>() / n,
                    100.0 * g.iter().filter(|x| x.flip).count() as f32 / n);
            }
            // Grok's falsifier: μ_t-split WITHIN matched margin bins (PR ~flat, so margin-matching ≈ (Δ,PR)-matching).
            let mut ms: Vec<f32> = recs.iter().map(|x| x.margin).collect();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f32| if ms.is_empty() { 0.0 } else { ms[(((ms.len() - 1) as f32) * p) as usize] };
            let (t1, t2) = (q(0.333), q(0.667));
            let fl = |g: &[&A]| if g.is_empty() { f32::NAN } else { 100.0 * g.iter().filter(|x| x.flip).count() as f32 / g.len() as f32 };
            let mpr = |g: &[&A]| if g.is_empty() { f32::NAN } else { g.iter().map(|x| x.pr).sum::<f32>() / g.len() as f32 };
            let tal = |g: &[&A]| if g.is_empty() { f32::NAN } else { 100.0 * g.iter().filter(|x| x.talign).count() as f32 / g.len() as f32 };
            println!("\n  (Grok falsifier) flip@k1 split by μ_t WITHIN matched margin bins:");
            println!("  {:<10}{:>8}{:>22}{:>22}", "margin bin", "mean Δ", "μ_t≥2  n flip PR t→", "μ_t=0  n flip PR t→");
            for (lbl, lo, hi) in [("low ", f32::MIN, t1), ("mid ", t1, t2), ("high", t2, f32::MAX)] {
                let inb = |x: &&A| x.margin >= lo && x.margin < hi;
                let hi_m: Vec<&A> = recs.iter().filter(|x| inb(x) && x.mu_t >= 2).collect();
                let lo_m: Vec<&A> = recs.iter().filter(|x| inb(x) && x.mu_t == 0).collect();
                let mm = recs.iter().filter(inb).map(|x| x.margin).sum::<f32>() / recs.iter().filter(inb).count().max(1) as f32;
                // per cell: n, flip%, mean PR, and t→ = % of ablated top circuits that were themselves t-aligned (the
                // "which circuit we knock out" confound — higher in μ_t≥2, which inflates its flip% vs μ_t=0).
                println!("  {lbl:<10}{mm:>8.2}{:>8} {:>3.0}% {:>3.0} {:>3.0}%{:>8} {:>3.0}% {:>3.0} {:>3.0}%",
                    hi_m.len(), fl(&hi_m), mpr(&hi_m), tal(&hi_m), lo_m.len(), fl(&lo_m), mpr(&lo_m), tal(&lo_m));
            }
            println!("⇒ DECOUPLING (Grok) predicts μ_t≥2 flip% ≈ μ_t=0 flip% within a bin; a large gap (high-μ_t flips less) refutes it = redundancy is causally protective.");
            println!("  (t→ = % of ablated circuits that were themselves t-supporters: if μ_t≥2's higher flip tracks higher t→, the reverse gap is the which-circuit confound, not protection failing.)");
            // (B-clean) hold the which-circuit confound FIXED: restrict to t→=1 (we ALWAYS ablate a confirmed
            // t-supporter), then split flip by μ_t WITHIN matched margin bins. μ_t=1 = we removed the ONLY supporter
            // (none left); μ_t≥2 = we removed one, ≥1 backup remains. This is the airtight "do backups protect?" test —
            // both arms ablate a genuine t-supporter, so the only difference is whether redundant backups exist.
            // DECOUPLING predicts μ_t=1 flip% ≈ μ_t≥2 flip% (backups inert); μ_t≥2 flipping LESS = redundancy protects.
            let md = |g: &[&A]| if g.is_empty() { f32::NAN } else { g.iter().map(|x| x.margin).sum::<f32>() / g.len() as f32 };
            let nta = recs.iter().filter(|x| x.talign).count();
            println!("\n  (B-clean) within t→=1 ONLY (always ablate a CONFIRMED t-supporter, n={nta}): flip by μ_t in matched margin bins:");
            println!("  {:<10}{:>22}{:>22}", "margin bin", "μ_t=1 (none left)  n Δ flip", "μ_t≥2 (backups)  n Δ flip");
            for (lbl, lo, hi) in [("low ", f32::MIN, t1), ("mid ", t1, t2), ("high", t2, f32::MAX)] {
                let one: Vec<&A> = recs.iter().filter(|x| x.talign && x.margin >= lo && x.margin < hi && x.mu_t == 1).collect();
                let many: Vec<&A> = recs.iter().filter(|x| x.talign && x.margin >= lo && x.margin < hi && x.mu_t >= 2).collect();
                println!("  {lbl:<10}{:>8} {:>5.2} {:>4.0}%{:>12} {:>5.2} {:>4.0}%",
                    one.len(), md(&one), fl(&one), many.len(), md(&many), fl(&many));
            }
            println!("⇒ DECOUPLING predicts μ_t=1 flip% ≈ μ_t≥2 flip% (backups don't catch the loss); μ_t≥2 flipping LESS at matched Δ = redundancy is causally protective. Both arms remove a genuine t-supporter, so this isolates μ_t from the which-circuit confound.");
            // (D_j regression) the LINEAR flip identity: ablating the top circuit flips iff Δ < D_j (D_j = dla - dla_v,
            // the circuit's pivotality = how much it shifts the t-vs-v* margin). s = D_j - Δ is the linear flip score
            // (>0 ⇒ linear predicts flip). Three things at once: (1) does actual forward-flip rise as a step at s=0
            // (linear identity holds causally)? (2) the confusion of sign(s) vs actual flip (the indirect-effect gap);
            // (3) is μ_t inert once we bin on s (μ_t≥2 vs μ_t=0 flip% within s-bins should match — μ_t a noisy proxy).
            let s_of = |x: &A| x.dj - x.margin;
            println!("\n  (D_j regression) linear flip score s = D_j - Δ  (D_j = dla-dla_v of the ablated circuit):");
            println!("  {:<14}{:>6}{:>10}{:>12}", "s bin", "n", "mean s", "flip%");
            for (lbl, lo, hi) in [("s<-1   ", f32::MIN, -1.0), ("-1..-.3", -1.0, -0.3), ("-.3..0 ", -0.3, 0.0),
                                  ("0..+.3 ", 0.0, 0.3), ("+.3..1 ", 0.3, 1.0), ("s>+1   ", 1.0, f32::MAX)] {
                let g: Vec<&A> = recs.iter().filter(|x| { let s = s_of(x); s >= lo && s < hi }).collect();
                if g.is_empty() { continue; }
                let n = g.len() as f32;
                println!("  {lbl:<14}{:>6}{:>10.2}{:>11.0}%", g.len(), g.iter().map(|y| s_of(y)).sum::<f32>() / n, 100.0 * g.iter().filter(|x| x.flip).count() as f32 / n);
            }
            let pred_flip = |x: &A| s_of(x) > 0.0; // linear identity's prediction
            let (mut tp, mut tn, mut fp, mut fn_) = (0u32, 0u32, 0u32, 0u32);
            for x in &recs { match (pred_flip(x), x.flip) { (true, true) => tp += 1, (true, false) => fp += 1, (false, true) => fn_ += 1, (false, false) => tn += 1 } }
            let n = recs.len() as f32;
            println!("  linear identity sign(s)>0 vs actual flip: acc {:.0}%  [tp {tp} tn {tn} | fp {fp} fn {fn_}]  (fp+fn = indirect-effect / new-winner≠v* gap)",
                100.0 * (tp + tn) as f32 / n);
            println!("  ⇒ μ_t inert once s is fixed? flip% by μ_t WITHIN |s| bins (linear-score-matched, the cleanest control):");
            let mut ss: Vec<f32> = recs.iter().map(|y| s_of(y)).collect();
            ss.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let sq = |p: f32| if ss.is_empty() { 0.0 } else { ss[(((ss.len() - 1) as f32) * p) as usize] };
            let (s1, s2) = (sq(0.333), sq(0.667));
            // per-cell mean Δ and mean D_j: the CONFOUND CHECK. μ_t≥2 flipping less at matched s is genuine indirect
            // μ_t-protection ONLY if its Δ (and D_j) are ~equal to μ_t=0's; if μ_t≥2 has much higher Δ, the "protection"
            // is a margin/scale effect (more total cushion), not redundancy. (s = D_j - Δ, so matched s lets Δ co-vary.)
            let mg = |g: &[&A]| if g.is_empty() { f32::NAN } else { g.iter().map(|x| x.margin).sum::<f32>() / g.len() as f32 };
            let mdj = |g: &[&A]| if g.is_empty() { f32::NAN } else { g.iter().map(|x| x.dj).sum::<f32>() / g.len() as f32 };
            println!("  {:<8}{:>8}{:>26}{:>26}", "s bin", "mean s", "μ_t≥2  n flip Δ D_j", "μ_t=0  n flip Δ D_j");
            for (lbl, lo, hi) in [("low s ", f32::MIN, s1), ("mid s ", s1, s2), ("high s", s2, f32::MAX)] {
                let inb = |x: &&A| { let s = s_of(x); s >= lo && s < hi };
                let hi_m: Vec<&A> = recs.iter().filter(|x| inb(x) && x.mu_t >= 2).collect();
                let lo_m: Vec<&A> = recs.iter().filter(|x| inb(x) && x.mu_t == 0).collect();
                let ms = recs.iter().filter(inb).map(|y| s_of(y)).sum::<f32>() / recs.iter().filter(inb).count().max(1) as f32;
                println!("  {lbl:<8}{ms:>8.2}{:>8} {:>4.0}% {:>5.2} {:>5.2}{:>10} {:>4.0}% {:>5.2} {:>5.2}",
                    hi_m.len(), fl(&hi_m), mg(&hi_m), mdj(&hi_m), lo_m.len(), fl(&lo_m), mg(&lo_m), mdj(&lo_m));
            }
            println!("  (μ_t≥2 flips less at matched s ⇒ indirect μ_t-protection IF its Δ≈μ_t=0's; if its Δ is much higher, it's a margin/scale effect, not redundancy.)");
            // (logistic) the principled control: fit flip ~ Δ + D_j + 1[μ_t≥2]. Does μ_t add predictive power AFTER the
            // two real causal variables? Standardize Δ,D_j (coeffs comparable); GD. The mean-log-loss penalty from
            // DROPPING μ_t is its independent value: ≈0 ⇒ μ_t inert (proxy); large ⇒ μ_t independently causal.
            let ys: Vec<f32> = recs.iter().map(|r| if r.flip { 1.0 } else { 0.0 }).collect();
            let raw: Vec<[f32; 3]> = recs.iter().map(|r| [r.margin, r.dj, if r.mu_t >= 2 { 1.0 } else { 0.0 }]).collect();
            let nn = raw.len() as f32;
            let (mut mu, mut sd) = ([0f32; 2], [1f32; 2]);
            for j in 0..2 {
                mu[j] = raw.iter().map(|x| x[j]).sum::<f32>() / nn;
                sd[j] = (raw.iter().map(|x| (x[j] - mu[j]).powi(2)).sum::<f32>() / nn).sqrt().max(1e-6);
            }
            let z: Vec<[f32; 3]> = raw.iter().map(|x| [(x[0] - mu[0]) / sd[0], (x[1] - mu[1]) / sd[1], x[2]]).collect();
            let fit = |use_mu: bool| -> ([f32; 4], f32) {
                let mut w = [0f32; 4]; // bias, Δ, D_j, μ_t≥2
                for _ in 0..6000 {
                    let mut g = [0f32; 4];
                    for (zi, &y) in z.iter().zip(&ys) {
                        let lin = w[0] + w[1] * zi[0] + w[2] * zi[1] + if use_mu { w[3] * zi[2] } else { 0.0 };
                        let e = 1.0 / (1.0 + (-lin).exp()) - y;
                        g[0] += e; g[1] += e * zi[0]; g[2] += e * zi[1]; if use_mu { g[3] += e * zi[2]; }
                    }
                    for k in 0..4 { w[k] -= 0.3 * g[k] / nn; }
                }
                let ll = z.iter().zip(&ys).map(|(zi, &y)| {
                    let lin = w[0] + w[1] * zi[0] + w[2] * zi[1] + if use_mu { w[3] * zi[2] } else { 0.0 };
                    let p = (1.0 / (1.0 + (-lin).exp())).clamp(1e-6, 1.0 - 1e-6);
                    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                }).sum::<f32>() / nn;
                (w, ll)
            };
            let (wf, llf) = fit(true);
            let (_, ll0) = fit(false);
            println!("\n  (logistic) flip ~ Δ + D_j + 1[μ_t≥2]  (Δ,D_j standardized → coeffs comparable; sign: +D_j/−Δ expected):");
            println!("    coeffs  bias {:+.2}   Δ {:+.2}   D_j {:+.2}   μ_t≥2 {:+.2}", wf[0], wf[1], wf[2], wf[3]);
            println!("    mean log-loss  full {llf:.3}  drop-μ_t {ll0:.3}  (Δ {:+.4} = μ_t's INDEPENDENT predictive value; ≈0 ⇒ proxy)", ll0 - llf);
            // (A/B) Grok's incoherence-boundary + Δ-cushion tests. ρ = cos(U_t, U_v*). Among s>0 (linear predicts flip)
            // a RESCUE = forward keeps t (indirect recomposition). Predictions: (A) P(rescue|s>0) FALLS as ρ↑
            // [=1-Φ(s/σ), σ∝√(1-ρ²)→0], with |D_j|→0 mechanically as ρ→1; (B) rescue RISES with Δ (the cushion).
            let mut rs: Vec<f32> = recs.iter().map(|x| x.rho).collect();
            rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let rq = |p: f32| if rs.is_empty() { 0.0 } else { rs[(((rs.len() - 1) as f32) * p) as usize] };
            let (rq1, rq2, rq3) = (rq(0.25), rq(0.5), rq(0.75));
            let rbins = [("Q1 lo ρ", f32::MIN, rq1), ("Q2     ", rq1, rq2), ("Q3     ", rq2, rq3), ("Q4 hi ρ", rq3, f32::MAX)];
            println!("\n  (A) incoherence boundary: ρ = cos(U_t, U_v*) by quartile [pred: |D_j| ↓ as ρ ↑; rescue ↓ as ρ ↑]:");
            println!("  {:<10}{:>5}{:>9}{:>11}{:>9}{:>20}", "ρ quart", "n", "mean ρ", "mean|D_j|", "flip%", "s>0: n rescue%");
            for (lbl, lo, hi) in rbins {
                let g: Vec<&A> = recs.iter().filter(|x| x.rho >= lo && x.rho < hi).collect();
                if g.is_empty() { continue; }
                let n = g.len() as f32;
                let sp: Vec<&A> = g.iter().copied().filter(|x| s_of(x) > 0.0).collect();
                let resc = if sp.is_empty() { f32::NAN } else { 100.0 * sp.iter().filter(|x| !x.flip).count() as f32 / sp.len() as f32 };
                println!("  {lbl:<10}{:>5}{:>9.2}{:>11.2}{:>8.0}%{:>12} {:>5.0}%", g.len(),
                    g.iter().map(|x| x.rho).sum::<f32>() / n, g.iter().map(|x| x.dj.abs()).sum::<f32>() / n,
                    100.0 * g.iter().filter(|x| x.flip).count() as f32 / n, sp.len(), resc);
            }
            println!("  (B) Δ-cushion: among s>0 (linear predicts flip), rescue% by Δ [pred: rescue ↑ as Δ ↑]:");
            for (lbl, lo, hi) in [("Δ<.3  ", f32::MIN, 0.3), ("0.3-.6", 0.3, 0.6), ("0.6-1.", 0.6, 1.0), ("Δ>1.0 ", 1.0, f32::MAX)] {
                let g: Vec<&A> = recs.iter().filter(|x| s_of(x) > 0.0 && x.margin >= lo && x.margin < hi).collect();
                if g.is_empty() { continue; }
                let n = g.len();
                println!("  {lbl:<10}n {:>3}  rescue {:>4.0}%  (mean Δ {:.2}, mean s {:.2})", n,
                    100.0 * g.iter().filter(|x| !x.flip).count() as f32 / n as f32,
                    g.iter().map(|x| x.margin).sum::<f32>() / n as f32, g.iter().map(|y| s_of(y)).sum::<f32>() / n as f32);
            }
            // (coalition additivity) ablate the top-k JOINTLY; the linear coalition identity is flip ⟺ Δ < ΣD_j, i.e.
            // sk = ΣD_j − Δ > 0. Per k: flip%, linear-identity accuracy (sign(sk) vs forward flip), and the rescue rate
            // among sk>0 (forward keeps t). Predictions: identity stays a good NECESSARY condition; rescue rate FALLS as
            // k grows (more pivotality stripped ⇒ less downstream headroom = cushion exhaustion); the residual (fp) is
            // indirect-effect + new-winner≠v* (grows with k as other facets enter).
            let nrec = recs.len() as f32;
            println!("\n  (coalition additivity) ablate top-k JOINTLY; linear identity flip ⟺ Δ < ΣD_j (sk=ΣD_j−Δ):");
            println!("  {:>4}{:>10}{:>12}{:>16}{:>16}", "k", "flip%", "mean sk", "ident-acc[fp fn]", "sk>0: n rescue%");
            for (ki, &k) in KS.iter().enumerate() {
                let flippct = 100.0 * recs.iter().filter(|x| x.flipk[ki]).count() as f32 / nrec;
                let msk = recs.iter().map(|x| x.sk[ki]).sum::<f32>() / nrec;
                let (mut tp, mut tn, mut fp, mut fnn) = (0u32, 0u32, 0u32, 0u32);
                for x in &recs { match (x.sk[ki] > 0.0, x.flipk[ki]) { (true, true) => tp += 1, (true, false) => fp += 1, (false, true) => fnn += 1, (false, false) => tn += 1 } }
                let acc = 100.0 * (tp + tn) as f32 / nrec;
                let sp = recs.iter().filter(|x| x.sk[ki] > 0.0).count();
                let resc = if sp == 0 { f32::NAN } else { 100.0 * recs.iter().filter(|x| x.sk[ki] > 0.0 && !x.flipk[ki]).count() as f32 / sp as f32 };
                println!("  {k:>4}{flippct:>9.0}%{msk:>12.2}{acc:>11.0}% [{fp:>2} {fnn:>2}]{sp:>10} {resc:>5.0}%", );
            }
            println!("  ⇒ additivity holds if ident-acc stays high; cushion exhausts if rescue% falls with k; fp rising with k = new-winner≠v* (other facets).");
            // (rescue localization) where does the indirect rescue δ live? (1) does rescue scale with L_top depth
            // (downstream headroom)? (2) layer sweep: ablate {top + a whole downstream layer's attention} → which ℓ
            // un-rescues, by relative depth ℓ−L_top.
            let nl = lm.dims().map(|d| d.0).unwrap_or(24);
            println!("\n  (rescue localization) is the rescue DOWNSTREAM? among s>0 (k=1), rescue% by L_top depth [pred: early L_top ⇒ more headroom ⇒ more rescue]:");
            println!("  {:<8}{:>8}{:>12}{:>10}", "L_top", "n(s>0)", "mean L_top", "rescue%");
            for (lbl, lo, hi) in [("early ", 0usize, nl / 3), ("mid   ", nl / 3, 2 * nl / 3), ("late  ", 2 * nl / 3, nl + 1)] {
                let g: Vec<&A> = recs.iter().filter(|x| x.sk[0] > 0.0 && x.l_top >= lo && x.l_top < hi).collect();
                if g.is_empty() { continue; }
                let n = g.len() as f32;
                println!("  {lbl:<8}{:>8}{:>12.1}{:>9.0}%", g.len(), g.iter().map(|x| x.l_top as f32).sum::<f32>() / n,
                    100.0 * g.iter().filter(|x| !x.flip).count() as f32 / n);
            }
            println!("  layer sweep over k=1 rescues: ablate {{top + downstream layer ℓ's ATTN | MLP}} → un-rescue% by relative depth (ℓ−L_top):");
            println!("  {:<10}{:>6}{:>14}{:>14}", "Δdepth", "n", "attn un-resc%", "MLP un-resc%");
            let maxd = recs.iter().flat_map(|x| x.sweep.iter().map(|s| s.0 - x.l_top)).max().unwrap_or(0);
            for d in 1..=maxd.min(10) {
                let (mut tot, mut una, mut unm) = (0usize, 0usize, 0usize);
                for x in &recs { for &(l2, ua, um) in &x.sweep { if l2 - x.l_top == d { tot += 1; if ua { una += 1; } if um { unm += 1; } } } }
                if tot == 0 { continue; }
                println!("  Δdepth {d:>2}{:>8}{:>13.0}%{:>13.0}%", tot, 100.0 * una as f32 / tot as f32, 100.0 * unm as f32 / tot as f32);
            }
            let nresc = recs.iter().filter(|x| !x.sweep.is_empty()).count().max(1);
            let brk_a = recs.iter().filter(|x| x.sweep.iter().any(|&(_, ua, _)| ua)).count();
            let brk_m = recs.iter().filter(|x| x.sweep.iter().any(|&(_, _, um)| um)).count();
            let brk_e = recs.iter().filter(|x| x.sweep.iter().any(|&(_, ua, um)| ua || um)).count();
            println!("  of {nresc} k=1 rescues, breakable by SOME single downstream block: attn {brk_a} ({:.0}%) · MLP {brk_m} ({:.0}%) · either {brk_e} ({:.0}%)  [residual = diffuse across layers].",
                100.0 * brk_a as f32 / nresc as f32, 100.0 * brk_m as f32 / nresc as f32, 100.0 * brk_e as f32 / nresc as f32);
            if head_sweep {
                // Grok's PR→localizability lemma: P(single-MODULE un-rescue) ≈ 1/PR. Per-head un-rescue rate, pooled over
                // all downstream single-head ablations of all rescues, vs the measured mean 1/PR at those rescues.
                let tried: usize = recs.iter().map(|x| x.head_tried).sum();
                let unre: usize = recs.iter().map(|x| x.head_unresc).sum();
                let prr: Vec<f32> = recs.iter().filter(|x| x.head_tried > 0).map(|x| x.pr_at).collect();
                let mean_pr = if prr.is_empty() { f32::NAN } else { prr.iter().sum::<f32>() / prr.len() as f32 };
                let perhead = if tried == 0 { f32::NAN } else { 100.0 * unre as f32 / tried as f32 };
                println!("\n  (Grok lemma) per-MODULE un-rescue: ablate {{top + a SINGLE downstream head}} over ALL such heads:");
                println!("    {unre}/{tried} single-head ablations un-rescue = {perhead:.1}% per head   vs   1/PR = {:.1}% (mean PR {mean_pr:.0} at rescues)",
                    100.0 / mean_pr);
                println!("  ⇒ lemma predicts per-module un-rescue ≈ 1/PR; match ⇒ repair diffuse in a high-PR substrate (no surgical head).");
            }
            return;
        }

        // --serve / --server <PORT> (accept both spellings — a common typo). The API server (no ids needed).
        let serve_port = flag(&args, "--serve")
            .or_else(|| flag(&args, "--server"))
            .and_then(|s| s.parse::<u16>().ok());
        // --explain[=MODE]: in the chat REPL it turns ON per-reply explanations (toggle live with /explain). MODE is the
        // EXPLAIN level — `route` (default, free: per-token RETRIEVED/SELECTED/COMPOSED), `circuits` (route + DLA
        // breakdown only on COMPOSED tokens), `all` (DLA on every token). `--explain` alone = route.
        let explain: Option<explain::ExplainMode> = if has_flag(&args, "--explain") {
            Some(flag(&args, "--explain").and_then(explain::ExplainMode::parse).unwrap_or(explain::ExplainMode::Route))
        } else {
            None
        };

        // Chat (interactive REPL) is the DEFAULT when no other mode/input is given — the quickest "does it work?"
        // human interface — and also runs on explicit --chat. (--serve / --generate / --ids take precedence; bare
        // --explain with no ids falls through to chat with explanations enabled.)
        let chat_mode = has_flag(&args, "--chat")
            || (ids.is_empty() && serve_port.is_none() && flag(&args, "--generate").is_none());
        #[cfg(feature = "api")]
        if chat_mode {
            let want: Option<usize> = flag(&args, "--max-tokens").and_then(|s| s.parse().ok());
            match api::TextGen::load(&stem, eos.clone()) {
                // default reply cap depends on the model (reasoning models get a bigger budget); --max-tokens overrides.
                Some(tg) => {
                    let max_tokens = want.unwrap_or_else(|| tg.default_max_tokens());
                    // --store loads the KB rules so explain can attribute each token to an idiom (RETRIEVED/SELECTED);
                    // without it, routing is induction-only. The candidate set bounds the SELECTED-vs-COMPOSED line.
                    let kb = flag(&args, "--store").and_then(|p| Store::load(p).ok());
                    let cand = retrieval::CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true };
                    let bucket = has_flag(&args, "--bucket").then(|| bucketing::BucketOpts {
                        k: flag(&args, "--decomp-k").and_then(|s| s.parse().ok()).unwrap_or(4),
                        experts: flag(&args, "--experts").and_then(|s| s.parse().ok()).unwrap_or(8),
                    });
                    api::chat(lm, tg, max_tokens, explain, kb, cand, has_flag(&args, "--raw"), &arch, bucket);
                }
                None => eprintln!("[fieldrun] no tokenizer next to {stem} — re-run `convert` (it copies tokenizer.json). \
                                   Meanwhile: --ids <holdout.json> to score, or --serve <PORT>."),
            }
            return;
        }
        #[cfg(not(feature = "api"))]
        if chat_mode {
            eprintln!("[fieldrun] chat isn't in this build (built --no-default-features). Rebuild with default features \
                       for chat/API, or use --ids <holdout.json> to score / --serve <PORT>.");
            return;
        }

        // --serve PORT: start the HTTP API over this loaded model (no ids needed).
        if let Some(port) = serve_port {
            if explain.is_some() {
                eprintln!("[fieldrun] note: --explain toggles per-reply explanations in the chat REPL. The API server \
                           ignores it — POST /explain for the structured form, or pass \"explain\":true to the chat \
                           endpoints. (Use --chat --explain for an explained REPL.)");
            }
            #[cfg(feature = "api")]
            let textgen = api::TextGen::load(&stem, eos);
            #[cfg(not(feature = "api"))]
            let textgen: Option<api::TextGen> = None;
            // KB rules for the typed `"explain"` field (route/circuits/all): --store enables full RETRIEVED/SELECTED
            // attribution; without it, the route is induction-only. The candidate set bounds SELECTED-vs-COMPOSED.
            let explain_opts = api::ExplainOpts {
                store: flag(&args, "--store").and_then(|p| Store::load(p).ok()),
                cand: retrieval::CandCfg { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, skel: 8, uni: 128, closed: true },
            };
            api::serve(lm, &arch, port, textgen, explain_opts);
            return;
        }

        // --explain WITH --ids: standalone "explain the prediction at the end of the first --ctx tokens" (circuits +
        // features). Without ids we'd have already gone to chat above; guard anyway so an empty stream can't index out.
        if explain.is_some() {
            if ids.is_empty() {
                eprintln!("[fieldrun] --explain standalone mode needs --ids <token stream>. For explained chat replies, \
                           run with --chat --explain (or just --explain) and toggle /explain in the REPL.");
                return;
            }
            let ctx = &ids[..ctx_window.min(ids.len())];
            match lm.explain(ctx) {
                Some(ex) => {
                    let dec = load_decoder(flag(&args, "--vocab"));
                    let ctx_show = flag(&args, "--explain-context").and_then(|s| s.parse().ok()).unwrap_or(10);
                    println!("{}", explain::render(&ex, &dec, ctx_show));
                    if let Some(p) = flag(&args, "--out-json") {
                        if let Err(e) = std::fs::write(p, serde_json::to_string_pretty(&ex).unwrap()) {
                            eprintln!("[fieldrun] couldn't write --out-json {p}: {e}");
                        }
                    }
                }
                None => println!("[fieldrun] explain not implemented for arch {arch}"),
            }
            return;
        }

        // --gate-check N: the faithfulness measurement for --pruned-head. Generate N tokens through the GATED stream
        // decode vs the ungated full-head stream on the same prompts, report the identical prefix + accept rate.
        // This is how a --pruned-margin threshold is calibrated: raise it until the prefix holds at the length you
        // serve. (Past the first divergence the contexts differ, so only the prefix is the agreement metric.) The
        // reference is the ungated KV stream — itself gated byte-identical to the naive recompute by --gen-prefix /
        // validate_all.sh — so the check is N decode steps, not N full-context forwards. --gate-prompts P spreads P
        // prompts evenly across the --ids stream (closed-loop trajectories from one prompt are a sample size of 1).
        if let Some(n) = flag(&args, "--gate-check").and_then(|s| s.parse::<usize>().ok()) {
            if ids.len() < ctx_window {
                eprintln!("[fieldrun] --gate-check needs --ids with at least --ctx tokens (a prompt to generate from)");
                return;
            }
            let p_cnt: usize = flag(&args, "--gate-prompts").and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
            let span = ids.len() - ctx_window;
            let offsets: Vec<usize> = (0..p_cnt).map(|i| span * i / p_cnt).collect();
            let t0 = std::time::Instant::now();
            let gated: Vec<Vec<i64>> = offsets.iter().map(|&o| lm.generate_stream(&ids[o..o + ctx_window], n, &[], &mut |_| true)).collect();
            let gated_s = t0.elapsed().as_secs_f64();
            let stats = lm.head_gate_stats(); // capture before clearing for the reference pass
            lm.clear_head_gate();
            let t1 = std::time::Instant::now();
            let full: Vec<Vec<i64>> = offsets.iter().map(|&o| lm.generate_stream(&ids[o..o + ctx_window], n, &[], &mut |_| true)).collect();
            let full_s = t1.elapsed().as_secs_f64();
            println!("[fieldrun] gate-check · {arch} · {} prompts × {n} tokens (ctx {ctx_window})", offsets.len());
            let (mut tok_tot, mut agree_tot, mut exact) = (0usize, 0usize, 0usize);
            for (i, (g, f)) in gated.iter().zip(&full).enumerate() {
                let m = g.len().min(f.len());
                let agree = g.iter().zip(f.iter()).take_while(|(a, b)| a == b).count();
                tok_tot += m;
                agree_tot += agree;
                if agree == m { exact += 1; }
                println!("[fieldrun]   prompt@{:<6} identical prefix: {agree}/{m}{}", offsets[i],
                         if agree == m { String::new() } else { format!("  (diverged at token {agree})") });
            }
            println!("[fieldrun]   exact trajectories: {exact}/{} · mean identical prefix: {:.0}%",
                     offsets.len(), 100.0 * agree_tot as f64 / tok_tot.max(1) as f64);
            match stats {
                Some((acc, fb)) => {
                    let tot = (acc + fb).max(1);
                    println!("[fieldrun]   gate: {acc} pruned + {fb} full-head fallback ({:.0}% accepted)", 100.0 * acc as f64 / tot as f64);
                }
                None => println!("[fieldrun]   gate: none installed (pass --pruned-head --store <store.json>)"),
            }
            println!("[fieldrun]   gated: {gated_s:.2}s ({:.1} tok/s) · ungated full head: {full_s:.2}s ({:.1} tok/s) · {:.2}× decode",
                     (tok_tot as f64) / gated_s.max(1e-9), (tok_tot as f64) / full_s.max(1e-9), full_s / gated_s.max(1e-9));
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

        // --gen-prefix N: prefix-KV reuse gate. Warm a cache from `prompt` (leaving its full K/V resident), then
        // generate an EXTENDED prompt that shares that prefix two ways — reusing the warm cache (partial prefill of
        // only the new suffix) vs a cold cache (full prefill) — and against the naive recompute. Reuse must be
        // byte-identical: the chunked forward at the reuse boundary attends to the copied prefix rows exactly as a
        // fresh prefill would. Runs on the tiny no-tokenizer instances (it speaks raw ids), so validate_all.sh gates it.
        if let Some(n) = flag(&args, "--gen-prefix").and_then(|s| s.parse::<usize>().ok()) {
            use crate::model::PrefixKv;
            let prompt = &ids[..ctx_window.min(ids.len())];
            let mut warm = PrefixKv::default();
            let seed = lm.generate_stream_prefix(prompt, n, &[], &mut |_| true, &mut warm); // warm.ids = prompt ++ seed
            // an extended prompt that shares a non-trivial prefix with the warm cache
            let mut ext = prompt.to_vec();
            ext.extend_from_slice(&seed[..seed.len() / 2]);
            let reuse_l = warm.reuse_len(&ext);
            let reuse = lm.generate_stream_prefix(&ext, n, &[], &mut |_| true, &mut warm);
            let fresh = lm.generate_stream_prefix(&ext, n, &[], &mut |_| true, &mut PrefixKv::default());
            let naive = lm.generate_stream(&ext, n, &[], &mut |_| true);
            println!("[fieldrun] gen-prefix · {arch} · prompt {} → ext {} (reused {reuse_l} prefix tokens, {} new)",
                     prompt.len(), ext.len(), ext.len() - reuse_l);
            println!("[fieldrun]   reuse==fresh: {}  ·  reuse==naive: {}  ·  identical: {}",
                     reuse == fresh, reuse == naive, reuse == fresh && reuse == naive);
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

    // No --bundle, and no explicit --store: the user didn't pick a mode. DON'T silently fall through to Tier A on the
    // dev-default store path (that's how a `--convert`/`--bundl` typo ends up printing retrieval stats — baffling).
    // Point them at the right command; flag a likely-meant `convert` if --model is hanging around.
    if store_explicit.is_none() {
        if flag(&args, "--model").is_some() {
            eprintln!("[fieldrun] saw --model but no `convert` subcommand. Did you mean:\n  \
                       fieldrun convert --model {} --arch <arch> --dtype int8\n  \
                       (the subcommand is `convert`, not `--convert`/a flag.)",
                      flag(&args, "--model").unwrap());
        } else {
            eprintln!("[fieldrun] no mode selected. Pick one:\n  \
                       fieldrun --bundle <stem> [--chat | --serve PORT | --ids <ids.json>]   run a model\n  \
                       fieldrun convert --model <dir|hf-repo-id> --arch <arch>               build a bundle\n  \
                       fieldrun --store <store.json> --ids <ids.json>                        Tier A (retrieval)\n  \
                       fieldrun --help                                                       full flag list");
        }
        std::process::exit(2);
    }

    // Tier A (retrieval) — induction + n-gram + grammar over the flat store; positions scored in parallel.
    let store = Store::load(store_path).unwrap_or_else(|e| {
        eprintln!("[fieldrun] couldn't load --store {store_path}: {e}");
        std::process::exit(1);
    });
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

/// The default bundle cache dir (out of the cwd / dev tree), like the HF cache. Override with $FIELDRUN_BUNDLES;
/// per-convert, `-o <path>` overrides outright and `--bundle <path>` loads any explicit stem.
fn bundles_dir() -> String {
    if let Ok(d) = std::env::var("FIELDRUN_BUNDLES") {
        if !d.is_empty() {
            return d;
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    format!("{home}/.cache/fieldrun/bundles")
}

/// Bundle names present in the cache (`<dir>/<name>/<name>.fieldrun.json`) — for the "not found" hint.
fn available_bundles() -> Vec<String> {
    let mut out = Vec::new();
    for root in [bundles_dir(), "bundles".to_string()] {
        if let Ok(rd) = std::fs::read_dir(&root) {
            for e in rd.flatten() {
                if let Some(name) = e.file_name().to_str().map(String::from) {
                    if std::path::Path::new(&format!("{root}/{name}/{name}.fieldrun.json")).exists() {
                        out.push(name);
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Resolve a `--bundle` argument to a stem: an explicit `<raw>.fieldrun.json` if present, else the organized location
/// `convert` writes by default — the home cache `~/.cache/fieldrun/bundles/<raw>/<raw>` (or a legacy `bundles/<raw>/`
/// in the cwd) — else the raw value (load errors clearly).
fn resolve_bundle(raw: &str) -> String {
    if std::path::Path::new(&format!("{raw}.fieldrun.json")).exists() {
        return raw.to_string(); // explicit stem / path
    }
    // accept the bundle name OR the full HF repo id (org/name[@rev]) — convert names the bundle by the basename, so
    // resolve under the cache by both forms (and a legacy ./bundles).
    let name = raw.rsplit('/').next().unwrap_or(raw).split('@').next().unwrap_or(raw);
    for n in [raw, name] {
        for root in [bundles_dir(), "bundles".to_string()] {
            let cand = format!("{root}/{n}/{n}");
            if std::path::Path::new(&format!("{cand}.fieldrun.json")).exists() {
                return cand;
            }
        }
    }
    raw.to_string()
}

fn print_help() {
    let gpu = if cfg!(feature = "gpu") { "on" } else { "off (build --features gpu)" };
    let hub = if cfg!(feature = "hub") { "on" } else { "off (default; you built --no-default-features)" };
    print!(
        "fieldrun {ver} — run a decompiled LLM as a single native binary (pure-Rust, no framework at runtime).\n\
\n\
USAGE\n\
  fieldrun convert --model <dir|hf-repo-id> --arch <arch> [--dtype int8|f16|f32] -o <stem>\n\
  fieldrun --bundle <stem> --ids <ids.json> [--ctx N] [--n-eval N]        score next-token top-1 (Tier B)\n\
  fieldrun --bundle <stem> --ids <ids.json> --ctx N --generate M          greedy-generate M tokens\n\
  fieldrun --bundle <stem> --ids <ids.json> --ctx N --explain [--vocab vocab.json]   circuits + features\n\
  fieldrun --bundle <stem> --chat [--explain] [--raw]                     chat REPL (/explain, /format in-REPL)\n\
  fieldrun --bundle <stem> --serve <PORT>                                 HTTP API: token-id + OpenAI/Anthropic\n\
  fieldrun --store <store.json> --ids <ids.json>                          retrieval-only (Tier A)\n\
\n\
  --bundle takes a stem (<stem>.fieldrun.json) or a bare model name resolved under ~/.cache/fieldrun/bundles/<name>/.\n\
\n\
  --ids expects {{\"holdout_ids\": [<token ids>]}} from the model's tokenizer.\n\
\n\
CONVERT  (Hugging Face safetensors -> bundle, no torch)\n\
  --model <X>     local checkpoint dir, OR a HF repo id like Qwen/Qwen3-30B-A3B (org/name[@revision])   [hub: {hub}]\n\
  --arch <A>      gpt2 | neox (Pythia/GPT-NeoX) | rope (Llama/Qwen2.5/Mistral/Phi) | gemma | gemma3 | gemma4 | qwen3moe | mla (DeepSeek/Kimi) | minimax\n\
  --dtype <D>     int4 (group-wise Q4, smallest) | int8 (default, + expert-offload for MoE) | f16 | f32 (bit-exact)\n\
  -o, --out <S>   output bundle stem (default: ~/.cache/fieldrun/bundles/<name>/<name>, + a .tokenizer.json)\n\
  --hf-token <T>  token for gated models (else $HF_TOKEN, else `huggingface-cli login`)\n\
  --force         re-convert even if the bundle already exists (default: skip)\n\
\n\
RUN\n\
  --bundle <S>    the .fieldrun bundle stem to load          --ctx N         context window / prediction (default 64)\n\
  --n-eval N      positions to score (default 500)           --generate M    greedy-generate M tokens (KV-cache where wired)\n\
  --kv-int8       int8 KV cache during generate              --route-frac F  Tier C: compute only fraction F of MLP neurons\n\
  --explain       with --ids: explain that prediction;       --vocab <f>     gpt2 vocab.json for readable explain labels\n\
  \x20               in chat: per-reply explanations (toggle /explain on|off)\n\
  --serve <PORT>  start the HTTP API (--server also works)   --dump <f>      write predictions, one id per line\n\
  --pruned-head   serve/chat decode: margin-gated retrieval-pruned unembed (needs --store; rope arch). Scores only\n\
  \x20               the KB's ~540 candidate rows; falls back to the full head when the in-set normalized margin is\n\
  \x20               below --pruned-margin M (default 2.0). Opt-in + lossy: calibrate with --gate-check N (generates\n\
  \x20               N gated tokens vs the full head, reports the identical prefix + accept rate; --gate-prompts P\n\
  \x20               spreads P prompts across the --ids stream).\n\
  --raw           chat: stream raw text, no Markdown render   --max-tokens N  reply cap (default 512; 2048 if reasoning)\n\
  --device cpu|gpu|auto   --max-vram <GB>  override the RAM-fit budget (default: detected system RAM)   GPU: {gpu}\n\
\n\
LOGIC EXPORT  (LOGIC_EXPORT.md — the model as a semiring-Datalog program; rope arch, needs --ids)\n\
  export --logic [--out f.dl]   emit ONE next-token decision as a runnable .dl (Soufflé / `fieldrun eval`)\n\
  export --logic-whole [--out f.dl] [--maxpos N]  emit the CONTEXT-FREE WHOLE-MODEL forward pass — one .dl that\n\
  \x20             computes the next token for ANY token(pos,id) input (LO3a). Small rope bundles. Run: souffle f.dl -F <ctxdir> -D -\n\
  --export-logic <prefix>       emit a decode TRACE: one .dl per step (prefix.000.dl …); count via --steps N (default 8)\n\
  --candidates N  candidate-set cap (default 48)             --store <f>     add KB n-gram facts (Tier A)\n\
  eval <prog.dl> [--semiring max|log]   run an emitted program — max → greedy decode (T=0), log → distribution (T=1)\n\
  stitch <step.dl …> [-o out.dl]        merge per-step programs into ONE step-indexed .dl: decide(Step,T) over the trace\n\
  (in --chat: /export-logic [file.dl] <prompt> exports the WHOLE reply as a per-step trace, on demand)\n",
        ver = env!("CARGO_PKG_VERSION"), hub = hub, gpu = gpu
    );
}

fn dump_if(args: &[String], preds: &[i64]) {
    if let Some(path) = flag(args, "--dump") {
        let out: String = preds.iter().map(|p| format!("{p}\n")).collect();
        match std::fs::write(path, out) {
            Ok(()) => eprintln!("[fieldrun] wrote {} predictions to {path}", preds.len()),
            Err(e) => eprintln!("[fieldrun] couldn't write --dump {path}: {e}"),
        }
    }
}

/// Build an id→string decoder. With a GPT-2 `vocab.json` (token→id), invert it and show the raw BPE token (Ġ→space,
/// Ċ→newline for readability); without one, fall back to `[id]`.
/// Emit a step-indexed CONTRIB-OVER-EXPERT Datalog program: each decision's scored circuits' DLA contributions to the
/// candidate tokens, grouped by their corpus-expert, plus a catchall "rest" so Σ contrib == logit (faithful by
/// construction). Runs in `fieldrun eval --semiring max` → decode == the model's token at every step (the COMPOSITION,
/// not a lookup). The header reports the per-expert share of the winning margin + the catchall fraction (the
/// compactness / forge-tax meter). Recovers c_j^t = dla and c_j^v = dla − margins[v] from the descent substrate.
fn emit_contrib_dl(lm: &dyn model::Model, positions: &[&[i64]], k: usize, e_act: usize, expert_of: &HashMap<(u8, usize, usize), usize>, dec: &dyn Fn(i64) -> String) -> String {
    use std::fmt::Write as _;
    let blk = |e: usize| if e == e_act { "residual".to_string() } else { format!("e{e}") };
    let (mut steps, mut faithful) = (0usize, 0usize);
    let mut margin_by_block = vec![0f64; e_act + 1]; // Σ over steps of (contrib(block,t) − contrib(block,v*))
    let (mut margin_rest, mut margin_total) = (0f64, 0f64);
    let mut body = String::new();
    for c in positions {
        let sub = match lm.explain_decomp(c, k).and_then(|e| e.decomp) {
            Some(s) if !s.competitors.is_empty() => s,
            _ => continue,
        };
        let t = sub.predicted;
        let comp_idx: HashMap<i64, usize> = sub.competitors.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let cands: Vec<i64> = std::iter::once(t).chain(sub.competitors.iter().copied()).collect();
        let nb = e_act + 1;
        // contrib[block][cand]: c_j^t = dla, c_j^v = dla − margins[v]; each scored circuit added to its expert's block.
        let mut contrib = vec![vec![0f64; cands.len()]; nb];
        for s in &sub.sources {
            let e = *expert_of.get(&(s.kind, s.layer, s.idx)).unwrap_or(&e_act);
            for (ci, &u) in cands.iter().enumerate() {
                contrib[e][ci] += if u == t { s.dla as f64 } else { (s.dla - s.margins[comp_idx[&u]]) as f64 };
            }
        }
        // catchall rest[cand] = logit(u) − Σ_scored, with logit(t)=0, logit(v)=−full_margin[v] ⇒ Σ_block + rest == logit.
        let rest: Vec<f64> = cands.iter().enumerate().map(|(ci, &u)| {
            let logit_u = if u == t { 0.0 } else { -(sub.full_margin[comp_idx[&u]] as f64) };
            logit_u - (0..nb).map(|b| contrib[b][ci]).sum::<f64>()
        }).collect();
        // faithful decode check: argmax_cand (Σ_block contrib + rest) must be t.
        let total: Vec<f64> = (0..cands.len()).map(|ci| (0..nb).map(|b| contrib[b][ci]).sum::<f64>() + rest[ci]).collect();
        let argmax = (0..cands.len()).max_by(|&a, &b| total[a].partial_cmp(&total[b]).unwrap()).unwrap();
        if cands[argmax] == t { faithful += 1; }
        // margin attribution vs the runner-up v* (smallest full_margin = closest competitor).
        let vstar = (0..sub.competitors.len()).min_by(|&a, &b| sub.full_margin[a].partial_cmp(&sub.full_margin[b]).unwrap()).unwrap();
        let vci = 1 + vstar;
        for b in 0..nb {
            margin_by_block[b] += contrib[b][0] - contrib[b][vci];
        }
        margin_rest += rest[0] - rest[vci];
        margin_total += total[0] - total[vci];
        let _ = writeln!(body, "// step {steps}: decode {:?} (margin {:+.3} vs {:?})", dec(t), sub.full_margin[vstar], dec(sub.competitors[vstar]));
        for &u in &cands {
            let _ = writeln!(body, "candidate({steps},{u}).   // {:?}", dec(u));
        }
        for b in 0..nb {
            for (ci, &u) in cands.iter().enumerate() {
                if contrib[b][ci].abs() > 1e-6 {
                    let _ = writeln!(body, "contrib({steps},\"{}\",{u},{:.4}).", blk(b), contrib[b][ci]);
                }
            }
        }
        for (ci, &u) in cands.iter().enumerate() {
            let _ = writeln!(body, "contrib({steps},\"rest\",{u},{:.4}).", rest[ci]);
        }
        steps += 1;
    }
    let mut out = String::new();
    let _ = writeln!(out, "// fieldrun EXPERTS-DL-CONTRIB — composition decode over the partition (NOT a lookup).");
    let _ = writeln!(out, "// Each step: per-expert Σ contrib to the candidate tokens + a catchall \"rest\" so Σ == logit;");
    let _ = writeln!(out, "// decode(step) = argmax_token Σ contrib = the model's own token (faithful). Run:");
    let _ = writeln!(out, "//   fieldrun eval <this>.dl --semiring max   (argmax decode)   |   --semiring log   (softmax dist)");
    let _ = writeln!(out, "//");
    let _ = writeln!(out, "// {steps} steps · faithful decode {faithful}/{steps} ({:.0}%)", if steps > 0 { 100.0 * faithful as f32 / steps as f32 } else { 0.0 });
    if margin_total.abs() > 1e-9 {
        let _ = writeln!(out, "// per-expert share of the winning margin (t vs runner-up), summed over steps — the compactness meter:");
        for b in 0..=e_act {
            let _ = writeln!(out, "//   {:<9} {:+8.2}  ({:>3.0}% of margin)", blk(b), margin_by_block[b], 100.0 * margin_by_block[b] / margin_total);
        }
        let _ = writeln!(out, "//   {:<9} {:+8.2}  ({:>3.0}% of margin)  ← catchall: forge-tax / non-compact remainder", "rest", margin_rest, 100.0 * margin_rest / margin_total);
    }
    let _ = writeln!(out, "\n.decl candidate(step:number, t:number)\n.decl contrib(step:number, block:symbol, t:number, w:float)");
    out.push_str(&body);
    out
}

fn load_decoder(vocab: Option<&str>) -> Box<dyn Fn(i64) -> String> {
    if let Some(path) = vocab {
        if let Ok(txt) = std::fs::read_to_string(path) {
            let map: HashMap<String, i64> = serde_json::from_str(&txt).unwrap_or_default();
            let inv: HashMap<i64, String> = map.into_iter().map(|(k, v)| (v, k)).collect();
            return Box::new(move |id| {
                inv.get(&id)
                    .map(|s| format!("{:?} [{id}]", s.replace('\u{0120}', " ").replace('\u{010A}', "\n")))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_parsing() {
        let a: Vec<String> = ["fr", "--arch", "rope", "--chat"].iter().map(|s| s.to_string()).collect();
        assert_eq!(flag(&a, "--arch"), Some("rope"));
        assert_eq!(flag(&a, "--missing"), None);
        assert!(has_flag(&a, "--chat"));
        assert!(!has_flag(&a, "--nope"));
    }

    #[test]
    fn resolve_bundle_explicit_cache_and_repo_id() {
        let dir = std::env::temp_dir().join(format!("fr_rb_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // explicit stem: <stem>.fieldrun.json exists -> returned as-is
        let st = dir.join("x").to_string_lossy().into_owned();
        std::fs::write(format!("{st}.fieldrun.json"), "{}").unwrap();
        assert_eq!(resolve_bundle(&st), st);
        // cache: bare name + full repo id both resolve to <cache>/<name>/<name>
        std::env::set_var("FIELDRUN_BUNDLES", &dir);
        let mdir = dir.join("Qwen2.5-0.5B-Instruct");
        std::fs::create_dir_all(&mdir).unwrap();
        let bstem = mdir.join("Qwen2.5-0.5B-Instruct").to_string_lossy().into_owned();
        std::fs::write(format!("{bstem}.fieldrun.json"), "{}").unwrap();
        assert_eq!(resolve_bundle("Qwen2.5-0.5B-Instruct"), bstem);
        assert_eq!(resolve_bundle("Qwen/Qwen2.5-0.5B-Instruct"), bstem);
        assert_eq!(resolve_bundle("does-not-exist"), "does-not-exist"); // passthrough
        std::env::remove_var("FIELDRUN_BUNDLES");
    }
}
