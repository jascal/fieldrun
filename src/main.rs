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
mod bundle;
#[cfg(feature = "jit")]
mod jit;
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
        const ARCHS: &[&str] = &["gpt2", "rope", "gemma", "gemma3", "gemma4", "qwen3moe", "mla", "minimax", "dsv4"];
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
            "rope" => Box::new(Rope::new(bundle, route, kv_int8)),
            "gemma" => Box::new(Gemma::new(bundle, route, kv_int8)),
            "gemma3" => Box::new(Gemma3::new(bundle, route, kv_int8)),
            "gemma4" => Box::new(Gemma4::new(bundle, route, kv_int8)),
            "qwen3moe" => Box::new(Qwen3Moe::new(bundle, route, kv_int8)),
            "mla" => Box::new(Mla::new(bundle, route, kv_int8)),
            "minimax" => Box::new(MiniMax::new(bundle, route, kv_int8)),
            "dsv4" => Box::new(Dsv4::new(bundle, route, kv_int8)),
            other => panic!("unknown bundle arch {other:?} (have: gpt2, rope, gemma, gemma3, gemma4, qwen3moe, mla, minimax, dsv4)"),
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
            let cap = (end - ctx_window).min(n_eval).min(150); // 1 explain + 3 ablated forwards / position
            let positions: Vec<&[i64]> = (ctx_window..ctx_window + cap).map(|i| ctx(i)).collect();
            eprintln!("[fieldrun] --probe-ablate: {} positions — explain + ablated forwards…", positions.len());
            let ks = [1usize, 2, 4];
            // (route, margin, [flip@k1, flip@k2, flip@k4])
            let recs: Vec<(u8, f32, [bool; 3])> = positions.par_iter().filter_map(|c| {
                let ex = lm.explain(c)?;
                let t = ex.model_predicts;
                let mut circ: Vec<(f32, bool, usize, usize)> = ex.head_circuits.iter().map(|h| (h.dla, true, h.layer, h.head))
                    .chain(ex.mlp_features.iter().map(|m| (m.dla, false, m.layer, m.neuron))).collect();
                circ.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let (kb, _) = store.predict(c);
                let covered = store.candidates(c, &cfg).contains(&t);
                let route = if kb == t { 0u8 } else if covered { 1 } else { 2 };
                let mut flips = [false; 3];
                for (ki, &k) in ks.iter().enumerate() {
                    let topk = &circ[..k.min(circ.len())];
                    let heads: Vec<(usize, usize)> = topk.iter().filter(|c| c.1).map(|c| (c.2, c.3)).collect();
                    let neurons: Vec<(usize, usize)> = topk.iter().filter(|c| !c.1).map(|c| (c.2, c.3)).collect();
                    flips[ki] = lm.predict_ablated(c, &heads, &neurons) != Some(t);
                }
                Some((route, ex.predicted_logit - ex.runner_up_logit, flips))
            }).collect();
            println!("\n=== (causal) ablate the top-k DLA circuits in the forward pass → does the prediction FLIP? ===");
            println!("{:<12}{:>6}{:>10}{:>12}{:>12}{:>12}", "route", "n", "margin", "flip@k1", "flip@k2", "flip@k4");
            for (lbl, r) in [("RETRIEVED", 0u8), ("SELECTED", 1), ("COMPOSED", 2)] {
                let g: Vec<&(u8, f32, [bool; 3])> = recs.iter().filter(|x| x.0 == r).collect();
                if g.is_empty() { println!("{lbl:<12}{:>6}", 0); continue; }
                let f = |ki: usize| 100.0 * g.iter().filter(|x| x.2[ki]).count() as f32 / g.len() as f32;
                let mm = g.iter().map(|x| x.1).sum::<f32>() / g.len() as f32;
                println!("{lbl:<12}{:>6}{:>10.2}{:>11.0}%{:>11.0}%{:>11.0}%", g.len(), mm, f(0), f(1), f(2));
            }
            // de-confound margin: within matched margin bins, does coverage predict robustness (lower flip@k1)?
            let mut ms: Vec<f32> = recs.iter().map(|x| x.1).collect();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f32| if ms.is_empty() { 0.0 } else { ms[(((ms.len() - 1) as f32) * p) as usize] };
            let (t1, t2) = (q(0.333), q(0.667));
            println!("\n  (de-confound) flip@k1 WITHIN matched margin bins — COVERED (R+S) vs COMPOSED:");
            println!("  {:<12}{:>12}{:>20}{:>20}", "margin bin", "mean margin", "COVERED n flip%", "COMPOSED n flip%");
            for (lbl, lo, hi) in [("low ", f32::MIN, t1), ("mid ", t1, t2), ("high", t2, f32::MAX)] {
                let inb = |x: &&(u8, f32, [bool; 3])| x.1 >= lo && x.1 < hi;
                let cov: Vec<&(u8, f32, [bool; 3])> = recs.iter().filter(|x| inb(x) && x.0 != 2).collect();
                let cmp: Vec<&(u8, f32, [bool; 3])> = recs.iter().filter(|x| inb(x) && x.0 == 2).collect();
                let fl = |g: &[&(u8, f32, [bool; 3])]| if g.is_empty() { f32::NAN } else { 100.0 * g.iter().filter(|x| x.2[0]).count() as f32 / g.len() as f32 };
                let mm = recs.iter().filter(inb).map(|x| x.1).sum::<f32>() / recs.iter().filter(inb).count().max(1) as f32;
                println!("  {lbl:<12}{mm:>12.2}{:>14} {:>4.0}%{:>14} {:>4.0}%", cov.len(), fl(&cov), cmp.len(), fl(&cmp));
            }
            println!("⇒ if COVERED flip% < COMPOSED flip% WITHIN a margin bin, redundancy is causal BEYOND confidence.");
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
                    api::chat(lm, tg, max_tokens, explain, kb, cand, has_flag(&args, "--raw"), &arch);
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
  --arch <A>      gpt2 | rope (Llama/Qwen2.5/Mistral/Phi) | gemma | gemma3 | gemma4 | qwen3moe | mla (DeepSeek/Kimi) | minimax\n\
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
  --device cpu|gpu|auto   --max-vram <GB>  override the RAM-fit budget (default: detected system RAM)   GPU: {gpu}\n",
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
