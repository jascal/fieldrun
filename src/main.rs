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
        let lm: Box<dyn Model> = match arch.as_str() {
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

        // --serve / --server <PORT> (accept both spellings — a common typo). The API server (no ids needed).
        let serve_port = flag(&args, "--serve")
            .or_else(|| flag(&args, "--server"))
            .and_then(|s| s.parse::<u16>().ok());
        // --explain: in the chat REPL it turns ON per-reply explanations (toggle live with /explain). With --ids it's
        // the standalone "explain this prediction" mode. (It used to force standalone mode and panic without --ids.)
        let explain = has_flag(&args, "--explain");

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
                    api::chat(lm, tg, max_tokens, explain, has_flag(&args, "--raw"), &arch);
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
            if explain {
                eprintln!("[fieldrun] note: --explain toggles per-reply explanations in the chat REPL. The API server \
                           ignores it — POST /explain for the structured form, or pass \"explain\":true to the chat \
                           endpoints. (Use --chat --explain for an explained REPL.)");
            }
            #[cfg(feature = "api")]
            let textgen = api::TextGen::load(&stem, eos);
            #[cfg(not(feature = "api"))]
            let textgen: Option<api::TextGen> = None;
            api::serve(lm, &arch, port, textgen);
            return;
        }

        // --explain WITH --ids: standalone "explain the prediction at the end of the first --ctx tokens" (circuits +
        // features). Without ids we'd have already gone to chat above; guard anyway so an empty stream can't index out.
        if explain {
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
