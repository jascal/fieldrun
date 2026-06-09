//! Optional Hugging Face pull (default feature `hub`): fetch a model's `config.json` + safetensors (single-file or
//! sharded) by repo id, so `fieldrun convert --model org/repo …` works without a manual download. We download with a
//! small `ureq` client directly (no torch, no async) into a local cache dir and hand `convert` that directory, so it
//! then reads the files exactly like a local `--model <dir>`.
//!
//! Why we don't use the `hf-hub` crate: current HF returns a *relative* 307 redirect (`/api/resolve-cache/…`) for
//! files; hf-hub 0.3 fed that straight to its client and failed ("relative URL without a base"). `ureq`'s redirect
//! follower resolves relative `Location`s against the request base, so a plain GET handles the redirect chain
//! (huggingface.co → resolve-cache → signed CDN) correctly.
//!
//! Auth: token only (HF OAuth is a browser/web-app flow, wrong for a CLI). We add NO login of our own — we read the
//! standard token: `--hf-token` > `$HF_TOKEN` > `~/.cache/huggingface/token` (written by `huggingface-cli login`).
//! Most amateur-relevant models are ungated (no token); gated ones (Gemma, official Llama) need a one-time
//! `huggingface-cli login`, and a 401/403 prints exactly that hint. The token authorises the huggingface.co request;
//! ureq drops it on the cross-origin CDN redirect (correct — that URL is pre-signed).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Resolve the standard HF token: explicit flag, else `$HF_TOKEN`, else the `huggingface-cli login` cache file.
pub fn token(explicit: Option<&str>) -> Option<String> {
    if let Some(t) = explicit {
        return Some(t.to_string());
    }
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.trim().is_empty() {
            return Some(t.trim().to_string());
        }
    }
    let home = std::env::var("HOME").ok()?;
    std::fs::read_to_string(format!("{home}/.cache/huggingface/token"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn cache_root() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/fieldrun/hub")
}

/// GET `url` → `dest` (streaming, with a coarse progress line), following redirects. Adds the bearer token if given.
fn download(url: &str, dest: &Path, token: &Option<String>, name: &str) -> Result<(), String> {
    let mut req = ureq::get(url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req.call().map_err(|e| match e {
        ureq::Error::Status(401, _) | ureq::Error::Status(403, _) => format!(
            "{name}: HTTP {} — gated or unauthorized. Run `huggingface-cli login` (or set HF_TOKEN) and accept the \
             model's terms on huggingface.co.",
            if let ureq::Error::Status(c, _) = &e { *c } else { 0 }
        ),
        ureq::Error::Status(404, _) => format!("{name}: HTTP 404 — not found (check the repo id / filename)"),
        other => format!("{name}: {other}"),
    })?;
    let total: u64 = resp.header("Content-Length").and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut reader = resp.into_reader();
    let tmp = dest.with_extension("part");
    let mut f = std::fs::File::create(&tmp).map_err(|e| format!("{name}: create {}: {e}", tmp.display()))?;
    let mut buf = vec![0u8; 1 << 20];
    let (mut done, mut last) = (0u64, 0u64);
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("{name}: read: {e}"))?;
        if n == 0 {
            break;
        }
        f.write_all(&buf[..n]).map_err(|e| format!("{name}: write: {e}"))?;
        done += n as u64;
        if done - last > (64 << 20) {
            if total > 0 {
                eprint!("\r[hub] {name}: {} / {} MB", done >> 20, total >> 20);
            } else {
                eprint!("\r[hub] {name}: {} MB", done >> 20);
            }
            last = done;
        }
    }
    drop(f);
    std::fs::rename(&tmp, dest).map_err(|e| format!("{name}: finalize: {e}"))?;
    eprintln!("\r[hub] {name}: {} MB ✓                ", done >> 20);
    Ok(())
}

/// Fetch `repo_id`'s config + safetensors to a local cache dir and return that dir (what `convert` opens). Sharded
/// models pull every shard listed in the index; single-file models pull `model.safetensors`. `revision` defaults to
/// `main`. Files already present are reused (a simple cache).
pub fn fetch(repo_id: &str, token: Option<String>) -> String {
    let (repo, revision) = match repo_id.split_once('@') {
        Some((r, rev)) => (r, rev),
        None => (repo_id, "main"),
    };
    let base = format!("https://huggingface.co/{repo}/resolve/{revision}");
    let dir = cache_root().join(repo.replace('/', "--")).join(revision);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("[hub] mkdir {}: {e}", dir.display()));

    let get = |file: &str| -> Result<(), String> {
        let dest = dir.join(file);
        if dest.exists() {
            eprintln!("[hub] {file}: cached");
            return Ok(());
        }
        if let Some(p) = dest.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        download(&format!("{base}/{file}"), &dest, &token, file)
    };

    get("config.json").unwrap_or_else(|e| panic!("[hub] {repo}: {e}"));
    let _ = get("tokenizer.json"); // best-effort — needed for the OpenAI/Anthropic text API + --chat; absent on some repos
    // sharded if there's an index; else a single safetensors
    match get("model.safetensors.index.json") {
        Ok(()) => {
            let idx: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap()).unwrap();
            let wm = idx["weight_map"].as_object().expect("[hub] weight_map in index.json");
            let mut files: Vec<String> = wm.values().filter_map(|f| f.as_str().map(String::from)).collect();
            files.sort();
            files.dedup();
            eprintln!("[hub] {repo} ({revision}): {} shard(s)", files.len());
            for f in &files {
                get(f).unwrap_or_else(|e| panic!("[hub] {repo}: {e}"));
            }
        }
        Err(_) => {
            eprintln!("[hub] {repo} ({revision}): single safetensors");
            get("model.safetensors").unwrap_or_else(|e| panic!(
                "[hub] {repo}: no model.safetensors.index.json or model.safetensors (only safetensors checkpoints are supported): {e}"));
        }
    }
    dir.to_string_lossy().into_owned()
}
