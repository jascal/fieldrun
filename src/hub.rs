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
    // stream to the .part file; on ANY failure (e.g. ENOSPC) delete it so a half-download doesn't keep eating disk.
    // immediately-invoked closure so its &mut borrows of `reader`/`f` release before we drop/rename below.
    let result: Result<u64, String> = (|| {
        let mut buf = vec![0u8; 1 << 20];
        let (mut got, mut last) = (0u64, 0u64);
        loop {
            let n = reader.read(&mut buf).map_err(|e| format!("{name}: read: {e}"))?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n]).map_err(|e| format!("{name}: write: {e}"))?;
            got += n as u64;
            if got - last > (64 << 20) {
                if total > 0 {
                    eprint!("\r[hub] {name}: {} / {} MB", got >> 20, total >> 20);
                } else {
                    eprint!("\r[hub] {name}: {} MB", got >> 20);
                }
                last = got;
            }
        }
        Ok(got)
    })();
    let done = match result {
        Ok(n) => n,
        Err(e) => {
            drop(f);
            let _ = std::fs::remove_file(&tmp); // don't leave a partial download consuming space
            return Err(e);
        }
    };
    drop(f);
    std::fs::rename(&tmp, dest).map_err(|e| format!("{name}: finalize: {e}"))?;
    eprintln!("\r[hub] {name}: {} MB ✓                ", done >> 20);
    Ok(())
}

/// Fetch `repo_id`'s config + safetensors to a local cache dir and return that dir (what `convert` opens). Sharded
/// models pull every shard listed in the index; single-file models pull `model.safetensors`. `revision` defaults to
/// `main`. Files already present are reused (a simple cache).
///
/// On failure, the cache dirs this call left EMPTY are removed again (`<org>--<name>/<rev>`, then `<org>--<name>`), so a
/// mistyped repo id or a bad path doesn't leave stray folders in the cache. `remove_dir` only succeeds on an empty dir,
/// so already-cached files are never touched.
pub fn fetch(repo_id: &str, token: Option<String>) -> Result<String, String> {
    let (repo, revision) = match repo_id.split_once('@') {
        Some((r, rev)) => (r, rev),
        None => (repo_id, "main"),
    };
    let dir = cache_root().join(repo.replace('/', "--")).join(revision);
    let result = fetch_into(&dir, repo, revision, token);
    if result.is_err() {
        remove_if_empty(&dir);
    }
    result
}

/// Remove `dir` and then its parent, each only if empty (best-effort; errors ignored).
fn remove_if_empty(dir: &Path) {
    if std::fs::remove_dir(dir).is_ok() {
        if let Some(parent) = dir.parent() {
            let _ = std::fs::remove_dir(parent);
        }
    }
}

/// True if `--model` was written as a filesystem path (absolute, `./`, `../` or `~`), not as a hub repo id like
/// `org/name`. Such a value that lacks a `config.json` is a missing local checkpoint, not something to download.
pub fn looks_like_path(model: &str) -> bool {
    model.starts_with('/') || model.starts_with("./") || model.starts_with("../") || model.starts_with('~')
        || model == "." || model == ".."
}

fn fetch_into(dir: &Path, repo: &str, revision: &str, token: Option<String>) -> Result<String, String> {
    let base = format!("https://huggingface.co/{repo}/resolve/{revision}");
    std::fs::create_dir_all(dir).map_err(|e| format!("[hub] mkdir {}: {e}", dir.display()))?;

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

    get("config.json").map_err(|e| format!("[hub] {repo}: {e}"))?;
    let _ = get("tokenizer.json"); // best-effort — needed for the OpenAI/Anthropic text API + --chat; absent on some repos
    // sharded if there's an index; else a single safetensors
    match get("model.safetensors.index.json") {
        Ok(()) => {
            let txt = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
                .map_err(|e| format!("[hub] {repo}: read index.json: {e}"))?;
            let idx: serde_json::Value =
                serde_json::from_str(&txt).map_err(|e| format!("[hub] {repo}: parse index.json: {e}"))?;
            let wm = idx["weight_map"].as_object().ok_or("[hub] no weight_map in index.json")?;
            let mut files: Vec<String> = wm.values().filter_map(|f| f.as_str().map(String::from)).collect();
            files.sort();
            files.dedup();
            eprintln!("[hub] {repo} ({revision}): {} shard(s)", files.len());
            for f in &files {
                get(f).map_err(|e| format!("[hub] {repo}: {e}"))?;
            }
        }
        Err(_) => {
            eprintln!("[hub] {repo} ({revision}): single safetensors");
            get("model.safetensors").map_err(|e| format!(
                "[hub] {repo}: no model.safetensors.index.json or model.safetensors (only safetensors checkpoints are supported): {e}"))?;
        }
    }
    Ok(dir.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_like_models_are_not_repo_ids() {
        for p in ["/tmp/nope", "./ckpt", "../ckpt", "~/models/x", ".", ".."] {
            assert!(looks_like_path(p), "{p} should be treated as a path");
        }
        for r in ["Qwen/Qwen2.5-0.5B", "gpt2", "EleutherAI/pythia-160m@step1000"] {
            assert!(!looks_like_path(r), "{r} should be treated as a repo id");
        }
    }

    #[test]
    fn remove_if_empty_keeps_nonempty_dirs() {
        let root = std::env::temp_dir().join(format!("fieldrun-hub-test-{}", std::process::id()));
        // empty rev dir under an empty repo dir: both go
        let empty = root.join("org--empty").join("main");
        std::fs::create_dir_all(&empty).unwrap();
        remove_if_empty(&empty);
        assert!(!root.join("org--empty").exists());
        // rev dir holding a cached file: nothing is removed
        let full = root.join("org--full").join("main");
        std::fs::create_dir_all(&full).unwrap();
        std::fs::write(full.join("config.json"), "{}").unwrap();
        remove_if_empty(&full);
        assert!(full.join("config.json").exists());
        // empty rev dir next to another cached revision: the rev dir goes, the repo dir stays
        let other = root.join("org--full").join("step1");
        std::fs::create_dir_all(&other).unwrap();
        remove_if_empty(&other);
        assert!(!other.exists() && full.exists());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
