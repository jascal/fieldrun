//! The runtime kernel interface — one decompiled-LLM forward pass, dispatched by `arch` in the bundle manifest.
//! Every kernel (GPT-2, RoPE family, Gemma-2) mirrors its pylm numpy reference and is held behind this trait so the
//! scoring loop is architecture-agnostic. `Sync` so rayon can fan independent forwards across cores.

use ndarray::{s, Array2};

/// A single-slot KV cache carried across `generate_stream` calls so a growing chat reuses the K/V of its common prefix
/// instead of re-prefilling the whole context every turn (prefix caching — the dominant per-turn cost in multi-turn
/// serving). `ids` are the tokens whose K/V occupy rows `[0..ids.len())` of every `kc[l]`/`vc[l]`. Architecture-agnostic:
/// the per-layer K/V column widths live inside the `Array2`s, so the server holds it opaquely. Empty = cold start.
#[derive(Default)]
pub struct PrefixKv {
    pub ids: Vec<i64>,
    pub kc: Vec<Array2<f32>>,
    pub vc: Vec<Array2<f32>>,
}

impl PrefixKv {
    pub fn clear(&mut self) {
        self.ids.clear();
        self.kc.clear();
        self.vc.clear();
    }

    /// Longest common prefix length between the cached ids and `prompt`, capped at `prompt.len()-1` so there is always
    /// at least one suffix token left to prefill (a forward pass over it produces the next-token logits).
    pub fn reuse_len(&self, prompt: &[i64]) -> usize {
        let mut l = 0;
        while l < self.ids.len() && l < prompt.len() && self.ids[l] == prompt[l] {
            l += 1;
        }
        l.min(prompt.len().saturating_sub(1))
    }
}

/// Shared driver for prefix-cached greedy streaming — every KV-cache kernel's `generate_stream_prefix` delegates here.
/// The arch supplies three closures so this stays architecture-agnostic:
///   * `alloc(total)` → zeroed `(kc, vc)` with that arch's per-layer K/V column widths, sized for `total` rows;
///   * `fwd(ids, cur, &mut kc, &mut vc)` → run the block over `ids` placed at absolute position `cur` (writing their
///      K/V into rows `[cur..cur+ids.len())` and attending to the reused prefix `[0..cur)`), returning the hidden block;
///   * `argmax(&hidden)` → next-token id from the last row.
///
/// Reuse is byte-identical to a full recompute — K/V are deterministic functions of the prefix tokens and the chunked
/// forward at `cur=L` attends to the copied prefix rows exactly as a fresh prefill would — so this runs unconditionally;
/// the validation suite gates `prefix == full-recompute`. On exit `cache` holds the full (prompt + emitted) K/V for the
/// next turn.
#[allow(clippy::too_many_arguments)]
pub fn prefix_generate(
    prompt: &[i64],
    max_tokens: usize,
    eos: &[i64],
    emit: &mut dyn FnMut(i64) -> bool,
    cache: &mut PrefixKv,
    n_layer: usize,
    alloc: &dyn Fn(usize) -> (Vec<Array2<f32>>, Vec<Array2<f32>>),
    fwd: &mut dyn FnMut(&[i64], usize, &mut [Array2<f32>], &mut [Array2<f32>]) -> Array2<f32>,
    argmax: &dyn Fn(&Array2<f32>) -> i64,
) -> Vec<i64> {
    let total = prompt.len() + max_tokens;
    let l = cache.reuse_len(prompt);
    let (mut kc, mut vc) = alloc(total);
    if l > 0 {
        for layer in 0..n_layer {
            kc[layer].slice_mut(s![0..l, ..]).assign(&cache.kc[layer].slice(s![0..l, ..]));
            vc[layer].slice_mut(s![0..l, ..]).assign(&cache.vc[layer].slice(s![0..l, ..]));
        }
    }
    // Prefill the suffix in one chunked forward at position l (attends to the reused prefix rows [0..l)).
    let xb = fwd(&prompt[l..], l, &mut kc, &mut vc);
    let mut next = argmax(&xb);
    let mut out = Vec::new();
    let mut pos = prompt.len();
    loop {
        if eos.contains(&next) {
            break;
        }
        out.push(next);
        if !emit(next) || out.len() == max_tokens {
            break;
        }
        let xb = fwd(&[next], pos, &mut kc, &mut vc);
        next = argmax(&xb);
        pos += 1;
    }
    // Persist the full (prompt + emitted) K/V for the next turn — exactly `pos` rows are populated (the decode loop
    // writes row `pos` then increments, so a break before the write leaves `pos` valid rows); truncate ids to match.
    let mut ids: Vec<i64> = Vec::with_capacity(pos);
    ids.extend_from_slice(prompt);
    ids.extend_from_slice(&out);
    ids.truncate(pos);
    cache.ids = ids;
    cache.kc = kc;
    cache.vc = vc;
    out
}

pub trait Model: Sync {
    /// Top-1 next-token id for a context.
    fn predict(&self, ids: &[i64]) -> i64;

    /// Explain the prediction (composition-side circuits + features). Default None; GPT-2 implements it.
    fn explain(&self, _ids: &[i64]) -> Option<crate::explain::Explanation> {
        None
    }

    /// Greedy generation up to `max_tokens`, stopping early at any `eos` id (the stop token is NOT included in the
    /// output). `emit(id)` is called for each generated token *as it is produced* (for streaming / a live chat);
    /// returning `false` (e.g. the HTTP client disconnected) stops generation. Default: naive — recompute the whole
    /// context every token (O(n)/token); kernels with a KV-cache override this for O(1)/token decode.
    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let mut ctx = prompt.to_vec();
        let mut out = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let t = self.predict(&ctx);
            if eos.contains(&t) {
                break;
            }
            out.push(t);
            if !emit(t) {
                break;
            }
            ctx.push(t);
        }
        out
    }

    /// Fixed-length greedy generation (exactly `n_new` tokens, no early stop) — used by the CLI `--generate`
    /// KV-cache-vs-naive benchmark. KV-cache kernels override for speed; otherwise delegates to `generate_stream`.
    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        self.generate_stream(prompt, n_new, &[], &mut |_| true)
    }

    /// `generate_stream` with prefix-KV reuse across calls — the streaming/serve/REPL chat path threads a single
    /// `PrefixKv` here so a growing conversation only prefills the new suffix each turn. Default: no reuse — clear the
    /// cache and run the stateless path (correct, just no speedup). KV-cache kernels with an f32 `Vec<Array2<f32>>`
    /// cache override via `prefix_generate`; the int8-KV path also falls back here (its cache layout differs).
    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool, cache: &mut PrefixKv) -> Vec<i64> {
        cache.clear();
        self.generate_stream(prompt, max_tokens, eos, emit)
    }
}
