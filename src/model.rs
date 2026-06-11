//! The runtime kernel interface — one decompiled-LLM forward pass, dispatched by `arch` in the bundle manifest.
//! Every kernel (GPT-2, RoPE family, Gemma-2) mirrors its pylm numpy reference and is held behind this trait so the
//! scoring loop is architecture-agnostic. `Sync` so rayon can fan independent forwards across cores.

use ndarray::{s, Array2};

/// A single-slot KV cache carried across `generate_stream` calls so a growing chat reuses the K/V of its common prefix
/// instead of re-prefilling the whole context every turn (prefix caching — the dominant per-turn cost in multi-turn
/// serving). `ids` are the tokens whose K/V occupy rows `[0..ids.len())` of every cache layer. Holds EITHER the f32
/// cache (`kc`/`vc`, used by default) OR the int8 cache (`kc_q`/`vc_q` bytes + `ks_q`/`vs_q` per-head scales, used under
/// `--kv-int8` — 4× smaller AND prefix-reused, so a long chat is both memory- and latency-light). A given model uses one
/// set for its whole lifetime. Architecture-agnostic: per-layer widths live inside the rows, so the server holds it
/// opaquely. Empty = cold start.
#[derive(Default)]
pub struct PrefixKv {
    pub ids: Vec<i64>,
    pub kc: Vec<Array2<f32>>,
    pub vc: Vec<Array2<f32>>,
    pub kc_q: Vec<Vec<i8>>,
    pub vc_q: Vec<Vec<i8>>,
    pub ks_q: Vec<Vec<f32>>,
    pub vs_q: Vec<Vec<f32>>,
}

impl PrefixKv {
    pub fn clear(&mut self) {
        self.ids.clear();
        self.kc.clear();
        self.vc.clear();
        self.kc_q.clear();
        self.vc_q.clear();
        self.ks_q.clear();
        self.vs_q.clear();
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

/// `prefix_generate` for the int8 KV cache (`--kv-int8`): the cache is flat `Vec<i8>` K/V bytes (row-major, per-layer
/// width = nkv·hd) plus `Vec<f32>` per-head scales (per-layer width = nkv), so the prefix copy is a contiguous
/// `l·width` slice copy rather than an `Array2` row-slice. Reuse is byte-identical to a cold int8 prefill (the
/// quantisation is deterministic and the copied bytes are exactly what a fresh prefill would have produced). The arch
/// supplies an `alloc(total) → (kc, vc, ks, vs)` (zeroed at its per-layer widths) and an `fwd` wrapping `forward_block_q`.
#[allow(clippy::too_many_arguments)]
pub fn prefix_generate_q(
    prompt: &[i64],
    max_tokens: usize,
    eos: &[i64],
    emit: &mut dyn FnMut(i64) -> bool,
    cache: &mut PrefixKv,
    n_layer: usize,
    alloc: &dyn Fn(usize) -> (Vec<Vec<i8>>, Vec<Vec<i8>>, Vec<Vec<f32>>, Vec<Vec<f32>>),
    fwd: &mut dyn FnMut(&[i64], usize, &mut [Vec<i8>], &mut [Vec<f32>], &mut [Vec<i8>], &mut [Vec<f32>]) -> Array2<f32>,
    argmax: &dyn Fn(&Array2<f32>) -> i64,
) -> Vec<i64> {
    let total = prompt.len() + max_tokens;
    let l = cache.reuse_len(prompt);
    let (mut kc, mut vc, mut ks, mut vs) = alloc(total);
    if l > 0 {
        for layer in 0..n_layer {
            // Per-layer row widths derived from the fresh alloc. K and V widths can DIFFER (MLA: kdim=nh·qkh vs
            // vdim=nh·v_head), so compute each separately; the scale buffers share the per-head width (nkv).
            let kdim = kc[layer].len() / total;
            let vdim = vc[layer].len() / total;
            let nkvw = ks[layer].len() / total;
            kc[layer][..l * kdim].copy_from_slice(&cache.kc_q[layer][..l * kdim]);
            vc[layer][..l * vdim].copy_from_slice(&cache.vc_q[layer][..l * vdim]);
            ks[layer][..l * nkvw].copy_from_slice(&cache.ks_q[layer][..l * nkvw]);
            vs[layer][..l * nkvw].copy_from_slice(&cache.vs_q[layer][..l * nkvw]);
        }
    }
    // Prefill the suffix in one chunked forward at position l (attends to the reused prefix rows [0..l)).
    let xb = fwd(&prompt[l..], l, &mut kc, &mut ks, &mut vc, &mut vs);
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
        let xb = fwd(&[next], pos, &mut kc, &mut ks, &mut vc, &mut vs);
        next = argmax(&xb);
        pos += 1;
    }
    let mut ids: Vec<i64> = Vec::with_capacity(pos);
    ids.extend_from_slice(prompt);
    ids.extend_from_slice(&out);
    ids.truncate(pos);
    cache.ids = ids;
    cache.kc_q = kc;
    cache.vc_q = vc;
    cache.ks_q = ks;
    cache.vs_q = vs;
    out
}

pub trait Model: Sync {
    /// Top-1 next-token id for a context.
    fn predict(&self, ids: &[i64]) -> i64;

    /// Explain the prediction (composition-side circuits + features). Default None; GPT-2 implements it.
    fn explain(&self, _ids: &[i64]) -> Option<crate::explain::Explanation> {
        None
    }

    /// The final post-norm residual `r(x)` at the predicting position — the exact vector the unembedding dots against
    /// (`logits = U·r`). Exposed for the power-diagram geometry probe (`--probe-facet`): the token cells in r-space are
    /// the Laguerre power diagram of the unembedding rows. Default None; arches implement it where wired.
    fn final_residual(&self, _ids: &[i64]) -> Option<Vec<f32>> {
        None
    }

    /// Causal ablation: top-1 next token with the given attention heads (`heads`: layer, head) and MLP neurons
    /// (`neurons`: layer, neuron) ZEROED out of the forward pass. For the `--probe-ablate` redundancy test — is a
    /// covered token robust to knocking out its top circuits (redundant) vs a composed one fragile (emergent)?
    fn predict_ablated(&self, _ids: &[i64], _heads: &[(usize, usize)], _neurons: &[(usize, usize)]) -> Option<i64> {
        None
    }

    /// (n_layer, n_head) — for the rescue-localization layer sweep (`--probe-ablate`): ablate the top circuit + a whole
    /// downstream layer's attention to find where the indirect rescue δ lives. Default None; rope implements it.
    fn dims(&self) -> Option<(usize, usize)> {
        None
    }

    /// Cosine similarity between two unembedding rows U_a, U_b — the runner-up *coherence* ρ for the incoherence-
    /// boundary probe (`--probe-ablate`, problem A): the decoupling proof assumes a circuit's push toward `t` is ~⊥ its
    /// push toward the runner-up `v*`; that fails when U_t ≈ U_{v*} (near-synonym, high ρ), where D_j = c_j·(U_t−U_{v*})
    /// → 0. Default None; rope implements it.
    fn unembed_cos(&self, _a: usize, _b: usize) -> Option<f32> {
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
    /// cache and run the stateless path (correct, just no speedup). KV-cache kernels override: the f32 cache via
    /// `prefix_generate`, and the `--kv-int8` cache via `prefix_generate_q` (so int8-KV is BOTH 4× smaller AND
    /// prefix-reused — a long chat that's memory- and latency-light at once).
    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool, cache: &mut PrefixKv) -> Vec<i64> {
        cache.clear();
        self.generate_stream(prompt, max_tokens, eos, emit)
    }
}
