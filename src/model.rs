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
///   * `argmax(&hidden, ctx)` → next-token id from the last row. `ctx` is the full context (prompt + emitted so far)
///      at that step, so a margin-gated pruned head (`--pruned-head`) can build its KB candidate set; plain kernels
///      ignore it.
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
    argmax: &dyn Fn(&Array2<f32>, &[i64]) -> i64,
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
    let mut ctx: Vec<i64> = prompt.to_vec(); // running context for argmax (the gated head keys its KB lookup on it)
    let xb = fwd(&prompt[l..], l, &mut kc, &mut vc);
    let mut next = argmax(&xb, &ctx);
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
        ctx.push(next);
        let xb = fwd(&[next], pos, &mut kc, &mut vc);
        next = argmax(&xb, &ctx);
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
    argmax: &dyn Fn(&Array2<f32>, &[i64]) -> i64,
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
    let mut ctx: Vec<i64> = prompt.to_vec(); // running context for argmax (the gated head keys its KB lookup on it)
    let xb = fwd(&prompt[l..], l, &mut kc, &mut ks, &mut vc, &mut vs);
    let mut next = argmax(&xb, &ctx);
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
        ctx.push(next);
        let xb = fwd(&[next], pos, &mut kc, &mut ks, &mut vc, &mut vs);
        next = argmax(&xb, &ctx);
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

/// Per-position recursion substrate for `--recursion-explain`: the raw model-internal signals the recursion gate
/// reads — resolve-layer (logit-lens), the late-layer value-stack readout, and the dominant non-sink back-attention.
pub struct RecPos {
    pub pos: usize,
    pub final_top1: i64,
    pub resolve_layer: usize,        // first layer whose logit-lens argmax == final_top1 (deferred = late)
    pub n_layer: usize,
    pub lens_late: Vec<(usize, i64)>, // (layer, logit-lens top-1) at the late layers — the value stack
    pub lens_full: Vec<(usize, i64)>, // (layer, logit-lens top-1) at ALL layers — the value stack across depth
    pub back: usize,                  // dominant NON-SINK late-layer back-attention target (the frame it folds)
    pub conc: f32,                    // attention weight on `back` (max over late layers+heads); high = real bind
}

/// Assemble the per-position recursion trace from an arch's captured substrate. This is the **architecture-agnostic**
/// half of `recursion_trace`: only `recursion_capture` (the forward) and `lens_argmax` (the arch's final-norm +
/// unembed) are arch-specific, so every arch shares this one assembly and there is a single place to keep correct.
/// - `resids[l]` = the post-block residual at layer `l` (seq × d); `maxback` = the late-layer attention max (seq × seq).
/// - `lens_argmax(resids[l])` = the logit-lens top-1 token id per position for one layer (arch's final norm + unembed).
/// - `late0` = first "late" layer the binding signal is read from. Convention `2·nl/3` (last third): for Gemma 4 the
///   sliding-window layers can't carry long-range binding so a distant fold only registers on the global layers there;
///   for the rope/MoE families it is simply where return/fold attention concentrates. Bump it per-arch if needed.
pub fn build_rec_trace(
    resids: &[Array2<f32>],
    mut maxback: Array2<f32>,
    late0: usize,
    lens_argmax: impl Fn(&Array2<f32>) -> Vec<i64>,
) -> Vec<RecPos> {
    let nl = resids.len();
    let seq = resids.first().map(|r| r.nrows()).unwrap_or(0);
    if seq < 3 || nl == 0 {
        return vec![];
    }
    // per-layer logit-lens argmax per position (the only arch-specific step is `lens_argmax`)
    let lens: Vec<Vec<i64>> = resids.iter().map(&lens_argmax).collect();
    // zero the attention SINK (cols 0/1) so the binding signal is a real distant fold, not sink mass
    for i in 0..seq {
        maxback[[i, 0]] = 0.0;
        if seq > 1 {
            maxback[[i, 1]] = 0.0;
        }
    }
    let mut out = Vec::with_capacity(seq.saturating_sub(1));
    for p in 0..seq.saturating_sub(1) {
        // logit lens at p predicts token p+1; the model's prediction = the last-layer lens
        let final_top1 = lens[nl - 1][p];
        let resolve = (0..nl).find(|&l| lens[l][p] == final_top1).map(|l| l + 1).unwrap_or(nl);
        let lens_late: Vec<(usize, i64)> = (late0..nl).map(|l| (l + 1, lens[l][p])).collect();
        let lens_full: Vec<(usize, i64)> = (0..nl).map(|l| (l + 1, lens[l][p])).collect();
        let (mut back, mut conc) = (p, 0f32);
        for k in 0..p {
            if maxback[[p, k]] > conc {
                conc = maxback[[p, k]];
                back = k;
            }
        }
        out.push(RecPos { pos: p, final_top1, resolve_layer: resolve, n_layer: nl, lens_late, lens_full, back, conc });
    }
    out
}

/// The `recursion_lens_at` companion (arch-agnostic): the late-layer logit-lens reads at specific positions, for the
/// `--induce` value-stack sweeps. `lens_argmax_at(resids[l], p)` = the logit-lens top-1 for one position at one layer.
pub fn build_rec_lens_at(
    resids: &[Array2<f32>],
    positions: &[usize],
    late0: usize,
    lens_argmax_at: impl Fn(&Array2<f32>, usize) -> i64,
) -> Vec<Vec<(usize, i64)>> {
    let nl = resids.len();
    positions
        .iter()
        .map(|&p| {
            (late0..nl)
                .filter(|&l| p < resids[l].nrows())
                .map(|l| (l + 1, lens_argmax_at(&resids[l], p)))
                .collect()
        })
        .collect()
}

pub trait Model: Sync {
    /// Top-1 next-token id for a context.
    fn predict(&self, ids: &[i64]) -> i64;

    /// Per-position recursion substrate (`--recursion-explain`). Default None; the RoPE family implements it.
    fn recursion_trace(&self, _ids: &[i64]) -> Option<Vec<RecPos>> {
        None
    }

    /// CHEAP logit-lens at SPECIFIC positions, LATE layers only — for batched value-stack reads in --induce sweeps
    /// without paying the full-vocab argmax at every layer×position. Returns, per requested position (same order),
    /// the late-layer (layer, logit-lens top-1) reads. Default None; the RoPE family implements it.
    fn recursion_lens_at(&self, _ids: &[i64], _positions: &[usize]) -> Option<Vec<Vec<(usize, i64)>>> {
        None
    }

    /// Raw per-layer residual-stream vectors at SPECIFIC positions — feeds the supervised value-probe (B2): can a
    /// trained linear map read an intermediate subtree value off the residual, where the unembed-basis lens can't?
    /// Returns, per requested position (same order), a Vec over layers, each the d-dim residual. Default None.
    fn residuals_at(&self, _ids: &[i64], _positions: &[usize]) -> Option<Vec<Vec<Vec<f32>>>> {
        None
    }

    /// CAUSAL interchange: run the forward but REPLACE the residual at each `positions[i]` with `donors[i]` (residuals
    /// captured from another expr at the same slots) right after `layer`, then predict the top-1 next token. If swapping
    /// in another expression's intermediate state changes the OUTPUT toward it, those slots causally carry it.
    fn predict_patched(&self, _ids: &[i64], _layer: usize, _positions: &[usize], _donors: &[Vec<f32>]) -> Option<i64> {
        None
    }

    /// Explain the prediction (composition-side circuits + features). Default None; GPT-2 implements it.
    fn explain(&self, _ids: &[i64]) -> Option<crate::explain::Explanation> {
        None
    }

    /// Like `explain`, but also populate the Density-Minimization substrate (`Explanation::decomp`): every scored
    /// source's margins against the top-`k` competitors, for the `--probe-decompose` descent. Default = plain `explain`
    /// (no substrate); arches that support the descent override it. See `explain::decompose_descent`.
    fn explain_decomp(&self, ids: &[i64], _k: usize) -> Option<crate::explain::Explanation> {
        self.explain(ids)
    }

    /// Stream the explain-with-substrate at every position `pos` in `start..=ids.len()` — the decision predicting from
    /// the growing context `ids[..pos]`. Default: an uncached loop over `explain_decomp` (one full forward per position).
    /// KV-cache arches override this to reuse ONE growing cache — O(seq) attention work, byte-identical by causality.
    fn explain_stream(&self, ids: &[i64], decomp_k: usize, start: usize, f: &mut dyn FnMut(usize, crate::explain::Explanation)) {
        for pos in start.max(1)..=ids.len() {
            if let Some(ex) = self.explain_decomp(&ids[..pos], decomp_k) {
                f(pos, ex);
            }
        }
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

    /// Full next-token logit vector at the predicting position (the same vector `predict` argmaxes). For
    /// per-token loss / target-token-logit measurement (`ablate-eval`). Default None; arches wire it where they can.
    fn logits(&self, _ids: &[i64]) -> Option<Vec<f32>> {
        None
    }

    /// Logit vector with the given attention heads + MLP neurons ZEROED — the ablated counterpart of `logits`, so
    /// Δlogit / Δloss of a causal knockout is measurable, not just the top-1 flip (`predict_ablated`).
    fn logits_ablated(&self, _ids: &[i64], _heads: &[(usize, usize)], _neurons: &[(usize, usize)]) -> Option<Vec<f32>> {
        None
    }

    /// Logit vector with the K/V cache round-tripped through a quantizer (`--probe-kv-quant`): `turbo_bits=None` → the
    /// int8 per-head max-scale scheme (the existing `--kv-int8` runtime cache); `Some(b)` → a `b`-bit TurboQuant codec
    /// (SRHT rotation + Lloyd–Max levels). Measures the decision distortion the cache quant injects vs the f32 `logits`
    /// reference — the test of whether TurboQuant's isotropy enables a lower-bit KV cache than int8. Default None; rope wires it.
    fn logits_kvq(&self, _ids: &[i64], _turbo_bits: Option<u8>) -> Option<Vec<f32>> {
        None
    }

    /// Install a margin-gated retrieval-pruned output head (`--pruned-head`) on the DECODE loops — the serve/chat
    /// streaming paths only; `predict` (scoring, probes) always runs the full head. Returns false if the arch doesn't
    /// wire it (default). See `headgate::HeadGate`.
    fn set_head_gate(&mut self, _gate: std::sync::Arc<crate::headgate::HeadGate>) -> bool {
        false
    }

    /// (accepted, fallback) decode-step counts of the installed head gate, if any — for `--gate-check` / exit stats.
    fn head_gate_stats(&self) -> Option<(u64, u64)> {
        None
    }

    /// Remove the head gate (back to the full head on every step) — `--gate-check` uses this to run its ungated
    /// reference stream on the same model instance.
    fn clear_head_gate(&mut self) {}

    /// Decode with ONE residual-stream block's write quantized (per-row symmetric to `bits`, round-trip) — for
    /// `--probe-quant`: does a block's pivotality `D_b` predict how much quantizing it perturbs the decode? `block`
    /// indexes the residual writes as `residual_decomp` labels them (0 = embed; layer l → `2l+1` attn, `2l+2` mlp).
    /// Returns the new top-1 token. Default None; rope implements it.
    fn predict_block_quant(&self, _ids: &[i64], _block: usize, _bits: u8) -> Option<i64> {
        None
    }

    /// Per-block residual decomposition at the predicting position, for the LE-T5 / `--probe-reconstruct` test: returns
    /// `(labels, contrib)` where `labels[b]` names a residual-stream write (embedding, each layer's attention, each
    /// layer's MLP) and `contrib[b][i]` is that block's exact contribution to the logit of `toks[i]` (in true logit
    /// units, final-norm folded in). By residual-stream additivity `Σ_b contrib[b][i] == logit(toks[i])` exactly — so
    /// the reconstruction residual measures decompiler completeness, and the per-block concentration is the decision's
    /// "support number" (PIC O2: small for retrieved, large for composed). Default None; rope implements it.
    fn residual_decomp(&self, _ids: &[i64], _toks: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        None
    }

    /// Per-block *contribution vectors* at the predicting position: returns `(labels, dvec)` where `dvec[b]` is the
    /// block `b` write with the final norm folded in (`d̃_b`), living in unembed space (dim = d_model). It is the exact
    /// vector with `⟨d̃_b, U_v⟩ == contrib[b][v]` for every token `v` — i.e. `residual_decomp` is just this projected onto
    /// the `toks` rows. The raw `d̃_b` (not its projection onto the model's *own* `U`) is what the forge-tax certificate
    /// needs: re-representing the decoder frame `U → U'` plugs straight into `⟨d̃_b, U'_v⟩`. Emitted by `--source-dump`.
    /// Default None; rope implements it (and `residual_decomp` derives from it).
    fn residual_normed_writes(&self, _ids: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        None
    }

    /// LEAN single-forward decision + residual decomposition for the logic export — the fast corpus path. ONE forward
    /// (no `explain` circuit capture: no attention matrices, head/MLP attribution), a single full-vocab argmax for
    /// predicted/runner-up, then per-block contributions projected onto the candidate set `{predicted, runner-up} ∪
    /// extra` (deduped, capped at `cap`). Returns `(predicted, runner_up, pred_logit, ru_logit, candidates, blocks)`
    /// with `blocks[b] = (label, contrib-to-each-candidate)` and (by residual additivity) `Σ_b blocks[b].1[i] ==
    /// logit(candidates[i])`. Costs ~one forward instead of `explain`'s forward + ~14 s of capture — `logic::build_decomp`
    /// uses it and falls back to the `explain`+`residual_decomp` path when an arch returns None (default).
    fn decision_decomp(&self, _ids: &[i64], _extra: &[i64], _cap: usize)
        -> Option<(i64, i64, f32, f32, Vec<i64>, Vec<(String, Vec<f32>)>)> {
        None
    }

    /// ALL-POSITIONS teacher-forced decomposition from ONE forward — the corpus throughput path. The per-decision cost
    /// is a fixed (int4-dequant-bound) prefill that does NOT scale with context length, so amortize it: one forward over
    /// `ids` yields a decision at EVERY position (entry `p` = the model's prediction of token p+1 given ids[..=p]).
    /// Each entry is `(predicted, runner_up, pred_logit, ru_logit, candidates(top-`cap` by logit), blocks)` with
    /// `Σ_b blocks[b].1[i] == logit(candidates[i])`. Turns ~L prefills into one. Default None.
    fn decomp_all(&self, _ids: &[i64], _cap: usize)
        -> Option<Vec<(i64, i64, f32, f32, Vec<i64>, Vec<(String, Vec<f32>)>)>> {
        None
    }

    /// Like `predict_ablated`, but also zeroes a *whole* attention block (`attn_layers`) and/or *whole* MLP block
    /// (`mlp_layers`) of the listed layers — for the rescue-localization sweep (ablate {top circuit + downstream layer
    /// ℓ's MLP or attention}). Default None; rope implements it.
    fn predict_ablated_blocks(&self, _ids: &[i64], _heads: &[(usize, usize)], _neurons: &[(usize, usize)], _attn_layers: &[usize], _mlp_layers: &[usize]) -> Option<i64> {
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

    /// The raw unembedding row `U_id` (the frame element for token `id`) — for building the Gram kernel `G_{vw}=⟨U_v,U_w⟩`
    /// and rank diagnostics offline (PIC_PROPOSAL §2; the paper's "linear SVD rank cannot measure the gap" test). Default
    /// None; rope implements it via `weight_row(unembed_name, id)`.
    fn unembed_row(&self, _id: usize) -> Option<Vec<f32>> {
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
