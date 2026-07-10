# J-Lens — Jacobian-lens probe (fieldrun)

An **empirical** mid-stack read-out aid. Where the logit-lens reads an intermediate residual by unembedding it directly
(assuming the downstream map is the identity, `J_l = I`), the J-lens first routes it through the layer's **averaged
causal Jacobian** to the final residual,

```
J_l = E_{t, t'≥t, prompt} [ ∂h_final,t' / ∂h_l,t ]        read(h_l) = softmax( W_U · norm( J_l · h_l ) )
```

so a layer-`l` activation is scored by what the network is *disposed to make it emit*, not by the identity-path guess.
Motivated by Anthropic's *"Verbalizable Representations as Global Workspace"* (transformer-circuits.pub/2026/workspace).

> **Tag: `empirical`.** `J_l` is a first-order, context-averaged approximation. The J-lens **never touches the forward
> path or the faithfulness gate** — it only re-reads captured residuals. Treat its output as a probe, not a certificate.

## How it's fit (no autodiff needed)

fieldrun owns the forward pass, so it estimates `J_l` by a finite-difference JVP: perturb one row of a captured
post-block-`l` residual, run the forward from `l+1` (`Model::jlens_forward_from`), and read the change at the final
layer. The estimator is the unbiased Hutchinson outer-product `E_g[(J g) gᵀ] = J` for `g ~ N(0, I)`, central-differenced
and averaged over source positions `t`, downstream targets `t'≥t`, and the corpus. Reproducible (seeded PRNG),
checkpointed, and off the hot path.

Implemented on the **rope** family (`rope.rs`) and **neox / Pythia** (`neox.rs`) via three `Model` hooks
(`jlens_capture`, `jlens_forward_from`, `recursion_trace_lens`); `recursion_trace` delegates to the last with `J_l = I`.

## CLI

```bash
# 1. fit {J_l} over a corpus (offline; ~probes×prompts×src passes per layer, checkpointed)
fieldrun --bundle <model> --recursion-explain --jlens-fit \
    --jlens-corpus corpus.txt --jlens-out model.jlens \
    --jlens-probes 5 --jlens-max-src 4 --jlens-max-seq 24 --jlens-layers all

# 2. eval: J-lens vs logit-lens (resolve-layer, across-depth stability), sweeping the shrinkage knob
fieldrun --bundle <model> --text "…" --recursion-explain --jlens-eval \
    --jlens-in model.jlens --jlens-shrink 0.0,0.25,0.5,1.0

# 3. export {J_l} to the numpy channel (for pil / fieldrun_io.py)
fieldrun --jlens-export model.npz --jlens-in model.jlens

# 4. export the model's unembedding U (V,d) + final-norm gain gamma (d,) — the two constants pil's sweep also
#    needs (--U / --gamma). Needs the loaded model (reads weights), so pass --bundle (rope/neox archs).
fieldrun --bundle <model> --recursion-explain --tensors-export model.tensors.npz
```

**Shrinkage** (`--jlens-shrink λ`, eval-time, sweepable): reads through `J' = (1−λ)I + λ·J`. `λ=1` is the raw fit,
`λ=0` is exactly the logit-lens. An under-fit (noise-dominated) `J_l` degrades *gracefully* toward the logit-lens as
`λ→0` instead of scrambling the read-out — sweep to find the operating point.

## Export format (`--jlens-export`, for pil)

A `.npz` (stored zip of `.npy`, `np.load`-able — no numpy/zip dependency in fieldrun; hand-rolled + numpy-verified)
plus a `.meta.json` sidecar:

| array | dtype | shape | meaning |
|-------|-------|-------|---------|
| `J` | `float32` | `[n_layer, d, d]` | the averaged causal Jacobian per layer (row-major) |
| `fitted` | `int32` | `[n_layer]` | `1` where a Jacobian was fit; `0` = identity (reads as the logit-lens) |

**Apply convention:** route a layer-`l` residual `r` through `J[l] @ r` (numpy: `r @ J[l].T`), then the model's final
norm + unembed. The last layer is the identity by construction (`J_last = I`).

## Tensors export (`--tensors-export`, for pil's sweep)

pil's `jcorrect_sources` corrects the DLA vectors, but to turn the corrected read into token logits its sweep also
needs the model's **unembedding** `U`, and — for the *exact* correction — the **final-norm gain** `γ`. `--tensors-export`
writes both onto the same numpy channel (a stored-zip `.npz` + `.meta.json`):

| array | dtype | shape | meaning |
|-------|-------|-------|---------|
| `U` | `float32` | `[vocab, d]` | unembedding rows; `U[id]` scores token `id` (rows indexed by token id) |
| `gamma` | `float32` | `[d]` | the final-norm gain |

The meta carries `norm_type` and `gamma_exact`: `SourceBundle.D` is final-norm-**folded**, so pil applies `J` in that
basis via the conjugation `diag(γ) J diag(1/γ)`. For **RMSNorm** archs (rope) that fold is a pure diagonal, so the
conjugation is **exact**; for **LayerNorm** archs (neox) it omits the mean-centering rank-1 term and the `ln_f` bias, so
`gamma_exact=false` (a documented approximation). Unlike `--jlens-export` (a pure transcode), this reads the loaded
model's weights — pass `--bundle`. Implemented for the **rope** and **neox** families (others report unsupported).

Feed the two files to the sweep (`--U`/`--gamma` read the `U`/`gamma` arrays from the same `.npz`):
`jlens_correction_sweep.py run.source.jsonl --jlens model.npz --U model.tensors.npz --gamma model.tensors.npz`.

**Capture point:** `h_l` is the **post-block** residual of layer `l` (after the attn+MLP residual add, **pre** final-norm)
— the same tensor `recursion_capture` records; `h_final` is the post-last-block residual (pre final-norm). (Also in the
`.meta.json` `capture_point` field.)

## Causal validation (`--jlens-causal`, `--jlens-causal-jspace`)

The resolve-layer/flips readout turned out to be **context-fragile** (see findings), so the faithful test of the paper's
claims is causal — and fieldrun already owns the machinery (`residuals_at` + `logits_patched`, a real forward with a
swapped residual; no lens, no resolve-layer).

- **`--jlens-causal`** — interchange tracing (paper 5.3/5.4). Aligned prompt pairs ("capital of France is"→Paris vs
  "…China is"), patch the base's `(layer, last-pos)` residual with the source's, sweep layers, report **flip-to-source
  rate + source-logit shift**. On Qwen2.5-0.5B it gives a clean, monotone localization (0% early → 100% at L23; ≥50%
  band L21–23) — France→Spain flips *Paris→Madrid*. Knobs: `--causal-template "…{}…"`, `--causal-entities`, `--causal-pairs`.
- **`--jlens-causal-jspace`** — the J-space test (paper 5.1). At the causal layer, patch with only a subspace's slice of
  the swap `Δ`: the J-space `P_JΔ` (from `{J_l}`) vs its complement vs a diff-subspace **oracle** (top-PCA of the actual
  swaps) vs random; report flip + capture `‖PΔ‖/‖Δ‖`. Knobs: `--jlens-in`, `--causal-layer`, `--causal-jspace k1,k2,…`.

## Eval-time denoising knobs (`--jlens-rank`, `--jlens-logit-rank`, `--jlens-shrink`)

Applied to a loaded `{J_l}` before the read (no re-fit): `--jlens-shrink λ` blends toward the logit-lens;
`--jlens-rank k` keeps the top-k SVD of `J−I`; `--jlens-logit-rank k` keeps the logit-relevant rank-k (top eigenvectors
of `JᵀMJ`, `M` = sampled unembed Gram — the paper's `Wᵤ·J` J-space).

## Status & honest findings (2026-07)

The J-lens **instrument** is built, tested, and faithful (off the forward path). The J-space **phenomenon** does *not*
cleanly reproduce at 0.5B–410m on CPU — consistent with it being a frontier-scale result. Specifically:

- **Fit / lens read** (Qwen2.5-0.5B): `‖J_l−I‖_F` decays monotonically toward the output (right shape). The J-lens
  resolves the final token *earlier* than the logit-lens at **λ≈0.25–0.5** on some contexts — **but the resolve-layer
  metric is context-fragile** (full-rank 410m flips +2.1 later ↔ −1.1 earlier just by changing the sentence), so this is
  not a reliable signal.
- **Pythia ladder** (14m→410m): earlier-resolve is **non-monotonic in scale**; the largest rung regresses under a
  fixed-probe fit (read-noise ∝ σ√d).
- **Low-rank denoise** (`--jlens-rank`): removes the σ√d noise but also removes signal — rescues 410m, breaks Qwen.
  **Logit-weighted** (`--jlens-logit-rank`, G′): worse (the sampled NeoX Gram is outlier-dominated).
- **Causal (`--jlens-causal`)**: the tracer **works** — clean, monotone, faithful. This is the reusable win.
- **J-space causal test (`--jlens-causal-jspace`)**: the fitted-`{J_l}` J-space **does not house** the swappable concept
  at L22 — both definitions capture ≤half of the swap direction and **never suffice to flip** (0%), while the complement
  ≈ full. Caveats: our J-space (top-k of `JᵀMJ`) ≠ the paper's sparse-pursuit over the `Wᵤ·J` dictionary; partial-swap
  flips are confounded by downstream nonlinearity; 0.5B is below the expected scale.

**To push further** (both larger efforts): implement the paper-faithful sparse-pursuit J-space, or add
`predict_patched`/`residuals_at` to `neox` and run the causal tests up the Pythia ladder / on the big MoE archs.
