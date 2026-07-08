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

**Capture point:** `h_l` is the **post-block** residual of layer `l` (after the attn+MLP residual add, **pre** final-norm)
— the same tensor `recursion_capture` records; `h_final` is the post-last-block residual (pre final-norm). (Also in the
`.meta.json` `capture_point` field.)

## Status (2026-07)

Fit on Qwen2.5-0.5B (300-prompt corpus, all 23 layers): `‖J_l − I‖_F` decays monotonically toward the output (the
right shape — the downstream map → identity near the final layer). Eval shows the J-lens **resolves the final token
earlier** than the logit-lens (the paper's core claim), best at **λ ≈ 0.25–0.5** — on arithmetic contexts a clean win
on *both* earlier-resolve and fewer across-depth flips; raw `λ=1` goes earlier but is noise-dominated (hence shrinkage).
The Pythia ladder (14m→2.8b, learned-positional) is wired for the scale study.
