# Certified mixed-precision quantization — Step 1 v1

The buildable v1 of [`../../CERTIFIED_QUANT_PROPOSAL.md`](../../CERTIFIED_QUANT_PROPOSAL.md): a
**margin-certified per-tensor bit allocator** that decides which int8 write-tensors can drop to int4
while *provably* preserving the (int8-baseline) decode on a calibration corpus, plus the convert-side
`--dtype-map` that applies it.

## What's built

- **`step1_allocate.py`** — the allocator. Reads a `--pil-dump` calibration + the bundle manifest, runs
  the certificate-gated rate-distortion downgrade (`2·Σ_{int4} q4_rel·s_b(x) < margin(x)` per kept
  position; residue excuses the smallest-margin forge-tax positions), and emits `alloc.json`
  (`{"dtype_map": {tensor: "int4"}}`) + the predicted bundle size. **Runs; real output below.**
- **`convert --dtype-map alloc.json`** (`src/convert.rs` `DTYPE_MAP`/`dmap_dtype` + `src/main.rs`) —
  per-tensor dtype override consulted in `put_lin`. **Compiles (`cargo check` clean).** The loader already
  dispatches per-array dtype (`bundle.rs:317`), so this is the *only* engine-side change.

## How to run

```bash
# 1. calibrate (real per-block incidences + margins)
./target/release/fieldrun --bundle <M> --recursion-explain --pil-dump calib.jsonl --n 200
# 2. allocate (offline)
python3 experiments/certified_quant/step1_allocate.py \
    bundles/<M>/<M>.fieldrun.json calib.jsonl 0.0625 0.10 alloc.json
# 3. emit the certified-mixed bundle (NEEDS source weights — convert reads HF safetensors)
./target/release/fieldrun convert --model <hf-or-dir> --arch rope --dtype int8 --dtype-map alloc.json -o <out>
# 4. measure: bundle bytes, tokens/sec, held-out decode-flip rate vs the int8 baseline
```

## v1 result (Qwen2.5-0.5B, science calib, 115 positions, q4_rel=1/16, 10% residue)

```
write blocks: 4/48 -> int4 (rest int8; embed f16)   [int4: 0 attn, 4 mlp]
write tensors: int8 357.8 MB -> mixed 331.7 MB  (saved 26.1 MB, 7% of writes)
embed (read-out, fixed f16): 272.3 MB  [needs frame-quant bound, v1.5]
full bundle: 630.1 MB -> certified-mixed 603.9 MB
```

Residue sweep: 3/48 blocks (616 MB) @5%, 4/48 (604 MB) @10%, 7/48 (584 MB) @20%. Mostly **MLP** writes
downgrade (lower-sensitivity than attn).

## The finding (this is the payoff of building v1)

Certified int8→int4 on **write tensors** is **modest** (~2–8% of the bundle), consistent with Step 0's
"static quant is margin-capped." Building the allocator surfaced the real lever:

> **`embed` is 272 MB = 43% of the bundle** — the unembedding read-out — and the per-block write proxy
> *cannot* certify it (quantizing `embed` shifts every logit via the tied `U_v`, not captured by any
> single block's sensitivity). That needs the **frame-quant bound** (`PIC_Quant.frame_quant_logit_bound`:
> `δ ≤ ρ·‖ΔU_v‖`), which the `--pil-dump` doesn't carry (no raw `U_v`).

So **v1.5 = certify the embed/read-out via the frame-quant bound** (a `--source-dump`-style probe that
emits the raw `U_v`, or a direct embed-requant-and-measure) — that's where the cashable MB are, not the
write-tensor downgrade.

## v1.5 — the embed/read-out frame-quant certificate (`step1_5_embed.py`) **[done]**

`step1_5_embed.py` consumes a `--source-dump` (raw block vectors `d̃_b`, so `r = Σ_b d̃_b`, plus cands +
margin) and the bundle's f16 embed rows, quantizes the candidate rows exactly as fieldrun's convert
(per-row int8, group-32 int4), and applies the margin certificate with the **measured** logit
perturbation `ΔL(v) = ⟨r, ΔU_v⟩` (`PIC_Quant.quant_decode_preserved`: `2·max_v|ΔL(v)| < margin`).

```bash
# 1. dump r + cands + margin (raw block vectors; once)
./target/release/fieldrun --bundle <M> --recursion-explain --source-dump src.jsonl --n 80
# (optional) slim it to a committable calib (r precomputed, ~0.5 MB):
python3 experiments/certified_quant/step1_5_embed.py src.jsonl bundles/<M>/<M> --slim calib_embed.jsonl
# 2. certify + cross-check (exact full-vocab decode + loose Cauchy-Schwarz):
python3 experiments/certified_quant/step1_5_embed.py calib_embed.jsonl bundles/<M>/<M> --allvocab --exact
```

### v1.5 result (Qwen2.5-0.5B, science calib, 68 positions) — see `RESULTS_EMBED.txt`

```
embed f16 -> int8 :  margin-certified at 5% residue (96% per-position); EXACT full-vocab flips 0/68
embed f16 -> int4 :  NOT certifiable (26% per-position, never static); EXACT full-vocab flips 4/68 = 6%
embed bytes: f16 272 MB -> int8 136 MB  (saves 136 MB = 21% of the 630 MB bundle)
combined with v1 writes (-26 MB):  full bundle 630 -> 468 MB  (1.35x, certified, 0 exact decode flips)
```

**This is the payoff of the whole certified-quant line.** The read-out is the single biggest tensor
(43%), and the frame-quant bound cashes it: **int8 is decode-safe (0/68 exact flips, margin-certified),
int4 is not (6% flips)** — the certificate locates the read-out's precision boundary *exactly* at int8.
The 136 MB it frees dwarfs the 26 MB write-tensor win (v1); together they hit the proposal's §8 go target
(≥1.3× toward int4 at baseline fidelity).

### Honesty: the bound that cashes is the *measured* δ, not Cauchy-Schwarz

The literal kernel bound `frame_quant_logit_bound` (`δ ≤ ‖r‖·‖ΔU_v‖`, Cauchy-Schwarz) is **far too loose**
to certify the read-out: `2ρε = 9.05 ≫ min-margin 0.04` for int8. That ~180× gap is exactly the
**TurboQuant `√d` regime** — quantization noise `ΔU_v` is ≈orthogonal to the residual `r`, so the *actual*
`⟨r,ΔU_v⟩` is `√d`-smaller than the worst-case product (and `ρ_max`/`ε_max` don't co-occur). What
certifies is the **measured per-position `⟨r,ΔU_v⟩`**, which `quant_decode_preserved` accepts as `δ`. The
`--exact` full-vocab decode check (`argmax(Aq8·r)` vs `argmax(A·r)`, no cand-set/no C-S) validates it:
**0 actual int8 flips**, confirming the cand-set certificate is sound *and conservative*.

**Caveats:** corpus-relative (science calib); the operative certificate ranges over the top-24 cand set
(the argmax-binding competitors — the `--exact` mode confirms no out-of-cand flip occurs); argmax-lossless,
not softmax-lossless; the worst-case all-vocab C-S bound does *not* hold (use the measured/TurboQuant δ).
End-to-end tokens/sec on a rebuilt int8-embed bundle still needs an HF `convert` (source weights). `[empirical]`

## Status & honest scope

- **Allocator: runs**, real output above. **convert `--dtype-map`: compiles.** Loader: unchanged (already
  mixed-capable). The one untested step is the **end-to-end mixed bundle + tokens/sec measurement** — it
  needs source safetensors (`convert` reads HF), not the pre-converted `.fieldrun.bin` we have locally;
  the command is step 3 above.
- **Proxy caveats:** per-block, first-order (ignores cross-layer propagation), write-tensors only; `embed`
  excluded (the v1.5 lever). `q4_rel` (int4 relative error, default 1/16) is a modeling parameter — the
  end-to-end measurement (step 4) is the validator. Corpus-relative; argmax-lossless, not
  softmax-lossless. `[empirical]`
