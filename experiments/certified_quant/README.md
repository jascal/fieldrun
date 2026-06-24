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

## Status & honest scope

- **Allocator: runs**, real output above. **convert `--dtype-map`: compiles.** Loader: unchanged (already
  mixed-capable). The one untested step is the **end-to-end mixed bundle + tokens/sec measurement** — it
  needs source safetensors (`convert` reads HF), not the pre-converted `.fieldrun.bin` we have locally;
  the command is step 3 above.
- **Proxy caveats:** per-block, first-order (ignores cross-layer propagation), write-tensors only; `embed`
  excluded (the v1.5 lever). `q4_rel` (int4 relative error, default 1/16) is a modeling parameter — the
  end-to-end measurement (step 4) is the validator. Corpus-relative; argmax-lossless, not
  softmax-lossless. `[empirical]`
