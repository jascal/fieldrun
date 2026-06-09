# fieldrun

Run a decompiled LLM as a single native binary — the three [`pylm`](../lm-sae/pylm) tiers in pure Rust, no
deep-learning framework at runtime.

`pylm` (in the `lm-sae` repo) decompiles a small LLM into two halves: a flat-file **retrieval** store (n-gram /
induction / grammar / knowledge) that reproduces ~half the model with stdlib-only Python, and a **composition** kernel
(attention + MLP) that runs the rest as plain numpy matmuls over flat weight arrays — no torch. `fieldrun` is the
distribution form of that result: the same tiers, ported to Rust, built into one static binary you can hand someone.

## Tiers

| Tier | What it adds | Status |
|------|--------------|--------|
| **A · retrieval** | induction + n-gram backoff + grammar skeleton over the flat store | **done** — bit-for-bit faithful to `lm.py` |
| **B · composition** | the attention + MLP forward pass as Rust matmuls | **done — GPT-2, Llama/Qwen (RoPE), Gemma-2**, each exact vs the Python/torch reference |
| **C · router** | compute only the top fraction of MLP neurons/token | **done** — `--route-frac` (accuracy-vs-budget probe; see note) |
| `explain` | "explain this prediction": live circuits + named features | **done (GPT-2)** — byte-identical to `explain.py` |
| API | `/predict` · `/generate` · `/explain` HTTP server | **done** — `--serve PORT` |

Plus: **KV-cache generation** (all archs, tokens identical to naive), **fp16/int8 bundles** (Gemma-2-2b in 3.2 GB, fits
8 GB), and an **AVX-512 VNNI** int8 matmul (on-core int8 dot, runtime-detected + scalar fallback).

The weights + store load from a **fieldrun bundle** ([`FORMAT.md`](FORMAT.md)) — a flat manifest + raw blob (f32/f16/i8)
that the build side (`lm-sae`'s `pylm/export_bundle.py`, the one-time Hugging Face step) writes and the runtime reads.
**Runtime is pure Rust, no framework.**

## The faithfulness gate

Every tier is validated by **top-1 agreement against the Python/torch reference** on the same inputs:
- **Tier A** — 0 per-position mismatches vs `lm.py` over 500 positions (with and without grammar).
- **Tier B** — exact vs the numpy kernels (= torch): GPT-2 0/200, Qwen 0/32, Gemma-2-2b 0/18 (fp16/fp32); int8+VNNI
  matches on the sample once activations are outlier-aware-quantised.
- **KV-cache** generation produces tokens byte-identical to naive full-recompute on every arch.

## Performance (16-core box)

- **Generation** (single-stream, KV-cache): GPT-2 ~25 tok/s, Qwen2.5-0.5B ~9 tok/s, Gemma-2-2b int8+VNNI ~3 fwd/s.
- **KV-cache** turns O(n²) recompute into O(n): GPT-2 64→128 tokens is 4.8× over naive.
- **int8 + AVX-512 VNNI** (Gemma): 0.8 → 3.0 fwd/s (3.75×); **outlier-aware** activation quant keeps it lossless on
  the sample (100%).
- Scoring fans out over positions with rayon; the per-token matmul + unembed are parallel too.

**Tier C note:** `--route-frac` reduces the MLP FLOP *budget* and measures the accuracy-vs-budget curve (GPT-2 keeps 94%
at 60% MLP), but is **not** a wall-clock speedup as-is — the gate/up matmuls still run in full (only the down-proj is
sparsified), and the sparse path is scalar vs the dense SIMD matmul. A true speedup needs a router that predicts the
active set *before* gate/up (the informed-router / MoE direction) plus SIMD sparse kernels.

## Build & run

```bash
cargo build --release
B=../lm-sae/pylm                       # bundles + stores live here (built by pylm/export_bundle.py)

# Tier A — retrieval over the flat store
./target/release/fieldrun --store $B/store_gpt2.json --ids $B/holdout_gpt2.json
# Tier B — score the real forward pass over a bundle (gpt2 / qwen05b / gemma2_2b[_int8])
./target/release/fieldrun --bundle $B/gpt2 --ids $B/holdout_gpt2.json --n-eval 200   # --dump preds.txt for the diff
# Generate (KV-cache) — compares cached vs naive, prints tok/s
./target/release/fieldrun --bundle $B/gpt2 --ids $B/holdout_gpt2.json --ctx 64 --generate 128
# Tier C — conditional MLP (top fraction of neurons/token)
./target/release/fieldrun --bundle $B/gpt2 --ids $B/holdout_gpt2.json --route-frac 0.6
# Explain a prediction (GPT-2)
./target/release/fieldrun --bundle $B/gpt2 --ids ctx.json --ctx 12 --explain --vocab $B/vocab_gpt2.json
# Serve the HTTP API
./target/release/fieldrun --bundle $B/gpt2 --serve 8731
#   curl -s localhost:8731/predict  -d '{"ids":[...]}'
#   curl -s localhost:8731/generate -d '{"prompt":[...],"n":16}'
#   curl -s localhost:8731/explain  -d '{"ids":[...]}'
```

## License

Apache-2.0. This is a 0.x prototype — interfaces and the on-disk bundle format are not yet stable.
