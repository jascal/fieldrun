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
| **B · composition** | the attention + MLP forward pass as Rust matmuls | **done — GPT-2, Llama/Qwen2.5 (RoPE), Gemma-2/3/4** (incl. **Gemma-4 MoE**), **Qwen3-MoE**, each exact vs the Python/torch reference |
| **C · router** | compute only the top fraction of MLP neurons/token | **done** — `--route-frac` (accuracy-vs-budget probe; see note) |
| `explain` | "explain this prediction": live circuits + named features | **done — all archs**; byte-identical to `explain.py` |
| API | `/predict` · `/generate` · `/explain` HTTP server | **done** — `--serve PORT` |

**GPU backend** (opt-in, `--features gpu`, via **wgpu** → Metal/DX12/Vulkan): `--device cpu|gpu|auto` + `--max-vram`
budget (default 24 GB; exploits Apple unified memory), CPU default + fallback. A **GPU-resident GPT-2 forward** (weights
+ residual on-device, matmul/LayerNorm/GELU/residual as WGSL shaders) is **validated 20/20 top-1 vs CPU** on an RTX 5050
(Vulkan) — `--gpu-check`. It's correctness-first (not yet faster than the rayon CPU batch on a small model; the speedup
is the optimization pass: on-GPU attention, persistent buffers, tiled/fp16 matmul, and a bigger model). The default
build stays pure-CPU with no GPU dependency.

Plus: **KV-cache generation** (all archs, tokens identical to naive), **fp16/int8 bundles for all four archs** (embeddings
stay fp16, linear weights int8; GPT-2 164 MB, Qwen 631 MB, Gemma-2-2b 3.2 GB / fits 8 GB), and an **AVX-512 VNNI** int8
matmul with **outlier-aware** activation quant (on-core int8 dot, runtime-detected + scalar fallback; GPT-2 int8 = fp32
accuracy, 99% per-position).

The weights + store load from a **fieldrun bundle** ([`FORMAT.md`](FORMAT.md)) — a flat manifest + raw blob (f32/f16/i8)
that the build side (`lm-sae`'s `pylm/export_bundle.py`, the one-time Hugging Face step) writes and the runtime reads.
**Runtime is pure Rust, no framework.**

## The faithfulness gate

Every tier is validated by **top-1 agreement against the Python/torch reference** on the same inputs:
- **Tier A** — 0 per-position mismatches vs `lm.py` over 500 positions (with and without grammar).
- **Tier B (GPT-2 / RoPE / Gemma-2)** — exact vs the numpy kernels (= torch): GPT-2 0/200, Qwen2.5 0/32,
  Gemma-2-2b 0/18 (fp16/fp32); int8+VNNI matches on the sample once activations are outlier-aware-quantised.
- **Tier B (newer archs)** — each scored top-1 against a *tiny random-init* torch model (`scripts/gemma3_ref.py`,
  eager attention), sized to exercise every code path (both sliding+full layers, GQA, QK-norm, dual/partial RoPE,
  window masking, per-layer-type `head_dim`, PLE, MoE routing + experts, MLA latents). No gated download — the
  architecture *math* is what's validated, and a tiny instance exercises it identically to the full model. `f32` is
  the gate (the math); `f16`/`int8` are lossy by design. Reproduce: `scripts/validate_all.sh`.

  | arch | reference | f32 | f16 | int8 |
  |------|-----------|-----|-----|------|
  | `gemma3`   | Gemma3ForCausalLM        | 60/60 | 60/60 | 59/60 |
  | `gemma4`   | Gemma4ForCausalLM (dense) | 60/60 | 60/60 | 59/60 |
  | `gemma4` (MoE) | Gemma4ForCausalLM `enable_moe_block` | 60/60 | 59/60 | 59/60 |
  | `qwen3moe` | Qwen3MoeForCausalLM      | 60/60 | 60/60 | 60/60 |
  | `mla`      | DeepseekV3ForCausalLM    | 60/60 | 60/60 | 60/60 |

**Quality (precision sweep, `scripts/bench.sh`).** Aggregating 3 seeds × 120 positions per arch — `f32` holds 100%
(the math is exact); `f16` is 99.7–100%; `int8` is 95.6–100%. The int8 dips are near-tie argmax flips on *tiny random*
weights — real checkpoints quantize better (e.g. real GPT-2 int8 == f32 top-1, 50.0%). This is the announce-gate
quality signal, not the correctness gate (that's `f32` above).

  | arch | f32 | f16 | int8 |
  |------|-----|-----|------|
  | `gemma3`   | 100.0% | 100.0% | 95.6% |
  | `gemma4`   | 100.0% | 99.7%  | 98.3% |
  | `gemma4` (MoE) | 100.0% | 99.7% | 96.1% |
  | `qwen3moe` | 100.0% | 100.0% | 100.0% |
  | `mla`      | 100.0% | 100.0% | 100.0% |

**Expert offload (MoE).** MoE is what moves the memory–capability curve: per token only the router's top-k experts are
touched, so resident set ≠ total params. `convert` writes **each expert as its own int8 array**; the loader **mmaps**
the blob and keeps expert weights **on disk**, paging only the active experts in per token (the OS page cache holds the
hot working set) — so a model with far more expert params than RAM runs, resident set = shared layers + hot experts.
Non-MoE models are unaffected (no expert arrays). Validated on **Gemma-4 MoE** and **Qwen3-MoE** (f32 60/60).
Qwen3-MoE needs no new attention — it's the RoPE backbone + QK-norm + the MoE block — so it's the first frontier-MoE
family reachable end-to-end; the remaining kernel class for DeepSeek-V4 / Kimi is MLA.
- **KV-cache** generation produces tokens byte-identical to naive full-recompute on every arch.

## Supported models

`convert` reads a Hugging Face checkpoint (single-file or sharded safetensors) straight to a bundle, in pure Rust.
Pick `--arch` by family:

| `--arch` | Models | Notes |
|----------|--------|-------|
| `gpt2`   | GPT-2 (124M–1.5B) | learned pos, LayerNorm, tied wte |
| `rope`   | Llama-3.x, Qwen2.5, Mistral, Phi | RMSNorm + RoPE + GQA + SwiGLU; optional q/k/v bias |
| `gemma`  | Gemma-2 | √d embed, 4-norm sandwich, logit soft-cap, sliding window |
| `gemma3` | Gemma-3 (1B/4B/12B/27B) | + QK-norm, dual-base RoPE, 5:1 local/global, no soft-cap |
| `gemma4` | Gemma-4 (E2B/E4B dense, **26B-A4B MoE**) | + value-norm, per-layer-type `head_dim`, partial-rotary global RoPE, PLE, MoE |
| `qwen3moe` | Qwen3-MoE (e.g. 30B-A3B) | RoPE + QK-norm + sparse MoE; no new attention kernel |
| `mla`    | DeepSeek-V3, DeepSeek-V4, **Kimi-K2** | multi-head latent attention + group-limited sigmoid MoE + shared expert |

All validated to top-1 agreement vs the torch reference (see the gate above). Big MoE models (Gemma-4 26B, Qwen3-MoE,
DeepSeek/Kimi) use **expert offload** — the experts stay mmap'd on disk and only the active top-k page in per token, so
the resident set is the shared layers + a working set of hot experts, not the whole model. (Predict/score is fully
supported on every arch; KV-cache `generate` + `explain` are wired for GPT-2 / RoPE / Gemma-2 / Gemma-3 — the newer
archs fall back to naive recompute for generation, which is correct but slower.)

## Running a big model on a Mac (M3 / M4, unified memory)

The build is pure-CPU and cross-platform (the AVX-512 int8 path is x86-gated and falls back to a portable scalar dot on
ARM; no code change needed). On a 24 GB M-series the lever is **int8 weights + expert offload**:

```bash
cargo build --release                                   # builds on macOS ARM as-is
# 1. download a checkpoint (e.g. Qwen3-MoE) with the HF CLI or hf_hub; you just need its safetensors + config.json
# 2. convert to an int8 bundle — streams one tensor at a time, so RAM ≈ one tensor, not the whole model
./target/release/fieldrun convert --model <hf-model-dir> --arch qwen3moe --dtype int8 -o qwen3moe
# 3. run: score next-token over a held-out id stream (experts page in from the .bin on demand)
./target/release/fieldrun --bundle qwen3moe --ids holdout.json --ctx 64 --n-eval 50
```

Why a model far bigger than RAM still runs: `convert` writes each expert as its own int8 array; at load the blob is
`mmap`'d and only the small dense/shared layers are resident — each token's router picks its top-k experts, those pages
fault in (and stay warm in the OS page cache), the rest never touch RAM. Headroom = your free disk + page cache, not
just VRAM. Optional GPU backend (Metal via wgpu) is `--features gpu` (correctness-validated for GPT-2/RoPE; the newer
archs run on CPU).

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
# Convert a Hugging Face checkpoint -> bundle, pure Rust, no torch (single-file or sharded safetensors)
#   --arch gpt2 | rope (Llama/Qwen2.5/Mistral/Phi) | gemma | gemma3 | gemma4 (incl. MoE) | qwen3moe | mla (DeepSeek/Kimi)
#   --dtype int8 (default) | f16 | f32 (f32 = bit-exact bundle, used by the faithfulness gate)
./target/release/fieldrun convert --model ~/.cache/huggingface/hub/.../gemma-3-1b-it --arch gemma3 --dtype int8 -o $B/gemma3_1b
# Tier B — score the real forward pass over a bundle (gpt2 / qwen05b / gemma2_2b[_int8] / gemma3_*)
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
