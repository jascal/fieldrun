# AGENTS.md — fieldrun

Orientation for agents working in this repo. Read this first.

## What this is

`fieldrun` is the **Rust runtime** for the `pylm` decompiled-LLM track (which lives in the sibling `lm-sae` repo,
under `pylm/`, thread doc `docs/PYLM_TRACK.md`). `pylm` proves a small LLM decomposes into a flat-file **retrieval**
half (stdlib-only Python) + a **composition** half (numpy matmuls over flat weights, no torch). `fieldrun` packages
that same result as **one native binary, no framework at runtime** — the distribution form of "the minimum to run".

The Python kernels in `lm-sae/pylm` are the **reference spec** (deliberately small and readable). `fieldrun` is a
faithful re-implementation; it does not invent behaviour, it mirrors `lm.py` / `numpy_lm.py` / `numpy_rope.py` /
`numpy_gemma.py`.

## The one rule: the faithfulness gate

Anything you add must be validated by **top-1 agreement against the Python/torch reference** on the same inputs — the
same bar the Python kernels hold (GPT-2/Qwen/Gemma-2 all hit 100% vs torch). Tier A already passes exactly (0
per-position mismatches vs `lm.py` over 500 held-out positions, with and without grammar). Do not merge a tier that
diverges without explaining why.

## Layout

- `src/retrieval.rs` — Tier A: `Store` ports `lm.py` (induction + n-gram backoff + grammar).
- `src/bundle.rs` — the fieldrun bundle loader (f32/f16/i8), the matmul `mm` (parallel f32/f16 + VNNI int8 W8A8 with
  outlier-aware activation quant), `mm_routed_down` (Tier C), and the row-wise embed helpers.
- `src/composition.rs` / `src/rope.rs` / `src/gemma.rs` / `src/gemma3.rs` — Tier B forward passes (GPT-2 / Llama-Qwen /
  Gemma-2 / Gemma-3), each with a KV-cache `generate` (+ int8-KV) and `explain`. `gemma3.rs` adds QK-norm, dual-base
  RoPE (local θ for sliding layers / global θ for full), the 5:1 sliding:full pattern, and no soft-capping.
- `src/convert.rs` — `convert` subcommand: HF safetensors (single/sharded, mmap-streamed) → bundle, pure Rust, all
  archs (`--arch gpt2|rope|gemma|gemma3`), `--dtype int8|f16|f32` (f32 = bit-exact, for the faithfulness gate).
- `src/model.rs` — the `Model` trait (predict / generate / explain), arch-agnostic.
- `src/explain.rs` — head-circuit classification + feature naming + render.
- `src/api.rs` — the `tiny_http` server (`--serve PORT`).
- `src/device.rs` / `src/gpu_mm.rs` / `src/gpu_gpt2.rs` — the opt-in GPU backend (`--features gpu`, wgpu): device
  selection + budget/fallback, the validated matmul primitive, and the GPU-resident GPT-2 forward (`--gpu-check`).
  Default build excludes all of this (no GPU dependency).
- `src/main.rs` — CLI: scoring, `--generate`, `--route-frac`, `--explain`, `--serve`, `--dump`.

Done across the board: Tier A/B (**5 archs**: GPT-2, RoPE, Gemma-2, Gemma-3), KV-cache + **int8 KV cache**
(`--kv-int8`, all archs, ~4x smaller, lossy: near-lossless short-run, occasional greedy flips long-run), fp16/int8
bundles (int8 for all archs — embeddings stay fp16, linear weights int8 via VNNI W8A8 + outlier-aware quant), Tier C
(`--route-frac`), `explain` for all archs, the HTTP API, and a **pure-Rust `convert`** (no torch). Gemma-3 is validated
f32 60/60 / f16 60/60 / int8 59/60 vs a tiny `Gemma3ForCausalLM` (`scripts/gemma3_ref.py`).

Still open, with the honest catch on each (none are quick wins — they need hardware or a deep kernel, not more glue):
- **ARM NEON SDOT** (Peter's M2): can't be *validated* on this x86 box, and shipping unvalidated SIMD would break the
  faithfulness gate. ARM already runs correctly today via the scalar fallback; SDOT is a perf path to add + verify on
  real ARM hardware. (Note: ARM `sdot` is s8×s8, so it can skip the VNNI u8-offset/colsum trick — a *different* quant
  packing, which is exactly why it needs its own validation.)
- **Tier-C wall-clock speedup**: confirmed the bottleneck — sparse-scalar gather loses to dense-SIMD on CPU, and the
  gate/up still run full. A real speedup needs SIMD sparse kernels (gather into dense sub-weights with weights stored
  transposed for contiguous gather) + a predictor that skips gate/up. That's a deep build, and on CPU the payoff is
  uncertain (the gather read can cost as much as the dense pass).
- **KV-cache quant** (TurboQuant-style): a *memory* win (4× smaller cache) that only manifests at long context, which
  these short-context demos don't exercise — validatable by token-identity but not demonstrable as a real saving here.
- **Newer archs.** Gemma-3's backbone (just landed) is also the **Gemma-4 text** backbone; Gemma-4 adds three pieces on
  top, each its own validated increment: Per-Layer Embeddings (a parallel 256-dim conditioning stream added per layer),
  a partial-rotary (0.25) RoPE + larger `head_dim` on the *global* layers, and an MoE FFN (the 26B-A4B). The MoE FFN +
  the MLA attention class are the two unimplemented kernels that also gate the wider frontier-MoE roadmap (Qwen3.x,
  Kimi-2.x, DeepSeek-V4, MiniMax-M3) — most of those don't fit a ≤24 GB budget at f16, so they're "kernel exists,
  hardware permitting", not near-term. `transformers` 5.10 already exposes `Gemma4ForCausalLM`/`Gemma3ForCausalLM`, so
  each is validatable on a tiny random-init instance with no gated download (the gemma3 gate is the template).

## Conventions

- **Version stays 0.x** — prototype; the on-disk bundle format is not stable.
- **License: Apache-2.0** (matches the rest of the workspace).
- Runtime depends on **no ML framework**. Build-time tooling (weight/store export) lives in `lm-sae`, invoked once;
  cross-repo dependence is on published artifacts, not local paths (workspace rule).
- Keep the grand framing out of committed docs — describe what the code does and what was measured.
- Planned acceleration: CPU SIMD via a pure-Rust GEMM crate (keeps the single-binary story); NPU/ANE only ever as an
  opt-in, feature-gated backend.

## Build & test

```bash
cargo build --release
cargo test           # (add tests alongside each tier)
./target/release/fieldrun --store ../lm-sae/pylm/store_gpt2.json --ids ../lm-sae/pylm/holdout_gpt2.json
```
