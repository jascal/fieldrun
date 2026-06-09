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
- `src/composition.rs` / `src/rope.rs` / `src/gemma.rs` / `src/gemma3.rs` / `src/gemma4.rs` — Tier B forward passes
  (GPT-2 / Llama-Qwen / Gemma-2 / Gemma-3 / Gemma-4). GPT-2/RoPE/Gemma-2/Gemma-3 each have a KV-cache `generate`
  (+ int8-KV) and `explain`. `gemma3.rs` adds QK-norm, dual-base RoPE, the 5:1 pattern, no soft-capping. `gemma4.rs`
  adds (dense text path) value-norm, scaling=1, per-layer-type head_dim, partial-rotary global RoPE, and the PLE
  gated-residual block; norms use the weight directly (no (1+w)); `moe_branch` adds the MoE-FFN (router + experts,
  dense||expert sum). gemma4 `generate` falls back to naive recompute and `explain` is TBD (forward correctness is the gate).
- `src/bundle.rs` — also the **MoE expert-offload**: the blob is mmap'd; dense arrays parse into RAM (the resident set),
  but expert weights stay on disk and `expert_f32`/`expert_mm` read+dequant them on demand (per token only the active
  top-k fault in; the OS page cache holds the hot working set). Non-MoE models: no expert arrays, footprint unchanged.
- `src/convert.rs` — `convert` subcommand: HF safetensors (single/sharded, mmap-streamed) → bundle, pure Rust, all
  archs (`--arch gpt2|rope|gemma|gemma3|gemma4`), `--dtype int8|f16|f32` (f32 = bit-exact, for the faithfulness gate).
  For Gemma-4 MoE it writes each expert as its own int8 array (independently pageable).
- `src/model.rs` — the `Model` trait (predict / generate / explain), arch-agnostic.
- `src/explain.rs` — head-circuit classification + feature naming + render.
- `src/api.rs` — the `tiny_http` server (`--serve PORT`).
- `src/device.rs` / `src/gpu_mm.rs` / `src/gpu_gpt2.rs` — the opt-in GPU backend (`--features gpu`, wgpu): device
  selection + budget/fallback, the validated matmul primitive, and the GPU-resident GPT-2 forward (`--gpu-check`).
  Default build excludes all of this (no GPU dependency).
- `src/main.rs` — CLI: scoring, `--generate`, `--route-frac`, `--explain`, `--serve`, `--dump`.

Done across the board: Tier A/B (**6 archs**: GPT-2, RoPE, Gemma-2, Gemma-3, Gemma-4 text incl. **MoE**), KV-cache +
**int8 KV cache** (`--kv-int8`, GPT-2/RoPE/Gemma-2/Gemma-3, ~4x smaller, lossy: near-lossless short-run, occasional
greedy flips long-run), fp16/int8 bundles (int8 for all archs — embeddings stay fp16, linear weights int8 via VNNI W8A8
+ outlier-aware quant), **MoE expert-offload** (experts mmap'd, paged per token, never resident), Tier C
(`--route-frac`), `explain` (GPT-2/RoPE/Gemma-2/Gemma-3), the HTTP API, and a **pure-Rust `convert`** (no torch).
Gemma-3 / Gemma-4 dense / Gemma-4 MoE are each validated f32 60/60 (f16/int8 ≥59/60) vs a tiny
`Gemma3ForCausalLM`/`Gemma4ForCausalLM` (`scripts/gemma3_ref.py build {gemma3,gemma4,gemma4moe}`).

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
- **Gemma-4 remaining pieces** (dense text + MoE + expert-offload all landed, 60/60). Each its own validated increment:
  **attention_k_eq_v** (global layers reuse k as v, with `num_global_key_value_heads`); **KV-sharing**
  (`num_kv_shared_layers` — later layers reuse an earlier layer's K/V); a **KV-cache `generate` + `explain`** for gemma4
  (currently naive recompute / TBD); and an **explicit LRU + prefetch** over the page cache (perf, not correctness — the
  current path leans on the OS page cache, the chosen "no extra bookkeeping" strategy).
- **Frontier-MoE roadmap** (Qwen3.x, Kimi-2.x, DeepSeek-V4, MiniMax-M3). The **MoE-FFN kernel + expert-offload now
  exist** (Gemma-4), and generalise to **Qwen3-MoE with no new attention** (normal GQA) — the next reachable target. The
  remaining kernel class is **MLA** (multi-head latent attention, the DeepSeek/Kimi compressed-KV scheme). Beyond
  kernels, the binding constraint is **memory**: these are 100B–1T-param models — expert-offload is exactly the lever
  (resident set = shared layers + hot experts), but the shared layers + a usable working set still want a bigger box and
  fast disk. `transformers` 5.10 exposes the classes, so each kernel is validatable on a tiny random-init instance with
  no gated download (the gemma3/gemma4 gate is the template); the *full* weights are the hardware ask.

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
