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
- `src/bundle.rs` — the fieldrun bundle loader (f32/f16/i8), the matmul `mm` (parallel f32/f16 + int8 W8A8 with
  outlier-aware activation quant). The int8 dot (`i8dot`) is signed s8×s8: **stable NEON** `vmull_s8` → `vpadalq_s16`
  on aarch64 (`#[target_feature(enable="neon")]`, neon is baseline so the runtime check always passes) with a scalar
  fallback everywhere else; no feature flag, no nightly. NB: we deliberately do *not* use the one-instruction
  `sdot`/`vdotq_s32` — it's gated behind the unstable `stdarch_neon_dotprod` feature and would force nightly (the same
  trap the dropped x86 AVX-512 VNNI path hit). The f32/f16 GEMM goes through ndarray `.dot()`, which routes to a tuned
  **cblas (sgemm)** when built with a BLAS backend — `--features accelerate` (macOS) or `openblas` (Linux) — the lever
  for usable *dense* large-model speed on CPU; the pure-Rust column-block path is the default + faithful reference
  (int8 always uses `i8dot`). `mm_routed_down` (Tier C), and the row-wise embed helpers.
- `src/composition.rs` / `src/rope.rs` / `src/gemma.rs` / `src/gemma3.rs` / `src/gemma4.rs` — Tier B forward passes
  (GPT-2 / Llama-Qwen / Gemma-2 / Gemma-3 / Gemma-4). GPT-2/RoPE/Gemma-2/Gemma-3 each have a KV-cache `generate`
  (+ int8-KV) and `explain`. `gemma3.rs` adds QK-norm, dual-base RoPE, the 5:1 pattern, no soft-capping. `gemma4.rs`
  adds (dense text path) value-norm, scaling=1, per-layer-type head_dim, partial-rotary global RoPE, and the PLE
  gated-residual block; norms use the weight directly (no (1+w)); `moe_branch` adds the MoE-FFN (router + experts,
  dense||expert sum). gemma4 `generate` falls back to naive recompute and `explain` is TBD (forward correctness is the gate).
- `src/bundle.rs` — also the **MoE expert-offload**: the blob is mmap'd; dense arrays parse into RAM (the resident set),
  but expert weights stay on disk and `expert_f32`/`expert_mm` read+dequant them on demand (per token only the active
  top-k fault in; the OS page cache holds the hot working set). Non-MoE models: no expert arrays, footprint unchanged.
- `src/qwen3moe.rs` — Qwen3-MoE: the RoPE backbone + QK-norm (per-head RMSNorm on q/k) + per-layer MoE-or-dense
  (plain-gate router: softmax → top-k → optional renorm; SwiGLU experts read from the mmap). No new attention kernel —
  reuses the MoE-FFN + expert-offload. predict only (generate=naive / explain TBD).
- `src/mla.rs` — DeepSeek-V3 / Kimi-K2: MLA (q/kv low-rank latents + latent RMSNorms, a no-RoPE ‖ shared-RoPE key
  split, v_head_dim ≠ qk_head_dim) + DeepSeek MoE (group-limited sigmoid routing with bias-corrected *choice* and
  sigmoid *weight*, an always-on shared expert, first-k-dense layers). Experts paged from the mmap. predict only.
- `src/minimax.rs` — MiniMax-M2: RoPE backbone + FULL-WIDTH q/k-norm (RMSNorm over the whole nh·hd / nkv·hd, not
  per-head) + all-MoE with a sigmoid router (+bias for the choice, sigmoid renormed for the weight; no group, no shared
  expert). Mixtral-style expert weights on disk (`block_sparse_moe.experts.{e}.w1/w2/w3`). Experts paged. predict only.
- `src/convert.rs` — `convert` subcommand: HF safetensors (single/sharded, mmap-streamed) → bundle, pure Rust, all
  archs (`--arch gpt2|rope|gemma|gemma3|gemma4|qwen3moe|mla|minimax`), `--dtype int8|f16|f32` (f32 = bit-exact, for the
  faithfulness gate). `-o` defaults to `bundles/<name>/<name>` (grouped, not loose in cwd); copies `tokenizer.json` next
  to the bundle + records `eos` in the manifest (for chat / the text API). MoE experts written one int8 array each
  (independently pageable). lm_head (non-tied unembed) stored raw (vocab, d) low-precision — read row-wise by
  rowdot_f32, NOT transposed like the other Linears.
- `src/model.rs` — the `Model` trait (predict / generate / explain), arch-agnostic.
- `src/explain.rs` — head-circuit classification + feature naming + render.
- `src/mdfmt.rs` — dependency-free Markdown→ANSI for the chat REPL (headings/lists/bold/italic/code + LaTeX
  transliteration: `\theta`→θ, `\frac{a}{b}`→(a)/(b), `x^2`→x², math delimiters stripped). Line-buffered so it
  streams; TTY-gated (raw when piped or `--raw`/`/format off`).
- `src/api.rs` — the `tiny_http` server (`--serve PORT`): native token-id routes always (`/predict`,`/generate`,
  `/explain`,`/health`); under `--features api` (default-off) a `TextGen` (the `tokenizers` crate) adds the
  **OpenAI** (`/v1/chat/completions`,`/v1/completions`,`/v1/models`) + **Anthropic** (`/v1/messages`) text endpoints and
  the **`--chat`** REPL (ChatML prompt, greedy, EOS-stop). Tokenizer is loaded from `<stem>.tokenizer.json`. Also
  **tool/function calling**: `render_chat` renders the OpenAI/Anthropic message list (incl. prior tool_calls/results)
  into the ChatML prompt; tool requests are answered non-streaming and return structured `tool_calls`/`tool_use`.
- `src/tools.rs` — tool calling helpers (api): parse `tools` (OpenAI/Anthropic shapes) → a Hermes-style system
  preamble; parse the model's output back into calls across formats (Hermes/Qwen `<tool_call>`, Mistral `[TOOL_CALLS]`,
  Llama/generic JSON; `arguments`/`parameters` normalised, JSON-string args decoded). Best-effort, model-agnostic.
- `src/hub.rs` — `--features hub` (default): pull a model from HF by repo id with a small ureq client (token auth,
  relative-307-aware); used by `convert --model org/repo`.
- `src/device.rs` / `src/gpu_mm.rs` / `src/gpu_gpt2.rs` — the opt-in GPU backend (`--features gpu`, wgpu): device
  selection + budget/fallback, the validated matmul primitive, and the GPU-resident GPT-2 forward (`--gpu-check`).
  Default build excludes all of this (no GPU dependency).
- `src/main.rs` — CLI: scoring, `--generate`, `--route-frac`, `--explain`, `--serve`, `--dump`.

Done across the board: Tier A/B (**9 archs**: GPT-2, RoPE/Qwen2.5, Gemma-2/3/4 incl. **Gemma-4 MoE**, **Qwen3-MoE**, **MLA** (DeepSeek-V3/V4/Kimi-K2), **MiniMax-M2**), KV-cache +
**int8 KV cache** (`--kv-int8`, GPT-2/RoPE/Gemma-2/Gemma-3, ~4x smaller, lossy: near-lossless short-run, occasional
greedy flips long-run), fp16/int8 bundles (int8 for all archs — embeddings stay fp16, linear weights int8 W8A8 +
outlier-aware quant; signed int8 dot — stable NEON `vmull`/`vpadal` on aarch64, scalar elsewhere), **MoE expert-offload**
(experts mmap'd, paged per token, never resident), Tier C
(`--route-frac`), `explain` (GPT-2/RoPE/Gemma-2/Gemma-3), the HTTP API, and a **pure-Rust `convert`** (no torch).
Gemma-3 / Gemma-4 dense / Gemma-4 MoE are each validated f32 60/60 (f16/int8 ≥59/60) vs a tiny
`Gemma3ForCausalLM`/`Gemma4ForCausalLM` (`scripts/gemma3_ref.py build {gemma3,gemma4,gemma4moe}`).

Still open, with the honest catch on each (none are quick wins — they need hardware or a deep kernel, not more glue):
- **ARM NEON int8 validation**: the stable-NEON `vmull`/`vpadal` int8 dot is wired and *compiles* for aarch64
  (`cargo check --target aarch64-unknown-linux-gnu --no-default-features` on this x86 box), but its numerical output
  can't be *run-validated* here. The scalar path is proven bit-exact to the prior scheme (int8 dumps diff clean), and
  the NEON path is a straightforward vectorisation of that same signed s8×s8 sum — but per the faithfulness gate it
  still needs an on-device run: on an M-series, `scripts/validate_all.sh` (the int8 column) confirms NEON == torch.
  (The faster one-instruction `sdot`/`vdotq_s32` stays out: unstable `stdarch_neon_dotprod`, i.e. nightly-only.)
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
- **Frontier-MoE roadmap.** **Qwen3-MoE and MLA (DeepSeek-V3/V4/Kimi-K2) are done** (60/60) — both carried by the
  MoE-FFN + expert-offload; MLA is the last new attention *class*. Remaining tails: Qwen3-MoE **sliding window**
  (`use_sliding_window`, convert asserts off); **DeepSeek-V4 deltas** over V3 + **YaRN** long-context RoPE scaling;
  **MiniMax-M2 done** (60/60); **MiniMax-M3** (newer, weights/class not out yet — validate when they land); and
  **KV-cache `generate` + `explain`** for the newer archs (currently naive recompute). Beyond kernels the binding constraint is **memory** — these are 100B–1T-param models; expert-offload
  is the lever (resident = shared layers + hot experts), but the shared layers + a usable working set still want a
  bigger box + fast disk. Every kernel is tiny-instance-validatable (`scripts/validate_all.sh`); full weights are the
  hardware ask.

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
