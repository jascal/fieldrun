# fieldrun

A single static **binary** (the runtime/tool) that runs an LLM from a flat-file **bundle** — a raw weight blob
(`.fieldrun.bin`) plus a small JSON manifest (`.fieldrun.json`), and a copied `tokenizer.json` — in pure Rust, with no
deep-learning framework at runtime. (The model is *not* baked into the binary; it's a portable bundle you convert once
and point the binary at.)

`pylm` (in the [`lm-sae`](../lm-sae/pylm) repo) decompiles a small LLM into two halves: a flat-file **retrieval** store
(n-gram / induction / grammar / knowledge) that reproduces ~half the model with stdlib-only Python, and a
**composition** kernel (attention + MLP) that runs the rest as plain numpy matmuls over flat weight arrays — no torch.
`fieldrun` is the distribution form: those tiers ported to Rust as one static binary, plus the model as a portable
bundle (weights blob + JSON manifest + tokenizer) that `convert` produces once and the binary loads/mmaps at run time.

**New here?** [`QUICKSTART.md`](QUICKSTART.md) is a clean-machine walkthrough — install Rust, build, convert a small
model, chat — in five copy-paste steps.

## Tiers

| Tier | What it adds | Status |
|------|--------------|--------|
| **A · retrieval** | induction + n-gram backoff + grammar skeleton over the flat store | **done** — bit-for-bit faithful to `lm.py` |
| **B · composition** | the attention + MLP forward pass as Rust matmuls | **done — GPT-2, GPT-NeoX/Pythia, Llama/Qwen2.5 (RoPE), Gemma-2/3/4** (incl. **Gemma-4 MoE**), **Qwen3-MoE**, each exact vs the Python/torch (or pure-numpy) reference |
| **C · router** | compute only the top fraction of MLP neurons/token | **done** — `--route-frac` (accuracy-vs-budget probe; see note) |
| `explain` | "explain this prediction": live circuits + named features | **done — all archs**; byte-identical to `explain.py`. In chat: `--explain` / `/explain on`; over the API: native `/explain` or `"explain":true` |
| `logic` | export the decode as a **semiring-Datalog program** (greedy = max-product, sampling = sum-product) | **done — rope archs** (Qwen2.5/Llama). `export --logic` (one decision) · `--export-logic <prefix>` (per-step trace) · `/export-logic` in chat; run with the built-in `eval` (`--semiring max|log`) or Soufflé. See [`LOGIC_EXPORT.md`](LOGIC_EXPORT.md) |
| API | HTTP server + **OpenAI- & Anthropic-compatible** endpoints + **interactive chat** | **done** (default build) — `--serve PORT` (native `/predict`·`/generate`·`/explain` + `/v1/chat/completions`·`/v1/completions`·`/v1/messages`, streaming, **tool/function calling**) and `--chat` |

**GPU backend** (opt-in, `--features gpu`, via **wgpu** → Metal/DX12/Vulkan): `--device cpu|gpu|auto` + `--max-vram`
budget (default 24 GB; exploits Apple unified memory), CPU default + fallback. A **GPU-resident GPT-2 forward** (weights
+ residual on-device, matmul/LayerNorm/GELU/residual as WGSL shaders) is **validated 20/20 top-1 vs CPU** on an RTX 5050
(Vulkan) — `--gpu-check`. It's correctness-first (not yet faster than the rayon CPU batch on a small model; the speedup
is the optimization pass: on-GPU attention, persistent buffers, tiled/fp16 matmul, and a bigger model). The default
build stays pure-CPU with no GPU dependency.

Plus: **KV-cache generation** (all archs, tokens identical to naive, with an optional int8 KV cache in both the one-shot
and the streaming/serve decode loop, and **prefix-KV reuse** across turns in the chat/serve path — a growing
conversation re-prefills only the new suffix, byte-identical to a cold prefill), **fp16/int8/int4 bundles for every
arch** (embeddings stay fp16, linear weights
int8 or group-wise int4; GPT-2 164 MB, Qwen 631 MB, Gemma-2-2b 3.2 GB / fits 8 GB) with **outlier-aware**
activation quant (GPT-2 int8 = fp32 accuracy, 99% per-position). The int8 dot is vectorised on aarch64 (Apple Silicon /
ARM) with **stable NEON** intrinsics (`vmull_s8` → `vpadalq_s16`, 16 lanes/iter) and a portable scalar fallback
everywhere else — all on stable Rust, no feature flag or nightly. (Activations are quantised to *signed* int8; this is
bit-exact to the scalar dot, so the faithfulness numbers are unchanged. We avoid the one-instruction `sdot`/`vdotq_s32`
on purpose — it's still behind an unstable feature and would force nightly.)

Also: a **margin-gated retrieval-pruned output head** on the serve/chat decode loops (`--pruned-head`, needs `--store`;
rope arch). Per decode step the KB proposes ~540 candidate tokens and the unembed scores only those rows; the pick is
accepted iff the in-set normalized margin `(L_t − L_v)/‖U_t − U_v‖` (the exact distance to the nearest candidate
power-diagram facet — see [`FINDINGS.md`](FINDINGS.md) §5b) clears `--pruned-margin` (default 2.0), else the full
(vocab × d) head runs. Opt-in and deliberately lossy (an accuracy-vs-speed knob like `--route-frac`): measure it with
`--gate-check N`, which generates N tokens through the gated decode vs the ungated full head and reports the identical
prefix + accept rate. At threshold +∞ every step falls back and the output is byte-identical to the full head.

The weights + store load from a **fieldrun bundle** ([`FORMAT.md`](FORMAT.md)) — a flat manifest + raw blob (f32/f16/i8)
that the build side (`lm-sae`'s `pylm/export_bundle.py`, the one-time Hugging Face step) writes and the runtime reads.
**Runtime is pure Rust, no framework.**

## The faithfulness gate

Every tier is validated by **top-1 agreement against the Python/torch reference** on the same inputs:
- **Tier A** — 0 per-position mismatches vs `lm.py` over 500 positions (with and without grammar).
- **Tier B (GPT-2 / RoPE / Gemma-2)** — exact vs the numpy kernels (= torch): GPT-2 0/200, Qwen2.5 0/32,
  Gemma-2-2b 0/18 (fp16/fp32); int8 matches on the sample once activations are outlier-aware-quantised.
- **Tier B (newer archs)** — each scored top-1 against a *tiny random-init* torch model (`scripts/gemma3_ref.py`,
  eager attention), sized to exercise every code path (both sliding+full layers, GQA, QK-norm, dual/partial RoPE,
  window masking, per-layer-type `head_dim`, PLE, MoE routing + experts, MLA latents). No gated download — the
  architecture *math* is what's validated, and a tiny instance exercises it identically to the full model. `f32` is
  the gate (the math); `f16`/`int8` are lossy by design. Reproduce: `scripts/validate_all.sh`.

  | arch | reference | f32 | f16 | int8 |
  |------|-----------|-----|-----|------|
  | `gemma3`   | Gemma3ForCausalLM        | 60/60 | 60/60 | 59/60 |
  | `gemma4`   | Gemma4ForCausalLM (dense) | 60/60 | 60/60 | 58/60 |
  | `gemma4` (MoE) | Gemma4ForCausalLM `enable_moe_block` | 60/60 | 59/60 | 56/60 |
  | `qwen3moe` | Qwen3MoeForCausalLM      | 60/60 | 60/60 | 60/60 |
  | `qwen3moe` (sliding window) | Qwen3MoeForCausalLM `use_sliding_window` | 60/60 | 60/60 | 60/60 |
  | `mla`      | DeepseekV3ForCausalLM    | 60/60 | 60/60 | 60/60 |
  | `mla` (YaRN) | DeepseekV3ForCausalLM yarn `rope_parameters` | 60/60 | 59/60 | 54/60 |
  | `minimax`  | MiniMaxM2ForCausalLM     | 60/60 | 60/60 | 60/60 |

  The YaRN row is built deliberately *sharp* (mean-1 norm weights, large init) so it actually gates the rotary
  details — a wrong rope de-interleave agrees only ~11/60, a missing YaRN ramp ~32/60; its int8 dip is that same
  sharpness amplifying near-tie flips, not a kernel gap.

**Quality (precision sweep, `scripts/bench.sh`).** Aggregating 3 seeds × 120 positions per arch — `f32` holds 100%
(the math is exact); `f16` is 99.7–100%; `int8` is 90.8–100%. The int8 dips are near-tie argmax flips on *tiny random*
weights (largest on the deliberately-sharp YaRN config) — real checkpoints quantize better (e.g. real GPT-2 int8 ==
f32 top-1, 50.0%). This is the announce-gate quality signal, not the correctness gate (that's `f32` above).

  | arch | f32 | f16 | int8 |
  |------|-----|-----|------|
  | `gemma3`   | 100.0% | 100.0% | 95.6% |
  | `gemma4`   | 100.0% | 99.7%  | 98.3% |
  | `gemma4` (MoE) | 100.0% | 99.7% | 96.1% |
  | `qwen3moe` | 100.0% | 100.0% | 100.0% |
  | `qwen3moe` (sliding window) | 100.0% | 99.7% | 99.7% |
  | `mla`      | 100.0% | 100.0% | 100.0% |
  | `mla` (YaRN) | 100.0% | 99.7% | 90.8% |
  | `minimax`  | 100.0% | 100.0% | 100.0% |

**Expert offload (MoE).** MoE is what moves the memory–capability curve: per token only the router's top-k experts are
touched, so resident set ≠ total params. `convert` writes **each expert as its own int8 array**; the loader **mmaps**
the blob and keeps expert weights **on disk**, paging only the active experts in per token (the OS page cache holds the
hot working set) — so a model with far more expert params than RAM runs, resident set = shared layers + hot experts.
Non-MoE models are unaffected (no expert arrays). Validated on **Gemma-4 MoE**, **Qwen3-MoE**, **DeepSeek-V3/Kimi-K2
(MLA)**, and **MiniMax-M2** (each f32 60/60). Qwen3-MoE needs no new attention — it's the RoPE backbone + QK-norm +
the MoE block; MLA was the last new attention class. (DeepSeek-**V4** is *not* MLA — it ships a new
hierarchical/compressed-sparse attention; `convert` refuses it explicitly rather than mis-converting.)
- **KV-cache** generation produces tokens byte-identical to naive full-recompute on every arch.

## Supported models

`convert` reads a Hugging Face checkpoint (single-file or sharded safetensors) straight to a bundle, in pure Rust.
Pick `--arch` by family:

| `--arch` | Models | Notes |
|----------|--------|-------|
| `gpt2`   | GPT-2 (124M–1.5B) | learned pos, LayerNorm, tied wte |
| `rope`   | Llama-3.x, Qwen2.5, **Qwen3 (4B/8B dense)**, Mistral, Phi | RMSNorm + RoPE + GQA + SwiGLU; optional q/k/v bias; optional **QK-norm** (Qwen3) |
| `gemma`  | Gemma-2 | √d embed, 4-norm sandwich, logit soft-cap, sliding window |
| `gemma3` | Gemma-3 (1B/4B/12B/27B) | + QK-norm, dual-base RoPE, 5:1 local/global, no soft-cap |
| `gemma4` | Gemma-4 (E2B/E4B dense, **26B-A4B MoE**) | + value-norm, per-layer-type `head_dim`, partial-rotary global RoPE, PLE, MoE |
| `qwen3moe` | Qwen3-MoE (e.g. 30B-A3B) | RoPE + QK-norm + sparse MoE + optional sliding window; no new attention kernel |
| `mla`    | DeepSeek-V3/R1, **Kimi-K2** | multi-head latent attention (incl. interleaved rotary + YaRN long-context) + group-limited sigmoid MoE + shared expert |
| `minimax`| MiniMax-M2 | softmax attn + full-width q/k-norm + sigmoid-router MoE (no MLA, no shared expert) |

All validated to top-1 agreement vs the torch reference (see the gate above). Big MoE models (Gemma-4 26B, Qwen3-MoE,
DeepSeek/Kimi) use **expert offload** — the experts stay mmap'd on disk and only the active top-k page in per token, so
the resident set is the shared layers + a working set of hot experts, not the whole model. Predict/score, KV-cache
`generate`/`generate_stream` (+ optional int8 KV), and `explain` (live circuits + features) are wired for **every**
arch — the incremental KV-cache decode is byte-identical to the naive full recompute (the f32 generation gate in
`scripts/validate_all.sh`).

**`--recursion-explain`** (the recursion spectrum — COMPUTED / DEFERRED / BINDING long-range binds + nested-fold
detection — and the value-stack logit-lens) additionally needs the per-position substrate `recursion_trace`, currently
implemented for:

| arch | recursion tracing |
|------|-------------------|
| `rope` (Llama-3.x · Qwen2.5 · Qwen3 dense · Mistral · Phi) | ✅ |
| `qwen3moe` (Qwen3-MoE, e.g. 30B-A3B) | ✅ |
| `gemma4` (Gemma-4 E2B / E4B / 26B-A4B) | ✅ |
| all other arches (`gpt2`, `gemma`, `gemma3`, `mla`, `minimax`, …) | — (`--recursion-explain` prints `no recursion_trace`) |

For `gemma4` the long-range binding signal is read only from the **global** (full-attention) layers — the
sliding-window layers mask distant keys, so a distant fold can register only there.

## Running a big model on a Mac (M3 / M4, unified memory)

The build is pure-CPU and cross-platform — **`cargo build --release` compiles on stable Rust everywhere** (Apple
Silicon, Intel macOS, Linux); on Apple Silicon the int8 dot vectorises with stable NEON automatically (scalar fallback
elsewhere), no flags needed.

**For dense models, build with Apple Accelerate** — `cargo build --release --features accelerate`. This routes the
f32/f16 matmuls through Apple's tuned BLAS (a large speedup over the pure-Rust kernel; the pure-Rust path stays the
default and the faithful reference). Dense models run **f16** (`--dtype f16`); int8 is for memory-bound MoE, not dense
speed. (Linux: `--features openblas`, needs `libopenblas`.)

> macOS may print `ld: warning: … Accelerate … was built for newer 'macOS' version … than being linked` — it's
> **harmless** (Rust's default deployment target is older than the system SDK; the binary still builds and runs). Silence
> it by matching your macOS: `MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --release --features accelerate`.

On a 24 GB M-series the lever for *big MoE* is **int8 weights + expert offload**:

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

- **Generation** (single-stream, KV-cache): GPT-2 ~25 tok/s, Qwen2.5-0.5B ~9 tok/s.
- **KV-cache** turns O(n²) recompute into O(n): GPT-2 64→128 tokens is 4.8× over naive.
- **f32/f16 matmul**: pure-Rust (`matrixmultiply`, rayon over column blocks) by default; **`--features accelerate`**
  (Mac) / `openblas` (Linux) routes it through a tuned BLAS — the lever for usable *dense* large-model speed on CPU.
- **int8 dot**: stable NEON `vmull_s8`/`vpadalq_s16` (s8×s8) on aarch64, scalar fallback elsewhere; **outlier-aware** activation quant keeps it
  lossless on the sample (100%). On a Mac, prefer `--dtype f16` for the fastest first-token latency on dense models —
  f16 goes through the blocked SIMD GEMM, whereas int8 trades a little speed for a smaller resident set (its win is on
  big / memory-bound MoE models, where the matmul isn't the bottleneck).
- Scoring fans out over positions with rayon; the per-token matmul + unembed are parallel too.

**Tier C note:** `--route-frac` reduces the MLP FLOP *budget* and measures the accuracy-vs-budget curve (GPT-2 keeps 94%
at 60% MLP), but is **not** a wall-clock speedup as-is — the gate/up matmuls still run in full (only the down-proj is
sparsified), and the sparse path is scalar vs the dense SIMD matmul. A true speedup needs a router that predicts the
active set *before* gate/up (the informed-router / MoE direction) plus SIMD sparse kernels.

## Build & run

**Requirements:** stable **Rust ≥ 1.82** (`rustup update` if older) — the default build, `--features gpu`, and the
default `hub` (HF pull) feature are all pinned to build on 1.82. The runtime needs no ML framework. `convert` reads a
local checkpoint dir **or** pulls one from the Hugging Face hub by repo id (the default `hub` feature; build
`--no-default-features` for a fully offline, network-free binary). The validation/quality harness (`scripts/`) uses
Python + `transformers`, but the binary itself does not.

```bash
cargo build --release                    # default build: HF pull + OpenAI/Anthropic API + `--chat` all included
cargo build --release --features accelerate   # Mac: tuned BLAS matmul (much faster dense models; openblas on Linux)
cargo install --path .                   # optional: puts `fieldrun` on PATH (~/.cargo/bin) so you can drop ./target/release/

# 1. CONVERT — pull from HF by repo id (or a local dir). Bundles default to a home cache (~/.cache/fieldrun/bundles/),
#    NOT the cwd; a tokenizer.json is copied alongside (for chat / the text API). Gated models: `huggingface-cli login`.
fieldrun convert --model Qwen/Qwen2.5-7B-Instruct --arch rope --dtype int8     # -> ~/.cache/fieldrun/bundles/Qwen2.5-7B-Instruct/
#   --arch  gpt2 | rope (Llama/Qwen2.5/Mistral/Phi) | gemma | gemma3 | gemma4 (incl. MoE) | qwen3moe | mla (DeepSeek/Kimi) | minimax
#   --dtype int8 (default) | f16 | f32       --hf-token <t> (gated)      -o <stem> (override the default location)

# 2. CHAT — interactive REPL (text in/out). This is the DEFAULT: a bare `--bundle <name>` with no other mode opens chat
#    (Tab-completes slash commands, ↑/↓ history). Replies render from Markdown to ANSI in a terminal (bold/headers/
#    lists/code, LaTeX math transliterated to Unicode: \theta→θ, x^2→x²); `--raw` or `/format off` keeps it plain (piped
#    output is always raw). `/help` lists slash commands (/exit, /reset, /explain [on|off], /format [on|off]).
#    Instruct models use a ChatML template; base models (e.g. GPT-2) run as a plain text-completion REPL.
fieldrun --bundle Qwen2.5-7B-Instruct                      # bare name resolves under bundles/; defaults to chat
fieldrun --bundle Qwen2.5-7B-Instruct --explain            # chat + per-reply circuits/features (--chat is the explicit form)

# 3. SERVE — OpenAI- & Anthropic-compatible HTTP API
fieldrun --bundle Qwen2.5-7B-Instruct --serve 8731
#   curl -s localhost:8731/v1/chat/completions -d '{"messages":[{"role":"user","content":"Capital of France?"}]}'
#   curl -s localhost:8731/v1/messages         -d '{"max_tokens":64,"messages":[{"role":"user","content":"Hi"}]}'
#   add "stream":true for SSE; add "explain":true to attach the structured explanation (fieldrun_explanation field)
#   tool calling: pass "tools":[…] (OpenAI {type:"function",function:{…}} or Anthropic {name,input_schema}); the model's
#     calls come back as OpenAI "tool_calls" / Anthropic "tool_use", and you feed results back as role:"tool"/tool_result
#   native token-id API too:  /predict {"ids":[…]}  ·  /generate {"prompt":[…],"n":N}  ·  /explain  ·  /health

# 4. SCORE / GENERATE / EXPLAIN against a held-out token-id stream ({"holdout_ids":[…]} from the model's tokenizer)
fieldrun --bundle Qwen2.5-7B-Instruct --ids holdout.json --n-eval 200      # next-token top-1
fieldrun --bundle Qwen2.5-7B-Instruct --ids holdout.json --ctx 64 --generate 128

# 5. LOGIC EXPORT — the decode as a runnable semiring-Datalog program (rope archs; see LOGIC_EXPORT.md)
fieldrun --bundle Qwen2.5-0.5B-Instruct --ids holdout.json export --logic --out decision.dl   # ONE next-token decision
fieldrun --bundle Qwen2.5-0.5B-Instruct --ids holdout.json --export-logic trace --steps 8      # per-step decode trace → trace.000.dl, trace.001.dl, …
fieldrun eval decision.dl --semiring max                   # run it without Soufflé: max → greedy decode (T=0)
fieldrun eval decision.dl --semiring log                   #                         log → the distribution over candidates (T=1)
#   compile a standalone "expert":  souffle -o expert decision.dl && ./expert   (writes decide.csv)
#   in chat:  /export-logic decision.dl <prompt>           # emit the .dl for that decision on demand
```

A bare `fieldrun` (or `--help`) prints the full flag list. The default build includes HF pull (`hub`) and the
OpenAI/Anthropic API + `--chat` (`api`); build `--no-default-features` for a lean, offline, token-id-only binary.

## Background / further reading

fieldrun is the distribution form of an ongoing interpretability research program — decompiling an LLM into a flat
retrieval store plus a small composition kernel, and asking what the irreducible "thinking" part costs. The write-up:
**[A Field Guide to Attention](https://jascal.github.io/lm-sae/)**. (Not required to use fieldrun — it's here if you
want the why behind the tiers and the `explain` output.)

## License

Apache-2.0. This is a 0.x prototype — interfaces and the on-disk bundle format are not yet stable.
