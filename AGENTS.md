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
- `src/neox.rs` — GPT-NeoX/Pythia: LayerNorm(+bias), parallel residual, partial rotary (`rotary_pct` of each head),
  exact erf GELU, untied `embed_out` (the fused qkv is de-interleaved at convert). Full probe surface
  (`final_residual`/`explain`/`predict_ablated`), so the FINDINGS battery runs on the Pythia ladder. Faithfulness
  gate: top-1 exact vs the pure-numpy `scripts/neox_ref.py` (no torch installs on the dev box) — 160m 60/60,
  70m/410m 20/20. No int8-KV yet (`--kv-int8` warns and uses the f32 cache).
- `src/composition.rs` / `src/rope.rs` / `src/gemma.rs` / `src/gemma3.rs` / `src/gemma4.rs` — Tier B forward passes
  (GPT-2 / Llama-Qwen / Gemma-2 / Gemma-3 / Gemma-4). **Every arch now has a KV-cache `generate`/`generate_stream`
  (+ int8-KV) and `explain`** — the KV-cache decode is byte-identical to the naive full-recompute path (the f32
  generation gate, checked per arch by `scripts/validate_all.sh`). `gemma3.rs` adds QK-norm, dual-base RoPE, the 5:1
  pattern, no soft-capping. `gemma4.rs` adds value-norm, scaling=1, per-layer-type head_dim, partial-rotary global
  RoPE, and the PLE gated-residual block; norms use the weight directly (no (1+w)); `moe_branch` adds the MoE-FFN
  (router + experts, dense||expert sum). Its KV cache is per-layer-width (local/global head_dim differ); the PLE block
  is per-token so it recomputes for new rows only.
- `src/bundle.rs` — also the **MoE expert-offload**: the blob is mmap'd; dense arrays parse into RAM (the resident set),
  but expert weights stay on disk and `expert_f32`/`expert_mm` read+dequant them on demand (per token only the active
  top-k fault in; the OS page cache holds the hot working set). Non-MoE models: no expert arrays, footprint unchanged.
- `src/qwen3moe.rs` — Qwen3-MoE: the RoPE backbone + QK-norm (per-head RMSNorm on q/k) + per-layer MoE-or-dense
  (plain-gate router: softmax → top-k → optional renorm; SwiGLU experts read from the mmap) + optional **sliding
  window** (`use_sliding_window`: ONE window on every layer — no per-layer pattern, unlike Gemma 3). No new attention
  kernel — reuses the MoE-FFN + expert-offload. KV-cache `generate` (+ int8-KV) and `explain` wired; on a MoE layer the
  explain feature is the dominant expert's promoted neuron. **The same kernel also carries Mixtral and Qwen2-MoE** — the
  three families differ in exactly three places, all probed from the CHECKPOINT at convert (never inferred at runtime,
  so a re-upload under a different `model_type` still converts right): **QK-norm** is on only where `q_norm` exists
  (Mixtral/Qwen2-MoE have none, and `head_norm` reads a *required* array, so the guard is one call at each of the six
  forward sites rather than a per-site `if`); **Mixtral** names its block `block_sparse_moe` with `w1/w3/w2` experts,
  makes EVERY layer MoE, renormalises the top-k unconditionally (HF's `/= routing_weights.sum(-1)` has no config key, so
  `norm_topk` is forced), and gates sliding on `sliding_window` merely being SET (8x7B 4096 / 8x22B null); **Qwen2-MoE**
  adds an always-on shared expert whose SwiGLU output is scaled per token by `sigmoid(shared_expert_gate)` and *added*
  to the routed mixture. Qwen2-MoE q/k/v biases are stored and applied in every forward/scoring path; the tiny
  reference randomizes these biases so omission is observable. Validated f32 60/60 each vs `MixtralForCausalLM` / `Qwen2MoeForCausalLM`, plus the SWA and
  full-attention Mixtral regimes. Two of these bits are **load-bearing, and the sweep proves it by ablation**: dropping
  the shared expert or the forced renorm moves 2–8 of 60 positions, dropping the window moves 39.
- `src/qwen35moe.rs` + `src/deltanet.rs` — Qwen3.5 / Qwen3.6 (`model_type: qwen3_5` dense, `qwen3_5_moe` MoE; this single
  arch also covers the 27B Qwen3.6/3.8 dense releases, which are shape-identical): a **hybrid** stack — 3 of every 4 layers
  are Gated DeltaNet *linear* attention (`deltanet.rs`, pinned ~1e-7 vs `torch_recurrent_gated_delta_rule`), the 4th is gated
  full GQA attention with partial RoPE. `qwen35` (dense SwiGLU) and `qwen35moe` (softmax top-k, always-renormalized, plus a
  sigmoid-gated shared expert; packed `experts.gate_up_proj`/`down_proj` unpacked at convert) share the kernel.
  **Decode is incremental via `HybridState`** — the DeltaNet recurrence (per-value-head `S` + the conv1d rolling window) is
  carried alongside the full-attention K/V cache, 6.0x on Qwen3.5-0.8B and byte-identical to the naive path. `--kv-int8`
  quantises the full-attention K/V only (the recurrent state stays f32 — no natural per-token scale). Because that state is
  *recurrent*, not row-indexed, cross-turn prefix reuse resumes only when the new prompt extends the cached one exactly;
  otherwise it falls back to a cold prefill (correct, just not faster). `validate_all.sh`'s "HYBRID decode gate" asserts the
  resume counter advanced, since token equality alone is satisfiable by a cold prefill — see that gate's header.
- `src/mla.rs` — DeepSeek-V3/R1 / Kimi-K2: MLA (q/kv low-rank latents + latent RMSNorms, a no-RoPE ‖ shared-RoPE key
  split, v_head_dim ≠ qk_head_dim) + DeepSeek MoE (group-limited sigmoid routing with bias-corrected *choice* and
  sigmoid *weight*, an always-on shared expert, first-k-dense layers). **YaRN** rope scaling (every real
  DeepSeek-V3/R1/Kimi config ships it): the inv_freq ramp, the mscale attention factor on cos/sin, and the mscale²
  softmax-scale correction. **Interleaved rotary** (`rope_interleave`, the DeepSeek default) is de-interleaved at
  convert time (the permutation is baked into the q_b/q + kv_a rotary rows), so the kernel keeps split-half rope and
  matches exactly. Experts paged from the mmap. KV-cache `generate` (+ int8-KV) caches the *assembled* per-head K/V
  (byte-identical to naive) and `explain` reads heads + the shared-expert/dense FFN features.
- `src/minimax.rs` — MiniMax-M2: RoPE backbone + FULL-WIDTH q/k-norm (RMSNorm over the whole nh·hd / nkv·hd, not
  per-head) + all-MoE with a sigmoid router (+bias for the choice, sigmoid renormed for the weight; no group, no shared
  expert). Mixtral-style expert weights on disk (`block_sparse_moe.experts.{e}.w1/w2/w3`). Experts paged. KV-cache
  `generate` (+ int8-KV) and `explain` (dominant-expert feature) wired.
- `src/gptoss.rs` — GPT-OSS: biased attention projections, learned per-head sinks, alternating sliding/full
  attention, YaRN, and biased interleaved clamped SwiGLU experts. Experts remain mmap-paged. Validated f32/f16/int8/int4 60/60 against a tiny `GptOssForCausalLM` with nonzero biases. KV-cache generation,
  int8 KV, prefix reuse, explain, and routed blind-max are wired. The blind-max candidate and corpus paths recompute
  full causal history (no scorer cache). Conversion accepts floating-point expert tensors; packed MXFP4 is unsupported.
- `src/convert.rs` — `convert` subcommand: HF safetensors (single/sharded, mmap-streamed) → bundle, pure Rust, all
  archs (`--arch gpt2|rope|gemma|gemma3|gemma4|qwen3moe|mla|minimax` — `qwen3moe` also takes Mixtral/Qwen2-MoE), `--dtype int8|f16|f32` (f32 = bit-exact, for the
  faithfulness gate). `-o` defaults to `bundles/<name>/<name>` (grouped, not loose in cwd); copies `tokenizer.json` next
  to the bundle + records `eos` in the manifest (for chat / the text API). MoE experts written one int8 array each
  (independently pageable). lm_head (non-tied unembed) stored raw (vocab, d) low-precision — read row-wise by
  rowdot_f32, NOT transposed like the other Linears.
- `src/model.rs` — the `Model` trait (predict / generate / explain), arch-agnostic.
- blind-max (the "Neural Networks Are Already Clean" probe, Cammarata) — `Model::{blind_proposals, blind_climb,
  blind_corpus_best}` + `--blind-max L,N [--blind-pole pos|neg --blind-steps N --blind-cands K --blind-corpus F]`.
  Question: does a unit's own self-description (its output projection, what `explain` prints as `promotes`) point at
  what actually fires it? The `blind` arm greedily hill-climbs tokens against the raw activation; the `oracle` arm is
  the best position over a real corpus (`--blind-corpus sweeps/corpora/*.json`), swept ONCE as a causal
  pass in 32-row blocks against a carried cache — a block amortises one weight read over 32 positions
  (overlapping windows cost ~84 min on an 11.8k-token corpus; one-token-at-a-time is bandwidth-bound at
  ~0.9 s/token and would cost ~3 h), and each position still sees its true causal history. The block
  values are bit-identical to a naive per-position forward, asserted in tests.
  The oracle is a **weak, corpus-relative adversary**: `ratio < 1` bounds the search, it does NOT refute the
  explanation, and the CLI says so inline — including the search config that produced the number, because the
  ratio is a property of the search and the corpus, not of the unit. MoE layers return `None` at every entry
  point — a routed expert has no fixed "neuron n", and guessing would be worse than declining.
  `Model::blind_support` is what the CLI gates on, and it must stay **coherent with** the arms: an arch that
  implements `blind_proposals` but inherits the `NoScorer` default makes the harness refuse to run on a model
  it supports (this happened, and the message named the rope family as unsupported). Two decline reasons,
  not one: `NoScorer` (nobody implemented it) vs `RoutedLayers` (the DENSE unit is declined, but a routed
  one is addressable as `--blind-max L,E,N`). A routed unit is scored by **`w_router · act`** — not `act`,
  since the router is a real gate, and not `· down[n]`, which is a per-neuron constant that cancels in the
  ratio and is signed, so including it can invert the objective. Only positions whose top-k CONTAINS that
  expert have a value; the rest are skipped, which is what makes a MoE unit's reachable set routing-dependent.
  Its self-description is `experts.{e}.down` row `n` — read through `expert_f32`, because expert weights are
  PAGED and `weight_row` cannot see them. The two forms are the same flag: `--blind-max 7,42` names a dense
  unit (layer 7, neuron 42), `--blind-max 7,3,42` names expert 3's neuron 42 on layer 7 — so a MoE bundle
  accepts only the three-component form and a dense bundle only the two, and each says so rather than
  guessing. (The 3-component order is `<layer>,<expert>,<neuron>`, the order this doc, the CLI help and the
  readout `L/E/N` all use; the parser used to take the expert from the THIRD field, so a 3-field spec named
  the wrong unit rather than failing — on a many-expert model both indices are in range, which is why it
  went unnoticed.) Routed support now covers `qwen35moe`, the **`qwen3moe` family** (Qwen3-MoE, Mixtral,
  Qwen2-MoE), **`minimax`**, **`gemma4`** and **`mla`**: the cached scorer there is a KV-cache twin of the
  forward — `Qwen3Scorer` / `MiniMaxScorer` / `Gemma4Scorer` / `MlaScorer` clone K/V into a snapshot so
  candidates can't see each other's writes, where Qwen3.5 has to advance a hybrid DeltaNet state
  instead. `qwen3moe`'s test module pins the invariants: the scorer step is **bit-identical** to a
  naive per-position forward, candidates are independent, the routed objective equals
  `w_router · silu(gate_n)·up_n` and is `NaN` (never 0) where the router skipped the expert, a MoE
  bundle reports `RoutedLayers` rather than `NoScorer`, and the QK-norm bit is load-bearing.
  `minimax` adds the case where the two must NOT be conflated: its top-k is chosen on `sigmoid + bias`
  but the weight is the **raw sigmoid** renormalised over the kept set, so a test that reused the
  softmax formula would pass the bias and fail the weight (or vice versa). `gemma4` adds a router
  with FIVE terms (weightless RMSNorm, per-channel `router.scale`, 1/√d, softmax→top-k→renorm, and a
  per-EXPERT `router.per_expert_scale` that is part of the weight) and a **GEGLU** activation, not
  SwiGLU — dropping any term still "works", just wrongly. Its scorer caches per-LAYER-WIDTH
  (`nkv_of(l)·hd_of(l)`; sliding and global layers differ, and KV-shared layers point at an earlier
  layer's cache). A **dense** Gemma-4 bundle reports `NoScorer`, not `Supported`: only the routed arm
  exists there, so claiming support would promise a dense probe that does not exist. `mla` adds group-
  limited routing (the group score is the top-2 of the **biased** choice within each group; the weight
  is the **unbiased** sigmoid renormed and scaled) and a guard with teeth: its first `first_k` layers
  are dense, so a routed unit named there returns None rather than a fabricated value. `dsv4` closes the
  set with a **fourth** weight form (`sqrtsoftplus`, biased choice / unbiased weight) and — because it has
  no KV cache yet — a deliberately *uncached* probe: each candidate re-runs the whole forward, which is
  exact by construction and costs O(prefix). A cache twin would be faster and would add a way for the
  scorer to silently disagree with the forward; the assertion that keeps this honest is candidate
  independence, which becomes a real snapshot/restore property the moment `generate` lands. No MoE arch
  declines blind-max any more.
- `scripts/make_corpora.py` — corpus generator, **parameterised by tokenizer**. A corpus is a list of token IDs
  and is only meaningful for the tokenizer that produced it: Qwen3.5 has a 248070-entry vocabulary against
  Qwen2.5's 151665, so a Qwen2.5 corpus fed to a Qwen3.5 bundle indexes the wrong embeddings. Qwen3 and
  Qwen3-MoE share Qwen2.5's vocabulary and reuse `sweeps/corpora/*.json` as-is; Qwen3.5 uses
  `sweeps/corpora/qwen35/*.json`. The default invocation reproduces the committed corpora byte-for-byte
  (verified), so the flag is additive. The English slice is stored as IDs; the script recovers the TEXT with
  the source tokenizer and asserts the `decode`→`encode` round-trip is exact before using it — if that ever
  stops holding, the "English" corpus would quietly be a different corpus. A token id at or past the vocab
  is rejected at generation time. `pooled_big.json` is a legacy ad-hoc artifact (committed in 71cd5a8 with no
  generator); the script emits `pooled_wide.json` under a new name rather than silently replacing it.
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

Done across the board: Tier A/B (**10 archs**: GPT-2, GPT-NeoX/Pythia (neox), RoPE/Qwen2.5, Gemma-2/3/4 incl. **Gemma-4 MoE**, **Qwen3-MoE** incl. sliding window **and incl. Mixtral + Qwen2-MoE on that same kernel**, **MLA** (DeepSeek-V3/R1/Kimi-K2, incl. YaRN + interleaved rotary; V4 is a different attention class, see below), **MiniMax-M2**), KV-cache +
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
- **Gemma-4 remaining pieces** (dense text + MoE + expert-offload + KV-cache `generate`/`explain` all landed, 60/60).
  Each its own validated increment: **attention_k_eq_v** (global layers reuse k as v, with `num_global_key_value_heads`);
  **KV-sharing** (`num_kv_shared_layers` — later layers reuse an earlier layer's K/V); and an **explicit LRU + prefetch**
  over the page cache (perf, not correctness — the current path leans on the OS page cache, the chosen "no extra
  bookkeeping" strategy).
- **Frontier-MoE roadmap.** **Qwen3-MoE (incl. sliding window) and MLA (DeepSeek-V3/R1/Kimi-K2, incl. YaRN +
  interleaved rotary) are done** (60/60 each, with dedicated discriminating columns in `validate_all.sh`) — both
  carried by the MoE-FFN + expert-offload; MLA is the last new attention *class* among the *supported* set.
  **DeepSeek-V4 is rescoped**: transformers ships it as a NEW attention class (hierarchical/compressed-sparse
  attention — compressors, a lightning indexer, grouped o_proj, sliding-window shared-KV MQA), *not* "V3 plus
  deltas"; it would be its own kernel, and `convert` now refuses `model_type=deepseek_v4` explicitly instead of
  mis-routing it to `mla`. Remaining tails: **DeepSeek-V4 (HCA/CSA) as a new arch**; **MiniMax-M2 done** (60/60);
  **MiniMax-M3** (newer, weights/class not out yet — validate when they land). **KV-cache `generate`/`generate_stream`
  (+ int8-KV) and `explain` are now wired for every arch** (gemma4, qwen3moe, mla, minimax included) — KV==naive
  byte-identity gated in `validate_all.sh`. Beyond kernels the binding constraint is **memory** — these are
  100B–1T-param models; expert-offload is the lever (resident = shared layers + hot experts), but the shared layers +
  a usable working set still want a bigger box + fast disk. Every kernel is tiny-instance-validatable
  (`scripts/validate_all.sh`); full weights are the hardware ask.

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
