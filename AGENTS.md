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
- `src/composition.rs` / `src/rope.rs` / `src/gemma.rs` — Tier B forward passes (GPT-2 / Llama-Qwen / Gemma-2), each
  with a KV-cache `generate`. `composition.rs` also has GPT-2 `explain`.
- `src/model.rs` — the `Model` trait (predict / generate / explain), arch-agnostic.
- `src/explain.rs` — head-circuit classification + feature naming + render.
- `src/api.rs` — the `tiny_http` server (`--serve PORT`).
- `src/main.rs` — CLI: scoring, `--generate`, `--route-frac`, `--explain`, `--serve`, `--dump`.

Done across the board: Tier A/B (4 archs), KV-cache + **int8 KV cache** (`--kv-int8`, all archs, ~4x smaller, lossy:
near-lossless short-run, occasional greedy flips long-run), fp16/int8 bundles (int8 for **all** of GPT-2/RoPE/Gemma —
embeddings stay fp16, linear weights int8 via VNNI W8A8 + outlier-aware quant), Tier C (`--route-frac`), `explain` for
all three archs, the HTTP API.

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
