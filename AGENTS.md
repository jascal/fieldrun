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

What's intentionally still open: int8 for GPT-2/RoPE (only Gemma int8 so far), ARM NEON SDOT (VNNI is x86), a true
informed-router for a Tier-C wall-clock speedup, RoPE/Gemma `explain`, and a real KV-cache speedup at long context.

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
