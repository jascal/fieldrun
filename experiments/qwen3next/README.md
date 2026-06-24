# Qwen3.6 / Qwen3-Next port — test harness (verify the arch *without* the big machine)

Qwen3.6 (and the Qwen3-Next family it's built on) is a **hybrid**: Gated DeltaNet *linear* attention
interleaved with gated full attention + MoE (256 experts, 8 routed + 1 shared) + multi-token prediction.
fieldrun has no linear-attention path, so supporting it means a new Rust kernel. The worry — "how do I know
it works without running the 35B?" — is solved by one principle:

> **Architecture correctness is size-independent.** A 4-layer / hidden-64 toy exercises the *same* Rust
> code path the 35B uses. The big machine is only for the final confirmatory run, never for development.

## The test pyramid (everything except the last step runs on a dev box)

| # | test | needs | what it proves | runs today? |
|---|---|---|---|---|
| 1 | **`deltanet_ref.py`** — Gated DeltaNet recurrence + property tests | numpy | the hardest new kernel is mathematically right (recall, **delta-overwrite vs linear-attn**, decay, β=0). The **oracle** the Rust must match. | ✅ **yes** |
| 2 | **`make_tiny_qwen3next.py`** — shrink the *real* config, random-init, dump reference logits + per-layer states | transformers* | a faithful toy whole-model reference (arch faithful, dims tiny) | needs transformers |
| 3 | **`compare.py`** — fieldrun vs reference, **per layer** | numpy | parity, and *which block* diverges if not | after the Rust port |
| 4 | golden vectors (freeze a few `compare.py` passes into CI) | numpy | regression guard, forever, no big weights | after #3 |
| 5 | end-to-end on the smallest *real* hybrid checkpoint (if one exists) | the model | real weight/routing/numeric coverage | optional |
| 6 | one confirmatory big-machine run | the 35B | perf/memory + final sanity | last, not a dependency |

\* `transformers` recent enough for the Qwen3-Next/3.6 class (`model_type` ≈ `qwen3_next`). This box doesn't
have it yet: `pip install -U transformers` (or Qwen's build), then steps 2–3 run locally on the toy.

## Run order

```bash
# 1. kernel oracle — RUN THIS FIRST, it needs nothing but numpy
python experiments/qwen3next/deltanet_ref.py        # all 4 properties must PASS

# 2. faithful toy + reference dump (after `pip install -U transformers`)
python experiments/qwen3next/make_tiny_qwen3next.py --layers 4 --hidden 64 --out experiments/qwen3next/tiny

# 3. implement the arch in Rust, convert the toy, emit a per-layer debug dump (fr.npz), then:
./target/release/fieldrun convert --model experiments/qwen3next/tiny --arch qwen3next --dtype f32 -o /tmp/tiny
./target/release/fieldrun --bundle /tmp/tiny --debug-dump fr.npz --ids experiments/qwen3next/tiny/ref_ids.json
python experiments/qwen3next/compare.py experiments/qwen3next/tiny/ref.npz fr.npz --tol 1e-4
```

## Non-negotiables (the time-sinks, learned the hard way)

- **Develop in f32 first.** Get f32 parity *before* touching f16/int8 — else you'll chase quantization
  noise thinking it's an arch bug. (The int8 read-out certificate is a *separate* layer on top — see
  `../certified_quant/`. It applies to Qwen3.6 unchanged: the read-out is norm-pinned regardless of the
  hybrid attention.)
- **Per-layer, not just logits.** `compare.py` diffs every block so a failure says *which op* is wrong.
- **Recurrent is enough for decode.** fieldrun decodes one token at a time = the DeltaNet recurrence
  itself; no chunked-parallel form needed to be correct. Chunking is a *prefill* speed optimization — add
  it later and test it against `deltanet_ref.py`.
- **Pin the variant to transformers.** `deltanet_ref.py` validates the recurrence + delta semantics; the
  exact gate placement is a variant choice — `compare.py` against the real `transformers` impl is what
  fixes the variant. If `deltanet_ref` passes but `compare` diverges in a DeltaNet layer, it's a gate/
  normalization variant mismatch, not a chunking bug.
- **Do it in a git worktree/branch.** The existing arch support must stay green.

## Status

- `deltanet_ref.py`: **built, all properties pass** (numpy-only — the kernel oracle is ready).
- `make_tiny_qwen3next.py`, `compare.py`: **built, ready**; need `transformers` (toy) and the Rust arch
  (parity). No Rust written yet — this is the verification scaffold that makes the port checkable off the
  big machine before any engine work starts. `[scaffold]`
