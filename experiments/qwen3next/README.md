# Qwen3.6 / Qwen3-Next port — test harness (verify the arch *without* the big machine)

Qwen3.6 (and the Qwen3-Next family it's built on) is a **hybrid**: Gated DeltaNet *linear* attention
interleaved with gated full attention + MoE (256 experts, 8 routed + 1 shared) + multi-token prediction.
fieldrun has no linear-attention path, so supporting it means a new Rust kernel. The worry — "how do I know
it works without running the 35B?" — is solved by one principle:

> **Architecture correctness is size-independent.** A 4-layer / hidden-64 toy exercises the *same* Rust
> code path the 35B uses. The big machine is only for the final confirmatory run, never for development.

**Confirmed arch (from `transformers` 5.12, the real `Qwen/Qwen3.6-35B-A3B` config):** `model_type =
qwen3_5_moe` with a **nested `text_config`** (`qwen3_5_moe_text` — the 3.5/3.6 family is VL/omni-capable).
The text path: `layer_types` = **3× `linear_attention` → 1× `full_attention`** (×10 over 40 layers); each
linear layer is a **short causal depthwise conv (`linear_conv_kernel_dim=4`) → Gated DeltaNet** (32 value /
16 key heads, head-dim 128); full layers are GQA 16/2; MoE **256 experts, 8 routed + 1 shared**
(`moe_intermediate_size=512`); **1 MTP layer**; vocab 248320; RMSNorm eps 1e-6. So the Rust port needs:
short-conv + Gated DeltaNet + the 3:1 scheduler + shared-expert MoE + the MTP head. (The transformers
"fla fast path not available → torch fallback" is fine — the torch path is the reference we compare to.)

## The test pyramid (everything except the last step runs on a dev box)

| # | test | needs | what it proves | runs today? |
|---|---|---|---|---|
| 1 | **`deltanet_ref.py`** — Gated DeltaNet recurrence + property tests | numpy | the hardest new kernel is mathematically right (recall, **delta-overwrite vs linear-attn**, decay, β=0). The **oracle** the Rust must match. | ✅ **yes** |
| 1b | **`crosscheck_deltanet.py`** — oracle vs transformers' own `torch_recurrent_gated_delta_rule` | transformers* | the oracle **IS** the real Qwen3.6 recurrence (matched to ~1e-7) — **variant pinned** | ✅ **verified** |
| 2 | **`make_tiny_qwen3next.py`** — shrink the *real* config, random-init, dump reference logits + per-layer states | transformers* | a faithful toy whole-model reference (arch faithful, dims tiny) | ✅ **verified** |
| 3 | **`compare.py`** — fieldrun vs reference, **per layer** | numpy | parity, and *which block* diverges if not | after the Rust port |
| 4 | golden vectors (freeze a few `compare.py` passes into CI) | numpy | regression guard, forever, no big weights | after #3 |
| 5 | end-to-end on the smallest *real* hybrid checkpoint (if one exists) | the model | real weight/routing/numeric coverage | optional |
| 6 | one confirmatory big-machine run | the 35B | perf/memory + final sanity | last, not a dependency |

\* needs `transformers>=5.12` (has the `Qwen3_5Moe` class). **Verified working:** in an isolated venv
(`pip install torch --index-url …/cpu && pip install "transformers>=5.12"`), `make_tiny_qwen3next.py` built
a **0.99 M-param faithful toy** (4 layers `[lin,lin,lin,full]`, conv k=4, 8 experts) and dumped `ref.npz`
(logits (512,256) + per-layer states (5,512,64)) — the Python side of the harness runs end-to-end today.

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

- `deltanet_ref.py`: **built, all 4 properties pass** (numpy-only — the kernel oracle is ready).
- `make_tiny_qwen3next.py`: **built and RUN-VALIDATED** (transformers 5.12) — builds a faithful 0.99 M toy
  + reference dump; it confirmed the real arch (above).
- `crosscheck_deltanet.py`: **DONE — variant pinned.** The oracle matches transformers'
  `torch_recurrent_gated_delta_rule` to ~1e-7 in the real config (`qk_l2norm=True`). Pinned conventions:
  `α=exp(g)`, `q·(1/√d_k)`, L2-norm(q)&(k), decay→delta-correct-vs-decayed-state→write→read-after-write.
  `deltanet_ref.gated_deltanet_qwen36()` packages exactly this — the Rust reference.
- `compare.py`: **built, ready** — needs the Rust arch to produce `fr.npz`.
- **Next: the Rust port.** With the kernel verified, implement `conv1d(k=4)→Gated DeltaNet` + the 3:1
  linear/full scheduler + shared-expert MoE + the MTP head, convert the toy, and run `compare.py` (per-layer,
  f32 first). The DeltaNet math is no longer a risk — it's pinned and tested.
- This is the verification scaffold that makes the port checkable off the big machine *before* engine work.
  `[scaffold; kernel verified vs transformers; engine not yet written]`
