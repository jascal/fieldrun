# Certified-prune Step 0 — does the margin certificate have pruning ore on real models?

A measurement-only probe (no engine change, no retraining) that tests, on real `fieldrun --pil-dump`
data, how much of a frozen model's decode can be **certifiably** pruned within tolerance — i.e. dropping
DLA blocks while *provably* preserving the decoded token.

It operationalizes the i-orca certificates (`examples/pic_krein/PIC_Prune.thy`, `PIC_Quant.thy`): the
margin certificate says a logit perturbation `δ` cannot flip a token whose margin exceeds `2δ`; pruning a
block set `P` perturbs each candidate logit by exactly the dropped incidence `Σ_{j∈P} c_j(v)`.

## How to reproduce

```bash
# dumps (real Qwen2.5-0.5B, the built-in default text + a code snippet):
./target/release/fieldrun --bundle Qwen2.5-0.5B-Instruct --recursion-explain \
    --pil-dump qwen05_science.jsonl --n 200
./target/release/fieldrun --bundle Qwen2.5-0.5B-Instruct --recursion-explain \
    --pil-dump qwen05_code.jsonl --text "$CODE" --n 200      # CODE = the snippet in this dir's RESULTS
# probe:
python3 step0.py qwen05_science.jsonl qwen05_code.jsonl      # -> RESULTS.txt
```

`step0.py` reads `contrib[block][cand]` + margins and computes, per corpus: the **adaptive (per-token)**
certified prune ratio (β-budget and the tighter signed/cancellation bound), the **static
(corpus-intersection)** ratio with a residue sweep, a heuristic-vs-certified control, the cross-corpus
flip rate, and the early-vs-late structure of the droppable set. The committed `qwen05_*.jsonl` +
`RESULTS.txt` are the exact data and output behind the findings below.

## Findings (Qwen2.5-0.5B, 49 blocks, 2 corpora ≈110 positions each)

1. **Static ("ship a pruned model") certified prune ≈ 0.** `0/49` blocks at 0% residue; only `1–2/49`
   even after discarding 40% of positions. The smallest-margin position (≈0.02, a near-tie) caps it.
   → Static / corpus-global certified pruning is **not viable** — the empirical face of the activation-
   relative (not behavior-invariant) result (`pic/spec/PIC_SPEC.md` §7).
2. **Adaptive (per-token) prune is real but modest; the signed bound dominates the β-budget.**
   science: med **8%** (budget) → **18%** (signed, p90 57%); code: med **16%** → **53%** (p90 82%).
   Strongly domain-dependent (code margin med 3.5 prunes ~3× the science margin med 0.89).
3. **The certificate is load-bearing.** Unchecked magnitude pruning at 50% flips **26%** (science) /
   **12%** (code) of decodes; the certified sets flip **0** (sanity).
4. **Cost cash-in is the dampener.** Droppable blocks are predominantly **early** (late-half share
   0.27 / 0.33). Early blocks still have to be computed to build the residual, so the ore is mostly
   **decode-attribution sparsity, not skippable FLOPs**; only ~⅓ sits in the late/early-exit region.

## Verdict

- **Static `--certified-prune`: no ore — do not build.** (Step 0's job: it just saved that build.)
- **Adaptive per-token prune: thin**, capped by the small-margin residue (the forge-tax positions where
  the model is itself undecided — legitimately incompressible), with limited compute cash-in.
- **The lever is margins:** larger / better-conditioned margins (upstream `pil`) raise every ratio here.
  Within the unified certificate, **quantization** (`PIC_Quant`) is the more cashable knob than pruning —
  bit reduction saves storage/bandwidth on every weight regardless of block position.

## Scope / honesty

One small model (0.5B, 49 blocks), CPU, two ≈110-position corpora — **not** the Pythia/Qwen ladder. The
qualitative findings (static≈0, signed≫budget, droppable-mostly-early, certificate load-bearing) are
likely robust; magnitudes need the size/architecture sweep (Step 0-i). This is decode/argmax-lossless,
not softmax-lossless. `[empirical]`
