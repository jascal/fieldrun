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

## Step 0-i: ladder sweep (`SWEEP.txt`, `step0_sweep.py`)

`python3 step0_sweep.py "label=dump.jsonl" ...` — same metrics across models, one science corpus:

| model (corpus) | nb | N | margin med | adapt budget | adapt signed (p90) | static@10% | static@40% | heur-50% flip | signed late-share |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (sci) | 49 | 115 | 0.89 | 0.08 | 0.18 (0.57) | 0.00 | 0.02 | 0.26 | 0.27 |
| Qwen2.5-0.5B (code) | 49 | 111 | 3.50 | 0.16 | 0.53 (0.82) | 0.00 | 0.04 | 0.12 | 0.33 |
| Qwen2.5-Coder-0.5B (sci) | 49 | 115 | 0.94 | 0.08 | 0.20 (0.59) | 0.02 | 0.02 | 0.23 | 0.25 |
| Qwen2.5-7B (sci) | 57 | 80 | 1.01 | 0.17 | **0.42 (0.72)** | 0.02 | 0.05 | **0.10** | 0.26 |

- **Scale helps adaptive prune.** 7B prunes ~2.3× the 0.5B on the *same* prose (signed med 0.42 vs 0.18,
  p90 0.72) and is more prune-robust (heuristic-50% flip 0.10 vs 0.26) — at **near-equal margin**
  (1.01 vs 0.89). So the gain is **redundancy** (more blocks, more cancellation for the signed bound),
  not bigger margins. Adaptive certified prunability grows with capacity.
- **Training-data axis weak:** base-0.5B vs Coder-0.5B on prose are ≈identical (0.18 vs 0.20).
- **Domain/margin axis strong:** code (margin 3.5) prunes ~3× prose on 0.5B.
- **The two caps hold at every scale:** `static ≈ 0` (0–0.05) and `late-share ≈ 0.26` (droppable blocks
  mostly early → decode-attribution sparsity, not skippable FLOPs).

## Step 0-quant: certified quantization precision (`QUANT_SWEEP.txt`, `step0_quant.py`)

Same data, the quantization side of the unified certificate: how few bits of relative precision on the
per-block decode contributions does each position tolerate? `b = log₂((L1[w]+L1[v]) / gap(w,v))`
(worst-case; RMS uses L2 — independent rounding).

| model (corpus) | nb | N | cert bits med (worst-case) | RMS bits med (p10–p90) | static 0%res | static @10%res | static @40%res |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (sci) | 49 | 115 | 5.8 | 3.5 (2.0–5.8) | 11.4 | 7.7 | 6.0 |
| Qwen2.5-0.5B (code) | 49 | 111 | 4.3 | 2.0 (0.8–5.0) | 11.1 | 7.1 | 4.5 |
| Qwen2.5-Coder-0.5B (sci) | 49 | 115 | 5.5 | 3.2 (1.5–5.3) | 12.2 | 7.7 | 6.0 |
| Qwen2.5-7B (sci) | 57 | 80 | 5.4 | 3.4 (1.8–5.8) | 12.6 | 7.9 | 5.9 |

- **Adaptive:** per-position the decode tolerates ~**5–6 bit** (worst-case) / ~**3 bit** (realistic)
  contributions vs fp16's 16 — large, cashable headroom.
- **Static is VIABLE:** the worst (near-tie) position needs ~11–13 bits, but at a **10% residue the global
  bit-width is ~7–8** (≤8-bit ship-able), ~5–6 at 40%. The sharp contrast with static *prune* (~0 blocks).
- **Scale-flat:** ~5.4–5.8 cert / ~7.7–7.9 static@10% across 0.5B→7B — quantizability tracks the
  margin/sensitivity ratio (scale-stable), unlike prune's adaptive yield (grew with scale but stayed
  uncashable).

## Prune vs quant — which knob cashes in

| knob | static viable? | adaptive yield | cashes in as |
|---|---|---|---|
| **prune** | NO (~0 blocks) | grows w/ scale (signed 0.18→0.42) | FLOPs — but droppable blocks mostly *early* → little real saving |
| **quant** | **YES** (~7–8 bit @10% residue) | ~3–6 bits/position | **bandwidth/storage on every weight** ✓ |

**Quantization is the forge's cashable lever** — now measured: certified, scale-stable bit savings on
every weight, where pruning is static-dead and FLOP-cash-in-capped. (The contribution-bit figures are a
favorable proxy for weight-bits: a weight error spreads over the hidden dim, so true weight bit-width is
no worse.)

## Verdict

- **Static `--certified-prune`: no ore — do not build.** (Step 0's job: it just saved that build.)
  Static ≈ 0 at every scale (0.5B → 7B); robust.
- **Adaptive per-token prune: thin at 0.5B, grows with scale** (7B signed med 0.42, p90 0.72), but capped
  by the small-margin residue (forge-tax positions, legitimately incompressible) and by **limited compute
  cash-in** (droppable blocks mostly early → not skippable FLOPs) — a cap that does *not* shrink with scale.
- **Quantization is the cashable knob — measured (Step 0-quant).** ~3–6 certified bits/position adaptive,
  ~7–8 bit static (10% residue) global, scale-flat — real bandwidth savings on every weight, where
  pruning is static-dead and FLOP-cash-in-capped. Build the **quant** path, not static prune.
- **The upstream lever is margins** (`pil`): every ratio here — prune and quant — is margin-bound, capped
  by the small-margin forge-tax residue. Higher / better-conditioned margins raise all of them.

## Scope / honesty

One small model (0.5B, 49 blocks), CPU, two ≈110-position corpora — **not** the Pythia/Qwen ladder. The
qualitative findings (static≈0, signed≫budget, droppable-mostly-early, certificate load-bearing) are
likely robust; magnitudes need the size/architecture sweep (Step 0-i). This is decode/argmax-lossless,
not softmax-lossless. `[empirical]`
