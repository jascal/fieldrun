# FINDINGS_PYTHIA — cross-architecture replication of the FINDINGS battery on the Pythia ladder

The cross-architecture / cross-scale replication FINDINGS §5 lists as the publication blocker, run on
**GPT-NeoX (Pythia 70m / 160m / 410m; 1b pending)** via the new `neox` arch (faithfulness-gated top-1 exact
vs the pure-numpy `scripts/neox_ref.py`: 160m 60/60, 70m/410m 20/20), plus a **Qwen2.5-0.5B-Instruct
re-baseline under the identical store recipe**, so the KB is held constant across architectures.

**Setup (differs from FINDINGS — read before comparing numbers).** Store: a *corpus* n-gram store
(`scripts/build_store.py`) over ~2.4M tokens of Gutenberg novels, per-model tokenizer, NOT the
model-captured pylm store FINDINGS used; holdout: the disjoint 15% tail of the same corpus (in-domain
book prose — an easy, high-coverage regime). n: 500 (attribute/probe/probe-dla), 300 (facet),
150 (ablate). μ_t over top-12 DLA circuits (lower bound, as in FINDINGS). All numbers indicative.

## The replication table

Route split (`--attribute`, 500 positions):

| model | RETRIEVED | SELECTED | COMPOSED |
|---|---|---|---|
| FINDINGS (Qwen-0.5B×2, natural text, pylm store) | ~25% | ~60% | ~15% |
| Qwen-0.5B-Instruct (books store) | 26.4% | 60.8% | 12.8% |
| pythia-70m | 35.6% | 56.6% | 7.8% |
| pythia-160m | 35.8% | 54.2% | 10.0% |
| pythia-410m | 36.8% | 51.6% | 11.6% |

`--probe-dla` (PR / exact-margin proxy / μ_t; per route R·S·C):

| model | PR (route-invariant?) | margin R / S / C | μ_t mean R / S / C | μ_t=0 (strict emergence) R / S / C |
|---|---|---|---|---|
| Qwen-0.5B | 38.1 / 36.4 / 38.6 ✓ | 3.62 / 1.67 / 1.44 | 0.93 / 1.17 / 0.23 | 42% / 37% / **81%** |
| pythia-70m | 13.2 / 13.4 / 12.5 ✓ | 3.09 / 1.00 / 0.76 | 0.12 / 0.08 / 0.03 | 88% / 93% / 97% |
| pythia-160m | 26.7 / 27.6 / 26.6 ✓ | 2.74 / 0.88 / 0.77 | 0.09 / 0.15 / 0.06 | 91% / 86% / 94% |
| pythia-410m | 62.8 / 58.1 / 64.1 ✓ | 3.20 / 1.14 / 0.82 | 1.85 / 1.78 / 0.36 | 32% / 30% / **74%** |

`--probe-facet` (exact nearest facet, 300 positions) and `--probe-ablate` (150 positions):

| model | exact facet dist R / S / C | v\*==runner-up | v\*==KB-top1 on COMPOSED (killer) | flip@k1 R / S / C |
|---|---|---|---|---|
| Qwen-0.5B | 3.24 / 1.66 / 1.30 | 86% | 7% | 24% / 41% / 53% |
| pythia-70m | 3.22 / 0.92 / 0.86 | ~82% | 11% | 6% / 19% / 33% |
| pythia-160m | 2.78 / 0.76 / 0.84 | ~85% | 9% | 3% / 12% / 8% |
| pythia-410m | 3.29 / 0.97 / 0.89 | ~83% | 10% | 8% / 32% / 40% |

Conflict resolution (`--probe`): baseline H(pick) over SELECTED 5.6–5.9 bits on every model; the fixed
max-incidence strategy reproduces only 13–15% of SELECTED (FINDINGS: 11%, rank-2 modal) — a fixed
syntactic conflict-resolution strategy is ruled out cross-architecture.

## What replicates across architecture (the robust core)

1. **The three-way routing and its proportions.** Retrieval-dominated labour (R+S ≈ 87–92%) with a
   ~8–15% forge tax, on a *second architecture family* and under a *different KB construction*. The
   Qwen re-baseline (26/61/13 vs FINDINGS ~25/60/15) also shows the decomposition is **store-robust** —
   a corpus n-gram store reproduces the model-captured store's split almost exactly.
2. **Q4a — no magnitude dominance, route-invariant PR.** PR is flat across routes at every scale tested
   (and on both architectures). No selection-in-magnitude anywhere.
3. **The margin geometry: RETRIEVED ≫ {SELECTED, COMPOSED}.** Exact nearest-facet distance ~3.2 vs
   ~0.8–1.0 on every Pythia size; same shape as Qwen. The fine SELECTED-vs-COMPOSED ordering is present
   but small — exactly FINDINGS' "not robust" caveat.
4. **The killer check stays refuted.** For ~90% of COMPOSED tokens the nearest facet is NOT the KB's
   prediction (9–11% across the ladder; Qwen 7%) — composition is a non-local divergence from the KB,
   not a near-miss, on both architectures.
5. **The runner-up proxy** is the true nearest facet ~82–86% of the time (FINDINGS: 89%).
6. **Causal fragility is route-ordered and margin-governed.** flip@k1 R < S ≤ C at 70m and 410m; within
   matched margin bins flips collapse to ~0 in the high bin on every model and COMPOSED ≥ COVERED in the
   low bins — consistent with the §5c decoupling result (margin is the governor), not with
   redundancy-as-protection. (160m's COMPOSED column is n=13 — too sparse to read.)
7. **Conflict resolution is not a fixed strategy** (max-incidence 13–15%, H(pick) ≈ 5.6–5.9 bits).

## The new, scale-dependent result: μ_t redundancy is something models grow into

At 70m/160m, μ_t ≈ 0 *everywhere* — covered tokens have almost no individually-sufficient circuits
(μ_t≥1 on only 7–14% of covered tokens), so the covered-vs-composed redundancy *gap* exists only as a
compressed remnant against the floor. At 410m the full Qwen-shaped transition appears: covered tokens
μ_t≥1 68–70% vs COMPOSED 26%, strict emergence 74% (C) vs ~31% (R/S) — quantitatively matching
Qwen-0.5B (81% vs 37–42%). Meanwhile the margin geometry and route structure are already fully formed
at 70m. So the two axes of the FINDINGS characterization **decouple developmentally**: the power-diagram
geometry (margin) is scale-stable, while single-circuit readout multiplicity (μ_t) is **scale-emergent**,
appearing between 160m and 410m in this family — consistent with §5c's verdict that μ_t is a readout
property (a proxy), not the causal variable. PR also grows roughly with width/depth (13 → 27 → 63),
so the "~45-way distributed sum" of FINDINGS is itself a scale parameter, not a constant.

## Caveats

- Corpus n-gram store, book-prose holdout (high-coverage regime); FINDINGS' natural-text/code regime
  contrast not yet rerun here.
- pythia-1b battery still running at the time of writing; 154-checkpoint training-dynamics sweeps not
  started (the ladder bundles + store recipe make them mechanical now).
- probe-ablate COMPOSED cells are n=12–20; the 160m flip inversion is within that noise.
- μ_t over top-12 is a lower bound (as in FINDINGS); the 70m/160m "μ_t≈0 floor" should be re-checked
  full-spectrum before leaning on the developmental claim.
- neox lacks `residual_decomp`, so the logic-export/LE-T5 probes (`--probe-reconstruct`,
  `export --logic`) don't run on Pythia yet — a mechanical port.

## Reproduce

```bash
# bundle (validated): fieldrun convert --model EleutherAI/pythia-410m --arch neox --dtype f32
# gate:               python3 scripts/neox_ref.py <hub-dir> holdout.json --ctx 64 --n 60 --dump np.txt
# store/holdout:      python3 scripts/build_store.py --text corpus.txt --tokenizer <bundle>.tokenizer.json \
#                       -o store.json --holdout holdout.json
fieldrun --bundle pythia-410m --ids holdout.json --store store.json --attribute     # routes
fieldrun --bundle pythia-410m --ids holdout.json --store store.json --probe         # conflict set
fieldrun --bundle pythia-410m --ids holdout.json --store store.json --probe-dla --n-eval 500
fieldrun --bundle pythia-410m --ids holdout.json --store store.json --probe-facet
fieldrun --bundle pythia-410m --ids holdout.json --store store.json --probe-ablate --n-eval 200
```
