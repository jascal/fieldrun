# FINDINGS — KB attribution & the geometry of "conflict resolution" in a decompiled transformer

A research thread that grew out of Phase 8b (the retrieval-pruned output head). It uses fieldrun's
KB-vs-composition decomposition to ask a sharp question: **when an LLM picks a token, is it retrieving a
symbolic rule, selecting within a rule-proposed set, or computing something new — and what does the
*mechanism* look like in each case?** All of it is explain-only (inference untouched); the tooling is the
`--attribute`, `--prune-head`, `--probe`, and `--probe-dla` CLI modes.

Models: **Qwen2.5-Coder-0.5B-Instruct** and **Qwen2.5-0.5B-Instruct** (same Qwen2.5 vocab/tokenizer, so a
shared model-captured KB store, `store_Qwen2.5-1.5B`). Holdouts: a natural-text and a code token stream.
Numbers below are ~300–500 contexts at ctx-window 64; treat them as indicative, not high-precision.

## 1. The three-way routing of each next-token decision

Given the KB's candidate set for a context, classify the model's argmax `t`:

- **RETRIEVED** — a single KB idiom's top-1 == `t` (a pure symbolic lookup).
- **SELECTED** — `t` is in the candidate set but is *not* any idiom's top-1 (the set contains the answer;
  the choice within it is made elsewhere).
- **COMPOSED** — `t` is in no rule's output (the forge tax — genuinely computed).

Decomposition of the model's labour (`--attribute`, natural text): **~25% RETRIEVED, ~60% SELECTED, ~15%
COMPOSED.** Composition is mostly *disambiguation within a retrieved set*, not generation from nothing; only
~15% is from-scratch (and this is KB-relative — a richer KB shrinks it).

Regime dependence (candidate-set coverage of the model's argmax == top-1 fidelity of a pruned head, by an
exact subset identity): at ~540 KB candidates, coverage is **~85% on natural text vs ~63% on code** — code
is computed, not retrieved.

## 2. Is SELECTED a function of the rule-firing state? (`--probe`)

Forward-chaining framing: the candidate set is the *conflict set*, SELECTED is *conflict resolution*.

- **A fixed strategy doesn't reproduce it.** "Pick the highest-count successor" (max-incidence) reproduces
  only **11% of SELECTED on natural text, 1.1% on code**; rank-2 is the mode. Classical syntactic conflict
  resolution (recency/specificity/refractoriness) is ruled out a fortiori.
- **The conflict set carries most of the choice but underdetermines it.** Conditioning on the last token
  (which fixes the bigram conflict set) drops `H(pick)` from 5.56→1.75 bits (natural, ~68%), 6.14→1.43
  (code, ~77%). Refining the key lowers it further, but ~1.4–1.75 bits residual remain.

## 3. Combine vs select — the DLA concentration (`--probe-dla`)

Per token, decompose the predicted logit additively over circuits, `L_t = Σ_i c_i`, and measure
concentration over the **full** candidate spectrum (~245 head+neuron candidates).

**No selection primitive (magnitude).** The participation ratio `PR = (Σc_i)²/Σc_i² ≈ 42–49` and is
**route-invariant** (RETRIEVED ≈ SELECTED ≈ COMPOSED, both models). No single circuit dominates the logit
magnitude — *ever*, even for tokens a single n-gram rule reproduces perfectly. The mechanism is a uniformly
distributed ~45-way additive sum + argmax.

## 4. Two falsifiers (Grok collaboration) — what the routes DO separate on

The retrieval/composition split is **not** a magnitude distinction (uniform). It separates on two other axes:

- **Geometry — decision margin.** The normalized margin `(L_t − L_v)/‖U_t − U_v‖` (= distance to the nearest
  unembedding power-diagram facet) is large for RETRIEVED, small for COMPOSED. *Robust:* RETRIEVED ≫ rest on
  both models (2.4–2.9 vs ~1.0–1.5). *Not robust:* the fine SELECTED-vs-COMPOSED ordering (clean on the coder,
  within noise on the non-coder at n=500). So state it as **RETRIEVED ≫ {SELECTED, COMPOSED}**.

- **Redundancy — single-circuit readout multiplicity `μ_t`.** `μ_t(x) = #{top-12-by-DLA circuits whose isolated
  argmax is t}`. Means (coder / instruct): RETRIEVED 1.13/0.83, SELECTED **1.45/1.06**, COMPOSED 0.23/0.31.
  Strict-emergence fraction (`μ_t = 0`): COMPOSED **84%/76%**, covered 25–52%. So coverable tokens are
  **redundantly multiply-realized** (many *individually sufficient* circuits, none necessary — magnitude still
  distributed, PR~45); ~80% of COMPOSED are **emergent** (readable from no single circuit, present only in the
  ~45-way sum). Note: μ_t is *not* monotone with margin — SELECTED has the highest μ_t but RETRIEVED the highest
  margin. The ~16–24% of COMPOSED with `μ_t ≥ 1` is a real subclass: *the model has a single-circuit rule the
  n-gram KB lacks* (model-retrievable, not corpus-retrievable). Caveat: μ_t over the top-12 is a lower bound,
  so the strict-emergence fraction is an upper bound.

**De-confounding (is it just confidence?).** Within matched normalized-margin bins, the covered−composed
redundancy gap **persists** (low/mid bins, where COMPOSED n is adequate): coder 65/71% vs 17/16%; non-coder
52/50% vs 23/16%. So the split is **not** "the KB covers the confident predictions" — at matched confidence,
covered tokens are ~2–4× more single-circuit-readable. (Caveat: the high-margin bin has COMPOSED n≈16, too
sparse to trust; "COMPOSED flat across *all* margins" is coder-specific, not established.)

## 5. The characterization

> The mechanism is a uniformly-distributed ~45-way additive sum + argmax (no selection in magnitude). The
> symbolic reducibility of the output tracks an axis orthogonal to *both* magnitude (uniform PR) *and*
> confidence (controlled out): **single-circuit-readout multiplicity** `μ_t = #{circuits i : argmax c_i = t}`.
> Coverable tokens are redundantly readable (`μ_t ≫ 1`, redundant distributed agreement); COMPOSED tokens are
> emergent (`μ_t ≈ 0`, the answer is the argmax of the *sum* but of *no* summand). COMPOSED = near a
> power-diagram facet + emergent-from-combination + no rule = the cleanest "computed, not retrieved" we have.

This is a *kind of conflict resolution with no named precedent*: redundant distributed voting shading to
emergent combination. Ingredients have precedent (connectionist production systems / DCPS; superposition,
Elhage et al.; product-of-experts; Bundy's incidence calculus over a continuous learned space), the fusion
does not. **Tropical-geometry lens (Grok):** the unembedding induces a power diagram of ℝ^d; margin = facet
distance; high `μ_t` = many tropical monomials (circuit terms) sharing the winning term for `t`; emergence =
"the winning term of the tropical sum that wins in no summand." A good *framing* for the discussion, not
evidence by itself.

**Margin–μ_t (Grok's prediction that deeper cells recruit more redundancy):** confirmed but WEAK — per-position
corr(margin, μ_t) = +0.12/+0.18 (covered), up to +0.32 (SELECTED, instruct); positive on both models. The
route-level anti-correlation (RETRIEVED high-margin/low-μ_t vs SELECTED low-margin/high-μ_t) is a Simpson's
paradox. r≈0.15 (~2% shared variance) means margin (geometry) and `μ_t` (code-multiplicity) are **largely
independent axes** — good for the two-axis framing, with only a weak positive coupling.

**Publication status: strong preprint *direction*, not finished.** The novel core (the `μ_t` code-multiplicity
transition + the emergence definition, de-confounded against confidence, replicated within family) is real, but
before "publish" it needs: (1) a **causal ablation** — knock out the top circuits and show covered tokens
survive while composed collapse (the "redundancy" claim is currently observational, not causal); (2) a
**cross-architecture / cross-scale replication** (two Qwen-0.5B models is seed-replication, not family); plus
the full-spectrum `μ_t` (not top-12), derivations for the asserted training-dynamics claims, and verified cites.

## 6. Open math questions (with empirical status)

- **Q1 (tropical/Boolean boundary).** The retrieval/composition boundary as alignment between the U
  power-diagram and the KB cells; the margin = facet distance. *Status:* RETRIEVED-deep-in-cell confirmed and
  de-confounded vs confidence. Owed: the literal pushforward-`r#μ` PCA/alignment test.
- **Q4a (no magnitude dominance).** `PR(x) ≥ k` a.s. under a superposition/incoherence hypothesis. *Status:*
  PR~45 route-invariant, both models — supported.
- **Q4b (code-multiplicity transition — the new object).** `μ_t ≫ 1` for coverable, `μ_t ≈ 0` for composed,
  independent of margin and PR. The reconciliation question: how is a token the argmax of *many* circuits yet
  *no* circuit dominates the magnitude (geometry of redundant weak codes)? And the emergence definition:
  "argmax of a sum that is the argmax of no summand." *Status:* the live frontier; not yet a theorem.
- Q2 (incidence-granularity entropy rate / forge-tax as positive asymptotic residual), Q3 (continuous
  incidence calculus & failure of truth-functionality, measurable via SAE features), Q5 (rank of the
  resolution map), Q6 (MDL of the boundary / ILP-over-COMPOSED) — open, measurable on this decompile.

## 7. Reproduce

```bash
# attribution decomposition (RETRIEVED/SELECTED/COMPOSED) + per-idiom breakdown
fieldrun --bundle <qwen> --ids <holdout.json> --store <store.json> --attribute
# coverage sweep + conditional analysis (the pruned-head / forge-tax curve)
fieldrun --bundle <qwen> --ids <holdout.json> --store <store.json> --prune-head
# is SELECTED a function of the firing state? (rank dist + conflict-set entropy)
fieldrun --bundle <qwen> --ids <holdout.json> --store <store.json> --probe
# combine vs select + Grok's falsifiers (PR, normalized margin, single-circuit redundancy, margin-controlled)
fieldrun --bundle <qwen> --ids <holdout.json> --store <store.json> --probe-dla --n-eval 500
```

All modes are explain-only; the decode/forward path is untouched (no faithfulness-gate risk).
