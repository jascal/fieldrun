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
transition + the emergence definition, de-confounded against confidence, replicated within family; the exact
power-diagram geometry; the causal fragility of composed tokens) is real. The **causal ablation is now done**
(§5c) — it *confirms* composed-is-fragile but *tempers* the redundancy claim: redundancy-beyond-margin is weak
causally (the readout μ_t stays the strong evidence). Before "publish" it still needs: a **cross-architecture /
cross-scale replication** (two Qwen-0.5B models is seed-replication, not family — blocked on a non-Qwen rope
bundle + store/holdout); a bigger margin-matched ablation (several cells are n-starved); full-spectrum `μ_t`
(not top-12); derivations for the asserted training-dynamics claims; and verified citations.

## 5b. Exact power-diagram geometry — and composition is NOT a near-miss of the KB

`--probe-facet` exposes the final residual `r(x)` (`Model::final_residual`) and computes, over *all* 151,936
tokens, the **exact** nearest power-diagram facet `argmin_{v≠t} (L_t − L_v)/‖U_t − U_v‖` (the token cells in
`r`-space are the Laguerre power diagram of `{U_v}`; the normalized margin is the *exact* Euclidean distance to
the `t`–`v` bisector). Both models, 300 positions:

- **Exact nearest-facet distance is monotone RETRIEVED ≫ SELECTED > COMPOSED** (coder 2.23/1.34/1.03; instruct
  2.78/1.45/1.22). The runner-up proxy used elsewhere *is* the true nearest facet **89%** of the time.
- **Killer check — refuted.** Hypothesis: "composition = `r(x)` crossing the facet *out of the KB's cell*."
  The nearest facet is the bisector with the *KB's own prediction* only **14%/8%** of COMPOSED (15%/17% of
  SELECTED). So for **~85% of COMPOSED the KB's prediction isn't even the nearest competitor** — composition is
  a *non-local* divergence (KB's cell not adjacent), not a near-miss of the rule.
- **The ~14% near-miss subclass IS one thing: function-word & morphology competition.** The `pick ⟂ KB-pred`
  pairs are overwhelmingly closed-class glue picking *interchangeable* alternatives — `a⟂the` (×4, both models),
  `will⟂is`, `were⟂be`, `she⟂I`, `with⟂to`, `;⟂,`, `tell⟂say`; COMPOSED adds subword suffixes `-ler⟂-ling`,
  `-ful⟂-y`, `-quent⟂-quence`. The closed-class/grammar regime where the KB is strongest. So: RETRIEVED
  (model=KB, deep cell) / COMPOSED-85% (genuine divergence, KB geometrically far) / near-miss-15% (function-word
  coin-flip the rule also offered — *not* novel computation).

## 5c. Causal ablation — composed is fragile, but redundancy-beyond-margin is weak

`--probe-ablate` knocks out the top-k DLA circuits in the *forward pass* (`hidden_ab` re-runs with the heads/
neurons zeroed; `Model::predict_ablated`) and asks whether the prediction flips — converting the μ_t readout
into a causal intervention.

- **Route-ordered fragility, replicated.** flip@k1 RETRIEVED 25%/25% < SELECTED 36%/52% < COMPOSED **59%/75%**
  (coder/instruct). Knock out just the *top* circuit and COMPOSED flips ~2× as often as RETRIEVED — composed
  tokens are *causally* fragile (emergent), retrieved ones robust.
- **De-confound (flip@k1 within matched margin bins) is mixed/weak.** Covered flips less than composed in the
  *low*-margin bin on both models (coder 69<86, instruct 80<100), but weak/absent in mid and n-starved in high
  (composed n=3–5). So the flip-ordering is **largely margin** (composed near the boundary), with only a faint
  redundancy residual.
- **Resolution of the readout↔causal split.** The *readout* μ_t shows *strong* redundancy-beyond-margin; the
  *causal* ablation shows it's mostly margin. Why: **redundant encoding (high μ_t) ≠ causal robustness when the
  margin is thin.** The redundant supporters are individually *weak* (PR≈45, none > ~10% of the logit), so even
  though *many* circuits point at `t`, removing the top few can still drop it below the runner-up if the cushion
  is small. μ_t-redundancy and ablation-robustness are **distinct properties that decouple at thin margin** —
  which is exactly why the readout looked strong while the causal test looked margin-dominated.

## 6. Open math questions (with empirical status)

- **Q1 (tropical/Boolean boundary).** The retrieval/composition boundary as alignment between the U
  power-diagram and the KB cells; the margin = facet distance. *Status:* RETRIEVED-deep-in-cell confirmed and
  de-confounded vs confidence; the **exact** nearest-facet computed (§5b) — but the elegant "composition =
  crossing the KB's facet" is **refuted** (~14% only; the rest diverge non-locally). Owed: the literal
  pushforward-`r#μ` PCA/alignment test.
- **Q4a (no magnitude dominance).** `PR(x) ≥ k` a.s. under a superposition/incoherence hypothesis. *Status:*
  PR~45 route-invariant, both models — supported.
- **Q4b (code-multiplicity transition — the new object).** `μ_t ≫ 1` for coverable, `μ_t ≈ 0` for composed,
  independent of margin and PR. The reconciliation question: how is a token the argmax of *many* circuits yet
  *no* circuit dominates the magnitude (geometry of redundant weak codes)? And the emergence definition:
  "argmax of a sum that is the argmax of no summand." *Status:* the live frontier; not yet a theorem. NB the
  causal ablation (§5c) shows redundancy (μ_t) and robustness *decouple at thin margin* — they're distinct
  properties, so a theorem must relate `μ_t`, PR, and margin jointly (not μ_t ⇒ robustness).
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
# combine vs select + Grok's falsifiers (PR, normalized margin, μ_t multiplicity, margin-controlled, margin↔μ_t corr)
fieldrun --bundle <qwen> --ids <holdout.json> --store <store.json> --probe-dla --n-eval 500
# exact power-diagram nearest facet + the killer check + near-miss subclass (§5b; rope arch — needs final_residual)
fieldrun --bundle <qwen> --ids <holdout.json> --store <store.json> --probe-facet
# causal: ablate top-k DLA circuits in the forward pass → flip rate by route, margin-de-confounded (§5c; rope arch)
fieldrun --bundle <qwen> --ids <holdout.json> --store <store.json> --probe-ablate --n-eval 200
```

All modes are explain-only; the decode/forward path is untouched (no faithfulness-gate risk).
