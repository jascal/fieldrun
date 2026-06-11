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

## 5c. Causal ablation — composed is fragile, and μ_t-redundancy confers no causal protection (decoupling, margin-matched + confound-controlled)

`--probe-ablate` knocks out the single top-DLA circuit in the *forward pass* (`hidden_ab` re-runs with the head/
neuron zeroed; `Model::predict_ablated`) and asks whether the prediction flips — converting the μ_t readout into a
causal intervention. k=1 (cheap → enough positions for a μ_t × margin split), n=300, both Qwen2.5-0.5B models,
natural-text holdout, matched-vocab store.

- **Route-ordered fragility, replicated.** flip@k1 RETRIEVED 22%/26% < SELECTED 40%/48% < COMPOSED **54%/61%**
  (coder/instruct). Knock out just the *top* circuit and COMPOSED flips ~2.4× as often as RETRIEVED — composed
  tokens are *causally* fragile (emergent), retrieved ones robust. But this tracks margin (RETRIEVED Δ≈1.4–1.6 vs
  COMPOSED Δ≈0.7), so it must be de-confounded.

- **Grok's decisive falsifier — μ_t split WITHIN matched margin bins.** Grok's incoherence-regime proof predicts the
  flip is governed by margin Δ and PR, *not* μ_t — so at matched margin, μ_t≥2 (redundantly read) and μ_t=0 (strictly
  emergent) should flip at the *same* rate; a protective gap (high-μ_t flips *less*) would refute decoupling =
  redundancy is causally protective. Result — flip% | mean PR | `t→` (= % of ablated circuits that are themselves
  t-supporters, isolated argmax == t), both models:

  | bin (mean Δ) | μ_t≥2 flip / PR / t→ | μ_t=0 flip / PR / t→ | gap |
  |---|---|---|---|
  | coder low 0.19  | 76% / 41 / 80% | 69% / 53 / 0% | +7  |
  | coder mid 0.65  | 41% / 40 / 91% | 27% / 52 / 0% | +14 |
  | coder high 1.91 | 17% / 39 / 71% |  7% / 44 / 0% | +10 |
  | instr low 0.17  | 88% / 37 / 62% | 73% / 48 / 0% | +15 |
  | instr mid 0.68  | 46% / 37 / 54% | 40% / 46 / 0% |  +6 |
  | instr high 2.15 | 17% / 32 / 51% |  7% / 43 / 0% | +10 |

  **Margin is the governor** — flip collapses 76→41→17 (μ_t≥2) and 69→27→7 (μ_t=0) across margin terciles, *identically*
  for both μ_t levels. The residual μ_t gap is small and in the **anti-protective** direction (+6 to +15pp; high-μ_t
  flips *more*, not less), and the `t→` control explains it exactly: the μ_t≥2 group ablates a *confirmed* t-supporter
  51–91% of the time vs **0%** for μ_t=0 (structural — μ_t=0 has no individually-t-aligned circuit to remove), so the
  high-μ_t group strips more pivotal mass. PR is flat — even slightly *higher* in μ_t=0 (44–53 vs 32–41) — so PR doesn't
  drive the gap either. ⇒ **decoupling confirmed, redundancy-protection falsified.** The deepest reading: in the μ_t≥2
  cells we remove a confirmed t-supporter *and ≥2 such supporters exist*, yet flip still tracks margin alone — the
  redundant backups (PR≈40, individually < ~10% of the logit) provide essentially no cushion.

- **(B-clean) the airtight backup test — redundancy is *non-compensatory*.** Restrict to `t→`=1 (we *always* ablate a
  confirmed t-supporter), then split μ_t=1 (no backup left) vs μ_t≥2 (≥1 backup remains) at matched margin — this holds
  the which-circuit confound fixed by construction, so the *only* difference between arms is whether redundant backups
  exist. Pooled over both models (μ_t=1 / μ_t≥2): low-Δ 90% / 80%, mid 36% / 40%, high 4% / 21%. **Backups confer no
  robust protection** — flat in the bulk, *anti*-protective at high Δ (small n), and only a faint non-significant ~10pp
  protective hint at the very thinnest margin (the facet, where any cushion would matter most). ⇒ superposition
  redundancy is **non-compensatory**: removing one t-supporter is *not* caught by the others — no error-correction
  dynamics in the forward pass, so apparent agreement (many readers) ≠ fault tolerance. This is stronger than "μ_t
  doesn't predict robustness": by the linear flip identity (flip ⟺ Δ < D_j = c_j^t − c_j^{v*}, j = ablated circuit),
  μ_t is a property of circuits we *don't* touch, so it's *structurally* irrelevant to single-ablation — the real causal
  variable is the **ablated circuit's pivotality D_j vs the margin Δ**, of which μ_t is a noisy proxy. (The high-Δ
  anti-protective blip is almost certainly D_j selection — μ_t≥2 high-margin tokens happen to carry a more dominant top
  circuit — itself the next thread: regress flip on D_j/Δ directly.)

- **(D_j regression) the causal variable is the ablated circuit's pivotality, not μ_t.** Exposed each circuit's
  contribution to the *runner-up* (`dla_v`, explain.rs) → per-circuit pivotality **D_j = dla − dla_v** (ablating
  circuit j shifts the t-vs-v\* margin by −D_j). The **linear flip identity** flip ⟺ Δ < D_j holds as a near-perfect
  *necessary* condition: binning the linear flip score s = D_j − Δ, actual flip steps cleanly at s=0 (coder 0–4% below
  → 45–80% above; instruct 11–15% → 60–78%), and sign(s) mispredicts a *non*-flip only 3/300 (coder) / 17/300 (instruct)
  times — when D_j < Δ the token essentially never flips. It is *not sufficient* (fp 60/51): when s>0, indirect/
  downstream recomposition **rescues** t about half the time (indirect effects are overwhelmingly protective — ~60
  rescues vs ~3 betrayals). Matching on s, μ_t≥2 *appears* to flip less, but the per-cell Δ exposes the **margin
  confound** — μ_t≥2 sits at higher Δ at matched s (coder mid 0.72 vs 0.41; instruct high 1.00 vs 0.32). The principled
  control settles it: logistic `flip ~ Δ + D_j + 1[μ_t≥2]` (Δ,D_j standardized) gives Δ **−4.21/−3.11**, D_j
  **+2.82/+1.16**, μ_t≥2 **−0.60/+0.06** (opposite signs across models = noise around 0); **dropping μ_t costs
  +0.0035/+0.000 mean log-loss** — nothing. ⇒ **μ_t is a proxy for (Δ, D_j) position, not an independent cause**;
  decoupling confirmed at the regression level. Aside: |w_Δ| > |w_Dj| on both ⇒ the margin protects *beyond* the linear
  identity (the indirect-rescue channel scales with Δ) — which is *why* flip ⟺ Δ<D_j is necessary but not sufficient.

- **(A/B) the incoherence boundary + Δ-cushion (Grok's derivations, run on both models).** ρ = cos(U_t, U_{v\*});
  among the s>0 set (linear identity predicts a flip), a **rescue** = the forward pass keeps t (indirect recomposition).
  Grok's derivation is **2/3 confirmed**:
  - **(B) Δ-cushion — confirmed.** Rescue rate rises monotonically with Δ at ~matched s (coder 14→39→50→61%; instruct
    9→36→33→71%). Higher margin ⇒ more downstream rescue — the quantified reason flip ⟺ Δ<D_j is necessary-not-
    sufficient and |w_Δ| > |w_Dj|.
  - **(A) geometry — confirmed.** mean|D_j| and flip% both fall with ρ (coder |D_j| 1.47→0.86, flip 53→18%; instruct
    1.56→0.84, 54→28%): near-synonym competitors have small pivotality D_j = c_j·(U_t−U_{v\*}) (common-mode cancels).
  - **(A) stochastic-rescue collapse — falsified.** Grok predicted σ(ρ)∝√(1−ρ²)→0 ⇒ rescue→0 at high ρ; instead rescue
    does *not* fall with ρ (coder 31→44%, instruct 26→40% — flat-to-rising). At high ρ the *linear* lever (D_j) weakens
    but the *indirect* rescue does not — likely because high-ρ flips involve tiny *absolute* D_j perturbations the
    forward pass trivially compensates. So near-synonyms are hard to edit because D_j is small, **not** because rescue
    starves. (`Model::unembed_cos`, rope; explain-only.)

- **(coalition additivity) ΣD_j predicts joint ablation; the cushion is finite; a new-winner channel opens at large k.**
  Ablating the top-k circuits *jointly* (k=1,2,3,5), the coalition linear identity flip ⟺ Δ < ΣD_j (sk = ΣD_j − Δ):
  - **(1) additivity holds** — sign(sk) vs forward-flip accuracy stays flat at ~75–83% across k on both models. The
    *individually*-measured D_j's **add**; indirect effects don't corrupt the sum (I'd predicted additivity would break
    — it didn't).
  - **(2) cushion exhausts** — rescue rate among sk>0 falls monotonically with k (coder 35→25→16→16%, instruct
    31→22→17→11%): stripping more pivotality leaves the forward pass less headroom to rescue, so larger coalitions are
    more reliably destructive (Grok's "coalition exceeding the cushion", confirmed).
  - **(3) a new-winner channel opens** — fn (flip despite sk<0) rises with k (coder 3→17, instruct 17→32) while fp
    falls: at large k the post-ablation argmax becomes a *third* token the t-vs-v\* identity doesn't model (the global
    power-diagram "surprise", made measurable).
  ⇒ the editing-budget rule is **ΣD_j > Δ + cushion(Δ,ρ)**, with the cushion exhausting as the coalition grows, plus a
  multi-facet correction at large k.

- **(rescue localization) the rescue is downstream but DIFFUSE — not a localizable lever.** Two robust facts (both
  models): (1) the top-DLA circuit is **always late** — L_top mean 20–21 of 24 layers; the highest-direct-attribution
  circuit sits near the output, so the rescue must operate within the last ~4 layers + final norm. (2) Sweeping {top +
  a whole downstream layer's **attention** *or* **MLP** block} per k=1 rescue: the MLP carries rescue comparably to
  attention (Δdepth1 attn/MLP coder 32/36%, instruct 20/33%), and breakability by *some single* downstream block rises
  to attn 42/58% · MLP 56/67% · **either 66/82%** — but **18–34% of rescues survive every single-block ablation**
  (genuinely distributed across multiple blocks), with a per-depth profile that **does not replicate** across models. So
  the rescue is *not* concentrated in a small set of late heads/blocks. ⇒ Grok's "localize δ to a few heads → a cheap
  hardening/editing lever" is **not supported**; there is no surgical rescue target. This mirrors the rest of the thread:
  the repair is diffuse *for the same reason* μ_t-redundancy is causally inert and PR≈45 — the whole system is
  **distributed-superposition, readout AND repair**. Grok's **PR→localizability lemma** (P(single-module un-rescue) ≈
  1/PR) is **order-of-magnitude confirmed** (`--head-sweep`, 2198/1694 single-downstream-head ablations): per-head
  un-rescue **4.1% / 4.0%** vs 1/PR **2.5% / 2.8%** (PR 40/35) — a ~1.4–1.6× constant above 1/PR (mild within-substrate
  concentration; repair not *perfectly* equitable). Cross-check: the measured per-*layer* (≈14-head) rate ~32% is
  *below* the 1−(1−0.04)¹⁴ ≈ 44% that independent heads predict ⇒ a layer's heads **share** repair (positively
  correlated un-rescues), so the repair is diffuse *across* layers but partly *redundant within* a layer. (Caveats:
  whole-block ablation is destructive = upper-bound/non-specific; L_top-always-late limits depth dynamic range.
  `Model::dims`/`predict_ablated_blocks`, rope; explain-only.)

- **Resolution of the readout↔causal split.** The *readout* μ_t separates routes strongly (coverable redundantly read,
  μ_t≫1; composed strictly emergent, μ_t≈0). The *causal* ablation shows that redundancy is **inert** under
  intervention: **redundant encoding (high μ_t) ≠ causal robustness when the margin is thin**, because the redundant
  supporters are individually weak (PR≈40). μ_t-redundancy and ablation-robustness are **distinct properties that
  decouple at thin margin in the incoherent regime** — which is exactly why the readout looked strong while the causal
  test is margin-dominated. *(Hedge, per "no necessity claims": shown at matched margin in this thin-margin/incoherent
  regime; whether a clean multi-ablation removing **all** t-supporters surfaces protection — and whether decoupling
  **breaks** for near-synonym runner-ups, where the incoherence assumption fails — is the open follow-on, §6 Q4b.)*

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
  "argmax of a sum that is the argmax of no summand." *Status:* the live frontier; not yet a theorem. The causal
  ablation (§5c) now **confirms decoupling causally** — margin-matched + `t→`-controlled, *and* a logistic
  `flip ~ Δ + D_j + 1[μ_t≥2]` on both models isolates the causal variables as the margin **Δ** and the ablated
  circuit's pivotality **D_j = c_j^t − c_j^{v\*}** (μ_t's independent log-loss value ≈ 0) — so μ_t is a *proxy*, and the
  theorem to write is about **D_j vs Δ plus an indirect-cushion term that scales with Δ** (|w_Δ|>|w_Dj|), not μ_t. The
  cushion term and the geometry are now **empirically confirmed** (§5c A/B); the one **falsified** sub-model is Grok's
  σ(ρ)∝√(1−ρ²) rescue-collapse — rescue does not weaken at high ρ. The sharp open test:
  Grok's proof rests on an **incoherence assumption** (a circuit's push toward `t` is ≈ independent of its push toward
  the runner-up `v*`); it predicts decoupling should **break** — redundancy becoming protective — exactly when `v*` is
  a near-synonym of `t` (high `cos(U_t, U_{v*})`). Splitting the flip by runner-up coherence would confirm the proof
  *by finding its boundary*. Not yet run.
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
# causal: ablate the top DLA circuit → flip; Grok μ_t-falsifier (flip split by μ_t WITHIN matched margin bins,
# + per-cell PR and t→ which-circuit control) → decoupling confirmed (§5c; rope arch — needs predict_ablated)
fieldrun --bundle <qwen> --ids <holdout.json> --store <store.json> --probe-ablate --n-eval 300
```

All modes are explain-only; the decode/forward path is untouched (no faithfulness-gate risk).
