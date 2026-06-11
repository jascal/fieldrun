# Projective Incidence Calculus (PIC)

**A weighted, inner-product generalization of incidence calculus for the transformer core**

*Status: research proposal / different paper from the fieldrun decompiler work. The empirical
anchors live in [`FINDINGS.md`](./FINDINGS.md) §5c (the `--probe-ablate` causal-attribution
program on two Qwen2.5-0.5B models); this document is the theory that program motivates.*

---

## Abstract

The decision logic of a transformer's composition core is **not** a discrete weighted-vote or
Horn-clause conflict resolution. Empirically (see §1) it is an **additive signed-evidence
accumulation over correlated hypotheses, with a weighted-threshold (power-diagram) decision and a
non-additive, diffuse repair term**. Bundy's incidence calculus is the right ancestor — it was
invented precisely to track the *joint* that probability is not truth-functional over — but it must
be generalized in two ways the data forces: **set-valued → signed-measure-valued incidences**, and
**∩/∪ → weighted threshold**. We propose **Projective Incidence Calculus (PIC)**: incidences live in
an inner-product space, propositions are frame elements `U_v`, the **Gram kernel `G_{vw} = ⟨U_v,U_w⟩`
is the structural carrier of non-truth-functionality**, evidence accumulates additively in log-space
(a product-of-experts), and probability is recovered exactly as a Gibbs/softmax incidence frequency.
A second, **non-monotone fixpoint layer** captures the part of the core that is *computed, not
retrieved* (the "forge tax"), and is provably diffuse in the high-participation-ratio regime.

PIC's distinctive contribution is to **formally separate the retrievable fragment of a model's logic
(additive, exact, compactly representable) from the computed fragment (a high-PR distributed
fixpoint with no compact symbolic form)** — a separation that the fieldrun probes make measurable.

---

## 1. Empirical desiderata (measured, not assumed)

Every axiom below is constrained by a measured result from the `--probe-ablate` program (two
Qwen2.5-0.5B models, natural-text holdout, n=300; FINDINGS §5c).

| # | Desideratum | Measured anchor |
|---|---|---|
| D1 | **Additivity.** Source `j` contributes `c_j^v = ⟨d_j, U_v⟩`; margin between `t` and competitor `v*` is `Δ = Σ_j D_j`, `D_j = c_j^t − c_j^{v*}`. | Coalition test: `sign(ΣD_j − Δ)` predicts joint-ablation flips at ~75–83% across k=1,2,3,5. |
| D2 | **Cardinality-inertness.** The *count* of sources whose isolated argmax is `t` (`μ_t`) has ~0 causal weight given `{D_j, Δ}`. | Logistic `flip ~ Δ + D_j + 1[μ_t≥2]`: dropping `μ_t` costs +0.0035 / +0.000 log-loss. |
| D3 | **Non-truth-functionality = a kernel.** Competition `t` vs `v*` hardens as `ρ = cos(U_t,U_{v*})` rises; `D_j = ⟨d_j, U_t−U_{v*}⟩ → 0` (common-mode). | `|D_j|` falls 1.47→0.86 with ρ; flip% 53→18%. |
| D4 | **Weighted-threshold connective beyond Horn.** COMPOSED = argmax of a sum that is the argmax of no summand: head entailed by `Σ wᵢxᵢ > θ` with *no sufficient sub-conjunction*. | Route decomposition; `μ_t = 0` for ~76–84% of COMPOSED tokens. |
| D5 | **Two layers.** The linear identity `flip ⟺ Δ < ΣD_j` is a near-perfect *necessary* condition (mispredicts a non-flip 1–6%) but **not sufficient**: a diffuse repair rescues ~50% of would-be flips, scales with `Δ`, and resists localization. | Necessary-not-sufficient (fn 3–17/300); rescue 14→61% with Δ; diffuse (18–34% un-breakable by any single block). |

---

## 2. Objects

- **Inner-product space** `H`. **Sources** `S` (circuits) with embeddings `d_j ∈ H`. **Propositions**
  `V` (tokens/outcomes) with directions `U_v ∈ H` (the unembedding rows).
- **Projective pairing** `c : S × V → ℝ`, `c_j^v = ⟨d_j, U_v⟩` (the direct logit attribution, DLA).
- **Aggregated evidence** `r = Σ_j d_j` (the residual stream); **logits** `L_v = ⟨r, U_v⟩ = Σ_j c_j^v`.
- **Gram kernel** `G_{vw} = ⟨U_v, U_w⟩` — the proposition frame's inner-product structure.
- **Differential incidence (pivotality)** `D_j^{t,v} = ⟨d_j, U_t − U_v⟩` — the calculus's atomic causal
  quantity (the object `μ_t` was a noisy proxy for, per D2).

**Projective incidence of a proposition** `v` is the functional `⟨·, U_v⟩` on `H` (replacing Bundy's
incidence *set* `i(v) ⊆ I`). Two propositions' "incidence overlap" is **not** a set intersection but
the kernel value `G_{vw}` — propositions are frame elements, so their incidences overlap *intrinsically*
by `G`. This is the central move: **`G` is structural because propositions share an inner-product
space, not because of any contingent input correlation.**

---

## 3. Connectives and inference

- **Combination = signed linear accumulation in log-space** (not ∩/∪): `L_v = Σ_j c_j^v`. This is a
  **product-of-experts**: each source multiplies a proposition's incidence weight by `exp(c_j^v)` (§5).
- **Decision = weighted threshold / argmax** over competing propositions — a Laguerre power diagram
  with weights `‖U_v‖²`; the margin is the facet distance `Δ / ‖U_t − U_v‖` (measured: the normalized
  margin is the exact facet distance, FINDINGS §5b).
- **Bounding / resolution analog** (incidence calculus's actual inferential power): given a *known
  subset* `S' ⊆ S` of source contributions, bound the decision margin from the partial sum `Σ_{j∈S'} D_j`
  plus a residual bound on `S∖S'`. The coalition result is the empirical face of this "weighted
  incidence resolution"; soundness/completeness is **Open Problem O1**.
- **Sufficient support & support number.** Let `σ(t)` = size of the smallest `S' ⊆ S` whose partial
  sum already crosses threshold. **RETRIEVED ≈ small `σ`** (a Horn body exists); **COMPOSED ≈ large
  `σ`** (no small sufficient body, D4). Conjecture `σ ∼ PR` (**O2**).

---

## 4. Two-layer architecture

PIC has a **monotone additive core** and a **non-monotone fixpoint closure**.

- **Additive (incidence) core.** The map `r ↦ (L_v)` and the threshold decision. Fully a PIC formula,
  monotone, *exactly* reconstructed by summing source contributions (within a single forward pass the
  residual stream is literally additive; the final LayerNorm contributes a per-position positive
  scalar). This is the **retrievable fragment** and it is where T1–T3, T5 live.
- **Fixpoint closure (the forge tax).** Model the forward pass as an iterated operator `T` on the
  evidence vector; the additive core is its linearization at the operating point, and the **repair /
  cushion is the higher-order closure** `T^∞ − (linearization)`. This is the **computed fragment** — it
  appears only under *intervention* (the off-diagonal Jacobian: how `d_j` change when an upstream
  source is ablated), is diffuse, and is **provably non-localizable in the high-PR regime** (T4).

The cleanest one-line statement of the whole architecture:

> **T5 lives in the static decomposition (additive, exact). T4 lives in the intervention response
> (non-additive, diffuse). They are different layers, and the same model exhibits both.**

---

## 5. Theorems

Aim to *recover the measured desiderata as consequences*, not to posit them.

- **T1 (cardinality-inertness).** Under the projective pairing, the decision depends only on
  `{D_j, Δ}` and is invariant to `μ_t`. *[Recovers D2.]*
- **T2 (non-truth-functionality budget).** Competition hardness between `t` and `v` is a monotone
  function of `ρ_{tv} = G_{tv}/√(G_{tt}G_{vv})`; as `ρ → 1`, `D_j → ` common-mode and differential
  incidence collapses. PIC reduces to Boolean incidence calculus exactly when `G` is diagonal.
  *[Recovers D3; identifies the diagonal-`G` limit with classical incidence calculus.]*
- **T3 (weighted-threshold expressivity).** COMPOSED conclusions (`σ > 1`, no sufficient
  sub-conjunction) are exactly the formulae **not** expressible in Horn/∩-∪ incidence calculus but
  expressible with the weighted-threshold connective. *[Formalizes D4 / "beyond Horn".]*
- **T4 (diffuseness / no compact rule).** Any causal property realized as `E = Σ_m e_m` with equitable
  `e_m ∼ E/PR` has single-source influence `O(1/PR)`; hence no bounded-size PIC formula localizes it,
  and `P(single-module intervention alters E) ≈ 1/PR`. *[Recovers D5 / the diffuse, PR-bounded repair;
  the `--head-sweep` probe measures the `1/PR` constant directly.]*
- **T5 (recovered probability).** The Gibbs measure `P(v) ∝ exp(L_v)` is recovered exactly as a PIC
  incidence frequency (§6). *[Bridge to classical "probability as proportion of worlds".]*

T1–T3 are fully pinned by current data; T4 is being measured now; T5 is constructed below.

---

## 6. T5 — recovered probability, and where `G` is structural

### 6.1 Product-of-experts recovery (exact)

Maintain a measure over proposition labels with uniform base `M_0`. Each source `j` reweights every
proposition's mass multiplicatively by `exp(c_j^v)`. After all sources,
`m(v) = M_0 · exp(Σ_j c_j^v) = M_0 · exp(L_v)`, so uniform sampling gives
`m(v)/Σ_w m(w) = exp(L_v)/Z` — **exactly the Gibbs/softmax measure** (the model's output). Each source
is an "expert"; the core is a **product-of-experts / log-linear model**. This is exact and
parameter-free.

### 6.2 Where `G` is structural — the resolution

A correlated discrete-choice (random-utility) representation `U_v = L_v + η_v + γ_v` with
`η_v = ⟨ξ, U_v⟩`, `ξ ∼ N(0, Id_H)` makes `Cov(η_v, η_w) = G_{vw}` and `γ_v` iid Gumbel — a probit/GEV
model. It is a useful **interpretive lens** (random utility, hard near-synonyms = correlated
alternatives) and connects PIC to discrete-choice econometrics (McFadden; the GEV theorem; nested
logit).

**But the added-noise route has a tension worth stating plainly:** exact softmax requires
`Var(ξ) → 0`, at which point the *added* `G`-covariance vanishes; for `Var(ξ) > 0`, `G` is present but
the recovered measure is a *smoothed* softmax, not the exact one. So "exact recovery **and**
structural `G` simultaneously" cannot come from added correlated noise.

The resolution is that **`G` is already structural without any added noise — it is the Gram of the
proposition frame `{U_v}`.** Because propositions are elements of `H`, the logits `L_v = ⟨r, U_v⟩` are
*intrinsically* correlated across `v` by `G`, and the competition hardness (T2) is read directly off
the frame Gram. The exact measure is recovered by §6.1 (multiplicative reweighting, equivalently iid-
Gumbel argmax). Therefore:

> **T5 (final form).** Keep iid-Gumbel / multiplicative reweighting for *exact* Gibbs recovery; take
> `G` to be the proposition-frame Gram, which structures both the static logits (T5) and the
> competition geometry (T2) **at once, with no variance trade-off**. The correlated-noise probit/GEV
> model is an *equivalent representation* useful for importing econometric machinery, not the locus of
> `G`.

This unifies T5 and T2 into one object — the frame `{U_v, G}` — without the added-noise tradeoff.

### 6.3 Empirical hook (runnable; delimits T5 vs T4)

For a forward pass, let `Ĺ_v = Σ_{j ∈ measured circuits} c_j^v` be the reconstructed logit and
`e_v = L_v^true − Ĺ_v` the residual.

- **Static / exact (T5).** `softmax(Ĺ)` vs the model's true distribution, and `‖e‖` vs the number of
  circuits summed. Prediction: the static residual is ~0 once *all* components are summed (residual-
  stream additivity), so the curve `‖e‖` vs top-`k`-circuits measures **decompiler completeness /
  logit sparsity** — how few sources reconstruct the logit to within ε. (Proposed mode:
  `--probe-reconstruct`.)
- **Interventional / diffuse (T4).** The same reconstruction after ablating a coalition; the residual's
  growth quantifies the diffuse fixpoint closure. Ablating any small downstream set changes the
  residual by `O(1/PR)` (T4).

The static residual being ~0 *confirms* T5 empirically and cleanly delimits it from T4 (where the
residual is the forge tax).

---

## 7. Related work

- **Incidence calculus** (Bundy 1985): the truth-functional-incidence / non-truth-functional-
  probability split. PIC is its signed-measure, inner-product generalization; the diagonal-`G` limit
  recovers it.
- **Log-linear / energy-based models & Markov logic networks**: PIC's additive core *is* a log-linear
  model; PIC adds the inner-product (frame-Gram) coupling and the retrievable/computed split.
- **Random-utility / discrete choice (GEV, nested logit, probit; McFadden)**: §6.2's interpretive lens;
  the natural home for the correlated-alternatives (high-ρ) regime.
- **Dempster–Shafer / valuation algebras**: signed-measure combination machinery.
- **ProbLog with aggregates / neural predicates**: the weighted-threshold connective (D4/T3).
- **Tropical / Laguerre geometry**: the argmax-of-sum = power diagram (already used in FINDINGS §5b).

PIC's stake: **incidence calculus with signed-measure incidences in an inner-product space, the Gram
kernel as the explicit non-truth-functionality operator, and a fixpoint closure that formally
separates the additive (retrievable) logic from the diffuse (computed) remainder.**

---

## 8. Open problems

- **O1** Soundness/completeness of weighted incidence resolution (the coalition bound as inference).
- **O2** `σ(t) ∼ PR`: is the support number (smallest sufficient circuit set) the participation ratio?
  Ties D2 (cardinality-inertness) to D4 (no small sufficient body).
- **O3** T4 under the §6 model: prove `P(single-module repair) ≈ 1/PR` from the fixpoint closure; match
  the `--head-sweep` constant.
- **O4** A PIC *syntax*: compile PIC formulae directly from DLA traces (the retrievable fragment as an
  extractable program), with COMPOSED positions flagged as "no compact formula" (the forge tax).
- **O5** Cross-architecture / cross-scale invariance of `G`'s spectrum and the retrievable/computed
  split (the program-wide thesis).

---

## 9. Provenance

This proposal is the product of a theory–experiment loop: the fieldrun `--probe-ablate` program
supplied the measured desiderata (D1–D5); a collaborator (Grok) contributed the product-of-experts T5
recovery and the correlated discrete-choice framing; the structural-frame-Gram resolution of §6.2 and
the static/interventional (T5/T4) delimitation are refinements from that exchange. Every quantitative
claim traces to a probe in [`FINDINGS.md`](./FINDINGS.md) §5; nothing here is posited that the probes
do not constrain.
