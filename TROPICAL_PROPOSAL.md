# Tropical Geometry of the Decision Surface

**The (max,+) algebra and tropical rank of the transformer core — and the forge tax as a tropical-rank floor**

*Status: research proposal / a third paper, distinct from both the fieldrun decompiler work and the
[Projective Incidence Calculus](./PIC_PROPOSAL.md) (PIC) proposal. Where PIC is the probabilistic
**logic** of evidence accumulation (soft, temperature 1, the recovered measure), this is the
**geometry/algebra** of the decision surface (hard, temperature 0, the argmax and its complexity).
The two are the same object at two temperatures (§6). Measured anchors live in
[`FINDINGS.md`](./FINDINGS.md) §5b (`--probe-facet`).*

---

## Abstract

A transformer's next-token decision is `argmax_v ⟨r, U_v⟩` over the unembedding frame `{U_v}`. The
**max-logit function** `M(r) = max_v (⟨r, U_v⟩ + b_v)` is therefore a **tropical polynomial** in the
residual `r`: its monomials are the unembedding rows, its **tropical hypersurface is the decision
boundary**, and its linear regions are the **Laguerre power-diagram cells** (one per token) — a
structure fieldrun already measures exactly (`--probe-facet`: the normalized margin is the Euclidean
distance to the nearest facet). This proposal develops the consequences: (i) the forward map
input→logit is a **tropical rational function** (the ReLU/PWL-net → tropical-geometry lineage), (ii)
**emergence (COMPOSED tokens) = interior tropical points** whose winning region is dominated by no
single source's monomial, and (iii) — the distinctive thesis — the **tropical rank of the core's
decision map lower-bounds any retrieval table that reproduces it, so the "forge tax" is a tropical-
rank floor**: the gap between the model's tropical rank and the (tropical-rank-1) lookup a flat KB can
express. Finally (§6), the tropical decision is the **zero-temperature Maslov dequantization of PIC's
Gibbs measure** (`log-sum-exp → max`, `softmax → argmax`), making the two papers exact complements
rather than overlapping accounts.

---

## 1. Why tropical, and why a separate paper from PIC

The decision layer is *literally* tropical: `argmax` over a sum of linear forms is the (max,+)
semiring (`a ⊕ b = max(a,b)`, `a ⊗ b = a + b`). PIC covers the *soft* accumulation and its recovered
measure; it stops at the weighted-threshold decision. Tropical geometry is the right tool for the
**hard decision surface itself** — its cells, boundaries, vertex/region *count*, and *rank* — none of
which PIC develops. The two share exactly one object (the power diagram = PIC's weighted-threshold =
the tropical variety) and are otherwise disjoint in method: PIC borrows probabilistic-logic /
discrete-choice machinery; this paper borrows tropical algebra and the geometry of piecewise-linear
maps. The forge-tax-as-tropical-rank thesis (§5) is unique to this paper and ties directly to the
program's rank-`r` entangled-core findings.

---

## 2. The decision surface as a tropical variety (measured)

**Setup.** `L_v(r) = ⟨r, U_v⟩ + b_v`. The decision is `argmax_v L_v(r)`. Define the **max-logit**

> `M(r) = ⊕_v (b_v ⊗ x^{U_v}) = max_v (⟨r, U_v⟩ + b_v)`,

a tropical polynomial in `r` with monomial exponents `U_v` (the unembedding rows) and tropical
coefficients `b_v`. `M` is convex, piecewise-linear; its **Newton polytope** is `conv{U_v}` (which
tokens can ever win), and its **tropical hypersurface** `T(M)` (the locus where the max is attained by
≥2 monomials) is the decision boundary.

**Measured anchors (FINDINGS §5b, `--probe-facet`, two Qwen2.5-0.5B models, all 151,936 tokens):**
- **TT1 (cells = power diagram).** The linear regions of `M` are the Laguerre power diagram of `{U_v}`
  (weights from `b_v`, `‖U_v‖²`); the cell containing `r` is the predicted token. *[The exact nearest
  facet is computed over the full vocabulary.]*
- **TT2 (margin = tropical distance).** The normalized margin `(L_t − L_v*)/‖U_t − U_v*‖` is the exact
  Euclidean distance from `r` to the nearest facet of `T(M)`. *[Measured: monotone RETRIEVED ≫ SELECTED
  > COMPOSED — coder 2.23/1.34/1.03, instruct 2.78/1.45/1.22; the runner-up proxy is the true nearest
  facet 89% of the time.]*

So §2 is not a conjecture — the decision surface *is* the tropical variety of `M`, and fieldrun
already measures its facet distances. This is the paper's solid floor.

---

## 3. The forward map as a tropical rational function

The input→logit map is a composition of linear maps with piecewise-linear nonlinearities (SiLU/GELU
are smooth but PWL-approximable; attention softmax is the soft part). For the **decision** (the hard
argmax), the relevant object is the PWL skeleton. Following the ReLU-net → tropical-geometry lineage
(Zhang–Naumann–Lim 2018: a ReLU network computes a tropical rational function `p ⊘ q` of tropical
polynomials, and its number of linear regions is bounded by Newton-polytope vertex counts), the core's
decision map is (approximately, on its PWL skeleton) a **tropical rational map**, and:

- **TT3 (region count).** The number of distinct decision behaviors the core can express is bounded by
  the vertices of the Newton polytope of its tropical-rational representation — a *capacity* statement
  about the composition core, parallel to (and finer than) parameter counts.

*Status: structural (inherited from the PWL-net→tropical lineage); the softmax/attention part is the
caveat — quantifying how much of a real transformer's decision map is captured by its tropical skeleton
is **Open Problem TO1**.*

---

## 4. Emergence as interior tropical points

PIC frames COMPOSED as "argmax of a sum that is the argmax of no summand" (`σ > 1`, no sufficient
sub-conjunction). The tropical translation is sharp:

- **TT4 (emergence = non-monomial interior).** Decompose `M(r) = max_v Σ_j c_j^v`. A position is
  **RETRIEVED** when the winning cell is already selected by a *single source's* monomial (some `d_j`
  whose isolated argmax is the winner — a dominated vertex); it is **COMPOSED** when the winning region
  is interior to the tropical variety in the sense that *no single source's monomial attains the max* —
  the cell exists only in the *sum* of monomials. This is the tropical face of `μ_t = 0` and of PIC's
  weighted-threshold-beyond-Horn (T3).

*Runnable test (proposed `--probe-tropical`):* per position, check whether any single circuit's
isolated argmax equals the model's token (the dominated-monomial case) vs none (the interior case);
this is exactly the `μ_t` machinery already in `--probe-ablate`, re-read geometrically. So TT4 is
*measurable now* and largely *already measured* (the μ_t = 0 fraction is the interior-point fraction).

---

## 5. The distinctive thesis — forge tax as a tropical-rank floor

This is what makes the tropical view its own paper rather than PIC's geometry chapter.

**Tropical rank.** A tropical matrix factorization `A = B ⊗ C` (`B`: `n×r`, `C`: `r×m`, tropical
product) of rank `r` expresses `A` with `r` "tropical components." For a decision map, `r` ≈ the number
of distinct linear pieces / monomials needed to reproduce its cell structure (Develin–Santos–Sturmfels
tropical rank; Barvinok rank).

**The retrieval baseline is tropical-rank-bounded.** A flat retrieval table (a KB lookup: "context key
→ stored next-token logits") is a tropical map whose monomials are exactly the *stored keys* — one
tropical term per row. Composition (the forge tax) is precisely the decision regions that require
**monomials not in the table** — sums/combinations of stored keys that create new cells (TT4's interior
points). Hence:

- **TT5 (forge tax = tropical-rank gap, *conjecture*).** Let `ρ_trop(core)` be the tropical rank of the
  core's decision map and `ρ_trop(KB)` the tropical rank of the best flat retrieval table at matched
  coverage. The **forge tax is the irreducible region of `ρ_trop(core) − ρ_trop(KB)`** — the decision
  cells that no lookup table reproduces because they are composed (interior) monomials. The COMPOSED
  fraction (measured ~15% / ~37% natural/code) is the empirical shadow of this gap.

**Tie to the program's rank-`r` findings.** This connects the tropical rank to the *measured* entangled-
core results (the `min_to_run` rank ladder; the finding that a frozen-linear core plateaus at a Θ(d)
floor that **retraining a rank-8 update beats losslessly**; data-aware low-rank beating plain SVD at
matched rank). The tropical reading predicts *why* a linear (SVD) rank misranks the core: the core's
complexity is **tropical**, not linear — its hardness is the number of *tropical* monomials (decision
cells), which a Frobenius/linear rank does not measure. **TO2:** is the gap between linear rank and
tropical rank of the core exactly the data-aware-vs-SVD gap we measured?

*Status: §5 is the conjectural spine. It is the contribution; it is also the least pinned. Mark it
clearly as a program, with TT5/TO2 as the falsifiable core.*

---

## 6. The bridge to PIC — Maslov dequantization (exact)

PIC recovers the Gibbs measure `P(v) ∝ exp(L_v / T)`. As the temperature `T → 0`:

> `T · log Σ_v exp(L_v / T) → max_v L_v` (log-sum-exp → max), and `softmax(L/T) → argmax`.

This is **Maslov dequantization** (idempotent analysis): the tropical (max,+) semiring is the `T → 0`
limit of the log-semiring that PIC lives in. Therefore:

- **TT6 (dequantization).** The tropical decision surface of this paper is the **zero-temperature limit
  of PIC's competition geometry**. The power diagram = `lim_{T→0}` of the softmax cells; PIC's
  non-truth-functionality kernel `ρ_{tv} = cos(U_t,U_v*)` (T2) becomes the **tropical facet angle** (how
  sharply two monomials cross); PIC's smoothed-softmax competition is the `T > 0` "viscosity"
  regularization of the tropical variety.

So the two papers are *one object at two temperatures*: PIC = soft logic at `T=1` (the measure, the
forge-tax-as-residual), Tropical = hard geometry at `T=0` (the cells, the rank, the forge-tax-as-
tropical-rank). They cite each other across this limit; neither subsumes the other.

---

## 7. Theorems / claims, by status

| Claim | Content | Status |
|---|---|---|
| TT1 | Decision cells = Laguerre power diagram of `{U_v}` | **Measured** (§5b) |
| TT2 | Margin = exact tropical-hypersurface distance | **Measured** (§5b) |
| TT3 | Region-count bounded by Newton-polytope vertices | Structural (PWL→tropical lineage); softmax caveat = TO1 |
| TT4 | Emergence = interior (non-monomial) tropical points = `μ_t=0` | **Measurable now** (largely measured) |
| TT5 | Forge tax = tropical-rank gap `ρ_trop(core) − ρ_trop(KB)` | **Conjecture** (the thesis) |
| TT6 | Tropical = `T→0` Maslov dequantization of PIC | Exact (idempotent analysis) |

---

## 8. Open problems

- **TO1** Quantify how much of a real transformer's decision map is captured by its tropical (PWL)
  skeleton vs the soft attention/softmax part — i.e. the fidelity of the `T→0` approximation per layer.
- **TO2** Linear rank vs tropical rank of the core: is their gap the measured data-aware-vs-SVD gap
  (the entangled-core rank ladder)? This is the bridge from TT5 to the program's measured rank results.
- **TO3** Compute (or bound) the tropical rank of a real unembedding+core; estimate the number of
  decision linear regions empirically (sample `r`, count distinct argmax cells visited).
- **TO4** `--probe-tropical`: measure the interior-point (COMPOSED) fraction as the dominated-monomial
  test, and the tropical facet angle as the `T→0` image of PIC's `ρ` (cross-validates TT4/TT6 against
  the existing `μ_t` and `--probe-facet` data).
- **TO5** Newton-polytope structure of `{U_v}`: which tokens are *vertices* (can win a cell on their own,
  retrievable) vs *interior* (only ever composed)? A vocabulary-level retrievable/computed map.

---

## 9. Related work

- **Tropical geometry of neural networks** (Zhang, Naumann, Lim, ICML 2018): ReLU nets = tropical
  rational maps; linear-region counts via Newton polytopes. The structural backbone of §3/TT3.
- **Idempotent analysis / Maslov dequantization** (Litvinov, Maslov): the `T→0` log-semiring → (max,+)
  limit; the exact bridge to PIC (§6/TT6).
- **Tropical rank** (Develin–Santos–Sturmfels; Barvinok rank): the rank notions for TT5.
- **Power / Laguerre diagrams** (Aurenhammer): the decision-cell geometry (TT1), already measured.
- **PIC companion** ([`PIC_PROPOSAL.md`](./PIC_PROPOSAL.md)): the `T=1` soft-logic dual; the power
  diagram = PIC's weighted-threshold decision.

The stake this paper claims: **the transformer decision surface as an explicit tropical variety whose
cells, margins, and *rank* are measurable, with the forge tax identified as a tropical-rank floor that
linear (SVD) rank structurally cannot see — and the whole thing the zero-temperature limit of PIC.**

---

## 10. Acknowledgment & provenance

This is the geometric/`T=0` dual of [`PIC_PROPOSAL.md`](./PIC_PROPOSAL.md), and through the Maslov
bridge (§6) it shares that paper's lineage: **the whole two-temperature program descends from Alan
Bundy's incidence calculus (1985)** — PIC removes Bundy's orthogonality assumption at temperature 1,
and this paper takes the resulting object to temperature 0, where the incidence cells become a tropical
variety. The tip of the hat is Bundy's; we have only added a thermometer.

Same theory–experiment loop. The power-diagram / facet-distance results (TT1/TT2) are measured in
`--probe-facet`; the tropical-monomial framing of emergence appears as a "lens" in FINDINGS §4/§6
(explicitly flagged there as framing, not theorem); the tropical-rank thesis (§5) and the Maslov-
dequantization bridge (§6) are this proposal's contributions. Conjectural sections are marked; the
measured floor (§2) stands on the existing probes.
