# Tropical Geometry of the Decision Surface

**The (max,+) algebra and tropical rank of the transformer core — and the forge tax as a tropical-rank floor**

*Status: research proposal / a third paper, distinct from both the fieldrun decompiler work and the
[Projective Incidence Calculus](./PIC_PROPOSAL.md) (PIC) proposal. Where PIC is the probabilistic
**logic** of evidence accumulation (soft, temperature 1, the recovered measure), this is the
**geometry/algebra** of the decision surface (hard, temperature 0, the argmax and its complexity), and
[Logic Export](./LOGIC_EXPORT.md) is the **executable** form (the same object as a semiring-Datalog
program). The three are one theory in three categories (semantics / geometry / computation); a result
in any is a result in the others (§6). Measured anchors live in [`FINDINGS.md`](./FINDINGS.md) §5b
(`--probe-facet`).*

---

## Abstract

A transformer's next-token decision is `argmax_v ⟨r, U_v⟩` over the unembedding frame `{U_v}`. The
**max-logit function** `M(r) = max_v (⟨r, U_v⟩ + b_v)` is therefore a **tropical polynomial** in the
residual `r`: its monomials are the unembedding rows, its **tropical hypersurface is the decision
boundary**, and its linear regions are the **Laguerre power-diagram cells** (one per token) — a
structure fieldrun already measures exactly (`--probe-facet`: the normalized margin is the Euclidean
distance to the nearest facet). This proposal develops the consequences with the three papers that
ground it — Zhang–Naitzat–Lim (the PWL-net → tropical lineage), Pachter–Sturmfels (polytope propagation
= the geometric sum-product), and Maragos–Charisopoulos–Theodosis (the constructive max-plus toolkit:
convex regression, zonotope pruning, sparse max-plus solutions): (i) the forward map input→logit is a
**tropical rational map** (Zhang–Naitzat–Lim Thm 5.4), so its decision behaviour is bounded by
**Newton-polytope vertex counts**; (ii) **emergence (COMPOSED tokens) = interior tropical points** whose
winning region is dominated by no single source's monomial; (iii) the forward pass's Newton polytope
**propagates by Minkowski-sum / convex-hull exactly as fieldrun's semiring-Datalog logic export runs**
(Pachter–Sturmfels polytope propagation — TT7); and (iv) — the distinctive thesis — the **tropical rank
of the core's decision map lower-bounds any retrieval table that reproduces it, so the "forge tax" is a
tropical-rank floor** (TT5), with Maragos's sparse max-plus solution giving a *constructive* handle on
that floor (TT8). Finally (§6), the tropical decision is the **zero-temperature Maslov dequantization of
PIC's Gibbs measure** (`log-sum-exp → max`, `softmax → argmax`), making the three papers exact
complements rather than overlapping accounts. §11 specifies `--probe-tropical` against the existing
`--probe-facet` / `--probe-ablate` machinery.

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

a tropical polynomial in `r` whose monomial exponents are the unembedding rows `U_v ∈ ℝ^d` (real
exponents → affine monomials; this is the PWL/degree-1 special case of Zhang–Naitzat–Lim **Def. 2.3**)
and whose tropical coefficients are the biases `b_v`. `M` is convex and piecewise-linear (a tropical
polynomial is convex PWL — Z-N-L §2). Two classical objects pin it exactly:

- **Newton polytope** (Z-N-L **Def. 3.2**): `Δ(M) = conv{U_v}` — *which tokens can ever win*. Lift each
  `U_v` to `(U_v, b_v) ∈ ℝ^{d+1}`; the **dual subdivision** `δ(M)` is the projection of the **upper
  faces** of `P(M) = conv{(U_v, b_v)}`. Each vertex of `δ(M)` is one linear region (one token cell), so
  *the number of upper-hull vertices of `P(M)` bounds the number of decision cells* (Maclagan–Sturmfels
  Prop. 3.1.6, cited by Z-N-L). This is the precise meaning of "retrievable vocabulary" in §5/TO5: a
  token has a non-empty cell **iff** its lifted point `(U_v, b_v)` is an upper-hull vertex of `P(M)`.
- **Tropical hypersurface** (Z-N-L **Def. 3.1**): `T(M) = {r : the max is attained by ≥2 monomials}` —
  the decision boundary; it is the `(d−1)`-skeleton of the polyhedral complex dual to `δ(M)`.

The cells `{r : c_v + ⟨U_v, r⟩ ≥ c_w + ⟨U_w, r⟩ ∀w}` are exactly the **Laguerre power diagram** of
`{U_v}` with weights `‖U_v‖² + 2b_v` (Aurenhammer).

**Measured anchors (FINDINGS §5b, `--probe-facet`, two Qwen2.5-0.5B models, all 151,936 tokens):**
- **TT1 (cells = power diagram).** The linear regions of `M` are the Laguerre power diagram of `{U_v}`;
  the cell containing `r` is the predicted token. *[`--probe-facet` computes the exact nearest facet over
  the full vocabulary; `headgate.rs` already exploits this geometry to gate heads.]*
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
argmax), the relevant object is the PWL skeleton, and the lineage is now precise rather than gestural:

**Zhang–Naitzat–Lim (PMLR v80, 2018), made explicit.** A tropical rational function is a difference of
two tropical polynomials `f ⊘ g = f − g` (**Def. 2.4**; the set of these is a semifield, and each is a
*difference-of-convex* function). Their **Theorem 5.4** is an exact equivalence: *`ν : ℝ^d → ℝ` is a
tropical rational function **iff** it is a feedforward ReLU network (integer/rational weights, linear
output, assumptions (a)–(c))*, and any `f ⊘ g` is realizable by an `L`-layer net with
`L ≤ max{⌈log₂ r_f⌉, ⌈log₂ r_g⌉} + 2` (where `r_f, r_g` are the monomial counts). The layer recurrence
that builds it (**Prop. 5.1**) propagates the convex/concave parts `F^{(l)}, G^{(l)}` by
`A⁺F + A⁻G`-style updates — i.e. the network *is* a tropical rational map by construction. Hence:

- **TT3 (region count).** The number of distinct decision behaviours the core can express is bounded by
  the **upper-hull vertex count of the Newton polytope** of its tropical-rational representation
  (**Def. 3.2** + **Thm. 6.3**, the `L`-layer linear-region bound; the single-hidden-layer case is a
  **zonotope** whose vertices are counted by **Cor. 3.4**, `Σ_{j=0}^d C(m,j)` for `m` generators). This
  is a *capacity* statement finer than parameter counts: depth multiplies regions (Z-N-L: a deeper net
  is exponentially more expressive than a shallow one), and **Lemma 6.2** identifies zonotopes as the
  building blocks composed by Minkowski sum across depth.

*Status: structural (inherited from the PWL-net→tropical lineage); the softmax/attention part is the
caveat — quantifying how much of a real transformer's decision map is captured by its tropical skeleton
is **Open Problem TO1**.*

### 3b. Polytope propagation = the logic export's evaluation (Pachter–Sturmfels)

The link to fieldrun's [`LOGIC_EXPORT.md`](./LOGIC_EXPORT.md) is not analogy — it is the *same algorithm*.
Pachter–Sturmfels (PNAS 101(46):16132–16137, 2004) show that a graphical model is an algebraic variety,
that the **sum-product algorithm evaluates a coordinate of that variety**, and that **parametric
inference** — how the MAP/decode depends on parameters — is governed by the **Newton polytope of the
model**, computed by *polytope propagation*: the same dynamic program as sum-product, but with **numbers
replaced by polytopes**, `×` by **Minkowski sum**, and `+` by **convex hull**. Tropicalizing
(`+ ↦ max`, `× ↦ +`) turns sum-product into max-product (Viterbi) — the MAP decode.

This is exactly the two-semiring picture LOGIC_EXPORT already runs:

> Forward accumulation along the residual stream = bottom-up semiring evaluation of a Datalog program `Π`.
> Under the **log-semiring** (`⊕ = log-sum-exp`, `⊗ = +`) `Π` evaluates to the softmax measure (PIC,
> `T=1`, sum-product); under the **tropical** semiring (`⊕ = max`, `⊗ = +`) to the MAP decode (this
> paper, `T=0`, max-product / Viterbi). Maslov dequantization is the homomorphism between them.

- **TT7 (decode = polytope propagation).** fieldrun's `export --logic` (LO3, one-decision partial
  evaluation, `(max,+)` argmax decode) and `export --logic-whole` (LO3a, context-free whole-model emit)
  are the **tropicalization of Pachter–Sturmfels polytope propagation** specialized to the unembedding
  layer. The Newton polytope that P–S propagate *is* `conv{U_v}` of §2 at the final layer; the
  high-treewidth "dense-Gram wall" (LOGIC_EXPORT LE-T4: the `vocab × d` embed-fact blow-up) is the
  statement that this terminal Newton polytope has no compact propagation, which is the geometric face
  of the forge tax (§5). *Status: structural/exact (a restatement, not a new claim); its value is that
  it gives the logic export a **named prior-art algorithm** and makes the `T=0`/`T=1` duality a semiring
  homomorphism on one polytope-propagation recurrence.*

---

## 4. Emergence as interior tropical points

PIC frames COMPOSED as "argmax of a sum that is the argmax of no summand" (`σ > 1`, no sufficient
sub-conjunction). The tropical translation is sharp:

- **TT4 (emergence = non-monomial interior).** Decompose `M(r) = max_v Σ_j c_j^v`. A position is
  **RETRIEVED** when the winning cell is already selected by a *single source's* monomial (some `d_j`
  whose isolated argmax is the winner — a dominated vertex); it is **COMPOSED** when the winning region
  is interior to the tropical variety in the sense that *no single source's monomial attains the max* —
  the cell exists only in the *sum* of monomials. This is the tropical face of `μ_t = 0` and of PIC's
  weighted-threshold-beyond-Horn (T3). Geometrically: RETRIEVED tokens are **Newton-polytope vertices**
  reachable by a single circuit; COMPOSED tokens win a cell that is only created by the Minkowski-sum of
  several circuits' sub-polytopes (the §3b composition), never by one alone.

*Runnable test (`--probe-tropical`, §11):* per position, check whether any single circuit's isolated
argmax equals the model's token (the dominated-monomial case) vs none (the interior case); this is
exactly the `μ_t` machinery already in `--probe-ablate`, re-read geometrically. So TT4 is *measurable
now* and largely *already measured* (the `μ_t = 0` fraction is the interior-point fraction; the COMPOSED
fraction measured ~15%/~37% natural/code).

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

**Maragos gives a constructive handle (the minimal max-plus table).** Maragos–Charisopoulos–Theodosis
(*Proc. IEEE* 109(5):728–755, 2021) solve the **max-plus / tropical linear system** `A ⊗ x = b`: when no
exact solution exists, the **greatest (principal) subsolution** `x̂_j = min_i (b_i − A_{ij})`
(Cuninghame-Green) is the tightest max-plus fit, and its residual `b − A ⊗ x̂` is the part of the target
*not expressible* by the dictionary `A`. Reading `A` as a candidate retrieval table (rows = stored keys)
and `b` as the core's logits:

- **TT8 (sparse max-plus residual = forge tax, *constructive*).** The greatest-subsolution residual of
  the best max-plus retrieval table is a *computable* lower bound on the forge tax — the per-position
  logit mass that no lookup over the dictionary can reproduce. This makes TT5 testable without computing
  a full tropical rank: fit the table by the closed-form principal solution, measure the residual, and
  correlate it with the COMPOSED / `μ_t=0` fraction (§11, E5). It also connects directly to the
  density-minimization expert work ([`DENSITY_BUCKETING.md`](./DENSITY_BUCKETING.md)): the
  irreducible-atom buckets are a *learned* dictionary `A`, and the residual expert (the catch-all that no
  hub absorbs) is exactly the max-plus residual — the forge tax that no number of buckets dissolves.

**Tie to the program's rank-`r` findings.** This connects the tropical rank to the *measured* entangled-
core results (the `min_to_run` rank ladder; the finding that a frozen-linear core plateaus at a Θ(d)
floor that **retraining a rank-8 update beats losslessly**; data-aware low-rank beating plain SVD at
matched rank). The tropical reading predicts *why* a linear (SVD) rank misranks the core: the core's
complexity is **tropical**, not linear — its hardness is the number of *tropical* monomials (decision
cells), which a Frobenius/linear rank does not measure. **TO2:** is the gap between linear rank and
tropical rank of the core exactly the data-aware-vs-SVD gap we measured?

*Status: §5 is the conjectural spine. TT5 is the thesis; TT8 is its first falsifiable, computable
shadow. Mark §5 clearly as a program, with TT5/TT8/TO2 as the falsifiable core.*

### 5b. Constructive corollaries (Maragos): regression, pruning, approximation

The same toolkit makes the tropical view *actionable*, not just descriptive:

- **Tropical / convex regression** fits a tropical polynomial (a max of affine pieces) to data with an
  optimal solution and an efficient algorithm (Maragos et al., § regression). Applied here: fit a
  **compact PWL surrogate** of the core's decision map and read its monomial count as an *empirical*
  tropical rank (TO9). A surrogate that matches the model only on RETRIEVED positions but diverges on
  COMPOSED ones quantifies the forge tax as a regression residual.
- **Zonotope / Newton-polytope pruning.** Network compression by **minimizing the number of linear
  regions** (zonotope vertices, Cor. 3.4) is Maragos's route to pruning/approximation. fieldrun's
  inverse use: a *retrievable* token is a surviving Newton-polytope vertex; pruning to the vertex set is
  exactly the "fall back to flat lookup on dominated/vertex tokens" hybrid of §7/E7.
- **Morphological (max-plus) layers.** A max-plus perceptron `y = max_j (x_j + w_j)` is one tropical
  monomial; the unembedding `M(r)` is a one-layer morphological network over `{U_v}`. This frames the
  "feature-native model" goal (sae-forge, cross-program) as building *more* of the model in the
  morphological/tropical layer where retrieval is native and the forge tax is explicit.

---

## 6. The bridge to PIC — Maslov dequantization (exact)

PIC recovers the Gibbs measure `P(v) ∝ exp(L_v / T)`. As the temperature `T → 0`:

> `T · log Σ_v exp(L_v / T) → max_v L_v` (log-sum-exp → max), and `softmax(L/T) → argmax`.

This is **Maslov dequantization** (idempotent analysis): the tropical (max,+) semiring is the `T → 0`
limit of the log-semiring that PIC lives in. The same homomorphism is what Pachter–Sturmfels use to pass
from sum-product (the partition function) to max-product (MAP/Viterbi) — so §3b and §6 are one statement
seen from the logic and the geometry sides. Therefore:

- **TT6 (dequantization).** The tropical decision surface of this paper is the **zero-temperature limit
  of PIC's competition geometry**. The power diagram = `lim_{T→0}` of the softmax cells; PIC's
  non-truth-functionality kernel `ρ_{tv} = cos(U_t,U_v*)` (T2) becomes the **tropical facet angle** (how
  sharply two monomials cross — `--probe-tropical` reports it directly, §11); PIC's smoothed-softmax
  competition is the `T > 0` "viscosity" regularization of the tropical variety.

So the three papers are *one object at three views*: PIC = soft logic at `T=1` (the measure), Tropical =
hard geometry at `T=0` (the cells, the rank), Logic Export = the executable semiring program whose
choice of `K` *is* the temperature. They cite each other across this limit; none subsumes the others.

---

## 7. Theorems / claims, by status

| Claim | Content | Status |
|---|---|---|
| TT1 | Decision cells = Laguerre power diagram of `{U_v}` | **Measured** (§5b, `--probe-facet`) |
| TT2 | Margin = exact tropical-hypersurface distance | **Measured** (§5b) |
| TT3 | Region-count bounded by Newton-polytope vertices (Z-N-L Thm 6.3 / Cor 3.4) | Structural (PWL→tropical lineage); softmax caveat = TO1 |
| TT4 | Emergence = interior (non-monomial) tropical points = `μ_t=0` | **Measurable now** (largely measured) |
| TT5 | Forge tax = tropical-rank gap `ρ_trop(core) − ρ_trop(KB)` | **Conjecture** (the thesis) |
| TT6 | Tropical = `T→0` Maslov dequantization of PIC | Exact (idempotent analysis) |
| TT7 | Logic-export decode = tropicalized Pachter–Sturmfels polytope propagation | Exact (restatement; names the prior-art algorithm) |
| TT8 | Greatest-subsolution residual of the best max-plus table = computable forge-tax lower bound | **Constructive / testable** (Maragos; §11 E5) |

---

## 8. Open problems

- **TO1** Quantify how much of a real transformer's decision map is captured by its tropical (PWL)
  skeleton vs the soft attention/softmax part — i.e. the fidelity of the `T→0` approximation per layer.
- **TO2** Linear rank vs tropical rank of the core: is their gap the measured data-aware-vs-SVD gap
  (the entangled-core rank ladder)? The bridge from TT5 to the program's measured rank results.
- **TO3** Compute (or bound) the tropical rank of a real unembedding+core; estimate the number of
  decision linear regions empirically (sample `r`, count distinct argmax cells visited — `--probe-tropical
  --tropical-cells`, §11).
- **TO4** `--probe-tropical`: measure the interior-point (COMPOSED) fraction as the dominated-monomial
  test, and the tropical facet angle as the `T→0` image of PIC's `ρ` (cross-validates TT4/TT6 against
  the existing `μ_t` and `--probe-facet` data).
- **TO5** Newton-polytope structure of `{U_v}`: which tokens are *vertices* (can win a cell on their own,
  retrievable — non-empty Laguerre cell) vs *interior* (only ever composed)? A vocabulary-level
  retrievable/computed map (`--tropical-vertices`, §11).
- **TO6** *(raised by Maragos / P–S)* Practical computation of tropical rank at real vocab sizes
  (`|V| ≈ 150k`, `d ≈ 900`): exact vertex enumeration of `conv{U_v}` is infeasible, so what are the
  sampling/LP estimators, and their error bars? (The greatest-subsolution residual, TT8, sidesteps full
  rank — is it a tight enough proxy?)
- **TO7** *(raised by the program)* Stability of the tropical skeleton under **quantization** (int8/int4
  bundles) and **MoE routing**: does the power-diagram combinatorics survive quantization, and do
  tropical cells align with `--route-frac` expert selection? Tokens within `ε` of a facet (small tropical
  margin) should be the ones that flip under quantization noise — a falsifiable prediction (§11, E7/E8).
- **TO8** *(raised by Logic Export LE-T2)* Non-scalar provenance under the dense frame geometry
  `G_{vw} = ⟨U_v, U_w⟩`: the clean monomial-sum decomposition of TT4 assumes near-diagonal `G`; when the
  frame entangles, how is the interior-point test defined? (The tropical hypersurface is still exact —
  it is the *attribution to single circuits* that blurs.)
- **TO9** *(raised by Maragos)* Can tropical/convex regression fit a compact PWL surrogate of the core's
  decision map, and is the surrogate's monomial count the empirical tropical rank / forge-tax floor of
  TT5?

---

## 9. Related work

- **Tropical geometry of neural networks** — **L. Zhang, G. Naitzat, L.-H. Lim**, "Tropical Geometry of
  Deep Neural Networks," *Proc. ICML 2018*, PMLR 80:5824–5832. ReLU nets ⟺ tropical rational maps
  (**Thm. 5.4**); tropical rational function = difference of tropical polynomials (**Def. 2.4**);
  tropical hypersurface = decision boundary (**Def. 3.1**, **Prop. 6.1**); linear-region counts via
  Newton-polytope upper-hull vertices (**Def. 3.2**, **Thm. 6.3**) and zonotopes (**Cor. 3.4**, **Lemma
  6.2**); depth is exponentially more expressive. *The structural backbone of §2/§3/TT3.*
- **Tropical geometry of statistical models** — **L. Pachter, B. Sturmfels**, *PNAS* 101(46):16132–16137,
  2004. Graphical models as algebraic varieties; the sum-product algorithm evaluates a coordinate;
  **polytope propagation** (Minkowski-sum/convex-hull dynamic program) as the geometric sum-product, and
  its tropicalization = MAP/Viterbi; the Newton polytope of a statistical model governs parametric
  inference. *The backbone of §3b/TT7 and the logic-export bridge.*
- **Tropical geometry and machine learning** — **P. Maragos, V. Charisopoulos, E. Theodosis**, "Tropical
  Geometry and Machine Learning," *Proc. IEEE* 109(5):728–755, 2021 (and **P. Maragos**, *Tropical
  Algebra and Geometry for ML / Optimization*, ICASSP 2024 tutorial). Morphological (max-plus) perceptrons
  and networks; tropical/convex regression with optimal solution and efficient algorithm; Newton-polytope
  and zonotope methods for NN pruning/approximation (minimizing linear-region count); sparse / greatest
  solutions of max-plus equations (Cuninghame-Green). *The constructive toolkit of §5/§5b/TT8.*
- **Idempotent analysis / Maslov dequantization** (Litvinov, Maslov): the `T→0` log-semiring → (max,+)
  limit; the exact bridge to PIC (§6/TT6).
- **Tropical rank** (Develin–Santos–Sturmfels; Barvinok rank): the rank notions for TT5.
- **Power / Laguerre diagrams** (Aurenhammer): the decision-cell geometry (TT1), already measured.
- **PIC & Logic Export companions** ([`PIC_PROPOSAL.md`](./PIC_PROPOSAL.md), [`LOGIC_EXPORT.md`](./LOGIC_EXPORT.md)):
  the `T=1` soft-logic dual and the executable semiring-Datalog form.

**Cross-program forward-pointers** (separate repos, published-package boundaries — not implemented here):
`polygram`'s hierarchical polysemantic-dictionary geometry is a candidate *factored* tropical dictionary
(its Q-Orca machines a structured `A` for the TT8 max-plus fit); `n-orca`'s typed-DAG architecture specs
are where a tropical-rank capacity budget (TT3) would attach as a verifiable per-layer constraint.

---

## 10. Acknowledgment & provenance

This is the geometric/`T=0` dual of [`PIC_PROPOSAL.md`](./PIC_PROPOSAL.md) and the geometric face of
[`LOGIC_EXPORT.md`](./LOGIC_EXPORT.md); through the Maslov bridge (§6) it shares their lineage: **the
whole program descends from Alan Bundy's incidence calculus (1985)** — PIC removes Bundy's orthogonality
assumption at temperature 1, and this paper takes the resulting object to temperature 0, where the
incidence cells become a tropical variety. The tip of the hat is Bundy's; we have only added a
thermometer.

Same theory–experiment loop. The power-diagram / facet-distance results (TT1/TT2) are measured in
`--probe-facet`; the tropical-monomial framing of emergence (TT4) appears as a "lens" in FINDINGS §4/§6
and is re-read as `μ_t=0`; the polytope-propagation identity (TT7) is a restatement of what LOGIC_EXPORT
already runs; the tropical-rank thesis (§5/TT5), its constructive shadow (TT8), and the Maslov bridge
(§6) are this proposal's contributions. Conjectural sections are marked; the measured floor (§2) stands
on the existing probes.

---

## 11. Implementation — `--probe-tropical` and the vocab/cell estimators

`--probe-tropical` is a near-clone of `--probe-facet` (`src/main.rs`, rope-only via
`Model::final_residual`) plus the interior-point test from `--probe-ablate`'s `μ_t` machinery. Three
flags, all reusing existing kernels (`Bundle::rowdot_f32`, the `‖U_t−U_v‖²` Gram trick, the
`decomp_k`/contrib decomposition, and `retrieval::Store` routing). **Design only — no compiled code in
this change.**

### 11.1 Per-position probe — `--probe-tropical [--decomp-k K] [--eps E]`

For each position with residual `r = final_residual(ctx)` and model token `t = argmax_v L_v(r)`:

| Quantity | Meaning | Source |
|---|---|---|
| `route ∈ {RETRIEVED, SELECTED, COMPOSED}` | KB-coverage class | `retrieval::Store::{predict,candidates}` (as `--probe-facet`) |
| `facet_dist` | Euclidean distance to nearest facet of `T(M)` (= normalized margin) | `--probe-facet`'s `(L_t−L_v)/‖U_t−U_v‖` min over `v≠t` |
| `facet_angle = cos(U_t, U_v*)` | sharpness of the binding crossing = `T→0` image of PIC's `ρ` (TT6) | `⟨U_t,U_v*⟩ / (‖U_t‖‖U_v*‖)` at the argmin facet |
| `interior` (bool) | **TT4**: does *no* single circuit's isolated argmax equal `t`? | `decompose`/contrib (`explain.rs`): `c_j^v`, check `∄ j: argmax_v c_j^v = t` |
| `local_rank` | # tokens within `eps` of the max at `r` (active monomials near the cell) | `count_v (L_t − L_v) ≤ eps` |

Output mirrors `--probe-facet`'s table, grouped by `route`, adding `interior%` (the COMPOSED/interior
fraction — TT4), `facet_angle` mean, and `local_rank` mean. **Validation built in:** `facet_dist` must
equal `--probe-facet`'s number bit-for-bit (same computation), and `interior%` must equal the
`--probe-ablate` `μ_t=0` fraction (E1/E2).

### 11.2 Vocab-level Newton-polytope map — `--tropical-vertices [--samples N] [--store S]`

Estimates **TO5**: which tokens have a non-empty Laguerre cell (retrievable vertices) vs never win
(interior, only-composed). Exact enumeration of `conv{U_v}` vertices is infeasible at `|V|≈150k, d≈900`
(TO6), so two estimators, both already-have-the-kernels:

1. **Empirical (sampling):** stream `N` real residuals (the eval corpus); `won = {argmax_v L_v(r)}`. The
   *won set* is a lower bound on the non-empty-cell (retrievable) vocabulary; its complement is
   candidate-interior. Cheap (one `rowdot_f32` per position, already computed in eval).
2. **LP feasibility (exact, per-token, optional `--exact V`):** token `v` is an upper-hull vertex iff
   `∃ r: ⟨U_v−U_w, r⟩ > b_w − b_v ∀w≠v` is feasible — one LP in `d` vars, `|V|−1` constraints. Run for a
   sampled token subset to calibrate the empirical estimator's miss rate.

Report: `|won| / |V|` (retrievable-vocab fraction), the frequency distribution of won vs never-won
tokens (E6: are never-won tokens the rare/composed ones?), and the per-token win-count (cell "mass").

### 11.3 Global cell-count / tropical-rank estimator — `--tropical-cells [--samples N]`

Estimates **TO3**: sample `N` residuals (empirical, or Gaussian in the residual subspace), count
**distinct argmax cells visited** and the **distinct binding pairs** `(t, v*)` — a lower bound on the
number of decision cells / facets actually used (the *effective* tropical complexity, ≪ the worst-case
Newton-vertex bound of TT3). Optionally fit the **greatest max-plus subsolution** (TT8): given a
candidate table `A` (the bucketing experts of `DENSITY_BUCKETING.md`, or top-frequency keys),
`x̂_j = min_i(b_i − A_{ij})` and report the residual `‖b − A⊗x̂‖` per route class — the computable
forge-tax lower bound.

### 11.4 New helpers / data structures (sketch)

```rust
// src/tropical.rs (new) — pure geometry over the unembedding frame; no forward pass.
struct Facet { v: usize, dist: f32, angle: f32 }       // nearest-facet record (token, distance, cos-angle)

/// k nearest facets of T(M) to residual r, with crossing angles. Reuses ‖U_t−U_v‖² = ‖U_t‖²+‖U_v‖²−2⟨U_v,U_t⟩.
fn nearest_facets(r: &[f32], un: &Bundle, name: &str, unorm: &[f32], t: usize, k: usize) -> Vec<Facet>;

/// # monomials within eps of the max at r (local active-monomial count ≈ local tropical rank).
fn active_monomials(logits: &[f32], t: usize, eps: f32) -> usize;

/// Interior-point test (TT4): true iff no single circuit's isolated argmax is the model token.
/// Consumes the per-circuit contribution rows c_j^v already produced by explain::decompose (decomp_k).
fn is_interior(contrib: &[Vec<f32>], t: usize) -> bool;

/// Greatest (principal) subsolution of the max-plus system A ⊗ x = b (Cuninghame-Green) and its residual.
fn maxplus_principal(a: &[Vec<f32>], b: &[f32]) -> (Vec<f32>, f32);   // (x̂, ‖b − A⊗x̂‖)
```

CLI flags to add: `--probe-tropical`, `--tropical-vertices`, `--tropical-cells`, `--decomp-k`, `--eps`,
`--samples`, `--exact` (LP calibration), reusing `--store`/`--vocab`/`--ctx`/`--n-eval` as `--probe-facet`
does. `headgate.rs` already implements the nearest-facet geometry for head gating — `nearest_facets`
should be factored out of it and `headgate`/`--probe-facet`/`--probe-tropical` should share it.

---

## 12. Experimental plan (runnable on current rope models — Qwen/Llama/Gemma; MoE via `--route-frac`)

| # | Experiment | Method | Success criterion |
|---|---|---|---|
| **E1** | Validate probe vs `--probe-facet` | run both on the same positions | `facet_dist` identical to ≤1e-5 (it is the same computation) |
| **E2** | Interior fraction = `μ_t=0` | `--probe-tropical --decomp-k K` vs `--probe-ablate` | `interior%` matches the `μ_t=0` fraction within sampling noise; monotone RETRIEVED < SELECTED < COMPOSED |
| **E3** | Facet angle = `T→0` of PIC `ρ` (TT6) | distribution of `facet_angle` by route vs PIC's `ρ_{tv}` | COMPOSED facets are *sharper* (higher `cos`) — the soft-competition cells collapse to the hardest crossings |
| **E4** | COMPOSED = interior fraction (TT4) | `--probe-tropical` over natural vs code corpora | interior% ≈ measured ~15% natural / ~37% code; reproduces the FINDINGS gradient |
| **E5** | Forge tax = max-plus residual (TT8) | `--tropical-cells` with `A` = bucketing experts vs top-freq keys | residual ‖b−A⊗x̂‖ is ~0 on RETRIEVED, large on COMPOSED; tracks the `DENSITY_BUCKETING` residual-expert mass |
| **E6** | Newton-vertex vocab map (TO5) | `--tropical-vertices --samples 1e5` | never-won tokens are the rare/specialised tail; `|won|/|V|` ≪ 1 and grows with corpus diversity |
| **E7** | Tropical margin predicts quant sensitivity (TO7) | small `facet_dist` vs int8/int4 prediction flips | flip rate is monotone-decreasing in `facet_dist`; near-facet tokens flip first under quantization |
| **E8** | Cells vs MoE routing (TO7) | `--tropical-cells` under `--route-frac` vs full | the cells/binding pairs an expert subset can realise ⊆ the full set; routing error concentrates on interior points |
| **E9** | Tropical vs linear rank (TO2) | `--tropical-cells` effective-rank vs SVD rank of the core | the tropical-effective-rank, not SVD rank, predicts the min_to_run / data-aware-vs-SVD ladder |

Priority order for a first pass: **E1 → E2 → E4** (cheap, validate the probe and TT4 against existing
data), then **E5/E6** (the TT8/TO5 contributions), then **E3/E7/E8/E9** (the cross-validations with PIC,
quantization, MoE, and the rank ladder). E1–E6 run on a single 0.5B rope model in minutes with the
KV-cached `explanation_stream`; E7–E9 need the int8/int4 bundles and an MoE bundle respectively.
