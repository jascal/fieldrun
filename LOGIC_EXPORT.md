# Logic Export of the Core

**The model as a provenance-semiring Datalog program — greedy decode is max-product, sampling is sum-product**

*Status: research proposal / a third paper. Where [PIC](./PIC_PROPOSAL.md) is the probabilistic
**logic** (soft, `T=1`, the measure) and [Tropical](./TROPICAL_PROPOSAL.md) is the decision **geometry**
(hard, `T=0`, the argmax), this is the **executable form**: the artifact you can run, statically check,
and compile back to kernels. The three are one theory in three categories (semantics / geometry /
computation); a result in any is a result in the others. Empirical anchors: FINDINGS §5 (the retrievable
fragment — induction/n-gram/grammar — and the `--probe-ablate` causal program). Lineage: incidence
calculus (Bundy 1985), via PIC.*

---

## Abstract

The core's next-token computation is **semiring accumulation over a finite relational domain, evaluated
bottom-up**: contributions sum along the residual stream, then a single competitive aggregation picks
the token. That is exactly what **Datalog evaluated under a provenance semiring** *is*. We export the
model as a semiring-weighted Datalog program `Π` and read it as a **Functional Aggregate Query (FAQ) /
sum-product** over a semiring `K`. The two theory papers are then two semirings over the *same* `Π`:

> **`K` = log-semiring (`⊕ = log-sum-exp`, `⊗ = +`) ⇒ sum-product ⇒ the softmax measure (PIC, `T=1`).
> `K` = tropical (`⊕ = max`, `⊗ = +`) ⇒ max-product ⇒ the greedy argmax decode (Tropical, `T=0`).**

Maslov dequantization is the semiring homomorphism between them; the temperature knob *is* the semiring
choice. Two theorems carry the paper: **LE-T5 (soundness)** — `Π` under the log-semiring evaluates to
the model's distribution, under (max,+) to its MAP decode — and **LE-T2 (the provenance gap)** — the
Gram coupling `G_{vw} = ⟨U_v, U_w⟩` forces provenance *valued in the frame geometry*, not scalars; scalar
semiring Datalog is exact only on diagonal `G`. The retrievable fragment exports to compact stratified
Datalog (induction = recursive clause, n-gram = fact); the computed fragment (the forge tax) exports to a
dense, high-treewidth recursive aggregate with no compact extension — so the export **localizes** the
forge tax to a named program region rather than dissolving it. This is the logic-programming analog of
`larql` ("the model IS the database"): here, *the model IS a semiring-weighted Datalog program, and
decoding is provenance evaluation.*

---

## 1. Why Datalog and not Prolog

This is a semantics decision, not export convenience; performance follows from it.

1. **The semiring *is* the model's arithmetic, and provenance semirings are a Datalog construct.**
   Provenance/semiring Datalog (Green–Karvounarakis–Tannen; FAQ / semiring Datalog, Khamis–Ngo–Rudra)
   is defined on Datalog's least-fixpoint semantics: finite, well-defined derivation sets, with `⊗`
   along a derivation and `⊕` over alternatives. **Prolog has no semiring knob** — its semantics is
   fixed Boolean SLD — and its machinery (cut, assert/retract, negation-as-failure, clause order, an
   infinite SLD tree) *breaks the semiring laws* (associativity/commutativity of `⊕`). The two-temperature
   unification (LE-T5) is impossible without the semiring parameterization, which only Datalog has.
2. **Forward accumulation matches Datalog's bottom-up evaluation.** The residual stream accumulates
   bottom-up then thresholds; Datalog's semi-naive least fixpoint *is* forward accumulation. Prolog
   backward-chains from a goal — the wrong dataflow for "sum all contributions then argmax."
3. **Aggregation/threshold is native** in semiring Datalog (the aggregate *is* `⊕`); in Prolog it is a
   meta-level `findall`+fold, outside the declarative semantics and order-sensitive.
4. **Decidability / static analyzability** (the "verify-before-execute" ethos, and `larql`'s relational
   stance): Datalog terminates, is order-independent, PTIME data complexity, a unique least fixpoint —
   a mathematical object you can bound and verify. Prolog is Turing-complete (function symbols → infinite
   Herbrand universe), can loop, is cut/order-dependent — a procedure, not an object.
5. **Performance follows:** bottom-up + semiring = set-at-a-time relational algebra over `K` = **sparse
   semiring matrix multiply** — the kernels fieldrun already has ((max,+) and log are kernels). Prolog's
   tuple-at-a-time SLD does not vectorize.

**What the restriction costs:** Prolog's extra power is function symbols / unbounded structured-term
construction. The retrievable rules (induction, n-gram, grammar) are finite-domain over tokens/positions
and need none of it; the forge tax is dense arithmetic (T4), which Prolog cannot compact *either*. So
Prolog's extra expressivity targets a gap neither fragment has. *Live caveat:* if a retrievable rule ever
required unbounded structured memory (a stack/parse built during inference), Datalog could not express it
and that would be the case to climb to a Prolog fragment — empirically the retrievable set is flat, so it
is not a current limit.

---

## 2. The exported program `Π`

A `.logic` bundle is:
- **Propositions** `V` (tokens) with directions `U_v` and **Gram** `G_{vw} = ⟨U_v, U_w⟩`.
- **Sources** `S` (circuits) with pairings `c_j^v = ⟨d_j, U_v⟩`.
- **Retrievable fragment** → compact **stratified** Datalog clauses (§3).
- **Computed fragment** → a flagged dense aggregate (the forge tax, §5).
- A **provenance evaluator** with a semiring/temperature knob.

Read `Π` as an FAQ / sum-product: variables = propositions, factors = sources; the decision is the
aggregation `⊕_v ⊗_j (factor)`.

---

## 3. Clause shapes (and why induction is the clean rule)

- **induction head** = a **recursive** Datalog clause — copy from the matched position:
  `next(T) :- match_prefix(P), follows(P, T).` Recursion + lookup, cleanly Horn. *(This is exactly why
  induction was the one high-precision retrievable idiom in the data: it has a finite recursive clause;
  the others do not.)*
- **n-gram** = a weighted **fact**: `succ(Ctx, Tok)` with provenance `c`.
- **grammar / closed-class** = a **unary constraint** on the candidate set.
- **attention (general)** = a **provenance-annotated join over context positions** — soft selection is a
  semiring-weighted aggregation across the sequence (sum-product over positions). Quadratic, soft.
- **composition / forge tax** = the **dense weighted residual**: a body of ~PR coupled literals with no
  sufficient sub-conjunction (D4). Datalog *holds* it (a recursive aggregate); it does not *compact* it.

---

## 4. LE-T5 — soundness (the central, established theorem)

Let `[[·]]_K` be provenance evaluation of `Π` over semiring `K`, with sources combined by `⊗` along the
residual derivation (so the accumulated value of proposition `v` is `⊗_j c_j^v = Σ_j c_j^v = L_v`) and the
competing propositions combined by `⊕`.

> **LE-T5 (Soundness / two-temperature decode).** Reading `Π` as a semiring FAQ:
> - under the **log-semiring** (`⊕ = log-sum-exp`, `⊗ = +`), `[[Π]]_log` is the **sum-product**: the
>   aggregate value is `log Σ_v exp(L_v) = log Z`, and the per-proposition share is
>   `exp(L_v)/Z = P(v)` — **the model's softmax distribution (PIC, `T=1`)**;
> - under the **tropical semiring** (`⊕ = max`, `⊗ = +`), `[[Π]]_max` is the **max-product**: the
>   aggregate value is `max_v L_v` and its witness is `argmax_v L_v` — **the model's greedy decode
>   (Tropical, `T=0`)**.

**Status: established.** It is the standard graphical-model duality (sum-product vs max-product over a
semiring) instantiated on `Π`, plus PIC's T5 (the product-of-experts recovery `Π_j exp(c_j^v) = exp(L_v)`).
The log↔max homomorphism is Maslov dequantization, so **the export is correct at both temperatures by one
identity.** This is the contract that makes the logic export *faithful*: `Π` is a correct executable form
of the model iff LE-T5 holds, and it does for the additive (static) fragment exactly (residual-stream
additivity; FINDINGS §5c D_j-reconstruction is the empirical face).

---

## 5. LE-T2 — the provenance gap (the central open problem)

LE-T5 is clean when the propositions' factors are independent. They are not: `G_{vw} = ⟨U_v, U_w⟩`
couples every pair (PIC's T2 / the measured ρ-boundary, FINDINGS §5c). Two consequences:

> **LE-T2 (Provenance gap).** Faithful correlated evaluation of `Π` requires provenance **valued in the
> frame geometry** — carrying the `U_v` directions (or the Gram operator `G`) — not scalars in a
> commutative `K` over ℝ. **Scalar semiring Datalog is exact only on diagonal `G`** (orthogonal frame =
> the independent / ProbLog / classical-incidence-calculus case). For dense `G`, scalar provenance drops
> the joint exactly as classical incidence calculus drops the correlation `i(A)∩i(B)` — which is where
> PIC begins.

**Status: open. This is the export's hard problem, and it is *the same object* as PIC's T2.** A solution
likely lives in the non-scalar provenance line (semiring Datalog° / FAQ over richer value structures,
Khamis–Ngo–Rudra) — provenance valued in an operator semiring whose product carries `G`. Solving it here
solves PIC's T2; that 1:1 correspondence is the strongest evidence the three docs are one theory.

**LE-T2 ties to tractability (and to T4).** A dense `G` coupling is a **high-treewidth factor graph**:
sum-product is exponential in treewidth, so the correlated evaluation is intractable *and* non-compact —
which is precisely the forge tax (T4) seen from the export side. The retrievable fragment is
low-treewidth / stratified (compact, tractable); the computed fragment is the dense-`G` core (the
irreducible cost). So **LE-T2 (the coupling) and T4 (the non-compact forge tax) are the same wall** viewed
as provenance structure vs intervention diffuseness.

---

## 6. Theorems / claims by status

| Claim | Content | Status |
|---|---|---|
| LE-T5 | log-semiring `Π` = softmax; (max,+) `Π` = argmax (sum-product / max-product) | **Established** (semiring duality + PIC T5) |
| LE-T2 | Gram coupling needs non-scalar provenance; scalar exact only on diagonal `G` | **Open** (= PIC T2) |
| LE-T4 | dense-`G` = high-treewidth = non-compact/intractable = the forge tax | Inherited (= T4, the export-side view) |
| LE-1 | retrievable fragment = compact stratified Datalog; induction = recursive clause | Measured-adjacent (the idiom data) |

---

## 7. Open problems

- **LO1** Define the non-scalar (geometry-valued) provenance semiring that carries `G` faithfully (LE-T2);
  prove it reduces to the scalar log/tropical semiring on diagonal `G`.
- **LO2** `--probe-reconstruct`: measure `Σ_j c_j^v` (export accumulation) vs the true logit — the static
  residual (decompiler completeness, LE-T5 exact) and its growth under intervention (forge tax, LE-T4).
- **LO3** Compile the retrievable fragment to an executable semiring-Datalog engine; benchmark
  sparse-(max,+)-matmul decode vs the dense forward (the performance face of §1.5).
- **LO4** Treewidth of the core's factor graph as a quantitative forge-tax measure; relate to PR and to
  the Tropical paper's tropical rank (one wall, three measures: PR, treewidth, tropical rank).
- **LO5** A static verifier over `Π` (the "verify-before-execute" payoff): which tokens are decided by the
  retrievable fragment alone (provably, no dense-`G` term) vs require the computed fragment.

---

## 8. Related work

- **Provenance semirings** (Green, Karvounarakis, Tannen 2007): the framework `Π` is read in.
- **FAQ / semiring Datalog / Datalog°** (Khamis, Ngo, Rudra): semiring sum-product over a database; the
  natural home for LE-T5 and the non-scalar generalization LE-T2 needs.
- **Sum-product / max-product & the semiring view of inference** (graphical models): LE-T5's duality;
  treewidth as the tractability/forge-tax measure (LE-T4).
- **larql** (the program's "model IS the database" tool): this is its logic-programming analog — *model
  IS a semiring-weighted Datalog program, decode IS provenance evaluation.*
- **PIC & Tropical companions**: the `T=1` and `T=0` semiring instances of `Π`.
- **Incidence calculus** (Bundy 1985): the ancestor (via PIC); the diagonal-`G` exact case is classical
  incidence calculus run as Datalog provenance.

The stake: **a transformer exported as a semiring-weighted Datalog program whose two temperatures are
sum-product and max-product over one program, whose retrievable fragment is compact verifiable Datalog,
and whose forge tax is exactly the dense-Gram / high-treewidth region — soundness by LE-T5, the open
frontier by LE-T2.**

---

## 9. Acknowledgment & provenance

The executable/computation category of a three-category theory with [PIC](./PIC_PROPOSAL.md) (logic) and
[Tropical](./TROPICAL_PROPOSAL.md) (geometry). Through LE-T5's log↔max duality it shares their lineage:
**all three descend from Alan Bundy's incidence calculus (1985)** — PIC removes his orthogonality
assumption (`G` off-diagonal), Tropical takes it to `T=0`, and this paper runs it as a program; the
diagonal-`G` case here is *literally Bundy's incidence calculus evaluated as Datalog provenance*. The tip
of the hat is his; we have given his calculus a semiring and a fixpoint engine.

Provenance: the same theory–experiment loop. LE-T5 rests on the semiring-inference duality and PIC's T5
(both established); LE-T2/LE-T4 are open and pinned to PIC's T2 and the measured forge tax. Every empirical
claim traces to a probe in [`FINDINGS.md`](./FINDINGS.md) §5.
