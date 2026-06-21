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
choice. This bottom-up polytope/semiring recurrence is exactly **Pachter–Sturmfels polytope propagation**
(the geometric sum-product; PNAS 2004) specialized to the unembedding layer — see
[`TROPICAL_PROPOSAL.md`](./TROPICAL_PROPOSAL.md) §3b/TT7, which names the decode's prior-art algorithm.
Two theorems carry the paper: **LE-T5 (soundness)** — `Π` under the log-semiring evaluates to
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
  prove it reduces to the scalar log/tropical semiring on diagonal `G`. **The decidable crux of the whole
  forge-tax question (Grok), with a candidate and a likely self-defeating obstruction:**
  - *Construction.* Take `K` = the semiring of operators (matrices/endomorphisms) over the frame `span{U_v}`;
    value proposition `v` at the rank-1 operator `U_v ⊗ φ(v)` (`φ` = local circuit state). The inner-product
    couplings `⟨U_v, U_w⟩` are then realized by matrix product / trace *inside* the algebra rather than scalar
    multiplication of independent facts, so composition internalizes the dense `G` without spawning new
    high-treewidth cliques; trace / diagonal projection recovers ordinary tropical/log provenance, and on
    diagonal `G` the off-diagonal operators vanish (the scalar case). If this evaluates the dense fragment at
    low treewidth, **the forge tax is almost entirely a *scalar-lens* artifact — the precise structural twin
    of Minsky's single-LAYER restriction (single-layer ↦ scalar-provenance; "go multilayer" ↦ this `K`).**
  - *Obstruction (most plausible; the escape may be self-defeating).* The **frame geometry** itself: the
    `ρ`-boundary (`ρ = cos(U_t, U_{v*})`) and the curvature of the power-diagram hyperplane arrangement imply
    any *local* operator assignment consistent with the dense couplings cannot be made *globally* consistent
    (across the `μ_t=0` non-decomposable loci) without re-introducing a clique. I.e. **the minimal width of any
    geometry-valued valuation that faithfully carries `G` is lower-bounded by a function of the tropical rank /
    PR of the dense fragment — exactly the quantity we were trying to reduce.** If that bound holds, `Δ_descr`
    cannot compact the forge tax; it only relocates the cost into the valuation's width.
  - *Status: open; existence plausible but obstructed by frame geometry; **decidable with current algebraic
    tools** via an explicit construction or a valuation-width rank lower bound. Highest-leverage target in the
    program — a positive construction collapses the forge tax to a lens artifact; a clean obstruction proves
    the descriptive move cannot escape the wall either.*
  - **First measurement (`lo3a/lo1_matrix.py`).** The whole construction reduces to one quantity: the
    operator-valuation width = the **effective rank** of the dense fragment's Gram (synthetic check: width
    tracks true rank ρ exactly while the scalar clique stays at `k`). On SmolLM-135M the **token-coupling**
    Gram `G_{vw}=⟨U_v,U_w⟩` over the top-K candidates is **low-rank — effrank/K ≈ 0.34**, so the matrix
    valuation carries it at ~3× compression: the descriptive escape *has traction on the token axis*. **But
    that rank is essentially invariant to the margin** (forge-tax 10.9 vs retrievable 11.0 over 150 decisions)
    — so **the forge tax is NOT in the token-coupling Gram**; it is a constant property of the unembedding
    geometry. The dense fragment therefore lives in the **circuit-coupling** axis (within-block PR≈45), which
    the candidate Gram cannot see — and that is exactly where Grok's "rank-tracks-PR" obstruction would bite.
  - **Circuit-axis test — the obstruction HOLDS (`lo3a/lo1_circuit.py`, SmolLM-135M).** Per decision, capture
    every head's and neuron's residual-write *vector*; for the ~92 effective circuits (scalar `PR_dla`), the
    **write-energy rank is ≈7** — so the write *geometry* is low-rank and the operator valuation compresses the
    *bulk*. **But the decode-faithful rank** (the minimal write-subspace that preserves the argmax) **is ≈213** —
    it *exceeds* the scalar PR, approaches the residual dimension, and **rises as the margin shrinks**
    (forge-tax 226 vs retrievable 206). The decision lives in the **low-energy tail**, not the high-energy bulk
    (the geometric face of `μ_t=0`). So the descriptive escape compresses the write *energy* but **not the
    decision**: Grok's self-defeating obstruction holds on the circuit axis, and its margin-bounded rise is the
    **PO4 margin certificate seen from the rank side** (LO1 and PO4 are bounded by the *same* margin — one wall).
    Verdict shift: **`Δ_descr` does NOT dominate the forge tax's decision-relevant geometry** — that part is
    intrinsic-or-`Δ_repr` (a real wall), not a scalar-lens artifact. *(Caveats: single model; `effrank` =
    energy PR vs `faithful_r` = a coarse rank sweep truncated to the top-256 components; a margin-aware /
    non-orthogonal valuation beating SVD-truncation is not yet ruled out — the open refinement.)*
  - **Decision-direction span — the deciding number (Grok), and it PARTIALLY REVERSES the above
    (`lo3a/lo1_span.py`, SmolLM-135M, 800 decisions).** A *context-free* valuation has fixed operators and
    cannot adapt per input, so the intrinsic object is the span `S = span{ΔU(x)/m(x)}` of the
    (margin-normalized) decision directions `ΔU = gain⊙(U_pred − U_runnerup)`. Measured: **effrank(unit ΔU)
    ≈ 16** of 576 (forge-tax 14.5, retrievable 23.2), **90%-energy rank ≈ 115**, margin-normalized effrank
    ≈ 5. This is **LOW** — so the energy-basis `faithful_r ≈ 213` was the *wrong basis*: the decision
    directions live in a shared low-rank subspace `S`, and a **context-free valuation whose fixed basis is
    `S` can preserve decodes at moderate rank without per-input adaptation**. By Grok's own criterion
    (low span ⇒ escape hatch open), **`Δ_descr` is alive on the decision axis** — the forge tax is more a
    *basis-choice (scalar-lens) artifact* than intrinsic, **provided the valuation is built in the decision
    basis, not the energy basis**. Two tensions keep it from a clean win: the 90%-coverage rank (~115) ≈ the
    scalar PR (~92), so the escape is *partial*; and forge-tax span is *lower* than retrievable (against the
    "forge tax needs more width" prediction). *Net: the basis choice (decision vs energy) is the lever Grok
    flagged — choose it right and the descriptive escape mostly works; the clean closer is to BUILD the
    fixed-`S` rank-r valuation and measure decode preservation vs r across inputs.* (Caveat: one model;
    random contexts may understate `dim(S)` vs real text.)
  - **The closer — fixed-`S` decode-preservation curve (`lo3a/lo1_curve.py`, `lo1_curve.png`).** Fit `S`
    (PCA of the decision directions) on a TRAIN split; on HELD-OUT decisions, project the residual onto the
    rank-`r` subspace of `S` and re-decode. The genuinely context-free valuation **preserves ~70% of decodes
    at peak** (ALL 70%, forge-tax 65%, retrievable 72%), at **r ≈ 128–192 — i.e. ≈ the scalar PR (~92)** —
    and **never reaches 90%**; beyond r≈192 it *crashes* (the low-variance tail of `S` lets the residual's
    non-decision energy corrupt the argmax, so the decode-relevant subspace is *finite*, ~128–192-dim). **Net,
    and this is the LO1 landing:** the effrank-16 read was optimistic; the *operational* decode-preservation
    needs rank **≈ the scalar PR** and **caps at ~70%**. So the descriptive escape is **PARTIAL and
    PR-bounded** — `Δ_descr` buys ~70% of the decodes with a fixed ~PR-rank subspace, but a **~30% residual +
    the high-coverage tail is the `τ*`/`Δ_repr` floor**. All four axes converge on the **same scalar-PR /
    margin wall** as PO4 and the circuit `faithful_r`: *choosing the decision basis helps, but the decode-
    faithful rank of any fixed valuation is ≈ the participation ratio.* (Open refinement: a non-orthogonal /
    readout-aligned / margin-weighted basis might lift the 70% ceiling; real-text contexts; multiple models.)
  - **The hatch is CLOSED — the decode-optimal basis does not beat the ceiling (`lo3a/lo1_readout.py`,
    `lo1_readout.png`).** Rebuilding `S` from the **readout diffs** `gain⊙(U_pred − U_v)` over the top-8
    competitors (Grok's decode-optimal basis) vs the variance-PCA of the runner-up direction: **at rank ≈ PR
    both give ~70%** (readout 67–71% vs variance 64–70% — only ~1–3% higher). The readout basis's only edge is
    that it is *monotonic* (no crash) and reaches 100% at **r = d** (full rank, identity — not compression);
    >90% coverage needs `r` approaching `d`. So the **~70%-at-PR ceiling is BASIS-INDEPENDENT** — it is a real
    floor, not a basis-misalignment tax (Grok's "low-80s" prediction is refuted). **LO1 final landing:** the
    descriptive escape (`Δ_descr`) buys ~70% of decodes with a fixed ≈PR-rank valuation *regardless of basis*;
    the remaining ~30% requires ~the full residual dimension and is **not escapable by any fixed low-rank
    description** — that is the `τ*`/`Δ_repr` intrinsic floor, pinned at the participation ratio. The forge tax
    on the decode axis is therefore **mostly intrinsic, with a real but bounded descriptive escape**, and
    `Δ_descr` / `PO4` / the circuit `faithful_r` all bottom out at the **same scalar-PR / margin wall**.
- **LO2** `--probe-reconstruct` — **DONE** (FINDINGS §5d). Per-block residual decomposition: `Σ_blocks == logit`
  **exact** (mean err 6–7e-6 both models) ⇒ LE-T5 confirmed numerically, the static export is faithful. The decision is
  **block-sparse** (≈8–10 effective of 49 blocks, σ≈1.1–1.6) but **circuit-dense** within a block (PR≈45). So the
  emitted readable fragment is compact at *block* granularity and bottoms out there for composed tokens (below = the
  dense forge-tax sum). NB the source-level support number (PIC O2 / `σ(t) ∼ PR`) is *not* this block-level σ — at block
  granularity it mildly reverses (composed slightly more concentrated, being thin-margin); O2 is the circuit-coalition
  question. *Remaining:* the interventional residual (forge-tax growth under ablation) = the D_j-regression's indirect
  gap (FINDINGS §5c), already measured.
- **LO3** — **emitter DONE** (`fieldrun ... export --logic`, rope). Emits a runnable, Soufflé-compatible semiring-
  Datalog program SPECIALIZED to one next-token decision: candidate set (facts), Tier-A retrievable fragment (induction
  = recursive clause, n-gram = `ngram_succ` fact), Tier-B composition as `contrib(Block, Token, Weight)` per-block facts
  (`|W|≥0.1` shown, dense remainder folded into block `"rest"` = the forge tax), and the decode as
  `decide(T) :- logit(T,S), S = max … : { logit(_,S2) }` (max-product / `T=0`). Verified: `Σ contrib == logit` to
  floating point and the `(max,+)` decode `== the model's token` (LE-T5 round-trip self-check, "FAITHFUL ✓", both
  Qwen2.5-0.5B). **A built-in evaluator runs it without Soufflé:** `fieldrun eval prog.dl --semiring max|log` parses the
  candidate/contrib facts and applies the cross-candidate `⊕` — `max` → `decide(T)` (greedy decode), `log` → the
  softmax distribution. Verified: `eval --semiring max` on the emitted program returns the model's token; `--semiring
  log` returns the distribution — **one program, two semirings, two temperatures, run.**
  **Three entry points, one builder** (`logic::build` — so each is the provenance of the actual decode, faithful by
  construction, not a reconstruction): (i) `fieldrun … export --logic [--out f.dl]` — one decision; (ii) `fieldrun …
  --export-logic <prefix> [--steps N]` — a multi-step decode **trace**: one *independent* program per generated token
  (`prefix.000.dl`, `prefix.001.dl`, …), the context advancing by the model's own greedy pick each step (deliberately
  per-step files, not one concatenated program — merged programs redeclare relations and make `eval` sum `contrib`
  across tokens); (iii) `/export-logic [file.dl] <prompt>` inside the `--chat` REPL — emit the `.dl` for a chosen
  decision on demand, in the live chat context (ChatML template + history). *Remaining:* (a) the whole-model
  (context-free) emit — the trace is still a *sequence* of per-context programs, not one program over all contexts;
  (b) the sparse-`(max,+)`-matmul performance face of §1.5.
- **LO3a** — **emitter DONE at small scale** (`fieldrun … export --logic-whole`, rope family). The CONTEXT-FREE
  whole-model emit: one `.dl` whose only input is `token(pos,id)` and that *computes* the next token from scratch —
  weights are facts, the forward pass (RMSNorm, RoPE/GQA attention, SwiGLU MLP, tied/untied unembed, argmax) is rules.
  Not a partial evaluation: swap the token facts and Soufflé recomputes, answering contexts the emitter never saw.
  It is **plain Datalog, no FFI** — Soufflé's `^` gives `sqrt(x)=x^0.5` and `exp(x)=E^x`, and RoPE `sin`/`cos` depend
  only on position so they are precomputed model-constant facts (never token-dependent). Verified: across base /
  +bias / untied / bias+untied tiny rope bundles, Soufflé's `decide` == fieldrun's forward == an independent numpy
  reference on every held-out context (`lo3a/verify_all.py`). **What stays open is LE-T2/LE-T4, exactly as predicted:**
  the program exists and is correct for *any* model, but the dense `embed`/`unembed` fragment costs `vocab×d` facts —
  the non-compact dense-Gram wall. So `export --logic-whole` refuses full-scale bundles by default (naming LE-T4) and
  needs `--force`. LO3a moved the frontier from *can a context-free program be emitted?* (yes) to *can the dense
  fragment be emitted COMPACTLY?* — which is LE-T2, still open. See [`SOUFFLE.md`](./SOUFFLE.md) §8.
  - **Two partial compact-unembed levers exist on the lossy/certified side (R4/R5).** (1) The PR-core
    *factored* readout (`lo3a/pr_core_export.py --datalog`) replaces the dense `vocab×d` embed facts with the
    rank-`r` factored pair `proj(i)=Σ_j xraw(j)·sbasis(i,j)`, `corelogit(v)=Σ_i proj(i)·acore(i,v)` —
    `r(d+vocab)` facts, decode-kept-67% on SmolLM, labeled-lossy. (2) The `--pruned-head` KB-proposed
    candidate shortlist (~540 tokens) with the PO-T3 margin gate is a **compact *certified* unembed on the
    tokens where `m > 2δ`**: where the margin clears, the argmax provably equals the full-vocab argmax over
    just the shortlist, so the dense `vocab×d` emit shrinks to `shortlist×d` with a Soufflé-checkable
    certificate. Wiring this shortlist into `export --logic` (so the whole-model emit is *compact-and-certified
    where the margin clears, full only on the thin-margin tail*) is the concrete next step against the LE-T4
    wall — noted here as the achievable increment; the lossless whole-`vocab×d` emit stays blocked.
  - **The margin-routing principle is now wired into the decode trace** (`--export-logic --residue-strategy
    {ring|pic|edb|margin} [--tau t]`): per generated token, high-margin / retrieved tokens emit the *compact* decode-only
    form (Tier B elided — decode-safe above 2δ by PO-T3) and the low-margin tail keeps the full per-block Π. Both round-
    trip the model under `eval --semiring max` (argmax = the model token) and `log` (the softmax); on a 16-token prose
    trace the `edb` (all-compact) artifact is 26 KB vs `ring` (all-Π) 71 KB, with `margin` in between — the forge tax is
    paid only where the margin is thin. (`edb` = *extensional database*, the Datalog name for a fact table: the compact
    form memorises the decode as a fact rather than recomputing it.) This realizes the increment *for the per-context
    trace*; lifting the same router onto the **context-free `--logic-whole`** unembed (the `shortlist×d` + PO-T3
    certificate above) is the remaining LE-T4 step.

    | strategy | per-token emission | 16-tok prose trace | faithful (`eval` max / log) |
    |---|---|---|---|
    | `ring` / `pic` (default) | full per-block Π every token | 0 compact / 16 Π · **71 KB** | argmax = model token / softmax ✓ |
    | `edb` | decode memorised every token | 16 compact / 0 Π · **26 KB** | model token asserted ✓ |
    | `margin --tau 2` | high-margin → memorise, low → Π | 2 compact / 14 Π · ~66 KB | ✓ (PO-T3 on the compact set) |
- **LO4** Treewidth of the core's factor graph as a quantitative forge-tax measure; relate to PR and to
  the Tropical paper's tropical rank (one wall, three measures: PR, treewidth, tropical rank). *Caveat
  (Grok): treewidth `τ` is the load-bearing invariant (the forge tax = the no-compact-sub-conjunction
  fragment = high `τ` by definition), but the three measures may **diverge on the dense fragment** — tropical
  rank can be low while `τ` stays high (a max-plus low-rank factorization the elimination order misses), and
  PR can drop while `τ` stays elevated (spectrum concentrates without shrinking the minimal-clique support).
  They coincide on diagonal-`G`/retrievable fragments and move together in the Pythia checkpoints, but a
  counter-example separating them on dense `G` would falsify the "one wall" claim — so LO4 is **open**, and
  the Pythia PR signal (PO-T7) is a treewidth **proxy**, not treewidth.**
- **LO6** **Is the forge tax intrinsic or escapable?** (Grok's formalization & verdict.) Decompose, for a
  competent next-token function `f`, the decision-hypergraph treewidth `τ(M,x) = τ*(f,x) + Δ_repr(M) +
  Δ_descr(lens)`: `τ*` = inf over **all** architectures computing `f` to accuracy `1−ε` of the induced
  treewidth (architecture-independent, the task's intrinsic dense-composition complexity); `Δ_repr` = excess
  from superposition in a width-`d` bottleneck (removable by wider/sparser/modular/symbol-augmented archs —
  converts forge tax → size); `Δ_descr` = excess from scalar-`ℝ` vs geometry-valued provenance (LO1).
  **Verdict (open, leaning escapable):** the forge tax is *most likely dominated by* `Δ_repr + Δ_descr`; a
  small `τ* > 0` floor cannot yet be ruled out. **Minsky/Papert placement:** no known reduction embeds a hard
  function (parity / inner-product / set-disjointness) into a language sub-task to *force* high `τ*` — and
  attention models sit in TC⁰ for many pattern-matching subtasks — so the perceptron-style escape (enlarge
  the class: depth, sparsity, or the LO1 valuation) is more likely than an AC⁰-parity-style intrinsic wall.
  A genuine wall would require showing **every** high-accuracy architecture induces high-`τ` provenance on a
  positive-density input set; that theorem is not available and looks harder than the escapability
  constructions. **Decisive experiments (all on instruments in hand):** (a) cross-architecture `τ` (PR proxy)
  at *matched* accuracy — lower PR for the same `f` ⇒ `Δ_repr > 0`; (b) the `lim_{train→∞}` PR floor as an
  *estimator of `τ*`* (the PO-T7 late-consolidation endpoint, ~12 on Pythia-70m — if it stays `>1` across
  architectures at matched accuracy, that number lower-bounds the intrinsic forge tax); (c) correlate the
  late PR drop with superposition metrics (SAE recon error, effective weight rank) — if it's superposition
  cleanup with no accuracy gain, it is `Δ_repr`, and its endpoint estimates the residual `τ*`. **Honest
  caveat (Grok):** `τ*` is only well-posed once `(f, ε)` are fixed (`f` is a *distribution*, not a Boolean
  function), and NNs do **approximate**, not exact, inference — so "no compact extension" may not be the right
  operationalization of hardness; the decomposition risks mixing representational capacity, descriptive power,
  and inference complexity in one equation. *(Continuing the Tropical / ρ-boundary collaboration — FINDINGS §4.)*
- **LO5** A static verifier over `Π` (the "verify-before-execute" payoff): which tokens are decided by the
  retrievable fragment alone (provably, no dense-`G` term) vs require the computed fragment. *Now the
  subject of a companion proposal — [`PROVABLE_OPT_PROPOSAL.md`](./PROVABLE_OPT_PROPOSAL.md) — which casts
  LO5 as a certified Datalog reduction (magic-sets demand pruning) and adds margin-certified pruning/quant.*

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
