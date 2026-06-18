# Hybrid Split-Execution: Lookup-in-C++, Composition-in-Ternary, Combine-in-Soufflé

**The retrievable/computed seam as a system architecture — *exact* (not approximate) to the
fixed-point model, with cost as the only variable to optimize**

*Status: research proposal / architecture synthesis. The convergence point of five prior threads in
this repo: [`FINDINGS.md`](./FINDINGS.md) §5 (the measured retrievable/computed split),
[`LOGIC_EXPORT.md`](./LOGIC_EXPORT.md) (the model as a semiring-Datalog program — the combine),
[`TERNARY` via `--verify-ternary`/`src/ternary.rs`] (the balanced-ternary lossless-via-expansion lemma),
[`TROPICAL_PROPOSAL.md`](./TROPICAL_PROPOSAL.md) (the facet margin — the gate, TT2), and
[`TURBOQUANT.md`](./TURBOQUANT.md) (the distortion bound). External lineage: BitNet b1.58 (ternary
LLMs), GPTQ/AWQ (post-hoc quant), retrieval-/cache-augmented generation, and `larql` ("the model IS a
database"). This document is written to be self-contained for external review — the measured numbers
are cited inline and flagged as measured vs conjecture.*

---

## Abstract

A transformer's next-token decision factors into two tiers that fieldrun **measures**: a **RETRIEVABLE**
tier (induction / n-gram / grammar — a lookup) and a **COMPUTED** tier (attention + MLP entanglement —
the "forge tax"). This proposal runs each tier in its natural substrate and glues them declaratively:

> **Lookup → a pure C++ KB facility** (exact, O(1) hash/FST). **Composition → balanced-ternary matmuls**
> (lossless via expansion, sparse sign-flip-accumulate). **Combine → a Soufflé semiring program**
> (`argmax` over the summed contributions = the `T=0` tropical decode; `log-sum-exp` for the `T=1`
> measure).

The key consequence: **both tiers are exact**, so the combine is exact up to the single float→int
fixed-point step. "Get as close as possible to the original" therefore becomes "**exactly equal to the
chosen-precision model**," and the cost (the ternary `K×` width blowup) — not accuracy — is the only
variable to optimize. The split also **sidesteps the dense-Gram wall** (`LOGIC_EXPORT` LE-T4) that blocks
the whole-model Datalog emit: retrieved logits enter Soufflé as **EDB facts** from the C++ lookup, so the
Datalog program is the small *combine*, not the dense forward. The lookup **short-circuits** the RETRIEVED
majority; the ternary tier is the exact **fallback** that carries the COMPOSED tail, gated by the tropical
**facet margin** (TT2). The hard part is honest and stated (§7, §9): the split is per-*decision*, not a
static per-*weight* partition.

---

## 1. The seam — why this cut, and not another

The retrievable/computed boundary is **the model's own structure**, not an imposed one:

- **Measured.** `FINDINGS.md` §5 classifies every decision RETRIEVED / SELECTED / COMPOSED by whether a
  context-keyed KB (induction + n-gram + grammar) reproduces the model's pick. On the 7B "monster" run
  (30k tokens, 10 languages + code + math): **~57% span-1 routable** (the deciding atom fits one expert),
  and the deciding circuits are **~93% sparser** than the full set (oracle-router proxy). The COMPOSED
  fraction is **~15% on natural text / ~37% on code**.
- **Partitioned.** The density-bucketing work (`DENSITY_BUCKETING.md`) clusters the model's circuits into
  retrievable hub-experts + a dense residual; the residual is the part no hub absorbs.
- **Already emitted in two tiers.** `LOGIC_EXPORT` (LO3) emits **Tier A** (the retrievable fragment as
  compact clauses/facts) and **Tier B** (composition as per-block weighted `contrib` facts), with
  `Σ contrib == logit` (LE-T5) and a max-plus argmax decode.

So "lookup → C++, entangled → ternary" assigns each *measured* tier to the substrate it's actually shaped
like. The cut is the model's grain.

---

## 2. The architecture

```
context tokens
   │
   ├─►  [Tier A — C++ lookup facility]       retrieval / selection
   │      n-gram / induction / grammar KB  →  candidate set + retrieved logits
   │      hash / FST / perfect-hash;  EXACT (stored values);  no quantization
   │      (the tropical-rank-1 table — what a flat KB can express, TT5)
   │
   ├─►  [Tier B — ternary composition engine]  composition / forge tax
   │      attn + MLP as balanced-ternary matmuls (lossless via expansion;
   │      sparse sign-flip-accumulate)  →  computed logit contributions
   │
   └─►  [Soufflé semiring combine]
          predict(v) = argmax_v ( retrieved(v) ⊕ computed(v) )      T=0  (decode)
                     = softmax  ( retrieved(v) ⊕ computed(v) )      T=1  (measure)
          retrieved(v) arrives as EDB facts; computed(v) as rules; LE-T5 round-trip self-check
```

---

## 3. The reframe — *exact*, not approximate

This is the load-bearing idea. Naive ternarization (and the lossy-quant line in `TURBOQUANT.md`)
*approximates* the model and then bounds the error. Divide-and-conquer + **lossless** expansion removes
the approximation:

- **Tier A is exact** — it *stores* the retrieved values; a lookup has no quantization error.
- **Tier B is exact** — the balanced-ternary expansion `Σᵢ wᵢxᵢ = Σⱼ 3ʲ (Σᵢ tᵢⱼxᵢ)` is an identity
  (`tᵢⱼ ∈ {−1,0,+1}`), verified **byte-identical on real int8 weights** by `--verify-ternary`
  (`src/ternary.rs`; PASS, i64-exact).
- Therefore **the combine is exact**, up to the one lossy step that *any* finite-precision scheme has:
  `float → w_int = round(w/s)` (choosing the fixed-point precision `s`).

So the objective flips: **"as close as possible" → "exactly equal to the chosen-precision model,"** and
the cost — the ternary `K×` width blowup (`K = ⌈log₃(2·max|w_int|+1)⌉`: int4→3, int8→6, fp16→11) — becomes
the only thing to optimize. And it is *amortizable*, not fixed:

- **Sparsity.** Measured on a real int8 layer: **52.5% of trits are zero**, **mean 2.85 nonzero
  trits/weight** vs the uniform `K=6` (zeros are *free* in Datalog's closed world — absent facts).
- **Short-circuit.** The lookup answers the RETRIEVED majority, so the ternary `K×` is paid only on the
  COMPOSED tail (`~15–37%`).

Trading accuracy-loss for amortizable compute is a strictly better place to stand than minimizing a loss.

---

## 4. Tier A — the C++ lookup facility (retrieval / selection)

**What it is.** A function `context → (candidate set, retrieved logits)`. Concretely the n-gram /
induction / grammar KB: given the recent context, return the tokens a flat table predicts plus their
stored scores. This is the **tropical-rank-1** fragment (TT5) — the part a lookup table *can* express.

**What exists.** `retrieval::Store` already is this KB (quad/tri/bi/uni successor tables keyed on token
ids, the induction/recency candidates, the grammar skeleton). `candidates()` + the pruned-head already
produce the per-context candidate set. So Tier A is **largely built**; "pure C++ facility" is a
packaging/scale choice — a succinct index (FST / perfect hash / sorted-array binary search) for a
standalone, embeddable, exact lookup with no float arithmetic.

**Why it's exact & cheap.** It returns *stored* values (no matmul, no quantization), `O(1)`–`O(log n)`
per query. It carries the high-margin decisions (RETRIEVED tokens have the largest facet margins —
FINDINGS §5b: ~2.1–2.8 vs ~0.9–1.2 for COMPOSED).

---

## 5. Tier B — the ternary composition engine (the forge tax)

**What it is.** The attention + MLP composition — the part that genuinely *computes* and that no compact
lookup reproduces (the forge tax; TT5's tropical-rank-gap). Represented as **balanced-ternary matmuls**:
each integer weight `w` becomes `K` trits, and the layer's dot is the power-of-3-weighted sum of `K`
ternary dots — exact (§3). The kernel is a **sparse sign-flip accumulate** (`±x`, skip zeros), cheaper
than int4 (`i4_dot`) and it exploits the 52.5%-zero sparsity directly.

**What's new.** A `Tern` bundle dtype + `tern_dot` kernel (mirrors the existing `I4w`/`i4_dot`), the
convert path, and the forward wiring. `src/ternary.rs` already has the expansion + the trit-sparsity
accounting; `--verify-ternary` already proves the identity on real weights.

**The optimization (the hard, valuable half).** The uniform `K` is the existence bound; the *sparse* trit
budget (mean 2.85, with small weights needing fewer — measured histogram: most weights use 3–5 trits,
1.3% are exact zero) is what you minimize. "Fewest trits preserving behaviour (exactly, or within `ε` on a
dataset)" is an integer program / MaxSAT over the relational form — analytically well-posed, NP-hard in
general. This is where the analytical/optimization work lives; existence is the easy part.

---

## 6. The Soufflé combine

The decode is a semiring aggregation over the **sum** of the two tiers:

```
.decl retrieved(v:number, logit:float)      // EDB — facts from the C++ lookup (Tier A)
.decl computed(v:number, logit:float)       // from the ternary engine (Tier B): Σ_j 3^j Σ_i t_ij x_i
.decl score(v:number, s:float)
score(v, lr + lc) :- retrieved(v, lr), computed(v, lc).
score(v, lc)      :- computed(v, lc),  !retrieved(v, _).     // computed-only candidates
predict(v)        :- score(v, s), s = max { s2 : score(_, s2) }.   // T=0 argmax (max-plus)
```

- This is precisely LOGIC_EXPORT's `LE-T5` (`Σcontrib == logit`) and the tropical `T=0` argmax; swapping
  the aggregate for `log-sum-exp` gives the `T=1` measure (PIC). Soufflé is the **verifiable spec** — a
  statically-checkable, terminating, least-fixpoint program — that compiles to parallel C++.
- **It sidesteps the dense-Gram wall (LE-T4).** The whole-model emit was non-compact because the
  unembedding is `vocab × d` dense weight facts. Here the *retrievable* logits arrive as **EDB facts from
  the C++ lookup** (no dense Gram emitted for them), and only the *computed* contributions + the combine
  are rules. The Datalog program is the small combine, not the dense forward — which is exactly the
  blocker that the split dissolves.
- **Round-trip self-check.** As LE-T5 does today: emit, run Soufflé on a held-out context, and confirm
  `predict` equals the model's decode (exactly, in the fully-lossless setting; within the gate's tolerance
  otherwise).

---

## 7. The gate — per-token routing between tiers (the honest hard part)

The retrievable/computed split is **per-decision, not per-weight**: the same attention head is "induction
lookup" for one token and "composition" for the next. So you cannot statically slice weights into "lookup
weights" vs "ternary weights." The realization that makes the architecture work:

- **Tier A is a fast-path short-circuit** keyed on context (the n-gram/induction KB *is* a table). On the
  high-margin RETRIEVED majority it answers directly; Tier B is skipped.
- **Tier B is the exact fallback** that runs the full composition and carries the COMPOSED tail.
- **Amortized cost** `= P(retrieved)·lookup + P(composed)·ternary`, dominated by the cheap lookup; the
  `K×` ternary blowup is paid only on the hard minority.

**The gate is the tropical facet margin (TT2).** Short-circuit on the lookup's top candidate only when its
margin exceeds the largest swing the computed tier could contribute. In the **fully-lossless** setting the
computed tier contributes its *exact* value, so the gate is just "is the lookup's top provably the argmax
given the (bounded) computed term" — and `TURBOQUANT` TT2 gives the closed-form threshold when the
computed term is itself quantized. The margin probe (`--probe-tropical`) is the instrument that calibrates
the gate's false-accept rate.

---

## 8. Claims by status

| Claim | Content | Status |
|---|---|---|
| **HY-T1** | The combine is **exact** up to the float→int fixed-point step (both tiers exact) | Follows from Tier-A lookup exactness + the ternary identity (`--verify-ternary` PASS) |
| **HY-T2** | The split **sidesteps the dense-Gram wall** (LE-T4): retrieved logits are EDB facts, not dense weight facts | Structural (LOGIC_EXPORT) |
| **HY-T3** | Amortized cost `= P(retr)·lookup + P(comp)·ternary`, lookup-dominated (~57% span-1 routable) | Measured inputs; cost model is **conjecture** (HY-O4) |
| **HY-T4** | The facet margin (TT2) is a sound short-circuit **gate** | **Conjecture** (calibrate via `--probe-tropical`, HY-O2) |
| **HY-T5** | Tier B **is** the localized forge tax (the computed residue no lookup captures = TT5's tropical-rank gap) | Measured-adjacent (TT5 + the bucketing residual) |

---

## 9. Open problems — the analysis targets

These are the questions worth digging into; HY-O1/O2/O4 are the cruxes.

- **HY-O1 (the modeling crux).** The split is per-decision, not a static weight partition. Is there a clean
  factorization `M = Lookup ⊕ Compute` where Lookup is a genuine table and Compute the complement — or is
  the only honest form the *fast-path/fallback* of §7? What's the right object: a learned router, the
  induction/n-gram KB as-is, or a distilled "retrievable head" set?
- **HY-O2 (gate soundness & coverage).** When can Tier A short-circuit without Tier B disagreeing? The
  margin threshold, the false-accept rate, and the coverage (what fraction of tokens safely skip Tier B) as
  functions of the margin and the corpus. Is there a *certified* gate (provably never wrong) vs a
  *calibrated* one (wrong with bounded probability)?
- **HY-O3 (Soufflé: runtime vs spec).** Is Soufflé in the hot loop, or does it only *generate/verify* the
  C++ combiner? The per-token combine is small (candidates × contributions); measure whether the Datalog
  engine is competitive or whether it's best as the statically-checked source of a hand-tunable C++ kernel.
- **HY-O4 (the cost model).** Quantify `P(retr)·lookup + P(comp)·ternary` on real models: does the COMPOSED
  tail's `K×` ternary cost dominate, and how far does the sparse-trit optimization (HY-O5) push it down?
- **HY-O5 (minimize the blowup).** The integer-program/MaxSAT for fewest nonzero trits preserving behaviour
  — exact, and the `ε`-relaxation on a dataset (where the facet margin says which decisions have slack).
  Variable-length per-weight expansion (small weights → fewer trits) already drops the mean to 2.85; how
  much further with structure (shared trit planes, block sparsity)?
- **HY-O6 (lossless vs lossy Tier B).** Keep Tier B fully lossless (`K×`, exact) — or allow a *lossy*
  ternary tier + a small **data-aware low-rank residual** (the `TURBOQUANT` line; the measured "rank-8
  update beats the frozen-linear Θ(d) floor losslessly") for a smaller, slightly-approximate engine? The
  accuracy/cost frontier, in *decision-fidelity* currency, not weight-MSE.
- **HY-O7 (margin-adaptive precision — *developed*).** The only loss is `float → w_int`, so choose
  precision per layer to spend bits where decisions are tight. The advance over standard mixed-precision
  (HAWQ-style Hessian-trace sensitivity) is to use the **tropical facet margin** as the sensitivity signal
  — it is *decision-aware* (what actually flips the argmax, TT2/E7) rather than loss-curvature. fieldrun
  already computes **both** ingredients: per-circuit DLA (`--probe-decompose`, the contrib decode) and the
  facet margin (`--probe-tropical`), so a layer's bit budget can be set from its **tightness exposure** —
  how often its contribution is pivotal (high DLA) on a small-margin decision, e.g.
  `Σ_decisions DLA_layer · (1/margin)`. In this architecture the precision knob is concrete: the
  **trit-truncation depth `K′ ≤ K`**. Keep all `K` trits ⇒ lossless (no flip); truncate to `K′` where the
  residual (`~3^{K′}` scale) stays below the margin's tolerance (the `TURBOQUANT` TT2 threshold). So
  **HY-O6 and HY-O7 are the same knob** — per-weight/per-layer trit depth, set by the margin; the
  fully-lossless tier is just "all trits everywhere." *Remaining crux:* aggregating the per-decision
  margins to a static per-layer/per-group `K′`, and the closed form `K′(margin)`.
- **HY-O8 (MoE composition — *developed*).** It composes cleanly, and the monster-tree result *is* the
  decomposition. **Tier A** = a shared retrieval backbone (the recurring hub-experts — the bucketing
  anchors that fire across domains, e.g. the monster tree's depth-0 hubs / the recurring late-layer hub)
  **plus expert-specific n-gram tables** (per-domain). **Tier B** = the routed sparse experts expressed in
  ternary (the monster tree's per-language / code / math leaves). The short-circuit then **bypasses the
  router itself**, not just expert compute, for high-margin retrieved cases — a strictly bigger win, since
  routing is its own cost. This rides fieldrun's existing expert-offload (`bundle.rs` paged experts): cold
  ternary experts stay paged, the lookup backbone stays resident. *Remaining crux:* the per-token gate must
  stay sound under routing — the margin has to bound the contribution of the **unselected** experts, not
  just the chosen one.

---

## 10. Related work & provenance

- **The retrievable/computed split** — `FINDINGS.md` §5 (measured), `DENSITY_BUCKETING.md` (the partition),
  `LOGIC_EXPORT.md` (Tier A/B, LE-T5 `Σcontrib==logit`, the LE-T4 dense-Gram wall this dissolves).
- **Lossless ternary** — the balanced-ternary expansion lemma (`src/ternary.rs`, `--verify-ternary`,
  byte-identical PASS; the i-orca `bitnet/ternary` corpus kernel-checks the existence half). BitNet b1.58
  (Ma et al.) for trained ternary LLMs; GPTQ/AWQ for post-hoc quant.
- **The gate** — `TROPICAL_PROPOSAL.md` TT2 (margin = facet distance), TT5 (forge tax = tropical-rank gap);
  `TURBOQUANT.md` TT2 (the closed-form distortion threshold when Tier B is quantized).
- **The combine** — `LOGIC_EXPORT.md` (the model as a semiring-Datalog program); `larql` ("the model IS a
  database" — Tier A is literally that).

The stake: **the model split along its own measured seam — an exact C++ lookup for what's retrievable, a
lossless ternary engine for what must be computed, and a verifiable Soufflé program as the semiring glue —
so that the result is not an *approximation* of the original but an *exact* reconstruction of the
chosen-precision model, with cost the only thing left to minimize.**
