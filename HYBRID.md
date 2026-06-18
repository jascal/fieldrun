# Hybrid Split-Execution: Lookup + Ternary Composition in Rust, certified by Datalog

**The retrievable/computed seam as a system architecture — a pure-Rust engine (exact, not approximate, to
the fixed-point model, with cost the only variable to optimize), with Soufflé/Datalog kept as the *offline
certificate*, not the runtime glue**

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
the "forge tax"). This proposal runs each tier in its natural substrate — **all within fieldrun's existing
framework-free pure-Rust runtime** — and keeps Datalog as the proof, not the plumbing:

> **Lookup → a Rust KB facility** (exact, O(1); `retrieval::Store` today, an `fst`/perfect-hash index for
> scale). **Composition → balanced-ternary matmuls** (lossless via expansion, sparse sign-flip-accumulate;
> a Rust kernel mirroring `i4_dot`). **Combine → a trivial Rust join+argmax** over the two tiers' summed
> contributions (the `T=0` tropical decode; `log-sum-exp` for the `T=1` measure). **Soufflé/Datalog is the
> *offline certificate*** that proves this engine is faithful (the `LE-T5` round-trip), **not** the runtime.

The key consequence: **both tiers are exact**, so the combine is exact up to the single float→int
fixed-point step. "Get as close as possible to the original" therefore becomes "**exactly equal to the
chosen-precision model**," and the cost (the ternary `K×` width blowup) — not accuracy — is the only
variable to optimize. The combine itself is a relational join + aggregate with **no recursion and no
fixpoint**, so it is a handful of Rust lines, not a Datalog-engine workload — Datalog's value here is
verification and framing (§6), which are offline. The split also **dissolves the dense-Gram wall**
(`LOGIC_EXPORT` LE-T4) that blocks the whole-model Datalog *export*: retrieved logits enter as **EDB
facts** from the lookup, so the certified program is the small *combine*, not the dense forward. The lookup
**short-circuits** the RETRIEVED majority; the ternary tier is the exact **fallback** that carries the
COMPOSED tail, gated by the tropical **facet margin** (TT2). The hard part is honest and stated (§7, §9):
the split is per-*decision*, not a static per-*weight* partition.

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

So "lookup → table, entangled → ternary" assigns each *measured* tier to the substrate it's actually shaped
like. The cut is the model's grain — and both substrates are native Rust (a KB index and a sparse integer
kernel), so the engine never leaves fieldrun's framework-free pure-Rust runtime.

---

## 2. The architecture

```
context tokens                                            ─── all Rust (one process, no FFI) ───
   │
   ├─►  [Tier A — Rust lookup facility]      retrieval / selection
   │      n-gram / induction / grammar KB  →  candidate set + retrieved logits
   │      retrieval::Store today / fst|perfect-hash for scale;  EXACT (stored values); no quantization
   │      (the tropical-rank-1 table — what a flat KB can express, TT5)
   │
   ├─►  [Tier B — Rust ternary composition engine]  composition / forge tax
   │      attn + MLP as balanced-ternary matmuls (lossless via expansion;
   │      sparse sign-flip-accumulate; a kernel mirroring i4_dot)  →  computed logit contributions
   │
   └─►  [Rust combine — a join + argmax]
          predict(v) = argmax_v ( retrieved(v) + computed(v) )      T=0  (decode)
                     = softmax  ( retrieved(v) + computed(v) )      T=1  (measure)

   ┄┄┄ offline ┄┄┄
   [Soufflé/Datalog certificate]  emit the combine (retrieved as EDB facts, computed as rules) and
   run it to PROVE the Rust engine is faithful (LE-T5 round-trip) — not in the hot loop.
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

## 4. Tier A — the Rust lookup facility (retrieval / selection)

**What it is.** A function `context → (candidate set, retrieved logits)`. Concretely the n-gram /
induction / grammar KB: given the recent context, return the tokens a flat table predicts plus their
stored scores. This is the **tropical-rank-1** fragment (TT5) — the part a lookup table *can* express.

**What exists.** `retrieval::Store` already is this KB (quad/tri/bi/uni successor tables keyed on token
ids, the induction/recency candidates, the grammar skeleton). `candidates()` + the pruned-head already
produce the per-context candidate set. So Tier A is **largely built — in Rust already**; scaling it is a
data-structure choice *within* Rust (the `fst` crate / perfect hashing / sorted-array binary search) for a
compact, embeddable, exact lookup. No C++ is needed or wanted — fieldrun is framework-free pure Rust, and a
key→value table is the last thing that would justify an FFI boundary.

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

### 5.1 Tier B on accelerators — native low-precision bulk + a small exact ternary residual

*(developed with Grok.)* Pure ternary maps poorly to accelerator matrix units (NPUs/GPUs want dense
int4/int8 GEMM, not `{−1,0,+1}` + power-of-3 + sparsity). A **residual split** fixes this while keeping
exactness *on demand*, and it is the practical form of Tier B (full ternary, §5, is then the exact
*reference*):

> `w = ŵ_q + r`  ⇒  `w·x = ŵ_q·x + r·x`. Run the **bulk** `ŵ_q·x` as a dense matmul on whatever the
> hardware does fast, and compute the **residual** `r·x` with the lossless balanced-ternary expansion
> (`src/ternary.rs`, `--verify-ternary`) only where it matters.

The residual `r = w − ŵ_q` is the *low-order* quantization error, so it is **small and sparse** — `|r|` is
bounded by the int-`q` step, giving `K_r ≈ 2–3` trits, not 6. So even a *full* exact residual is just the
native bulk + a cheap 2–3-trit pass.

**Exact vs approximate is a spectrum — a per-*decision* precision ladder (the per-decision extension of
HY-O7).** Two regimes, both keyed by the facet margin `m`:
- **Full residual (all weights) ⇒ bit-exact** — `w·x` reconstructed exactly (HY-T1 holds).
- **Subset residual (the tight-margin weights, selected by `DLA·(1/m) > τ`) ⇒ approximate**, with error
  bounded by the *omitted* residuals: `δ = ‖r_omitted‖∞ · ‖x‖₁` (tighter from the calibration activation
  stats). The argmax is then **sound wherever `m > 2δ`** — each candidate's score shifts ≤ ±δ, so a margin
  past `2δ` cannot flip it. (Naming note vs Grok's sketch: `δ` is the *omitted*-residual swing, so full
  residual ⇒ `δ = 0` ⇒ exact; the more you include, the smaller `δ`, the more decisions are argmax-safe.)

| facet margin `m` | path | cost | exactness |
|---|---|---|---|
| `m >` full Tier-B bound | **Tier A short-circuit** (skip Tier B) | lookup only | gate-sound (§7) |
| `m > 2δ` | **native bulk + subset residual** (the common, accelerator-fast case) | int4/8 GEMM + small ternary | argmax-exact |
| `m ≤ 2δ` | **full exact residual** (or full ternary) for that decision | int4/8 GEMM + full residual | bit-exact |

Typical from the measured DLA/margin distributions: **~10–25% of weights** carry the exact residual; the
accelerator does the rest in native low precision.

**Hardware-parametric — and it subsumes "stick with int8."** The bulk precision is the target's native
unit: **int8 on fieldrun's CPU today** (the existing `i8_dot`), int4 on an NPU later; the ternary residual
runs on the scalar/CPU path either way. So this is exactly *"keep the native low-precision matmul, and
bring ternary back only as a small exact correction for the tight decisions"* — the synthesis of the
int8-first cut and the ternary exactness, with full ternary (§5) recovered by setting the bulk to zero.

**Calibration reuses existing tooling.** Per-layer `δ` and the `DLA·(1/m) > τ` residual mask come from
`--probe-decompose` (DLA) + `--probe-tropical` (margin). New work: the residual-selection calibration pass,
the sparse residual kernel applied to the small selected set (reuses `src/ternary.rs`), and an empirical
check of the `m > 2δ` rule's false-short-circuit rate on held-out data.

---

## 6. The combine — a trivial Rust step, certified by Datalog

**At runtime** the combine is a relational join + aggregate over the **sum** of the two tiers:
`predict(v) = argmax_v ( retrieved(v) + computed(v) )`. There is **no recursion and no fixpoint** — so in
Rust it is a hash-join over the candidate set and a `max` (≈ a dozen lines), and the `T=1` measure is the
same with `log-sum-exp`. A per-token combine over a few hundred candidates is the *last* thing that needs a
Datalog engine; Soufflé compiling to parallel C++ is for large recursive relational workloads, which this
is not. **So the runtime combine stays in Rust** alongside Tiers A and B — one process, no FFI.

**Offline**, the *same* combine is emitted as a semiring-Datalog program and run in Soufflé as a
**certificate** — this is where Datalog earns its place (verification + framing, not throughput):

```
.decl retrieved(v:number, logit:float)      // EDB — facts handed in from the Rust lookup (Tier A)
.decl computed(v:number, logit:float)       // from the ternary engine (Tier B): Σ_j 3^j Σ_i t_ij x_i
.decl score(v:number, s:float)
score(v, lr + lc) :- retrieved(v, lr), computed(v, lc).
score(v, lc)      :- computed(v, lc),  !retrieved(v, _).     // computed-only candidates
predict(v)        :- score(v, s), s = max { s2 : score(_, s2) }.   // T=0 argmax (max-plus)
```

- It is precisely LOGIC_EXPORT's `LE-T5` (`Σcontrib == logit`) and the tropical `T=0` argmax (`log-sum-exp`
  → the `T=1` measure, PIC) — one **semiring-parameterized** program where the temperature is the semiring
  choice. Its value is being a *statically-checkable, terminating, least-fixpoint object* that **proves the
  Rust engine faithful**, and that holds the "model IS a semiring program" claim (larql / LOGIC_EXPORT).
- **It dissolves the dense-Gram wall (LE-T4).** The whole-model *export* was non-compact because the
  unembedding is `vocab × d` dense weight facts. Here the *retrievable* logits arrive as **EDB facts from
  the lookup** (no dense Gram emitted), and only the *computed* contributions + the combine are rules — so
  the certified program is the small combine, not the dense forward.
- **Round-trip self-check.** As LE-T5 does today: emit, run Soufflé on a held-out context, and confirm
  `predict` equals the Rust engine's decode (exactly, in the fully-lossless setting; within the gate's
  tolerance otherwise). This is the byte-identity `--verify-*` ethos, expressed as a logic proof.

So: **Rust is the engine, Datalog is the certificate** — the two roles the earlier "use Soufflé to combine"
framing conflated. Nothing is lost by keeping the runtime pure Rust; the Datalog artifact is produced on
demand for verification.

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
- **HY-O3 (Soufflé: runtime vs certificate — *resolved*).** Settled toward **certificate**: the combine is
  a non-recursive join+argmax (no fixpoint), so the runtime is a dozen lines of Rust, and Soufflé/Datalog
  is the *offline* proof that the Rust engine is faithful (LE-T5 round-trip) — not the hot loop (§6). The
  residual question is only *cadence*: how often to re-certify (per build / per bundle / per release).
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

## 10. Relationship to TurboQuant — complementary, and worth doing first

[`TURBOQUANT.md`](./TURBOQUANT.md) (unbiased KV-cache quantization + the margin–distortion bound) and this
hybrid are **complementary — they quantize different objects** — and they share one instrument, which is
exactly why the practical sequencing is **TurboQuant first**.

- **Different objects.** TurboQuant compresses the **KV cache** (per-token vectors — the attention memory);
  the hybrid quantizes/represents the **weights** (Tier B) and the **retrieval KB** (Tier A). Orthogonal
  axes; apply each to its own object and they compose.
- **The one rule: don't TurboQuant the *weights*.** TurboQuant is lossy-unbiased; the hybrid keeps weights
  exact (int8 bulk + exact ternary residual, §5.1). So: TurboQuant the **KV cache** (lossy-unbiased is fine
  there), keep the **weights and unembedding exact** via the residual path. Double-applying is the only
  interaction to avoid — it would break the hybrid's exact-on-demand property.
- **They share the facet-margin gate.** TurboQuant's "stable iff `m > z·ρ_KV`" (TT2) and the hybrid's
  "argmax-sound iff `m > 2δ_weight`" (§5.1) are the *same* margin-as-error-budget argument on different
  error sources. Do both and the gate just sums the budgets: **`m > z·ρ_KV + 2δ_weight`** — one unified
  gate, not a conflict.
- **Use-case split.** The KV cache is needed *even for short-circuited tokens* (future positions attend to
  them), so the hybrid's whole-forward short-circuit is mainly a **scoring / single-decision** win; for
  autoregressive **generation** you still populate K/V and TurboQuant's KV compression is the dominant
  lever. So TurboQuant is, if anything, *more* central in the generation path.

**Sequencing — TurboQuant first.** Four reasons, the second being the load-bearing one:
1. **Lower risk / better understood** — KV-cache quantization is effectively an industry standard and the
   math is proven (the paper + the i-orca `turboquant` corpus). The hybrid's gate (HY-O2) is the *novel*
   research risk; do the safe, high-value thing first.
2. **It builds the shared instrument.** TurboQuant's deliverable (B) is the **margin–distortion probe** (the
   TO7/E7 settle) — *exactly* the gate-calibration tooling the hybrid's Phase 3 reuses. Building TurboQuant
   first produces the `ρ` / margin machinery; the hybrid then only adds the `δ_weight` half of the unified
   gate. So this isn't just "easier first" — it's "the prerequisite that yields the hybrid's gate."
3. **Standalone win on the memory-bound 7B.** The 7B is KV / bandwidth-bound on commodity hardware;
   TurboQuant attacks that directly and ships value before any hybrid machinery exists.
4. **De-risks the shared principle.** E-TQ2 (flip-rate vs margin/distortion) validates the
   margin-as-budget law that *both* gates rest on — confirm the foundation before betting the hybrid on it.

So the program order is: **TurboQuant (KV mode + margin–distortion probe) → the hybrid (Tier A lookup +
int8-bulk/exact-residual Tier B + the unified gate, reusing TurboQuant's margin instrument).**

---

## 11. Related work & provenance

- **The retrievable/computed split** — `FINDINGS.md` §5 (measured), `DENSITY_BUCKETING.md` (the partition),
  `LOGIC_EXPORT.md` (Tier A/B, LE-T5 `Σcontrib==logit`, the LE-T4 dense-Gram wall this dissolves).
- **Lossless ternary** — the balanced-ternary expansion lemma (`src/ternary.rs`, `--verify-ternary`,
  byte-identical PASS; the i-orca `bitnet/ternary` corpus kernel-checks the existence half). BitNet b1.58
  (Ma et al.) for trained ternary LLMs; GPTQ/AWQ for post-hoc quant.
- **The gate** — `TROPICAL_PROPOSAL.md` TT2 (margin = facet distance), TT5 (forge tax = tropical-rank gap);
  `TURBOQUANT.md` TT2 (the closed-form distortion threshold when Tier B is quantized).
- **The combine** — `LOGIC_EXPORT.md` (the model as a semiring-Datalog program); `larql` ("the model IS a
  database" — Tier A is literally that).

The stake: **the model split along its own measured seam — an exact Rust lookup for what's retrievable, a
lossless Rust ternary engine for what must be computed, a trivial Rust join+argmax to combine, and a
Soufflé/Datalog program kept as the offline certificate that the whole thing is faithful — so that the
result is not an *approximation* of the original but an *exact* reconstruction of the chosen-precision
model, in one framework-free pure-Rust process, with cost the only thing left to minimize.**
