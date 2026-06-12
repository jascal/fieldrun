# Provable Optimization of the Core

**Optimize the LLM *as a Datalog program* — and carry a proof. Lossless rewrites preserve every
derived fact; margin-certified rewrites preserve the decode.**

*Status: research proposal / a fourth paper. Where [LOGIC_EXPORT](./LOGIC_EXPORT.md) makes the model an
**executable** semiring-Datalog program `Π` (and LO3a emits the whole forward pass context-free), this
is what you **do** with that artifact: transform it for speed/size and **prove** the transform is safe.
[PIC](./PIC_PROPOSAL.md) is the measure (`T=1`), [Tropical](./TROPICAL_PROPOSAL.md) is the decision
geometry (`T=0`), LOGIC_EXPORT is the program — and the margin that certifies an optimization is exactly
Tropical's `T=0` margin, so this paper is their optimization-theory corollary, not a new object.
Empirical anchors: the LO3a whole-model emit (`fieldrun export --logic-whole`, `lo3a/`) and the existing
speed probes (`--pruned-head`, `--prune-head`/`--gate-check`, `--probe-quant`).*

---

## Abstract

Once the model **is** a Datalog program `Π` (LOGIC_EXPORT / LO3a — RMSNorm, RoPE/GQA attention, SwiGLU,
unembed, argmax, all as rules over weight facts), optimizing it stops being "compress and *measure* top-1
agreement" and becomes "transform and *prove* the result." Datalog is unusually generous here: its
least-fixpoint semantics makes a large class of source-to-source rewrites **semantics-preserving by
theorem**, and Soufflé ships several first-class. We split provable optimization of `Π` into two regimes:

> **Lossless (exact).** Preserve the entire least (stratified) model of `Π` — every derived tuple, hence
> `decide`, `logit`, and all intermediates, for *any* token context (EDB). Instruments: semi-naïve
> evaluation, **magic sets / demand transformation** (`souffle --magic-transform`), unfolding/inlining,
> common-subexpression elimination, constant-folding (the RoPE tables, the RMSNorm `^(-0.5)`), dead-stratum
> elimination, and Soufflé's compiled synthesis (Datalog → parallel C++ via Futamura projection). Each is a
> known `T_P`-equivalence; correctness is a theorem about the fixpoint, not a measurement.
>
> **Margin-certified (decode-lossless).** Preserve only the *queried answer* `decide` (not every
> intermediate), within a **proven margin**. The decode is the `(max,+)` argmax, so a rewrite that perturbs
> each logit by `≤ δ` preserves `decide` **iff** the margin `m = L(win) − L(runner-up) > 2δ`. Emit
> `certified :- margin(m), delta(d), m > 2*d` and Soufflé *checks* the certificate per input; quantified
> over inputs it is a guarantee. This is the provable upgrade of fieldrun's empirical `--pruned-head` margin
> gate and `--probe-quant` flip-rate.

The two regimes meet the rest of the program at one wall. Lossless demand transforms prune exactly the
**low-treewidth retrievable fragment** (induction = a demand-restricted recursive clause, n-gram = a fact)
and provably prune **nothing** on the dense computed fragment, because high treewidth means every
intermediate is demanded by `decide`. So *the magic-sets residual is a lossless, machine-checkable measure
of the forge tax* — a fourth instrument beside PR, treewidth, and tropical rank (LE-T4 / LO4). **PO-T1
(lossless preservation) is established; PO-T3 (the margin certificate) is established; PO-T2 (residual =
forge tax) is the measured-adjacent bridge; the certified-in-Lean/Coq pipeline (PO-T4) is the open
frontier of *formality*.**

---

## 1. Why Datalog optimization is *provable* (and Prolog's isn't)

LOGIC_EXPORT §1 chose Datalog over Prolog for static analyzability: terminating, order-independent, a
unique least fixpoint — `Π` is a mathematical **object**, not a procedure. That same property is what makes
its optimization provable. A transformation `Π ↦ Π'` is **lossless** when `Π'` has the same least model
(equivalently the same `decide`/`logit` for every EDB); because the semantics is a fixpoint of the
immediate-consequence operator `T_P`, "same least model" is a statement you can *prove by induction on the
fixpoint iteration*, not merely test. Prolog has no such handle: SLD resolution with cut/assert/order
dependence has no order-independent model to preserve, so "this rewrite is safe" is generally undecidable.

The payoff: **every speed/size win on `Π` can come with a correctness certificate**, ranging from a
differential test, through a Soufflé provenance proof, to a machine-checked Coq/Lean equivalence (§5).

---

## 2. The lossless toolbox (exact, every-tuple-preserving)

All of these preserve the *full* least model of `Π`, hence the whole forward pass, for any context.

| Transform | What it does to `Π` | Status on `Π` |
|---|---|---|
| **Semi-naïve evaluation** | delta-relations: each iteration touches only newly-derived tuples | core to Soufflé; **moot on `Π`** — the emitted forward pass is a *DAG* (unrolled layers), no recursion to accelerate. Relevant only if the retrievable fragment's induction clause is emitted recursively (§4). |
| **Magic sets / demand** (`--magic-transform`) | propagate the `decide`/`logit` demand backward; materialize only relevant tuples | **lossless, verified**: `souffle --magic-transform=* whole_base.dl` reproduces `decide` exactly. Prunes the retrievable fragment; **not** the dense core (§3). |
| **Dead-stratum elimination / demand** | `xf`, `ssf` are computed at every position but only `lastpos` is read by `logit` → restrict them to `lastpos` | lossless source-rewrite; a concrete saving (final-norm at one position, not all). |
| **Unfolding / inlining** | fuse a non-recursive intermediate into its consumer (e.g. fold `oproj` into the residual add) | lossless; trades relations for wider rules. |
| **Common-subexpression elimination** | the GQA `rep` query heads share one kv head — score/`v` reads of a kv head are recomputed per query head | lossless; the natural fix for GQA's `head/rep` sharing. |
| **Constant-folding** | the RoPE `rope_cos/rope_sin` facts, the literal `eps`, `1/√hd`, `E` are already folded at emit | lossless by construction (LO3a precomputes them). |
| **Compiled synthesis** (`souffle -c`) | Datalog → tuned parallel C++ (Futamura projection over the RAM machine) | Soufflé's documented codegen; the sparse-`(max,+)`-matmul performance face LOGIC_EXPORT §1.5/LO3(b) asks for is *this* applied to `Π`. |

> **PO-T1 (Lossless model optimization).** For any `T_P`-preserving Datalog transformation `Π ↦ Π'`,
> `[[Π']]` and `[[Π]]` assign the same `decide`/`logit` to every token context. **Status: established**
> (standard Datalog equivalence theory) and **anchored**: the magic-sets transform reproduces the
> emitted whole-model decode exactly on `lo3a/whole_base.dl`.

### 2.1 Measured: lossless compiled synthesis is ~200× faster (the `T=0` matmul-performance face)

Compiled synthesis is the lossless transform with the largest measured payoff on `Π`, and it is exactly
the sparse-`(max,+)`-matmul performance face LOGIC_EXPORT §1.5 / LO3(b) names. On `lo3a/whole_base.dl`
(the LO3a whole-model program), `souffle -o` turns the Datalog into a native binary; the result preserves
the decode and the logits to one ULP, at ~200× the speed:

| Execution of the *same* `Π` | ms / decode | decode | logit vs interpreter |
|---|---|---|---|
| Soufflé interpreter (naive bottom-up aggregate joins) | **4360** | 29 | — |
| Soufflé **compiled** (`-o`, native C++, semi-naïve + index selection) | **22** | 29 | max Δ = **4.4e-15 (1 ULP)** |
| fieldrun native kernel (dense f32 LA; incl. model load) | 255 | 29 | exact |

The compiled `decide` is **identical** to the interpreter on every held-out context (10/10 + the LO3a
48/48); the logit reassociation (4.4e-15) is summation-order from the compiled join plan, below any margin.
So the speedup is **lossless** in the decisive sense (the decode) and ULP-exact in the logits — a
semantics-preserving optimization (Soufflé's synthesis is a Futamura projection over the RAM machine,
provably the same least model), measured. Reproduce: `lo3a/bench.sh`. The interpreter's cost is the naive
aggregate join (near-cross-product); compilation selects indices and emits native loops — *the same
move that turns the matmul-aggregate into a kernel*, which is why fieldrun's hand-written dense kernel is
the limit of this same lossless ladder.

---

## 3. PO-T2 — the magic-sets residual *is* the forge tax

This is the connection that makes lossless optimization a research instrument and not just an engineering
win, and it is the *same wall* as LE-T2/LE-T4 seen from the optimizer.

Magic sets / demand transformation removes tuples that the queried output does **not** depend on. Its leverage
is therefore exactly the program's **treewidth / dependency sparsity**:

- The **retrievable fragment** is low-treewidth: induction is a recursive clause whose demand is a single
  matched position; n-gram is a fact keyed by a short context; closed-class is a unary constraint. Demand
  propagation prunes these hard — magic sets collapses them to the few tuples the query needs.
- The **computed fragment** (the dense residual, the forge tax) is high-treewidth: `decide` aggregates over
  all of `logit`, each `logit(v)` reads the full final residual, which (through dense matmuls + attention)
  depends on *every* intermediate at *every* position. Demand cannot prune what is universally demanded, so
  magic sets shrinks the dense core by **≈ nothing** — losslessly confirming there is nothing to drop.

> **PO-T2 (Lossless residual = forge tax).** The materialized-tuple count of `Π` *after* a maximal lossless
> demand transform is a **machine-checked, lossless measure of the computed fragment** (the forge tax): the
> pruned mass is the retrievable fragment, the residual is the dense-`G` / high-treewidth core. It is a fourth
> instrument for the one wall, beside provenance rank (PR), treewidth, and tropical rank (LE-T4 / LO4 / the
> Tropical paper). **Status: measured-adjacent** — the profiler already shows the dense matmul aggregates
> each materialize ~10² × the weight-fact count and survive demand pruning, while the structural
> (index/retrievable) relations do not.

So `--magic-transform` is not just a speedup on `Π`; run with an explicit retrievable-vs-computed split it is a
**falsifiable probe**: a token whose `decide` survives with the dense strata demand-pruned away is *provably*
retrievable (this is LO5, realized by a standard certified transform rather than a bespoke analyzer — §4).

---

## 4. PO-T3 — the margin certificate (decode-lossless approximate optimization)

Lossless transforms keep every tuple; the bigger wins (pruning, quantization) change the numbers and need a
different guarantee — preserve the **decode**, certified by the Tropical margin.

The decode is `decide(v) = argmax_v logit(v)`. Let `m = L(win) − L(runner-up)` be the margin (Tropical
`T=0`, the facet distance of FINDINGS §5b). If a rewrite perturbs every logit by at most `δ`, the argmax is
unchanged whenever `m > 2δ`. The bound `δ` for a dropped block / pruned candidate / quantized weight is
itself a Datalog aggregate (`sum |contrib|` over the removed terms), so the certificate is self-contained
and Soufflé-checkable:

```prolog
margin(M)    :- decide(W), logit(W,SW), M = SW - max S : { logit(V,S), V != W }.
delta(D)     :- D = sum A : { pruned_contrib(_, _, C), A = abs(C) }.   // bound on the perturbation
certified()  :- margin(M), delta(D), M > 2 * D.        // ⇒ pruned decode == full decode, PROVED for this EDB
.output certified
```

> **PO-T3 (Margin-certified decode invariance).** A transformation of `Π` that bounds the per-logit
> perturbation by `δ` preserves `decide` on every input where `margin > 2δ`. The bound and the check are
> Datalog; the witness is a Soufflé proof. **Status: established** (it is the `(max,+)` stability of the
> Tropical paper, instantiated on `Π` via LE-T5 faithfulness). This is the provable upgrade of:
> `--pruned-head` (heuristic margin gate → *certificate*), `--prune-head`/`--gate-check` (measured top-1
> agreement → *proof*), `--probe-quant` (measured flip-rate → *certified* per-block bit allocation, §6 PO4).

Quantified over a corpus the certificate becomes a guarantee — "for all inputs with `margin > 2δ`, the
pruned model decodes identically" — and the *uncertified* inputs are exactly the thin-margin, composed
(forge-tax) tokens, closing the loop with PIC O2 / FINDINGS §5.

---

## 5. The formality ladder (how strong a proof do you want?)

Three rungs, increasing in rigor and cost — all available because `Π` is a finite logical object:

1. **Differential testing.** Run `Π` and `Π'` over representative EDBs (token contexts), diff `decide`/`logit`.
   Cheap, catches most rewrite bugs. (This is exactly `lo3a/verify_all.py` for the *emitter*; the same harness
   certifies a *transform*.)
2. **Provenance audit.** Soufflé `-t explain decide(v)` emits a derivation tree — a checkable certificate of
   *why* the optimized program produced that token; compare trees pre/post transform.
3. **Machine-checked equivalence.** Encode the transform's `T_P`-preservation in a proof assistant: the CPP'21
   *"Developing and Certifying Datalog Optimizations in Coq/MathComp"* trace semantics (deduction trees,
   proven adequate to the immediate-consequence operator; clause-specialization and predicate-specialization
   transforms machine-checked by induction on iteration count), and the ITP'25 Lean formalization of Datalog
   model-theoretic vs proof-theoretic semantics with verified derivation-tree/graph checkers. End-to-end:
   *transform → prove equivalence in Lean/Coq → run optimized Soufflé → check the provenance tree against the
   formal semantics.*

> **PO-T4 (Certified pipeline).** A concrete transform on `Π` (e.g. the dead-stratum `lastpos` restriction,
> or GQA-CSE) carried with a machine-checked `T_P`-equivalence proof. **Status: open** — the Coq/Lean Datalog
> formalizations exist; instantiating one on a transform of `Π` is the formality frontier of this paper.

**Caveats (inherited from Datalog optimization theory).** Magic sets is *query-specific*: it preserves the
answers to the targeted outputs (`decide`/`logit`), not necessarily every IDB relation — fine here, since
the decode is the query. `Π` uses **aggregates** (sum/max) and is **stratified** (a DAG); demand/magic-set
variants over aggregation must respect stratification, so PO2/PO3 verify decode-preservation empirically
(rung 1) *and* aim for rung 3, rather than assuming it.

---

## 6. Theorems / claims by status, and open problems

| Claim | Content | Status |
|---|---|---|
| PO-T1 | lossless `T_P`-preserving rewrites keep `decide`/`logit` for every context | **Established** + anchored (`--magic-transform` on `whole_base.dl`) |
| PO-T3 | margin-certified decode invariance (`m > 2δ`) | **Established** (= Tropical `(max,+)` stability via LE-T5) |
| PO-T2 | lossless demand residual = the dense forge tax (a 4th LO4 measure) | Measured-adjacent |
| PO-T4 | machine-checked (Coq/Lean) equivalence for a transform of `Π` | Open (formality frontier) |

- **PO1 — certified reducer → smaller bundle + HF round trip, DONE (`lo3a/reduce.py`, `lo3a/to_safetensors.py`).**
  Scores FFN neurons over a calibration set; drops the **provably-dead** (a zero `down_proj` row writes
  nothing to the residual for *any* activation ⇒ `δ = 0`, exact on every input — the sound certificate core)
  and the margin-dominated ones; writes a structurally **smaller** fieldrun bundle; certifies decode
  preservation against fieldrun itself. Measured (tiny rope bundle, 12/64 dead FFN neurons/layer planted):
  certified-lossless reduction ffn 64→52 (**11% smaller bundle, 20/20 decode match**); margin-gated to
  ffn 64→36 (**27% smaller, still 20/20**), first flip at 64→28 (19/20 — where the `m>2δ` certificate refuses).
  The reduced model then **exports to Hugging-Face `safetensors` + `config.json` (`LlamaForCausalLM`) and
  round-trips back through `fieldrun convert` with 12/12 identical decodes** — closing the loop
  *bundle → Datalog → optimize → reduce → HF-publishable model → bundle*. *Remaining:* a static (a-priori)
  `δ` bound through the layers so deeper drops are certified, not just verified (the propagation caveat, §5).
- **PO2 — the magic-sets forge-tax measure.** Emit `Π` with an explicit retrievable fragment (induction =
  recursive clause, n-gram = fact) and the dense fragment; run `--magic-transform`; report lossless tuple
  reduction (= retrievable mass) and residual (= forge tax); correlate with PR / treewidth (LO4 bridge).
- **PO3 — LO5 as a certified reduction.** Per token, prove the dense strata are demand-irrelevant (the
  retrievable clauses suffice) → emit the reduced stratified program + an equivalence certificate. The static
  verifier LOGIC_EXPORT LO5 asks for, realized by a *standard, certifiable* transform.
- **PO4 — certified quantization.** Per-block perturbation bound `δ_b`; `certified :- margin > Σ_b δ_b`;
  principled *proven* per-block bit allocation — the provable face of `--probe-quant`.
- **PO5 — the end-to-end certified pipeline.** One transform on `Π`, proven equivalent in Lean/Coq, run in
  Soufflé, provenance-checked against the formal semantics (PO-T4 made real).

---

## 7. Related work

- **Magic sets / demand transformation** (Bancilhon–Maier–Sagiv–Ullman; Beeri–Ramakrishnan): the lossless,
  query-driven rewrite; proven query-equivalent for positive Datalog, with stratified/stable-model
  generalizations. The FGH-rule line derives magic sets and semi-naïve uniformly.
- **Semi-naïve evaluation** & **Soufflé synthesis** (Scholz et al.): delta-relation fixpoint + Datalog→C++
  via Futamura projection — lossless by construction.
- **Certified Datalog optimization**: CPP'21 *Developing and Certifying Datalog Optimizations in Coq/MathComp*
  (trace semantics + machine-checked transforms); ITP'25 Lean Datalog semantics + verified derivation
  checkers — the route to rung 3.
- **PIC / Tropical / LOGIC_EXPORT companions**: the measure (`T=1`), the margin (`T=0`, the certificate
  currency), and the program `Π` this optimizes. LE-T2/LE-T4 is the wall PO-T2 measures losslessly.
- **larql**: "the model IS the database" — this is its optimizer: *the model IS a Datalog program, and
  compressing it is provable program optimization.*

The stake: **a transformer optimized as a Datalog program, where every speedup carries a proof — lossless
rewrites preserve the whole forward pass by fixpoint equivalence, margin-certified rewrites preserve the
decode by the Tropical margin, the magic-sets residual measures the forge tax, and the dense core's
incompressibility is no longer a measurement but a theorem (LE-T2/LE-T4).**

---

## 8. Acknowledgment & provenance

The optimization-theory category of a four-category theory with [PIC](./PIC_PROPOSAL.md) (logic),
[Tropical](./TROPICAL_PROPOSAL.md) (geometry), and [LOGIC_EXPORT](./LOGIC_EXPORT.md) (computation). It adds
nothing to the object — it *uses* it: LO3a makes the model a Datalog program, Tropical's `T=0` margin
certifies the approximate rewrites, LE-T5 makes the certificate sound, and LE-T2/LE-T4 is exactly the
residual the lossless transforms cannot remove. Every empirical claim traces to a probe — the LO3a emit
(`lo3a/`), the `--magic-transform` lossless check, and the existing `--pruned-head` / `--probe-quant` speed
measurements in [`FINDINGS.md`](./FINDINGS.md) §5 — the same theory–experiment loop.
