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
(lossless preservation) is established; PO-T3 (the margin certificate) is established *locally* but its
global reach is *bounded by LE-T2* — a sound, machine-checkable certificate that exposes the forge-tax wall
rather than escaping it; PO-T2 (residual = forge tax) is the measured-adjacent bridge; the certified-in-Lean/Coq
pipeline (PO-T4) and the through-layers `δ` bound (PO-T6, likely LE-T2 again) are the open frontier of
*formality*; and the grokking order parameter (PO-T7) — certifiable-compressibility as a progress measure for
memorization→generalization — is the most interesting open, directly-testable direction.**

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
> Datalog; the witness is a Soufflé proof. **Status: established as a *local* certificate** — it is exactly
> Maslov-dequantization of the classical margin argument: in `(max,+)` a `δ`-swing moves winner vs. runner-up
> by at most `2δ`, so `m > 2δ` is **necessary and sufficient** for invariance on that input, inheriting LE-T5
> soundness. The provable upgrade of: `--pruned-head` (heuristic margin gate → *certificate*),
> `--prune-head`/`--gate-check` (measured top-1 agreement → *proof*), `--probe-quant` (measured flip-rate →
> *certified* per-block bit allocation, PO4).

**The certificate is sound but bounded by LE-T2 (Grok, PO4 review).** The margin `m` is the Euclidean facet
distance in the power diagram of the unembedding vectors; the allowable `δ`-cushion *is* that distance. But a
**scalar** `δ_b` is exact only on the diagonal of `G_{vw} = ⟨U_v, U_w⟩` — off-diagonal it under/over-estimates
the effective perturbation precisely in the high-PR / high-tropical-rank region, **the forge tax**. So the
certificate stays sound but goes **vacuous on exactly the thin-margin, dense-`G` tokens you most want to
compress** — `m > 2δ` is *false* there. That is not a flaw: it is the theory correctly refusing to certify
what it cannot certify without re-simulation. Quantified over a corpus the certificate is still a real
guarantee — "for all inputs with `margin > 2δ`, the model decodes identically" — but PO-T3 **makes the
LE-T2/T4 wall machine-checkable, it does not dissolve it** (PIC O2 / FINDINGS §5).

> **PO-T6 (Through-layers `δ` has no compact a-priori form — open, likely false in closed form; Grok).**
> A perturbation `Δ` at an earlier block reaches the logits through RMSNorm (directional gain `1/‖x‖`,
> anisotropic — amplifies low-norm directions), the softmax-attention Jacobian (`(I − softmax)` outer-product
> whose eigenvalues depend on the full Gram, so the local Lipschitz constant is input- and direction-dependent),
> and SwiGLU (piecewise-linear, state-dependent slope). The composite differential **inside the dense-`G`
> region has no low-treewidth factorization**, so any scalar `δ_b` that is sound for all inputs must essentially
> re-materialize the high-treewidth factor graph — **re-encountering LE-T4**. Hence the honest split: the
> certificate is **a-priori at the last layer / direct effects**, and **verified (re-decode), not certified,
> on deeper transforms**. The two ways forward are (a) keep `δ` local, or (b) accept verified deep transforms.

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
| PO-T3 | margin-certified decode invariance (`m > 2δ`) — sound **local** certificate | **Established locally**; **globally bounded by LE-T2** (vacuous on dense-`G`/forge-tax tokens) |
| PO-T2 | lossless demand residual = the dense forge tax (a 4th LO4 measure) | Measured-adjacent |
| PO-T6 | a compact a-priori through-layers `δ` would re-materialize the high-treewidth graph | Open; **likely false in closed form** (= LE-T2/T4) |
| PO-T4 | machine-checked (Coq/Lean) equivalence for a transform of `Π` | Open (formality frontier) |
| PO-T7 | certifiable-compressible fraction = a **grokking order parameter** (treewidth/PR/tropical rank = progress measures) | **Tested** (Pythia-70m, 28 ckpts): cert fraction rises then *saturates* (confidence-bound); **PR consolidates in TWO events — including a discrete late one (~step 70k) invisible to accuracy/margin/cert** — the dissociation is the certificate's boundedness, empirically |

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
  **Validated on a REAL small Llama** (SmolLM-135M, `lo3a/run_smollm.py`): `fieldrun convert` → certified
  FFN reduce → HF `safetensors` (`LlamaForCausalLM`, publishable) → `fieldrun convert` → bundle′, the round
  trip **bit-identical (Δ=0 weights, 18/18 decode)**. Two findings carry the thesis: (i) the *whole-model
  Soufflé emit refuses* at `vocab×d = 28M` facts — the LE-T4 wall in practice; (ii) a *trained* dense FFN has
  **≈0 exactly-dead neurons**, so the losslessly-removable set is ≈0 and zero-shot pruning trades decode
  fidelity (15/18 at 1–2% smaller, 12/18 at 4–6%). That is **PO-T2 measured on a real model**: the dense
  computed fragment does not compress losslessly — the forge tax is real, and the certifier names exactly where.
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
- **PO6 — grokking ⇒ certifiable-compressibility (the most interesting open direction; Grok).** Map: the
  **retrievable** fragment (induction = recursive clause, n-gram = fact — low treewidth, compact provenance) is
  the **grokked, generalizing circuit**; the **forge tax** (dense `G`, high PR, high tropical rank, no compact
  extension) is the **un-grokked, input-specific memorization**. `μ_t` ("deeper cells recruit more redundancy")
  fits: deeper blocks carry more forge-tax mass until grokking consolidates them into lower-treewidth circuits.
  So the **certifiable-compressible fraction** (tokens/blocks on which `certified` holds for a fixed `δ` schedule)
  is a **grokking order parameter**, and treewidth / PR / tropical rank are mechanistic **progress measures** for
  memorization→generalization. Predictions (status): (i) at the grokking transition the margin on algorithmic
  tokens grows (cleaner power-diagram facets) ⇒ `certified` fires more ⇒ certified-removable fraction rises
  (*falsifiable*); (ii) the two axes conflict — grokking **shrinks** forge tax along a run, but it **grows** with
  model size at convergence (the Pythia-ladder result) ⇒ net certifiable-compressibility vs scale is *open*,
  likely task-dependent (higher on reasoning, lower on factual recall) and non-monotonic; (iii) grokking-proper
  (sharp transition on modular arithmetic) is an **analogy**, not identity, for NL LLMs — distinguisher: NL shows
  gradual, overlapping, redundant circuit formation, not one clean transition. **Decisive experiments:**
  (a) plot certified-removable fraction + provenance treewidth/PR across a grokking run / Pythia checkpoints vs
  loss & generalization — a rise coinciding with circuit consolidation confirms "quantize the grokked retrievable
  circuits, protect the un-grokked forge tax"; flat-despite-grokking kills it (propagation gap too severe);
  (b) per-token `margin` vs measured quant-flip rate (strong −corr ⇒ use the certificate as a gate);
  (c) `D_b`-driven + margin-gated bit allocation vs pure-`D_b` and pure-KL baselines, scored on certified
  decode-preservation *and* KL/ppl. **Single cleanest falsifier (Grok):** if on real-scale post-grokking
  checkpoints `certified` stays ≈0 even for low-`D_b` blocks while moderate per-block quantization still passes
  re-decode, the a-priori certificate is too loose and PO4 collapses to "verify by re-running the forward" (LE-T4)
  without enlarging the compressible surface.

  **PO6 result — RUN on Pythia-70m (`lo3a/pythia_grok.py`, `--probe-margin`, 21 checkpoints step0→143000 on the
  real FINDINGS holdout).** Tracking the four order parameters across training (plot `lo3a/pythia_grok.png`):
  - **Certifiable-compressible fraction `P(m>2δ)` rises 0 → ~37% over steps ~8–2000, then *saturates*** — it
    tracks accuracy (0 → ~43%) and **plateaus with it**. So Grok's prediction (i) holds *in the growth phase*
    but the certificate's reach is **confidence-bound**: it stops rising once the model is confident.
  - **PR (DLA participation ratio — the LO4 concentration/treewidth proxy) shows the grokking shape and the
    decisive dissociation:** it *rises* 48 → 56 (steps 8–64: diffuse circuit engagement, the "build" phase),
    then *monotonically consolidates* 56 → 20 → **13** — and **keeps dropping long after accuracy / margin /
    cert have plateaued** (PR 23 at step 2k → 13 at step 143k, while acc/cert are flat). Genuine **ongoing
    circuit consolidation that the margin certificate does not see.**
  - **Reading:** the margin certificate is a *confidence* signal (saturates); PR is the *structure* signal
    (keeps consolidating) — the **temporal face of the LE-T2 boundedness Grok predicted**: the certifiable
    surface saturates while the dense circuit structure keeps evolving. Net: PO-T7's cert fraction is a valid
    order parameter *for the learning phase*; **PR / treewidth is the better grokking progress measure**, and
    the dissociation is itself the result — you cannot read consolidation off the certificate alone.
  - **Densified late tail (28 checkpoints) sharpens it into a DISCRETE second consolidation.** PR is not a slow
    drift: it holds a **plateau at ~20 (steps 6k–64k)**, then drops sharply (64k: 20.8 → 80k: 14.8 → 96k: 12.1)
    to a **second stable plateau at ~12.5 (steps 96k–143k, 7 checkpoints — confirmed, not a single point)**.
    Through that entire second consolidation, **accuracy (~45%), margin (~1.05) and cert (~36%) are flat** — a
    real structural reorganization with *zero footprint* in any confidence/certificate metric. So NL training
    here is *not* one gradual cleanup but **at least two consolidation events** (the early 56→23 during the
    learning ramp, and a late ~20→12 long after the loss plateaus), the second invisible to the certificate.
    *Remaining:* characterize *what* consolidates at step ~70k (which heads/circuits), and replicate up the
    ladder (160m/410m) — same script, change the repo id.

---

## 7. The two-knob linear policy — the practical recommendation (PR-core mode)

The engineering output of the LO1 investigation (token/circuit/decision/scale/polynomial/spectral axes).
**One sentence:** instead of a single fixed rank, keep **two operating points on the same readout-aligned
linear basis**, chosen by goal — a small scale-stable core for the bulk, a modestly larger rank for high
coverage. Both are pure linear context-free projections, so they compose with the Datalog export,
polygram/sae-forge dictionaries, q-orca/MPS encodings, and the margin certificate.

| knob | rank | size (135M) | scaling | use |
|---|---|---|---|---|
| **1 — PR / hard-rank core** | `≈ PR` (energy concentration) | ~18–90 | **flat** with scale | default/fast path, the compact datalog core, small circuits — baseline ~60–70% of decodes |
| **2 — span90 / soft-rank coverage** | `≈ span90` (decode-faithful) | ~65–100 | **sublinear** (65→96) | high-fidelity / critical / final decode — closer to the measured peak |

**The spectral reason two knobs are needed.** The decision-direction spectrum is asymmetric (Spectral
Scaling Laws regime): a concentrated head (`hard_rank`/PR ≈ 18 at 135M) plus a heavy power-law tail
(`α ≈ 0.97`), so `soft_rank ≈ 58 ≫ hard_rank`. Widening inflates the **tail**, not the head — so the
energy core (PR) stays flat while the rank for high cumulative coverage (`span90`) grows. A single fixed-PR
core therefore gives **great compression but eroding fidelity** with scale (measured: `d/PR` 6×→11×→**22×**
at 1.7B, while fixed-rank peak preservation falls **74%→58%** and `span90` grows 65→96). Two knobs respond
to the spreading without jumping to full `d`.

**Operational flow.** (1) Project the residual onto the readout-aligned basis (fixed linear step).
(2) Default = top-`PR` components (the compact PR-core). (3) Gate on coverage — **the margin is the gate**
(`m > 2δ` = PO-T3 ≡ this policy's router, for free at decode time). (4) If thin-margin, promote that
decision to the top-`span90` components — still linear, still `≪ d`. (5) Log the spectral triple
(`hard_rank`, `soft_rank`, `α`) to set/adapt the knobs per model.

**Why linear, for now.** The Volterra/polynomial probe on the PR core was **flat** (degree 1/2/3 ≈ 68/68/64%
vs 65% linear) — low-order interactions don't reach the `α≈1` heavy tail. A non-linear series re-opens only
if a **decode-targeted trained head** (not L2 reconstruction; torch-gated) recovers tail mass. Until then the
two-knob *linear* policy is the cheapest, most robust, verification- and MPS-compatible lever.

**Serves the size goal directly.** Default at the small stable PR-core ⇒ the `d/PR` win on the dense
embed/unembed (LE-T4) fragment, *growing* with scale; pay the sublinear `span90` cost only when coverage
demands it; the spectral triple is the offline/runtime diagnostic. *Status: evidence-backed engineering
recommendation, validated within the fixed-linear class (Grok, continuing the LO1 collaboration); the
ladder spectral triple and a decode-targeted head are the two confirming experiments.*

---

## 8. Related work

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

## 9. Acknowledgment & provenance

The optimization-theory category of a four-category theory with [PIC](./PIC_PROPOSAL.md) (logic),
[Tropical](./TROPICAL_PROPOSAL.md) (geometry), and [LOGIC_EXPORT](./LOGIC_EXPORT.md) (computation). It adds
nothing to the object — it *uses* it: LO3a makes the model a Datalog program, Tropical's `T=0` margin
certifies the approximate rewrites, LE-T5 makes the certificate sound, and LE-T2/LE-T4 is exactly the
residual the lossless transforms cannot remove. Every empirical claim traces to a probe — the LO3a emit
(`lo3a/`), the `--magic-transform` lossless check, and the existing `--pruned-head` / `--probe-quant` speed
measurements in [`FINDINGS.md`](./FINDINGS.md) §5 — the same theory–experiment loop.

The PO4 status (sound *local* certificate, *globally bounded by LE-T2*; PO-T6's through-layers `δ`
likely LE-T2 again; the grokking order parameter PO-T7) is an **adversarial review contributed by Grok**
— continuing the collaboration behind the Tropical power-diagram / facet-distance margin and the
incoherence-regime / ρ-boundary derivations (FINDINGS §4). The verdict it sharpens: *PO4 is honest
engineering that makes the LE-T2 limitation machine-checkable — most valuable when you already live in the
Datalog/provenance world and need per-input `T=0` guarantees rather than aggregate `T=1` KL — and it does
not solve the propagation gap.*
