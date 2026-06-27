# idiom discovery — finding *unnamed* computational dances (prototype)

The targeted probes (is-a, modal, statistical, planning…) only find idioms we **already thought to look for** — the
streetlight problem. This is the inverse: surface candidate idioms **unsupervised**, from the model's own disassembly,
and let a critic flag what no idiom yet explains. "Discover, don't checklist."

## The loop
1. **collect** — `./collect.sh` runs `fieldrun --recursion-explain --recursion-dump` over a diverse corpus
   (`corpus.txt`), writing one per-decision signature per line to `dumps/NNN.jsonl`.
2. **cluster** — `discover.py` builds a **content-agnostic mechanism signature** per decision and clusters in that
   space. Clusters = candidate idioms (emergent, not pre-named).
3. **critic** — a RESIDUAL test flags decisions no cluster explains tightly = the frontier of unnamed idioms.
4. **close the loop** — a frontier model (the agent) reads the cluster exemplars, **names** the dense idioms, and takes
   the **residual** exemplars as the next round's probe targets.

Signature (all content-agnostic, from the faithful forward pass):
`resolve_frac` (compute depth) · `reach_norm` (fold distance) · `conc` (binding strength) · `copy` (induction flat-copy) ·
`lens_churn` (value-stack instability = "thinking").

## Run
```bash
./collect.sh                                   # default Qwen2.5-0.5B (fast); pass a bundle stem for a bigger model
python3 discover.py [dumps_dir] [k]            # cluster + residual critic
```

## v0 result (Qwen2.5-0.5B, 12 prompts, 129 decisions, k=5)
Unsupervised, the loop recovered ~5 interpretable idioms — **finer than the RETRIEVED/SELECTED/COMPOSED route**:
- **local continuation / no-fold** (reach≈0, conc≈0) — immediate-context next token;
- **induction copy** (copy=1, high conc) — the classic in-context copy;
- **early-stable recall** (resolve early, low churn) — quick rule retrieval (counting `1 2 3→4`, syntax `(a,b →):`);
- **deferred local compute** (late resolve, low reach, high churn) — value computed late but bound locally;
- **long-range computed bind** (late resolve, high reach, not copy) — folds to a distant frame and transforms it.

The **residual** (completeness 95%) isolated, unprompted: the **deep subject-retrieval recursion fold** (`reach≈0.9`,
the is-a-chain answer point) and **bracket-matching** in the arithmetic — genuine distinct dances flagged as next targets.

## Honest scope
Prototype. **0.5B + noisy logit-lens + 129 decisions** — small. The signature is **recursion/binding-flavored**, so it
discovers idioms in that subspace; the path to *non-binding* idioms is a richer signature (per-decision **DLA
block/circuit profile** via `residual_decomp`). `k` is a fixed knob (not swept). "Completeness 95%" is *w.r.t. this
signature and k* — it does **not** mean 95% of the model's computation is understood. Some residual is tokenization
noise, not idioms — the critic must filter.

## Residual-driven enrichment — the loop closing on itself (7B)
The 7B residual flagged an **early-resolution** idiom (decisions committed in the first few layers, `resolve_frac` ≈
0.1–0.4) too rare to cluster. The loop's response: add a corpus *targeting* it and re-cluster —
`./collect.sh <bundle> corpus_enrich.txt 100` (append; formulaic / forced-continuation text). Re-clustering the
combined 24 prompts (257 decisions) **falsified the hypothesis and refined the idiom** — which is the loop working as
science, not just clustering:

- **Falsified:** "predictable ⇒ early resolution" is **wrong**. Predictable *content* (`…quick brown fox jumps over the
  → lazy`, `…four five → six`, `red orange yellow green → blue`) still resolves **late** (`rf` ≈ 0.9). Predictability
  did **not** move resolution earlier.
- **Refined — predictability shows up as low *churn*, not early resolve.** The enrichment populated a clean **new**
  cluster: *settled continuation* (n=30, `churn` ≈ 0.25 vs the deliberated-compute cluster's ≈ 0.65) — predictable
  sequences lock in with little layer-to-layer revision, but still over the full stack.
- **Early-resolution is a function/format-token reflex, and genuinely sparse.** The true `rf` ≈ 0.04–0.4 cases are
  syntactic-glue tokens (`…at the end → of` rf 0.04; `…bird → =` rf 0.11; `…not → to`; `…bread → and`) — predicted in
  the earliest layers. They stayed a 2-point cluster + residual: sparse *and* spread across `reach`, so **k-means is the
  wrong grouper** — a 1-D threshold on `resolve_frac` isolates them better. (Methodological note for the loop: rare
  residual idioms need targeted isolation, not just more `k`.)

Net: the residual → targeted-corpus → re-cluster loop is a **falsifiable hypothesis test** that self-corrected
(predictability = settled/low-churn, *not* early-resolve; early-resolve = a separate glue-reflex).

## Sparse-idiom isolation — naming the rare dances (axis pass)
k-means forces every decision into a ball, so an idiom dense along ONE dimension but spread across others lands in the
residual (the early-resolution glue-reflex: low `resolve_frac`, but spread in `reach`). `discover.py` now runs an **axis
pass** after clustering — a gap-based 1-D tail detector that flags tails whose members were mostly k-means residual as
**newly named**:
- `resolve_frac < 0.46` → **early-resolution glue-reflex**, NEWLY NAMED (4/6 were k-means residual): `…at the end → of`
  (rf 0.04), `…bird → =` (rf 0.11), `…not → to`, `…bread → and` — syntactic-glue tokens emitted in the first layers.
- `conc < 0.22`, `copy > 0.50` → correctly tagged *"overlaps a cluster"* (no-fold, induction-copy already named). The
  1-D `copy` axis even re-unifies the atypical copies k-means split off.

So the loop now has **three tiers**: k-means (dense, populated idioms) · axis isolation (sparse 1-D-separable idioms) ·
residual (genuine multi-D frontier outliers, e.g. the deep subject-retrieval fold — high reach *and* conc *and* context,
not 1-D-separable, correctly still flagged).

## Next
richer signature (DLA circuit profile) → idioms beyond binding · auto-route residual exemplars into the probe harness
(`PROBES.md`) so discovery and characterization compose.
