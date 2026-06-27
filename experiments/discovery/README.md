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

## Next
richer signature (DLA circuit profile) → idioms beyond binding · bigger/cleaner model · k-sweep + stability ·
auto-route residual exemplars into the probe harness (`PROBES.md`) so discovery and characterization compose.
