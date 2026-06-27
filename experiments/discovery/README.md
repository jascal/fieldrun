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

## DLA-profile signature — idioms beyond binding (`discover_dla.py`, `--dla-dump`)
The recursion signature lives in the binding/fold subspace. The **DLA profile** clusters each decision by its per-block
contribution to the predicted logit (`fieldrun --dla-dump`, via `decomp_all`/`residual_decomp`; gemma PLE blocks dropped
as gemma-specific structural). Signature = `conc` (peakedness / PIC support number), `embed/attn/mlp_frac` (block KIND),
`early/late_frac` (depth), `neg_frac` (suppression share).

**What it found (gemma-4-e4b-it, 24 prompts, 350 decisions, k=6) — a NEW axis the recursion signature lacked:**
the **attn-vs-MLP circuit split + suppression structure**, yielding circuit-level idioms —
- **copy/induction** = attn-heavy with a `L41.attn(+) / L41.ffn(−)` **push-pull** (`na`, `orp`, `cherry`);
- **MLP content-recall** = MLP-dominated, **high suppression** (neg ≈ 0.3–0.6) — the last MLP down-weights the winner
  (`six`, `raining`, `story`);
- **MLP format/function** = MLP-dominated, **low suppression** (neg ≈ 0.1), `L40.ffn`-driven (`.` `)` `,` `the` `a`).
The critic also (correctly) isolated an **artifact** cluster: every `pos:0` decision is identical (`L41.ffn −24.99 →`
",") — a first-token degeneracy, not an idiom.

**The honest limitation — direct logit attribution is depth-degenerate.** `early_frac ≈ 0`, `late_frac ≈ 0.99`,
`embed_frac = 0` for *every* cluster: ~all DIRECT contribution to the final logit comes from the last 2–3 layers (the
unembed-proximity / logit-lens effect — earlier writes are read out *through* later layers, so their direct projection
is tiny). So the DLA profile **cannot find early/mid circuit idioms**; the depth axis is dead. The recursion signature's
`resolve_layer` captured depth far better. **The two signatures are complementary: recursion = *when* (timing/depth),
DLA = *which block-kind* (attn vs MLP + suppression).** Finding early/mid circuits needs a different attribution
(activation/path patching), not direct-logit DLA.

## Causal signature — cracking the dead depth axis (`discover_causal.py`, `--causal-dump`)
DLA can't see early/mid circuits (direct logit attribution is late-biased). The **causal** signature asks the
counterfactual: ablate each layer's attn/mlp block at the decision; which **flip** the prediction? A load-bearing early
block shows up regardless of its logit share. Signature: `flip_frac`, `earliest` (shallowest flipping layer / n_layer),
`early/mid/late_frac` (where the critical blocks sit), `attn/mlp_frac`. (Rope family; `predict_ablated_blocks`.)

**Headline (Qwen2.5-0.5B, 24 last-position decisions) — causal is the *near-opposite* of DLA on depth:**
- **23/24** decisions have an **early-third** load-bearing block; **median earliest critical layer = L0.** DLA said
  `early_frac = 0`; causally the early layers are critical almost everywhere.
- Only **1/24** fully redundant (no single block flips).

**The decompilation insight:** *where the logit comes from ≠ where the computation that determines it happens.* Late
layers **read out** (high logit contribution, but ablating one rarely flips — redundant); early layers **compute** (low
logit contribution, but ablating flips — critical). **Attribution is late-biased, causal is early-biased — faithful
decompilation needs both.**

**Idioms by criticality-depth + redundancy (a decomposition neither prior signature gave):**
- **early-bottleneck / redundant-readout** (n=10): a few *early* blocks critical, deep stack redundant — copy, format,
  and the **is-a recursion** (the chain is resolved in L0–L2; the answer is "decided early").
- **early-attention-critical** (n=4): early *attention* is the bottleneck (`na`, `wet`, `September`).
- **distributed multi-depth** (n=4): critical blocks across the stack — deep recall/counting (`Paris`, `eleven`).
- **fragile / broadly-critical** (n=6): many blocks each necessary (n_flip up to 32/48) — weakly-held, full-context
  predictions (`purple`, `twinkle…`); low redundancy = the causal "forge tax" (no clean circuit).

**Honest caveats:** single-block ablation finds *bottlenecks*, not distributed computation (collective early circuits
with no single critical block read as redundant); 0.5B, last-position only, n=24 (thin for k=4, but the 23/24 headline is
robust).

## The signature triad
Three complementary lenses on one decision: **recursion = *when*** (resolve-timing / fold-depth) · **DLA = *which
block-kind*** (attn vs MLP + suppression) · **causal = *where the load-bearing computation is*** (early vs late,
bottleneck vs redundant). Each is blind where another sees: DLA misses early (late-biased), causal misses distributed
(bottleneck-only), recursion misses circuit-kind. Fused, they're a far fuller decompiler.

## Fused signature — re-clustering on WHEN × WHICH × WHERE (`fuse.py`)
All three collected on the SAME model (Qwen2.5-0.5B), aligned per prompt at the end-of-prompt decision (impurity:
recursion is the last *in-prompt* decision, DLA/causal the continuation — an off-by-one proxy). Fused vector = recursion
{resolve, reach, copy, churn} + DLA {conc, mlp_frac, neg} + causal {flip_frac, earliest, early_frac}. k=4 over 24 prompts:

- **induction / copy** (n=5): `copy=1` · MLP, low-suppression · **early-bottleneck** (flip 0.08, early 0.88). `na→na`,
  `banana→cherry`, is-a `→w(ug)`.
- **early-decided sequence/format** (n=7): no copy, **earliest resolve** (0.78) · MLP · early-bottleneck, low flip.
  counting/`=`/`August→September`.
- **fragile / distributed** (n=8): **high flip_frac (0.38)**, critical blocks spread across depth. factual recall
  (`Paris`), in-context maps (`bird`), `eleven`, `twinkle…`.
- **deliberated-local** (n=4): **highest churn (0.75)**, latest resolve, local reach. narrative/`Britain→and`.

**The payoff — fusion adds resolving power: each lens splits what the others conflate.**
- The **causal** lens splits *fragile* (cluster 2, flip 0.38) from *robust-early* (clusters 0/3, flip ~0.05) — and those
  look **identical** to DLA (all conc≈0.2, mlp≈0.7) and similar to recursion. Causal does the discriminating.
- The **recursion** lens splits *copy* (cluster 1) and *high-churn deliberated* (cluster 3) — which the causal lens
  conflates (both early-critical, low flip).
So the clusters are crisper than any single signature gives; the three lenses contribute **orthogonal** discriminations.
(Also confirmed: early-criticality is near-universal — `earliest≈0`, `early≈0.5–1.0` everywhere — so the discriminator
*within* "early matters" is `flip_frac`: how many blocks / how fragile.)

Caveats: n=24 (thin for k=4 / 10-D), the off-by-one alignment, 0.5B.

## Cross-architecture — is the taxonomy rope-specific? (0.5B / 7B / gemma-4)
Ran the full 3-way on **gemma-4-e4b-it** (Gemma arch, instruct) — the first non-rope full fusion (required porting
`predict_ablated_blocks` to gemma4). **Cost finding:** gemma causal is ~7 min/prompt (84 blocks × ~5 s forward); the 3
*longer* prompts timed out at 650 s — causal scales as `n_layer × forward-cost`, so it's borderline on deep models (this
is *why* causal is the practically-rope-restricted lens). The 3 short prompts completed and map to the same taxonomy:
is-a = attn-heavy + most-critical (real recursion); factual = MLP-recall + robust + high-suppression; code = near-redundant.

**Cross-arch causal `flip_frac` (the fragile ↔ redundant axis):**
| prompt | 0.5B | 7B | gemma-4 |
|---|---|---|---|
| factual recall (`…France is`) | 0.167 | 0.071 | **0.048** |
| is-a chain (`…zorp is a`) | 0.083 | 0.036 | **0.131** |
| code/format (`…return a +`) | 0.021 | 0.000 | 0.012 |

**Findings:**
1. **The lens structure is architecture-general.** gemma-4 shows the *same* behavior — early-critical causal
   (`earliest≈0`), MLP-dominated DLA readout, suppression on recall (neg 0.31 for `Paris`), copy/resolve recursion. The
   when × which × where decomposition is **not** rope-specific.
2. **Redundancy tracks CAPABILITY, not raw layer count.** Factual-recall robustness rises 0.5B → 7B → gemma-4
   (flip 0.167 → 0.071 → 0.048); gemma-4 (~4B, strong) is **robust like 7B, not fragile like 0.5B**.
3. **Causal corroborates behavior on is-a.** gemma commits **more** critical blocks to the chain (0.131) than 7B (0.036)
   — causal evidence that **gemma genuinely recurses while 7B base recency-shortcuts**, independently matching the GAPS
   behavioral probe (gemma cleared ≥4-hop is-a; 7B base did not). More load-bearing blocks = real computation; few = shortcut.

## Next
multi-block / path-patching for distributed early circuits (single-block ablation's blind spot) · cheaper causal for
deep models (layer-group ablation) · auto-route residual exemplars into the probe harness (`PROBES.md`).
