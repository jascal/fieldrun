# Density-minimization bucketing

Applying the **Density-Minimization** result from `i-orca/examples/complexity`
(`Density_Minimization.thy` + `Density.thy`, kernel-checked) to fieldrun's measured
DLA surface. The goal across three granularities — **per token → per query → per
corpus** — is to decompose each decision into its irreducible deciding core, then
(longer term) reuse those cores as smaller MoE experts.

This branch lands **Phase 1**: the per-token descent as an analysis probe
(`--probe-decompose`).

## The theorem (what we are applying)

For a token `t` decided by a source set `S` against competitors `V`, deciding is
positivity of the per-competitor margin sums (the `decides_via_margin`
reformulation):

> `decides c P V t  ⟺  (∀ v ∈ V, v ≠ t)  Σ_{j∈P} m_j^v > 0`,   `m_j^v = c_j(t) − c_j(v)`.

`decomposes c t V S A` (the algorithm, as an inductive relation) repeatedly
replaces a **reducible** deciding coalition by a strictly smaller deciding
sub-coalition, bottoming out at an irreducible **atom** `A` — *you never split an
irreducible coalition*. Kernel-checked about the result:

- `decomposes_subset` — `A ⊆ S`
- `decomposes_atom` — `A` is irreducible
- `decomposes_decides` — `A` still decides `t`
- `decomposes_firing_non_increasing` — the **total firing COUNT** is non-increasing
  along the descent (`Density.total_firing_mono`).

**Load-bearing subtlety** (`Density.thy`): the density *ratio* `firing / |P|` is
**not** monotone under subsets, so "minimize density" can only mean minimize the
firing **count** (or density at fixed size) — never the fraction. Irreducible
atoms are the **floor**, and a smaller deciding sub-coalition exists iff the
current one is reducible. This is the formal version of fieldrun's `τ*`/`Δ_repr`
floor and the participation-ratio support number (PIC O2).

The executable face is `MinimalDecider.minimal_decider` — a greedy that drops one
removable source at a time to a locally-minimal deciding subset. It is a **sound
poly UNDER-approximation** of the true irreducible core, *not* global
irreducibility (`all_necessary_not_irreducible`: the `c4` token is locally minimal
yet reducible — a two-source removal still decides). We implement exactly this
greedy.

## How it maps onto the measured surface

`explain` already scores every candidate head/neuron with its DLA to the predicted
token (`dla = c_j^t`). We extend `assemble` (`src/explain.rs`) so that, when
`decomp_k > 0`, it also computes each source's margin `m_j^v = c_j^t − c_j^v`
against the **top-K competitors** (`unembed_row(v)` supplies each competitor row —
cheap K dot-products per source, no full-vocab projection). The result is a
`DecompSubstrate`:

- `competitors` — the top-K competitor token ids (predicted token excluded);
- `sources[j].margins[v]` — the margin matrix;
- `const_v` — everything **not** in the scored candidate set (embeddings, biases,
  the un-shortlisted tail), so the cone test is exact w.r.t. the linear DLA
  decomposition: `Σ_{all scored j} m_j^v + const_v == full_margin_v > 0`.

`decompose_descent` runs the greedy over this substrate (weakest-DLA-first; running
per-competitor margin sums, so each removal is O(K) and one pass reaches local
minimality — removals only shrink the sums, so an un-removable source stays
un-removable). It returns the atom, the positive-DLA mass retained, and the
per-competitor slack.

**Multi-competitor by design (Route A).** `single_competitor_reducible` says
irreducible ⟹ ≥ 2 competitors, so the descent uses the *cone* over `--decomp-k`
competitors (default 4), not just the runner-up.

**Honest caveat.** `const_v` lumps the un-shortlisted candidate tail into a fixed
offset, so the atom is minimal *among scored candidates* and `σ(t)` is a **lower
bound** on the true support number. The descent is also over the linear DLA model
(the ~75–83 % coalition-test regime, PIC D1). The faithful **confirmation** below
converts each atom into a causal claim against the real ablated forward.

## Confirmation (Route B, `predict_ablated`)

The descent is a *linear* construction; `--probe-decompose` confirms each atom in
the real nonlinear forward (2 extra forwards/token; `--no-confirm` to skip). The
test is **necessity, not sufficiency**: ablate *only* the atom `A` (|A| ≪ |S|
circuits) and check whether the prediction flips. A flip ⇒ the irreducible core is
causally load-bearing — the §5c ablation methodology applied to the atom. (Why not
sufficiency? "Keep only `A`" means zeroing the other ~441 scored circuits, which is
so destructive that nothing survives — linear DLA ≠ causal ablation. So the
sufficiency/"keep-only" reading is *not* a valid faithful test; necessity is.) A
top-|A|-by-DLA control shows whether the cone descent picked a better core than
naive top-k.

```
route       σ(t)=|A|  necessary  ctrl flip
RETRIEVED      1.9       27%        33%
SELECTED       2.0       44%        44%
COMPOSED       4.8       82%        91%
```

**Necessity rises 27 % → 44 % → 82 % across RETRIEVED → SELECTED → COMPOSED**:
ablating the small irreducible atom flips composed tokens 82 % of the time but
retrieved tokens only 27 % (their decision is redundantly backed up elsewhere) —
the "composed is fragile, covered is redundant" result of §5c, now recovered from
the descent's atom. The atom's necessity tracks the top-|A|-by-DLA control closely
(the descent and DLA-ranking mostly agree on the load-bearing core).

## CLI

```bash
fieldrun --bundle <qwen> --ids <holdout.json> [--store <store.json>] \
         --probe-decompose [--decomp-k 4] [--n-eval N]
```

Per token: descend to the atom, report `σ(t) = |atom|`, the `|S| → |A|` reduction,
the positive-DLA mass retained, the participation ratio (for the `σ(t) ∼ PR`
comparison), and the atom's margin slack — bucketed by route when `--store` is
given. `rope`/Qwen exposes the substrate (`explain_decomp`); other arches inherit
the default (no substrate) and the probe reports "unsupported arch".

## Phase-1 result (Qwen2.5-0.5B-Instruct, K=4)

```
route             |S| src  σ(t)=|A|  reduction  retained  PR eff#  slack
RETRIEVED          448      1.0        100%        8%      38.2    0.97
SELECTED           448      1.7        100%       11%      39.0    0.73
COMPOSED           448      6.5         99%       21%      49.4    0.33
```

The support number tracks the retrieval→composition gradient: **RETRIEVED tokens
reduce to a ~single deciding source, COMPOSED tokens need a ~6.5-source irreducible
core** (the forge tax, now measured as a minimal deciding coalition rather than a
participation ratio), and composed atoms sit closest to flipping (slack 0.33 vs
0.97 — the "composed is fragile" finding). `corr(σ(t), PR) ≈ 0.35`: positively
related but `σ(t) ≪ PR` — the minimal irreducible core is far smaller than the
effective number of participating circuits (a measured, partial answer to PIC O2).

## Per-query aggregation (`--query-decompose`)

The ladder's middle rung: treat a contiguous run of positions as ONE query and
aggregate the per-token atoms into the query's circuit **working-set**
`W = ⋃_t A_t` — **entirely in-memory from the descent**, with *no*
`export --logic → .dl → stitch` disk round-trip (the existing per-step `.dl` emit
+ `fieldrun stitch` path writes N files and merges them; the atoms are already in
memory, so the union is direct and simpler). This is the **Hub.thy** decomposition
of a query:

- `Σ|A_t|` — total firings (the per-token floor summed; the monotone objective);
- `|W|` — distinct circuits (the query's budget);
- **reuse** `1 − |W|/Σ` — circuit sharing across the query's tokens;
- **hub** — circuits shared by ≥ `--hub-frac` of the tokens (the disentangling
  core / a candidate expert); **private** — single-token circuits;
- the **d-bounded budget** `|W| ≥ Σ|A_t| / d` (`d` = max reuse), i.e.
  `Hub.d_bounded_private_budget`.

Measured (Qwen2.5-0.5B, 50-token passage, K=4): `Σ|A_t| = 119` (avg atom
2.38/token) over `|W| = 73` distinct circuits — **39 % reuse**. The reusable core
is a small set of **late-layer neurons** (e.g. L23 #2539 in 10/50 atoms, L22 #1069
in 5/50; `d = 10`) — the block-sparse, circuit-dense, late-block decision surface
(§5d), now surfaced as the query's shared expert seed.

## Per-corpus expert clustering (`--corpus-decompose`)

The endgame: cluster the per-token atoms across the whole corpus into `E`
**experts** — again **in-memory** from the descent, no `.dl`. The corpus working
set `C = ⋃_t A_t` is partitioned into hub-anchored buckets: the top-`E` circuits by
corpus frequency are the expert **anchors** (the recurring hubs), and every other
circuit joins the anchor-expert it **co-fires with most** (co-fire = same atom);
circuits that never co-fire with a hub fall into a residual bucket. A token routes
to the expert(s) its atom touches. The MoE questions it answers:

- **span-1 routability** — does the deciding atom fit inside ONE expert (so top-1
  routing reproduces the decision)?
- **active circuits/token** — under routing a token computes only the experts its
  atom touches; how much smaller than the monolithic working set `|C|`?

Measured (Qwen2.5-0.5B, 200 tokens, K=4, `--experts 8`): `|C| = 256` distinct
circuits → 8 experts (+119 residual). **52 % of tokens are top-1 routable**, mean
1.76 experts/token, and **active circuits/token = 81 of 256 → 68 % fewer
computed**. The anchors are overwhelmingly **late-layer (L21–23) neurons**; the
dominant expert (anchor L23 #2539 — the same circuit that was the per-query hub)
routes 26 % of tokens. With `--experts 16` the per-token cost drops to 54/256
(79 % fewer) at a slightly lower top-1 rate (atoms split across finer experts) —
the expected `E` tradeoff.

**Scaling + the emitted partition.** The corpus streams in chunks (`CorpusBuckets`
in `bucketing.rs`), so a much bigger corpus is bounded by forward-pass time, not
memory (atoms are ~72 bytes/token). `--report-every N` prints the running
clustering at runtime; the final report lands at the end. `--experts-out <path>`
dumps the **concrete partition** as JSON — each expert's anchor + the full circuit
list (kind/layer/idx) + token routing, residual bucket last — i.e. the build
artifact a router / weight-chunk pager consumes, not just the summary stats.

**Proxy caveat.** The active-circuit figure is a *static, oracle-router*
circuit-count proxy: it assumes you already know each token's atom. A real
wall-clock saving needs (a) a learned router that predicts the expert from the
context (not the atom), and (b) the circuit-experts mapped onto pageable weight
chunks (fieldrun's expert-offload).

## The ladder

1. **per token** — `decompose_descent` over one decision (`--probe-decompose`).
2. **per query** — `--query-decompose`: `W = ⋃_t A_t` in-memory.
3. **per corpus** — `--corpus-decompose`: cluster atoms into `E` hub-anchored
   experts; route a token to its atom-bucket → a smaller/cheaper MoE. fieldrun
   already has the expert-offload machinery (`src/bundle.rs`,
   qwen3moe/gemma4-MoE/mla/minimax) for the routed experts to land in.

## Status

- [x] Per-token descent + multi-competitor substrate (`explain.rs`), unit-tested.
- [x] `--probe-decompose` harness; measured on Qwen2.5-0.5B.
- [x] Faithful necessity-confirmation of atoms (`predict_ablated`, Route B).
- [x] Per-query atom aggregation (`--query-decompose`, in-memory, no `.dl`).
- [x] Per-corpus atom clustering → hub-anchored expert buckets (`--corpus-decompose`).
- [x] Streaming over a bigger corpus + `--report-every` incremental runtime reports.
- [x] Concrete partition export (`--experts-out <path>`, JSON: expert→circuit sets).
- [ ] Incremental bucketing in serve/REPL (per-reply, `--bucket`).
- [x] Datalog export of the partition (`--experts-dl`): routing/selection as a
      runnable Soufflé program + per-expert pick-entropy (lookup-exact vs computed).
- [x] Incremental bucketing in serve/REPL (`--bucket`: per-reply atom ingest +
      running clustering; `/bucket on|off|experts N|k N|reset|dump`).
- [ ] contrib-over-expert Datalog (faithful composition decode + catchall `rest`),
      replacing the bigram lookup — the runtime-MoE blueprint.
- [ ] Per-expert interpretability: what tokens/contexts route to each expert.
- [ ] Realize the partition as a runtime MoE: experts = pageable weight modules
      (fieldrun expert-offload), loaded on correlated work.

## Datalog lookup/selection export (`--experts-dl`)

`--experts-dl <path>` emits the partition as a **Soufflé-compatible Datalog
lookup/selection model**: the expert partition is RELATIONS (`expert`/`anchor`),
routing `selected(sig,e)` and decision `predict(sig,tok)` are LOOKUP tables over a
context signature (the previous token id) compiled from the corpus, and rules apply
the lookup (`decode`) and check it reproduces the model's decode (`hit`). The header
reports per-expert decision entropy `H(pred|expert)` — ≈0 marks a **lookup-exact
(retrievable)** expert, >0 the **computed residue** (the forge tax). This is a
corpus-derived lookup model (generalizes by signature match), *not* the dense
forward-pass-as-Datalog (that is `logic_whole.rs` / LO3a — exact but non-compact);
the partition is the instrument that isolates the compactly-lookup-able fragment.

Validated end-to-end (Qwen2.5-0.5B): the emitted program **runs in Soufflé**
(`souffle <path>.dl -D-`) and `hit_train`/`hit_test` reproduce the in-Rust stats;
the residual bucket carries the highest `H(pred|expert)` (the computed core).

**In-sample vs held-out (don't trust the in-sample number).** The lookup is
optimistically in-sample on a small corpus: 120 tokens → 89 signatures → ~74 %
singletons, each trivially memorised. The in-sample accuracy **falls as the corpus
grows** and singletons stop dominating — measured: **80 % (150 tok) → 66 % (400) →
63 % (800)**, flattening near the true bigram-determinism. So `--experts-dl` now
does a **train/test split** (`--dl-test-frac`, default 0.2): the lookup is compiled
from the train head and scored on the held-out tail, and the header reports
IN-SAMPLE vs HELD-OUT `predict==decode` + coverage (fraction of test signatures
seen in train) + `hit_train`/`hit_test` relations. A richer signature
(`--dl-sig N` = last N context tokens) raises accuracy at the cost of a bigger
table (the compactness↔accuracy tradeoff). The held-out miss is the **computed
fragment** (the forge tax); the covered hits are the **retrievable** fragment.
