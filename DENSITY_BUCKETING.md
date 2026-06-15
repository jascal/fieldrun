# Density-minimization bucketing

Applying the **Density-Minimization** result from `i-orca/examples/complexity`
(`Density_Minimization.thy` + `Density.thy`, kernel-checked) to fieldrun's measured
DLA surface. The goal across three granularities — **per token → per query → per
corpus** — is to decompose each decision into its irreducible deciding core, then
(longer term) reuse those cores as smaller MoE experts.

All three rungs ship here (per-token `--probe-decompose`, per-query
`--query-decompose`, per-corpus `--corpus-decompose`), plus the faithful Datalog
decode, interpretability, residency, and recursive sub-bucketing below.

## Quick example (end to end)

```bash
B="--bundle bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct --ids holdout.json"

# per-token irreducible atoms (σ(t)) + Route-B necessity confirmation, by route
fieldrun $B --probe-decompose --n-eval 200

# per-corpus experts: interpretability + runtime residency + the faithful Datalog decode
fieldrun $B --corpus-decompose --experts 8 --n-eval 300 \
        --interpret --residency --experts-dl-contrib /tmp/model.dl
fieldrun eval /tmp/model.dl --semiring max     # → decode == the model's tokens (faithful)

# resolve the residual into the long tail of domain experts
fieldrun $B --corpus-decompose --experts 6 --recurse-depth 3 --interpret --n-eval 300
```

Expected (Qwen2.5-0.5B): `--probe-decompose` → σ(t) 1.0/1.7/6.5 and necessity
27/44/82% across RETRIEVED/SELECTED/COMPOSED; `--experts-dl-contrib` → `fieldrun
eval` decodes 100% (faithful); `--interpret` → grammar-role experts (e3=verbs, …).

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
- [x] Datalog export of the partition (`--experts-dl`): routing/selection as a
      runnable Soufflé program + per-expert pick-entropy (lookup-exact vs computed).
- [x] Incremental bucketing in serve/REPL (`--bucket`: per-reply atom ingest +
      running clustering; `/bucket on|off|experts N|k N|reset|dump`).
- [x] Per-expert interpretability (`--interpret`): the decoded tokens routed to
      each expert reveal **grammatical-role specialization** (see below).
- [x] contrib-over-expert Datalog (`--experts-dl-contrib`): faithful composition
      decode + catchall `rest`, runs in `fieldrun eval` — replaces the bigram lookup.
- [x] Runtime residency profile (`--residency`): hot resident core vs paged long tail.
- [x] Multilingual / multi-domain: grammar-role experts recover across EN/DE/code;
      pooled clustering gives shared universal-structural experts + domain-specific
      experts (German function words, code syntax) + content residual.
- [x] Recursive residual sub-bucketing (`--recurse-depth D`): resolve the collapsed
      tail into hierarchical domain experts (`e3`, `r.e1`, `r.r.e0`) — toward the 10k tail.
- [ ] Realize the partition as a runtime MoE: experts = pageable weight modules
      (fieldrun expert-offload), loaded on correlated work. Compactness is a
      RUNTIME property (resident working set), not `.dl` size; the catchall `rest`
      is the always-resident shared core.

## Interpretability (`--interpret`)

`--interpret` decodes the tokens routed to each expert (its "specialty"). Measured
(Qwen2.5-0.5B, 300 tokens, E=8): the partition recovers **closed-class grammatical
roles** — e0 = punctuation + sentence-initial pronouns, e1 = auxiliaries, e2 =
determiners/adjectives, **e3 = verbs**, e5/e7 = `"And"`/future conjunctions — while
**content words fall into the residual catchall** (the largest bucket, and the
highest per-expert decision entropy). This echoes the open- vs closed-class lexis
split in the LO findings: the closed/structural lexicon buckets into legible
experts; the open lexicon is the computed residue (the forge tax). The `.dl`
`obs(pos,sig,route,pred,split)` facts also let you query "what routes to e3"
directly.

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

## Composition decode (`--experts-dl-contrib`) — the faithful model

The bigram lookup above is a *retrievable-floor baseline*; it discards composition
(the decode is `argmax_v ⟨r,U_v⟩ = argmax_v Σ_j c_j^v`, an additive sum, not a
token→token table). `--experts-dl-contrib` emits the **composition** instead: a
step-indexed program where, per decision, each scored circuit's contribution to the
candidate tokens (`c_j^t = dla`, `c_j^v = dla − margins[v]`, recovered from the
descent substrate) is summed **per corpus-expert**, plus a catchall `contrib("rest",…)`
so `Σ == logit`. It runs in the existing `fieldrun eval` (`--semiring max` → argmax
decode; `log` → softmax) and is **faithful by construction** — verified 12/12
(100%) decode-matches the model. The header reports the **per-expert share of the
winning margin** — the compactness meter: measured (Qwen2.5-0.5B), the verb expert
e3 carries +68% of the margin, the catchall `rest` +36% (the non-compact forge-tax
remainder). This is the runtime-MoE blueprint: route → load that expert, sum its
contribution; the `rest` is the always-resident shared core.
