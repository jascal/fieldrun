# Rule-synth prototype — §8 results

End-to-end validation of `RULE_EXTRACTION_PROPOSAL.md` §8: a `(input, model-output)` dump (`fieldrun --recursion-explain --list-dump`) + an offline bottom-up synthesizer (`synth.py`) that fits the **model's output** (faithful, "wrong" allowed) over a typed list/numeric/selection DSL, with observational-equivalence pruning, MDL selection, a guarded decision-list pass (held-out-gated), and an output-type (0–9) constraint.

Run: `fieldrun … --list-dump <f> --n 120 --lmax 7` then `python synth.py <f> 4`. 6 list tasks, 120 lists each, 70/30 train/held-out. `faith` = **held-out** faithfulness to the model's output; `truth%` = how often the discovered program equals the textbook function (low ⇒ a faithful *broken* function); `resid` = `1 − g-faith`.

## Qwen2.5-1.5B (K=4)

| task | model-acc | discovered | faith(te) | truth% | residue |
|---|---|---|---|---|---|
| max  | 100% | `max(xs)` | **100%** | 100% | **0%** |
| min  | 98%  | `min(xs)` | 94% | 100% | 6% |
| last | 88%  | `last(xs)` | 92% | 100% | 8% |
| sum  | 90%  | `sum(xs)` | 86% | 100% | 14% |
| first| 76%  | `first(xs)` | 78% | 100% | 22% |
| len  | 51%  | `len(xs)` | 53% | 100% | 47% |

**mean residue 16%.**

## Qwen2.5-0.5B (K=4)

| task | model-acc | discovered | faith(te) | truth% | residue |
|---|---|---|---|---|---|
| first| 96% | `first(xs)` | 97% | 100% | 3% |
| last | 82% | `last(xs)` | 75% | 100% | 25% |
| max  | 61% | `max(xs)` | 61% | 100% | 39% |
| sum  | 66% | `sum(xs)` | 56% | 100% | 44% |
| min  | 55% | `min(init(init(xs)))` | 50→58% (guarded) | **67%** | 42% |
| len  | 38% | `len(take(xs,6))` | 58% | **80%** | 42% |

**mean residue 32%.** At K=5 the `len` site resolves to a heavily-broken obscure program `imax(4, len(tail(xs)))` → **83% faithful, 19% truth-match** (residue 42%→17%): a non-human-named function the model uses for "len".

## What this validates (against §8)

1. **Clean recovery** — where the model is competent, the synthesizer recovers the textbook function faithfully: 1.5B `max` → `max(xs)` at **100% / 0 residue**, `min` 94%, `last` 92%.
2. **Faithful broken/obscure discovery** — where the model is wrong, it recovers the model's *actual* (broken) function, not textbook: 0.5B `min` → `min(init(init(xs)))` (truth 67%), `len` → `len(take(xs,6))` / K=5 `imax(4,len(tail(xs)))` (truth 19%). Faithfulness, not correctness — the criterion the proposal insists on.
3. **Residue scales with competence (the forge-tax number).** Mean held-out residue **0.5B 32% → 1.5B 16%** — it roughly halves as the model implements cleaner functions. The residue *is* the measured "forge tax" for this DSL; a more capable model leaves less.
4. **DSL depth shrinks the residue** — K=4→5 drops `len`'s residue 42%→17% by reaching an obscure program. Bank grows 511→2162 (the observational-equivalence working set tracks distinct behaviours, not program count).

## OOD-length test (the discriminating one) — `python synth.py … --ood`

Train on lists of length ≤5, test on length ≥6. This separates a **real function** (generalises across length) from an **in-distribution coincidence** (matches on short lists but isn't the model's actual algorithm). Qwen2.5-1.5B:

| task | faith(random-split) | **faith(OOD)** | reading |
|---|---|---|---|
| max  | 100% | **100%** | real function |
| min  | 94%  | **96%**  | real function |
| first| 78%  | **96%**  | real function |
| last | 92%  | **79%**  | mostly real |
| **len**  | 53% | **26%** | breaks on long lists — *not* a clean function |
| **sum**  | 86% | **0%**  | **fakes sum on short lists; breaks entirely on long** |

**Key methodological finding:** the random held-out **underestimates the forge tax** — `sum` looks 86% faithful in-distribution but **0% OOD**, i.e. the model does not actually implement a sum *fold*; it pattern-matches short cases. Mean residue rises **16% (random) → 34% (OOD)** once the coincidences are stripped out. Observational match in-distribution is necessary but not sufficient (proposal §7); **OOD is the honest faithfulness signal**, and it should be the default residue/forge-tax number going forward.

## §6 — Datalog emission round-trip (the loop closes) — `python emit_datalog.py <dump> 4`

The discovered program → position-indexed recursive Soufflé (`elem(l,i,v)`/`len(l,n)` + a fold rule) + a `residue(l,o)` EDB for the lists the program gets wrong + a wrapper `answer = program unless residue, else residue`. Run with `souffle`; by construction `answer == model output` on every list. Qwen2.5-1.5B:

| task | discovered → Datalog | souffle reproduces model | residue (EDB) |
|---|---|---|---|
| max  | `max(xs)` → recursive `acc(L,I,M):-acc(L,I-1,M0),e(L,I,V),M=max(M0,V).` | **100%** | **0%** |
| min  | `min(xs)` → recursive min-fold | 100% | 2% |
| sum  | `sum(xs)` → recursive sum-fold | 100% | 10% |
| last | `last(xs)` | 100% | 12% |
| first| `first(xs)` | 100% | 24% |
| len  | `len(xs)` | 100% | 49% |

So the full pipeline runs end-to-end: **model I/O → faithful program (§2) → recursive Soufflé + residue EDB (§6) → runs in Soufflé and reproduces the model 100%**, with the residue fact-count = the per-task forge tax. `max` becomes a *pure rule* (0 EDB); `len` is mostly EDB (the model's "len" isn't a clean fold). (The §6 translator currently covers the fold + list-modifier ops; binary-int programs like `imax(4,len(tail(xs)))` are flagged `unsupported-op` and would route to residue.)

## Head-vs-tail: residue vs DSL depth (proposal §9) — `python synth.py <dump> <K>`

Regenerate (deterministic, default `--seed`): `for k in 2 3 4 5 6; do python synth.py <dump> $k | grep "mean residue"; done`.
Sweeping DSL depth K=2→6 (the program space grows from ~hundreds to thousands of distinct behaviours):

| mean residue | K2 | K3 | K4 | K5 | K6 |
|---|---|---|---|---|---|
| 0.5B | 33% | 33% | 32% | 30% | 30% |
| 1.5B | 16% | 16% | 16% | 16% | 16% |

**The residue is a floor, not a long tail of bigger programs.** Per-task, where the model does a *clean* function the residue is flat across K (1.5B max 0→0, min 6→6, last 8→8, sum 14→14, first 22→22) — the simple fold *is* the function, and the residue is the model's own **deviation/inconsistency**, which no deeper program can fix (`max` floors at 0 because the model is perfectly consistent; `first` floors at 22% because the model deviates that often). Where the model is *systematically broken*, deeper DSL **does** find more structure (0.5B `len` 36→22%, `min` 50→42%).

Reading for §9: the "head" (clean folds) is captured at K=2; growing the *same* DSL barely moves the residue. The forge-tax floor is **model-deviation outside this DSL family**, not a long tail of larger same-family programs — so cracking it needs a *different representation*, not bigger programs (consistent with the "novel encodings" intuition). And greedy decoding ⇒ the residue is *deterministic* (a consistent un-DSL-expressible answer), not sampling noise.

## Deterministic exhaustion — breadth too, not just depth (`synth.py --packs=base[,pos,hist]`)

The depth sweep (above) showed the floor is flat in K. The **breadth** sweep adds new operation *classes* — `pos` (`argmax`/`argmin`) and `hist` (`countmax`/`nuniq`/`maxcount`) — and asks whether a *different kind* of primitive cracks it. It does **not** (1.5B 10-task): mean residue base **30%** → +pos **30%** → +pos+hist **31%** (slightly worse). Per-task every floor is unchanged, and `mode`/`cmax` get *worse* (78→81, 33→39) — the extra primitives only add overfitting.

The killer detail: **`cmax` doesn't improve even though `--packs=hist` adds the exact `countmax` primitive** (and `mode` gets `maxcount`). It's not that the DSL lacks the right rule — **the model doesn't follow a clean rule there**, so the perfect primitive changes nothing.

**Scope (important — not over-claimed):** this exhausts the **list-fold DSL family, on these flat-list tasks**, along depth *and* breadth. It is **not** a general "deterministic rules are exhausted" claim. A whole deterministic class is still **untried: tree-traversal / catamorphism rules** (proposal §11) — structural recursion the flat-list DSL cannot express. The "different representation" the floor needs may well be **deterministic-structural (trees)**, not soft. So the order is: exhaust list-DSL (done, here) → **tree-traversal rules (next)** → only then a soft representation (PIC) for whatever survives *both*.

## Honest (OOD) forge tax on the full 10-task battery

| split | 0.5B | 1.5B |
|---|---|---|
| random 70/30 | 32% | 30% |
| **OOD-length** | **56%** | **43%** |

The harder tasks carry big in-distribution coincidences (OOD strips them), so the honest forge tax on the 10-task battery is **43–56%** — much higher than the random split suggests, and still scaling with competence (0.5B→1.5B). This is the floor that the **list-fold** DSL can't reduce (depth + breadth); the next deterministic class to try against it is **tree-traversal rules**, and only what survives that goes to a soft representation.

## Tree-traversal rules — the untried DETERMINISTIC class (proposal §11) — `tree_synth.py`

The list-floor needed a "different representation." The first thing to try is still **deterministic**: *tree catamorphisms*
(structural recursion over a parse tree), which the flat-list DSL provably cannot express. Dump: `fieldrun --tree-dump`
emits nested arithmetic exprs (depth 1–3, operands 0–9) + the model's answer for a battery of tree tasks. `tree_synth.py`
parses each expr into a binary tree and bottom-up-synthesizes the smallest **catamorphism** (`eval`/`maxleaf`/`depth`/…
over the tree, + subtree selectors `left`/`right` + int combinators) faithful to the model's output — same OE/MDL/held-out
machinery as the list synth. The contrast column is the best **flat-list** program on the *same* exprs' leaf sequence.

**`eval` is the discriminating task** (250 exprs, K=4). `eval` is zero-ICL — the model evaluates `(+ 3 (* 2 5))` natively
— and its value depends on the *operators and structure*, not just the leaf sequence, so the flat-list DSL cannot express
it. The headline is a **scaling/emergence** result: the tree-vs-list gap *diagnoses whether the model genuinely recurses*.

| model | split | discovered | faith | truth | **flat-list faith** | gap | reading |
|---|---|---|---|---|---|---|---|
| 1.5B | random 70/30 | `eval(t)` | **69%** | 100% | **23%** | **+46pp** | genuine tree recursion |
| 1.5B | OOD-depth (≤2/≥3) | `eval(t)` | 41% | 100% | **3%** | **+38pp** | recursion + depth cliff |
| 0.5B | random 70/30 | `5` (const) | 73% | 8% | 80% | **−7pp** | **not recursing** — constant 5 |
| 0.5B | OOD-depth | `imin(5,sumleaf)` | 99% | 14% | 99% | 0pp | the same constant, disguised |

- **The tree-vs-list gap diagnoses genuine recursion.** At **1.5B** the model truly evaluates (model-acc 72%; outputs
  spread across the range and even exceed 9 — values like 12/16/17 that *only* a real computation can produce). The tree
  catamorphism `eval(t)` (truth 100%) captures it at 69% held-out; the best flat-list program on the same exprs reaches
  only 23%. The **+46pp gap is exactly the operator-dependent structural recursion the flat-list family lacks** (OOD-depth
  widens it to 41% vs **3%**). At **0.5B** the model is *degenerate* — it emits the **constant 5 for 73% of all exprs**
  (model-acc 31%), so it isn't evaluating; the synth faithfully reports `5`, and *both* DSLs merely fit the constant
  (list 80% ≥ tree 73%, gap **negative**). **Operator-dependent recursion emerges between 0.5B and 1.5B**, and the new
  instrument measures the emergence: large positive gap ⟺ real recursion, ~zero/negative ⟺ a leaf-level or constant
  heuristic. (The 0.5B OOD `imin(5,sumleaf)`=99% is the same constant in disguise — on deep exprs `sumleaf≥5`, so the cap
  pins it to 5.)
- **The 1.5B residue is the depth cliff.** `eval(t)` faith 69%→41% (random→OOD-depth): the model evaluates shallow trees
  and breaks at depth≥3 — the same D*≈3 cut the abductive `--measure` finds. The residue *is* the model's broken deep
  evals: deterministic (greedy), just not correct. So even the right representation leaves a floor = the model's deviation.
- **§6 closes over trees.** At 1.5B `eval(t)` → recursive Soufflé over a tree ADT (`leaf`/`node` facts +
  `ev(t,v):-node(t,"+",l,r),ev(l,a),ev(r,b),v=a+b.` ×{+,−,*}) + a residue EDB (71 facts) → reproduces the model **100%**
  by construction (`tree_synth.py --emit`): a genuine recursive Datalog catamorphism, not a flat fold. At 0.5B the best
  program is a *constant*, not a catamorphism, so the round-trip is honestly **all-EDB** (250 residue facts, no rule) —
  100% by construction but vacuous, which correctly reflects "there is no tree algorithm to recover."

**The battery also separates structure-determined from leaf-determined tasks** (the honest scope of what trees add).
`maxleaf`/`leftleaf`/`rightleaf` are tree *traversals* but their value is a function of the leaf *sequence* alone — so the
flat-list DSL expresses them too, and the two DSLs tie. Only `eval` (and `depth`, which the model can't do cleanly) is
genuinely tree-only. So tree traversal's unique deterministic power over the leaf-list DSL is precisely the
**operator-dependent recursive fold**, not extremal/boundary leaf reads. Multi-task battery (1.5B, n=150):

| model | task | discovered | faith | truth | list faith | gap | reading |
|---|---|---|---|---|---|---|---|
| 1.5B | eval | `eval(t)` | 62% | 100% | 24% | **+38** | **tree-only** — operator-dependent recursion |
| 1.5B | maxleaf | `maxleaf(t)` | 44% | 100% | 44% | 0 | leaf-determined (exact tie) |
| 1.5B | leftleaf | `leftleaf(t)` | 84% | 100% | 84% | 0 | leaf-determined (exact tie) |
| 1.5B | rightleaf | `rightleaf(t)` | 67% | 100% | 67% | 0 | leaf-determined (exact tie) |
| 0.5B | eval | `imin(5,sumleaf)` | 84% | 15% | 84% | 0 | constant-5 in disguise (no recursion) |
| 0.5B | maxleaf | `minleaf(right(t))` | 33% | 19% | 29% | +4 | broken/obscure |
| 0.5B | leftleaf | `minleaf(t)` | 56% | 36% | 56% | 0 | substitutes "smallest" for "first" |
| 0.5B | rightleaf | `minleaf(t)` | 56% | 41% | 44% | +12 | substitutes "smallest" for "last" |

Reading: (1) **Only `eval` shows a tree-vs-list gap, and only at 1.5B (+38pp)** — the operator-dependent recursion. The
extremal/boundary traversals (max/left/right) are leaf-determined, so tree = list at *both* scales (the 1.5B ties are
exact: 44/44, 84/84, 67/67) — confirming what trees uniquely add is the recursive *fold*, not boundary leaf reads.
(2) The **0.5B does no genuine tree computation**: `eval`→constant-5-disguise, and it substitutes `minleaf` for *both*
"first" and "last" (it emits the smallest operand regardless of which traversal is asked — a tree-level obscure-
substitution, the analog of list `max2`→`max`); model-acc 29–42% everywhere. So the instrument cleanly localizes
structural recursion — present at `eval`/1.5B, absent at 0.5B — and everything else either ties the list DSL or is a
faithful broken substitution.

**Roadmap upshot:** the "different representation" the list-floor pointed at is — for tree-structured behavior —
**deterministic-structural (a catamorphism), not soft.** Tree traversal must be exhausted before PIC. What survives
*both* the list and tree deterministic DSLs (per problem) is the residue that goes to the soft (PIC) representation.

## Harder / non-textbook tasks — obscure-function discovery (1.5B, K=5)

Four tasks the model is *poor* at (`--list-dump` battery extended): second-largest (`max2`), most-common (`mode`), count-of-max (`cmax`), range (`max−min`). Faithful synthesis surfaces **what the model substitutes** when it can't do the task:

| task | model-acc | model *actually* computes | faith | truth | residue |
|---|---|---|---|---|---|
| max2  | 32% | **`max(xs)`** — falls back to the simpler related fn | **83%** | 19% | 17% |
| cmax  | 29% | **`imax(2,min(xs))`** — obscure min-heuristic | 61→67% (guarded) | 24% | 33% |
| range | 17% | `max(tail(xs))` — broken approximation | 31% | 44% | 67% |
| mode  | 44% | obscure `nth(sort(xs),min(xs))`/`last(take(sort(xs),4))` | 17→22% | 37% | **78%** |

Three behaviours, all surfaced faithfully:
1. **Simpler-function substitution** — asked `max2`, the model returns `max` **83%** of the time (it cannot compute "second", so it defaults to a related simpler fold). A clean, faithful "what it really does."
2. **Obscure broken approximation** — `cmax`→`imax(2,min(xs))`, `range`→`max(tail(xs))`: non-human-named functions, partially faithful.
3. **Outside the DSL (forge tax)** — `mode` 78% residue: the model's behaviour there is mostly not expressible as a list-fold; the discovered programs are obscure and only ~20% faithful. This is the genuine tail for that site.

Mean residue rises 16% (6 clean tasks) → **30%** (10 tasks incl. these) — the harder sites carry more forge tax. The synthesizer's value: it doesn't just say "the model fails max2" — it says **"the model computes `max` when you ask for max2,"** which is the mechanistic content.

## Honest caveats

- **The guarded pass rarely fires** on these tasks once the held-out margin is enforced (+4%) — the 0.5B models are too noisy to have *clean* blends, and the 1.5B does the clean functions outright. Guards earn their place only on genuinely piecewise behaviour (the 0.5B `min`/K=5 `len`). This is the conservative, honest setting.
- **Single seed; list-fold sweeps at n=120/task, K≤5; tree sweeps at n=150–250, K=4.** Tree recursion is now done (the
  catamorphism DSL + §6 tree-ADT round-trip above); **wild-site scoping** (proposal §9) is the remaining deterministic-side
  gap. The soft (PIC) representation is still future and only for what survives *both* the list and tree DSLs.
- **`first` is oddly low at 1.5B (78%)** — the model deviates from `first` 22% of the time (a real model quirk the synthesizer faithfully reports, not a synth bug).

## Scope coverage (step 2.5) — the real tail test across 30 problems (1.5B, OOD)

A broad battery (`fieldrun --scope-dump`, 30 list→int families: position / reduction / selection / comparison / count /
arithmetic) run through the synthesizer (`scope_report.py`). The question: across *many* problems, does a small DSL +
reused rule-library cover most (short head) or does each need bespoke rules (long tail)? **Scope mean forge tax (OOD) =
33%** — but that number is *deflated* by a degeneracy the broad battery exposes:

| band | count | tasks |
|---|---|---|
| HEAD resid≤15% — **genuine function** | **3** | `first` `max` `min` (real program, competent model) |
| HEAD resid≤15% — **degenerate constant-fit** | 8 | `gcdred=1@96%` `argmax=5@9%` `cmax=1@71%` `issorted=0@95%` `allsame=0@100%` `ndesc=1@36%` `prodmod=0@60%` `ceven=2@26%` |
| MID 15–50% | 9 | `last` `min2` `range` `nasc` `adiff` `sum` `nuniq` `second` `len` |
| TAIL resid≥50% | 10 | `max2` `maxcount` `median` `penult` `codd` `midval` `czero` `summod` `argmin` `mode` |

**The honest reading (a partial surprise vs the "short head" hope):** of 30 diverse problems the model *cleanly*
implements only **~3** as genuine crisp functions (the simplest folds/selections). 8 more land in the "head" only
because the model **can't do the task and emits a near-constant** (argmax 9%, ceven 26%, ndesc 36% accurate) which a
*constant* program trivially reproduces — that is model degeneracy, not DSL coverage, and it inflates the naive
coverage curve. So genuine crisp coverage is small; the tail (idiosyncratic + degenerate) is the bulk. The coverage
*curve* (`resid≤50%: 67%`) looks short-head-ish; the genuine-function curve is much shorter.

## PIC residue (step 3) — reducible vs irreducible on the 19 non-head problems — `pic_residue.py`

For each surviving problem: top-40 candidate programs → incidence over the best-1 residue → PR from the set-cover
marginal gains + **held-out** ensemble coverage + unexplained% (outside the crisp family). The labels:

- **7 ensemble-reducible** (a small *generalizing* family of ≤2–4 crisp rules; held-out cover ≥60%): e.g. `max2` (~2
  rules, 90%), `midval` (78%), `nasc` (69%), `mode` (~4 rules, 64%).
- **1 PIC-irreducible**: `summod` — **71% of held-out residue outside the crisp family** (modular sum is genuinely not
  a crisp fold in this DSL).
- **11 diffuse/noise**: the train-chosen cover does *not* generalize (held-out coverage low) — the residue is model
  inconsistency, not a coherent alternative algorithm.
- **program-PR is LOW everywhere (1.0–2.9)**: where the residue is coverable at all, a *small* ensemble suffices; mean
  unexplained-residue (outside the crisp family) over non-head = **10%**.

**Caveat — the two PRs (do not conflate).** This is the *surrogate* program-PR (effective number of synthesized
candidate programs), **not** the model's source-PR (the paper's PR≈45 over the model's own circuits). Thm 5
(Diffuseness, proved) is about the *source*-PR; low program-PR here says nothing about it. Testing Thm 5 needs the
model's DLA (track B, future) — see `PIC_LOSSINESS.md` §6.

## Track B — the digit-output Gram kernel (a direct test of proved Thm 2) — `gram_probe.py`

`fieldrun --dump-unembed` extracts the unembedding rows `U_v` for the digit tokens; `gram_probe.py` characterises
`G_{vw}=⟨U_v,U_w⟩`:

| model | Thm 2 `‖U_v−U_w‖²=2(1−ρ)` | mean off-diag ρ | Gram effective rank |
|---|---|---|---|
| 0.5B | **confirmed, err 7.1e-15** | +0.73 | **1.72 / 10** |
| 1.5B | **confirmed, err 7.6e-15** | +0.75 | **1.63 / 10** |

**Thm 2 confirmed numerically to machine precision at both scales** — a clean theory⟷experiment confirmation of a
kernel-proved theorem. And the digit-output frame is **strongly coupled**, spanning only a **~1.6–1.7-dimensional
number-line manifold** (not a 10-D one-hot space; the ρ matrix is a clean ordering — adjacent digits most similar, `0`
the outlier), scale-consistent. This is the regime where a per-token one-hot/EDB view sees up to rank 10 while the
kernel `G` reveals the decisions live in ~2 dimensions — direct evidence for the paper's "linear SVD rank cannot
measure the gap" / `pic`-win direction. **Honest scope:** digits are an unusually coherent semantic family, so this is
a digit-output-specific result, not a claim about all vocabulary.

## Alignment — does the surrogate residue line up with the model's *computed* tokens? (track A ↔ track B) — `align.py`

The decisive join: for each token, **track A** = does the best crisp synthesized program reproduce the model output
(captured) or not (residue); **track B** = the model's own per-token DLA from `fieldrun --source-pr-dump` — **source-PR**
`(Σ_b c_b)²/Σ_b c_b²` over the 57 residual-write blocks (the paper's diffuseness quantity), decode **margin**, and
**μ_t** (blocks already argmaxing to the chosen digit; μ_t=0 = composed). 480 tokens, 1.5B, 12 tasks spanning head→tail.

| signal | residue | captured | AUC | paper's "computed" ⇒ | verdict |
|---|---|---|---|---|---|
| **margin** | 0.46 | 1.53 | 0.22 | lower | ✓ **confirmed** |
| **μ_t** | 6.45 | 8.70 | 0.36 | lower | ✓ **confirmed** |
| source-PR (signed) | 4.87 | 5.51 | 0.30 | higher | ✗ **reversed** |
| PR-magnitude `(Σ\|c\|)²/Σc²` | 7.10 | 8.13 | 0.29 | higher | ✗ **reversed** |

**Confirmed (margin + μ_t):** the surrogate residue boundary is a *real mechanistic boundary* — where the crisp program
fails, the model is making a **low-margin, composed (μ_t-low)** decision, not a clean single-source retrieval. So the
export's crisp-head / residue split corresponds to something the model actually does. (Margin is the cleanest router,
AUC 0.22.)

**The surprise (source-PR, robust to the signed-vs-magnitude confound):** residue tokens are **more *concentrated*
(lower block-PR), not more diffuse.** The paper's high-PR diffuse computation (PR≈45) is a *natural-text* regime; on
these structured single-digit tasks the whole regime is low-PR (3–8) and the residue is *lower* still. **This does not
contradict Thm 5** (which is about the natural-text source-PR) — it says the structured-task residue is a *different,
concentrated regime*: a small coalition of blocks commits to a different (often wrong) answer at low margin, rather than
a diffuse dense-Gram repair.

**Two consequences for the export (both reassuring):**
1. "Outside the crisp DSL" ≠ "diffuse/incompressible." `summod` (modular sum) is outside our DSL only because it lacks
   a modular primitive — the model computes it *concentratedly* (low PR). So surrogate-irreducibility is a **DSL-
   expressiveness gap**, not mechanistic diffuseness; extending the DSL can convert more residue into crisp rules.
2. The genuinely-incompressible high-PR dense-Gram residue (the expensive `pic` case) is **rare in this regime** — most
   residue is low-margin + concentrated, so a small ensemble / low-rank `pic` / `edb` captures it cheaply. The costly
   diffuse-PIC part is a natural-text / open-vocabulary phenomenon, not the structured-task forge tax.

**Caveats:** one model (1.5B), single-digit structured tasks, n=480, source-PR over 57 residual-write blocks. The
margin/μ_t alignment is robust; the low-PR-residue claim is specific to this structured-task family — the natural-text
diffuse regime (where the proved Thm 5 lives) is the obvious next measurement.

## Next

Done this round: tree-recursion DSL + §6 round-trip; scope battery (2.5) + coverage/degeneracy split; PIC residue
labels (3) + the program-PR-vs-source-PR distinction; track-B Gram (Thm 2 confirmed, ~1.6-D digit frame); the **A↔B
alignment** (margin/μ_t confirm the residue boundary; source-PR reversed = structured residue is concentrated, not
diffuse). Remaining:
the **model source-PR / DLA** test of Thm 5 (the real diffuseness test); the **tropical-rank vs linear-rank** logit
experiment; the 7B scaling point; **wild-site scoping** (§9); then the `--residue-strategy` roll-in to LOGIC_EXPORT.
