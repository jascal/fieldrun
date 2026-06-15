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

## The ladder

1. **per token** — `decompose_descent` over one decision (this branch).
2. **per query** — union the per-token atoms across a prompt → the minimal circuit
   working-set that reproduces the whole query.
3. **per corpus** — cluster atoms across many queries → recurring buckets =
   candidate **experts**; route a token to its atom-bucket → a smaller/cheaper MoE.
   fieldrun already has the expert-offload machinery (`src/bundle.rs`,
   qwen3moe/gemma4-MoE/mla/minimax) for this to land in.

## Status

- [x] Per-token descent + multi-competitor substrate (`explain.rs`), unit-tested.
- [x] `--probe-decompose` harness; measured on Qwen2.5-0.5B.
- [x] Faithful necessity-confirmation of atoms (`predict_ablated`, Route B).
- [ ] Per-query atom aggregation.
- [ ] Per-corpus atom clustering → expert buckets.
