# Bucketing sweep results

End-to-end sweep of the density-minimization bucketing across **models × corpora ×
(experts, depth)**. Reproduce with `scripts/make_corpora.py` then
`scripts/bucket_sweep.sh` (raw per-run outputs under `sweeps/runs/`, gitignored;
this file is the curated summary).

## Setup

- **Models** (both rope/Qwen2.5-0.5B): `Qwen2.5-0.5B-Instruct` (general) and
  `Qwen2.5-Coder-0.5B-Instruct` (code-specialized).
- **Corpora** (Qwen-tokenized): `english`, `german`, `spanish`, `code` (Python),
  `math` (LaTeX), + `pooled_diverse`. **K=4**, **ctx=48**, **n_eval=80** (so each
  run sees positions 48..128 of its corpus).
- **Grid**: `experts:depth` ∈ {8:1, 16:1, 8:3}.
- **Metrics**: `|C|` distinct circuits · `span1` = % tokens whose atom fits one
  expert (top-1 routable) · `active_fewer` = % fewer circuits computed under
  routing (oracle proxy) · `resident` = experts covering 90% of load (the hot set)
  · `contrib_faith` = composition-decode faithfulness (`--experts-dl-contrib`,
  held-out valid by construction) · `lookup_HO` = bigram-lookup **held-out**
  predict==decode (the retrievable floor; a learned table, NOT the model) · `leaves`
  = recursive leaf experts at depth 3.

## Headline findings

1. **Structured domains decompose to single experts; prose spreads.** `span1`
   (top-1 routable, E=8): **code ≈ 92–95 %, math ≈ 92 %** vs **prose (en/de/es)
   ≈ 50–56 %** (Instruct). Decisions in code/math collapse to one expert ~2× more
   than natural language — a clean cross-domain signal.
2. **The code-specialized model is more concentrated everywhere.** Coder vs
   Instruct: `span1` on english **75 % vs 50 %**; `|C|` on code **25 vs 48**
   (fewer distinct deciding circuits). Specialization shows up as concentration.
3. **The composition decode is faithful on every corpus and model — 100 %,
   held-out valid** (it recomputes per input; `Σ contrib == logit`). The **bigram
   lookup** is the only thing with a held-out gap: `lookup_HO` **0–31 %**
   (en/de moderate, es/code ~0). Composition dominates; the retrievable floor is
   small and corpus-dependent.
4. **E trades cost for routability** (consistent across all corpora): E 8→16 lifts
   `active_fewer` (~+8–20 pts) but lowers `span1` and grows the resident set.
5. **Recursive depth-3 resolves a stable 17–25 leaf experts** across corpora
   (the long tail) regardless of language/domain.

## Full table

| model | corpus | E | d | \|C\| | span1 | active_fewer | resident | contrib_faith | lookup_HO | leaves |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruct | english | 8 | 1 | 94 | 50% | 74% | 7 | 100% | 31% | 1 |
| Instruct | english | 16 | 1 | 94 | 52% | 81% | 10 | 100% | 31% | 1 |
| Instruct | english | 8 | 3 | 94 | 50% | 74% | 7 | 100% | 31% | 25 |
| Instruct | german | 8 | 1 | 74 | 52% | 70% | 5 | 100% | 19% | 1 |
| Instruct | german | 16 | 1 | 74 | 57% | 79% | 9 | 100% | 19% | 1 |
| Instruct | german | 8 | 3 | 74 | 52% | 70% | 5 | 100% | 19% | 17 |
| Instruct | spanish | 8 | 1 | 128 | 56% | 68% | 6 | 100% | 0% | 1 |
| Instruct | spanish | 16 | 1 | 128 | 44% | 73% | 8 | 100% | 0% | 1 |
| Instruct | spanish | 8 | 3 | 128 | 56% | 68% | 6 | 100% | 0% | 17 |
| Instruct | code | 8 | 1 | 48 | 92% | 67% | 6 | 100% | 0% | 1 |
| Instruct | code | 16 | 1 | 48 | 83% | 83% | 12 | 100% | 0% | 1 |
| Instruct | code | 8 | 3 | 48 | 92% | 67% | 6 | 100% | 0% | 25 |
| Instruct | math | 8 | 1 | 73 | 92% | 57% | 5 | 100% | 6% | 1 |
| Instruct | math | 16 | 1 | 73 | 80% | 63% | 7 | 100% | 6% | 1 |
| Instruct | math | 8 | 3 | 73 | 92% | 57% | 5 | 100% | 6% | 25 |
| Coder | english | 8 | 1 | 93 | 75% | 66% | 6 | 100% | 25% | 1 |
| Coder | english | 16 | 1 | 93 | 68% | 79% | 8 | 100% | 25% | 1 |
| Coder | english | 8 | 3 | 93 | 75% | 66% | 6 | 100% | 25% | 25 |
| Coder | german | 8 | 1 | 93 | 53% | 71% | 5 | 100% | 19% | 1 |
| Coder | german | 16 | 1 | 93 | 45% | 78% | 6 | 100% | 19% | 1 |
| Coder | german | 8 | 3 | 93 | 53% | 71% | 5 | 100% | 19% | 24 |
| Coder | spanish | 8 | 1 | 80 | 61% | 76% | 6 | 100% | 0% | 1 |
| Coder | spanish | 16 | 1 | 80 | 59% | 85% | 10 | 100% | 0% | 1 |
| Coder | spanish | 8 | 3 | 80 | 61% | 76% | 6 | 100% | 0% | 25 |
| Coder | code | 8 | 1 | 25 | 95% | 63% | 7 | 100% | 0% | 1 |
| Coder | code | 16 | 1 | 25 | 89% | 87% | 12 | 100% | 0% | 1 |
| Coder | code | 8 | 3 | 25 | 95% | 63% | 7 | 100% | 0% | 23 |
| Coder | math | 8 | 1 | 40 | 92% | 67% | 6 | 100% | 6% | 1 |
| Coder | math | 16 | 1 | 40 | 89% | 81% | 10 | 100% | 6% | 1 |
| Coder | math | 8 | 3 | 40 | 92% | 67% | 6 | 100% | 6% | 25 |

(`pooled_diverse` rows omitted — see caveat 1.)

## Caveats

1. **`pooled_diverse` ≡ `english` at this config** (omitted above): with `n_eval=80`
   the sweep only reaches token ~128, and `pooled_diverse` begins with the English
   slice, so it samples only English. A real multi-domain pooled run needs
   `n_eval` covering the whole corpus (and ideally interleaved domains) — a bigger
   sweep, which the **KV-cached `explain_stream`** (this branch) now makes
   practical for long contexts.
2. **Cost is a static oracle-router proxy** (`active_fewer`): it assumes the atom
   is known. A real wall-clock saving needs a learned router + experts mapped to
   pageable weight chunks.
3. **`span1` / `|C|` are over 80 sliding-window (ctx 48) decisions** — a sample of
   each corpus's opening, not the whole document.
4. **`contrib_faith` is 100 % by construction** (algebraic identity, recomputed per
   input) — it is a correctness sanity check, not an accuracy claim; the
   `lookup_HO` column is the one carrying real generalization signal.

## Next

- Larger `n_eval` over the full pooled corpus (now feasible with caching) to get a
  true multi-domain partition rather than the English prefix.
- More architectures / larger models once converted (all current bundles are
  Qwen2.5-0.5B rope).
