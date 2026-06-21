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

## Honest caveats

- **The guarded pass rarely fires** on these tasks once the held-out margin is enforced (+4%) — the 0.5B models are too noisy to have *clean* blends, and the 1.5B does the clean functions outright. Guards earn their place only on genuinely piecewise behaviour (the 0.5B `min`/K=5 `len`). This is the conservative, honest setting.
- **Single seed, n=120/task, K≤5, list folds only.** No OOD-length test yet; tree recursion and wild-site scoping are future (proposal §9/§11).
- **`first` is oddly low at 1.5B (78%)** — the model deviates from `first` 22% of the time (a real model quirk the synthesizer faithfully reports, not a synth bug).

## Next

OOD-length held-out; the 7B scaling point (in progress); tree-recursion DSL; then wire discovered folds → `recursion_dl` recursive rules (proposal §6).
