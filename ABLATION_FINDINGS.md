# Expert-ablation findings — turning "routes-to" into "load-bearing"

Causal test of the density-minimization expert labels: do the clusters that *route*
a language/domain actually *carry* it? We zero-ablate each target expert's **anchor**
circuit and measure held-out next-token loss vs baseline. Reproduce with
`scripts/make_ablation_evals.py` then
`fieldrun --bundle <b> --ablate-eval ablate/manifest.json` (raw eval sets +
`ablate/results.tsv` are gitignored; this file is the curated summary).

## Setup

- **Model**: `Qwen2.5-0.5B-Instruct` (rope, **24 layers**, 14 heads, MLP 4864, vocab
  151936). NB: a 0.5B model — *not* 7B/28-layer.
- **Targets** (each expert's anchor, from the 512-expert tree): `e8` German
  (neuron L19#1273), `e17` Romance det (n.L21#3483), `e19` Spanish verb (n.L20#35),
  `e0` cross-lingual core (n.L22#1222), `e59` code (head L22#2), `e107` LaTeX
  (n.L20#661); + 5 random matched-unit controls (noise floor).
- **Eval sets** (held-out, disjoint from the 15k clustering corpus, n=128 each,
  ctx=32): de/fr/es/en = Wikipedia articles past the clustering slice; python =
  stdlib `argparse` source; latex = authored math.
- **Intervention**: **zero**-ablation of the anchor (single neuron/head). Metrics:
  Δloss (next-token CE, nats), Δlogit (target-token), flip% (top-1 changed).
- **Substitutions / blockers**: mean-ablation not implemented (zero used);
  full-circuit-set ablation needs the `--experts-out` partition (a 2h40m re-run) —
  so this tests **anchors only**, not each expert's full 21–300 circuits.

## Results (Δloss, nats; n=128/eval, ctx=32)

Rows = ablated anchor, cols = held-out eval set. Diagnostic cells annotated with
**flip%** and **Δlogit** (target-token). Control block = 5 random matched-type units.

| ablate ↓  eval → | de | fr | es | en | python | latex |
|---|---|---|---|---|---|---|
| **e8** German `n.L19#1273` | **+0.2943** (17.2%, −0.524) | −0.0031 | +0.0028 | −0.0018 | +0.0016 | +0.0016 |
| **e17** Romance det `n.L21#3483` | +0.0053 | −0.0008 | −0.0037 | +0.0001 | +0.0012 | +0.0009 |
| **e19** Spanish verb `n.L20#35` | −0.0001 | −0.0011 | −0.0139 (18.8%, −0.268) | −0.0005 | +0.0002 | +0.0004 |
| **e0** xling core `n.L22#1222` | +0.0071 | +0.0496 | −0.0146 | +0.0120 | +0.0046 | +0.0025  · Δlogit +0.06…+0.20 *everywhere* |
| **e59** code `head L22#2` | −0.0016 | +0.0116 | +0.0056 | +0.0065 | +0.0044 | −0.0002 |
| **e107** LaTeX `n.L20#661` | +0.0004 | +0.0010 | +0.0041 | −0.0024 | −0.0008 | −0.0082 |
| *control mean \|Δloss\|* | *0.0014* | *0.0025* | *0.0026* | *0.0019* | *0.0011* | *0.0015* |

Baselines (mean CE, nats): de 2.282 · fr 3.088 · es 3.661 · en 2.179 · python 2.927 · latex 1.038.

## Headline findings

1. **Causality is real but concentrated, not uniform.** Anchor ablation confirms a
   causal, language-specific role for the *sharpest* experts and reads as noise for
   the *broad/diffuse* ones. The correlational labels are **not all equally causal**.

2. **`e8` is a clean, near-isolated German circuit.** Zeroing one neuron (L19#1273)
   raises **German** loss **+0.2943 nats** (13% over baseline 2.282), drops the
   German target logit **−0.524**, and flips **17.2%** of German next-tokens — while
   French (−0.0031), Spanish (+0.0028), English (−0.0018), Python (+0.0016) and LaTeX
   (+0.0016) stay at the control floor (~±0.002). German effect ≈ **1400×** e8's mean
   elsewhere. At least one "expert" anchor is a genuine, isolated language circuit.

3. **The anchor is not the expert.** We ablated 1 circuit per expert, not its full
   set. Where the function concentrates in the anchor (e8), the result is decisive;
   where it's distributed (e0=300 circuits, e17, e59, e107), one circuit reads as
   noise. **A null here means "the anchor isn't load-bearing," not "the expert isn't
   real."** Full circuit-set ablation is the decisive test still owed.

4. **The "universal shared syntactic core" (`e0`) is not load-bearing — it's a
   competitor.** Ablating e0 *raises* the target logit everywhere (+0.06…+0.20) and
   even *helps* Spanish (Δloss −0.0146); it predicts commas/`the`/`de` that compete
   with content targets. It has the broadest reach (only unit above noise on all six
   sets) but acts as a regularizer, not a scaffold. H3 as stated: **not supported** —
   now confirmed directly by the function-vs-content split (e0 hurts function-word
   targets, spares content; see the e0 follow-up section).

5. **Metrics disagree below ~0.01 nats; trust flip%/Δlogit there.** `e19` (Spanish)
   flips **18.8%** of Spanish tokens and drops the Spanish target logit **−0.268**
   (clearly Spanish-specific) yet shows **negative** mean Δloss (−0.0139) — a
   logit-scale (`lse`) artifact. Single-unit Δloss alone is unreliable for small
   effects.

6. **Family boundary holds in the direction testable.** e8 is German-specific and
   spares Romance (and everything else) — damage respects, even exceeds, the family
   line. The symmetric Romance→spares-German test is a **null** (e17 anchor not
   load-bearing), so the family claim is only half-confirmed.

7. **`e59`/`e107` show no clean formal-language separation at the anchor.** e59's
   biggest effect is on French (+0.0116), not code; e107's biggest is on LaTeX
   (−0.0082, largest in its row — direction consistent with specialization) but tiny.
   H4: **not supported (e59) / inconclusive (e107)** — pending full-set ablation.

## e0 follow-up — the competitor test (function vs content targets)

Splitting e0's per-position effect by **target-token class** — function-word/punctuation
vs content word (`--ablate-rows` + `scripts/analyze_e0_split.py`; de/fr/es/en, n=200/lang,
ctx=32) — settles the "shared syntactic core vs competitor" question, decisively for
**competitor**:

| ablate **e0** `n.L22#1222` | function-word targets | content-word targets | gap (fn − content) |
|---|---|---|---|
| **mean Δloss, pooled (nats)** | **+0.0394** (n=370) | **−0.0066** (n=428) | **+0.0460** |
| per-lang Δloss de/fr/es/en | +0.049 / +0.082 / +0.023 / +0.014 | −0.012 / +0.028 / −0.037 / −0.012 | +0.061 / +0.054 / +0.060 / +0.026 |
| random control `n.L22` (pooled) | −0.0002 | −0.0023 | +0.0021 |

Ablating e0 **hurts function-word prediction** (+0.039) and **spares/helps content-word
prediction** (−0.007) — a **+0.046-nat swing**, with the function > content gap positive in
**all four languages**, while the matched random control shows no such divergence (~±0.002).
e0 is a **function-word promoter / competitor**: it boosts commas, articles, prepositions;
removing it costs function-word targets and relieves competition for content targets. This
**refutes the "universal load-bearing syntactic core" reading of e0** — its causal job is to
predict high-frequency function tokens, not to scaffold syntax across languages.

## Bottom line

The clustering's labels are **causally real where the deciding circuit concentrates
in a single hub** — German is the clean, emphatic case (and Spanish by flip/logit) —
but for broad or small experts the single anchor is **not** load-bearing, so those
labels remain correlational pending **full circuit-set ablation** (the highest-value
next step; the `logits_ablated` hook already accepts a circuit list). The "universal
core" `e0` is a function-word *competitor*, not a load-bearing syntactic scaffold.

## Caveats

zero- (not mean-) ablation overstates magnitude but preserves the specificity
pattern; anchor-only (not full-set); ctx=32 (vs clustering's 64) inflates the German
effect (+0.20→+0.29); 0.5B model; n=128, single control seed. **A null at the anchor
is not a null for the expert** — the broad/diffuse experts (e0/e17/e59/e107) are
under-read here because their function is distributed across the full circuit set,
which this pass does not ablate.

## Next steps (prioritized)

1. **Full circuit-set ablation** (decisive). Re-run `--corpus-decompose --experts-out`
   (~2h40m) to recover per-expert circuit membership, then ablate each expert's full
   21–300 circuits (the `logits_ablated` hook already accepts a circuit list;
   `make_ablation_evals.py --partition <experts-out.json>` now emits the full-set specs
   + size-matched random controls). This is
   the test that settles H2 (family boundary), H3 (universal core), and H4
   (formal-language separation) for the diffuse experts — the anchor pass only
   settled the *concentrated* ones (e8, partly e19).
2. **Statistical power**: n≥500 per eval set + several control seeds for a proper
   noise *distribution* (current n=128, single seed, supports the large effects but is
   weak for asserting "no effect" on diffuse experts).
3. **Mean-ablation** alongside zero-ablation (shrinks absolute Δloss, should preserve
   the specificity pattern — confirms the effects aren't a mean-removal artifact).
4. ~~e0 token-class breakdown (function vs content targets)~~ — **done**, see the e0
   follow-up section above: competitor confirmed (function +0.039 / content −0.007 pooled).
5. **ctx=64** to match the clustering regime (magnitudes are ctx-dependent; the
   pattern is not).
