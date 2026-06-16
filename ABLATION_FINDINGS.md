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
   sets) but acts as a regularizer, not a scaffold. H3 as stated: **not supported**.

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
effect (+0.20→+0.29); 0.5B model; n=128, single control seed. Tightening: full-set +
mean-ablation, ctx=64, n≥500, per-token-class loss (function vs content) for e0,
multiple control seeds.
