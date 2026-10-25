# R3: competing trigram-copy route bodies, 2026-10-24

**Empirical gate passed as a finite circuit-readout diagnostic.** Frozen task and
body generators: `R3_PREREG.md`, SHA256
`9fefff6efe408ff6bcf7f8e2d05d8fe58e36f3a39c3ca713be4c20792514c81e`.
Native capture `r3_reads.json`; full bodies, sources, frame and judgments
`r3_charts.json`. No explanation was selected using a host token.

| Quantity | Result |
|---|---|
| Contexts / bodies | 16 / 576 (36 bodies per context) |
| Viterbi exact agreement with full host token | 8/16 = 50% |
| Marginal T=1 exact agreement with full host token | 6/16 = 37.5% |
| Fragility | 2/16 = 12.5% |
| Viterbi / marginal task-label agreement | 10/16 / 10/16 = 62.5% each |
| Ties | 0 |
| Host restricted to the eight fixed rivals | Same agreement rates on this set |
| Independent NumPy full-vocabulary top-1 validation | 256/256 = 1.00 |
| Native observer hidden-state error | 0 |
| Final-LayerNorm folded decomposition max absolute error | 4.014e-5 |

The bodies are singleton attention-value routes at earlier matching trigram
continuations, projected through the actual host output matrices and read by
the frozen final-LN-folded frame. Every matching route in every selected head
is retained. The two readouts have a real measured disagreement rate, but
neither recovers the entire host computation. No causal sufficiency or
necessity is claimed. LN folding treats the measured scale as fixed, which is
the DLA convention; it is not a claim about recomputing LN after intervention.
The copy task label differs from host agreement and is reported separately.

```sh
target/release/fieldrun --bundle bundles/gpt2 --pdc-read-selection experiments/pdc/r3_selection.json --pdc-read-dump experiments/pdc/r3_reads.json
python3 scripts/pdc_copy_chart.py --dump experiments/pdc/r3_reads.json --selection experiments/pdc/r3_selection.json --bundle bundles/gpt2 --out experiments/pdc/r3_charts.json
```

The artifact includes the shared candidate unembedding, per-position fixed
source writes, their complement, and bias. Thus later consumers can check the
incidences without loading a sibling repo or inventing a body from the token.
It does not yet include an admission or external-theory certificate.
