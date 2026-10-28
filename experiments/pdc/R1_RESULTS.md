# R1: finite-body executor, 2026-10-24

**Empirical gate passed on finite arithmetic programs.** This is not a model measurement.
The semantics are the kernel statements in i-orca `examples/pdc/PDC_Derive.thy`:
`witness_fragile`, `one_body_agreement`, `one_body_temperature_invariance`, and
`pic.specialization_alpha_eq_logit`. There are no sibling-repository imports.

```sh
python3 -m unittest discover -s scripts -p test_pdc_derive.py -v
python3 scripts/pdc_derive.py experiments/pdc/r1_witness_input.json --temperature 1
```

Four tests pass. The generated family has seed 20261024, 256 nine-atom programs,
four dyadic source contributions plus a dyadic bias per atom, one absent rival,
and five temperatures: 0.000001, 0.125, 1, 8, 1000000. Across all 1,280 evaluations,
both readout winner sets equal the full-coalition PIC argmax; no fragility occurs.
The two-body witness has Viterbi winner B, marginal winner A, and fragility true.
The frozen witness input and output are adjacent JSON files.

Schema `pdc.finite-derive.v1` contains the named rival set, full body table, two
named queries with scores, winner sets, margin, and status, and a fragile bit.
`winning_bodies` lists best-bracket witnesses; for marginal queries the whole
body set contributes, so those witnesses alone are not the marginal explanation.
Ties and entirely absent tables do not select an atom. A missing margin means
fewer than two present candidates. This executor supplies no admission or truth
certificate and does not serve a DerivationBundle yet.

Scores use floating point and shifted logsumexp, not exact kernel arithmetic.
Nonfinite inputs/results and nonpositive temperatures are rejected. Near-tie
rounding is not certified by these tests. Overlapping subbodies are not disjointified.
