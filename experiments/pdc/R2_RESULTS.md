# R2: frozen GPT-2 read incidences, 2026-10-24

**Empirical gate passed for the preregistered set only.** Preregistration:
`R2_PREREG.md`, SHA256 `27734b2d8072c2889885b715d61e78f463a577c7bcb685a957cbaedadbfb4ee2`.
The selection, native capture (`r2_reads.json`), and independent report
(`r2_results.json`) are retained. The report hashes its exact inputs.

| Claim | Result |
|---|---|
| Algebraic identity under stated convention | Host assigns `softmax(mask(q kᵀ / sqrt(64))) v` to the pre-projection head write; forbidden-key probability mass is exactly 0 on this set |
| Numerical scaled-incidence max absolute error, f64 recomputation | 7.143e-6 |
| Numerical probability max absolute error | 2.983e-7 |
| Numerical head-write max absolute / RMS error | 6.683e-7 / 4.239e-8 |
| Incidence argmax-set vs native probability argmax-set agreement | 108/108 = 1.00 |
| Independent NumPy full-vocabulary top-1 agreement | 48/48 = 1.00 |
| Observer effect on native hidden states | max absolute error 0; predictions identical |

All 12 named layer/head pairs pass on their nine instances. No head is excluded.
This read identification uses scaled queries, causal keys, and pre-projection
values. Algebraic identity follows the inspected host assignment; numerical
agreement is a separate finite-arithmetic measurement. This is not a theorem
about arbitrary implementations, prompt distributions, or head decision accuracy.
The hard max key is compared to the head's actual highest-probability key, not
to a vocabulary decision or to a hard-max value-vector replacement.

Reproduce from the fieldrun root:

```sh
cargo build --release --no-default-features
target/release/fieldrun --bundle bundles/gpt2 --pdc-read-selection experiments/pdc/r2_selection.json --pdc-read-dump experiments/pdc/r2_reads.json
python3 scripts/pdc_read_check.py --dump experiments/pdc/r2_reads.json --selection experiments/pdc/r2_selection.json --bundle bundles/gpt2 --out experiments/pdc/r2_results.json
```

The native observer reads the very arrays used by `Gpt2::hidden_observed`.
The independent checker uses only emitted artifacts and the same published
bundle format; no cross-repo imports. NumPy 1.26.4; f16 stored weights converted
to f32 for the reference. Compiler checks pass with pre-existing warnings.
