# R8-H: actual subset service, zero answer coverage, 2026-10-25

**Empirical negative coverage result.** The host-only service executes real
native forwards and returns complete bundles or abstentions. Both readouts
abstain on all 75 fixed contexts. Host agreement and hallucination rates
among commitments are undefined, not 100% or 0%. The required nonempty
coverage gate fails, so this is not a successful R8 service result. R5's
retired epistemic claims remain retired.

Preregistration `R8H_PREREG.md`, SHA256
`92ee9f0a9245192246d40e8c9255961adf5254d917fc9067083f09143cae8cfa`.
Lexical probe SHA256
`2361b4e36efa7928c6ee5f7f1c9aaff1154f90d31b40a32b4c1b2635b045cc34`.
It contains 1,703 tokens selected by the frozen tokenizer-text rule, not by
their host scores. Body generators, the nine derive atoms, T=1 and gamma=0.5
are unchanged from R5. The certificate additionally covers every probe token.

| Result | Viterbi | Marginal |
|---|---:|---:|
| Contexts | 75 | 75 |
| Commitments | 0 | 0 |
| Abstentions | 75 | 75 |
| Readout-margin refusals | 33 | 37 |
| Body-margin refusals on expanded coverage | 20 | 23 |
| Pairwise-fidelity refusals | 22 | 15 |
| Non-abstaining full-host agreement | undefined | undefined |

The body and source values themselves agree with the full baseline. The
expanded competitor coverage defeats the earlier narrow certificates:
neither relaxing a guard nor removing inconvenient probe tokens was allowed
after observing this result. The 47 misses and three fragile truths from
the unchanged nine-atom body program remain independently reported.

## Measured computation and validation

The native service dispatches `rowdot_f32_subset`, never a full-vocabulary
scorer on its service path. An explicitly labeled full diagnostic baseline
uses the full scorer. Each arm performs one transformer forward per context.
Both construct the same three body/probe score tables. The independent
auditor checks all 383,175 body/probe annotations, full source writes,
both readouts and every guard/refusal. Maximum annotation, native probe-score,
and source-write discrepancies are all zero. All 75 full-baseline tokens
also match the frozen R5 reference. No transformer layer or position is skipped.

| Work over 75 contexts | Full baseline | Probe service |
|---|---:|---:|
| Native output-row dot calls | 3,769,275 | 127,725 |
| Additional body/probe dot calls | 383,175 | 383,175 |
| Distinct output rows per context | 50,257 | 1,703 |

There are 3,641,550 fewer native output-row calls (96.61% reduction).
Including the shared body projections, output/body dot calls fall by 87.70%.
These are dispatch counts measured in the executed native branch and
verified against the actual scorer inputs. Model loading, token embedding
lookups and transformer matrices are not counted as skipped work. This
compute saving produces no usable answers under the frozen guard policy.

The timing addendum was written before inspecting baseline scores or
running the service. The initial full capture lacked an outer wall timer;
it is retained as a validation run. One additional explicitly diagnostic
full baseline followed by the first live service batch forms the fixed
timing pair. There was no favorable-run selection.

| Descriptive timing | Full baseline | Probe service |
|---|---:|---:|
| Native forward plus decode, summed seconds | 9.412 | 8.785 |
| End-to-end subprocess wall seconds | 13.355 | 12.871 |

The outer timer includes loading, native capture, Python body/guard
construction, external grading and serialization. One fixed ordered pair
does not establish a general latency or throughput speedup. It is the
row-dispatch reduction, not these timings, that is directly demonstrated.

## Runtime interface and retained evidence

The live CLI performs a fresh native forward rather than selecting a cached
answer. It loads only selected output rows and final-LN vectors for Python
body scoring. It validates the pinned domain, probe and model identities.
The external addition checker annotates already-selected responses; it
does not select answers or filter hallucinations. Claims explicitly set
`theory_soundness=false`. Derive and certificate rival scopes are separate.

```sh
python3 scripts/pdc_service.py --runtime target/release/fieldrun --selection experiments/pdc/r8h_service_selection.json --case-id t0-a0-b0 --domain experiments/pdc/r5_domain.json --probe experiments/pdc/r8h_probe.json --bundle bundles/gpt2 --out /tmp/pdc-response.json --grade-addition
```

A post-benchmark single-request integration check returns one response for
`t0-a0-b0`, with 1,703 actual native row calls, no full-host token obtained,
and explicit atom-null abstentions for both named readouts. It is retained
separately from the timing pair. The interface rejects held-out case IDs
and accidental full-scoring service requests before invoking the native
binary. `--allow-full-baseline` is an explicit diagnostic override; its
output is labeled a full baseline, never a subset service response.

Evidence: `r8h_results.json`, `r8h_timing.json`, the two served bundles,
the two timed native captures, the retained initial baseline, and the
single-request capture/bundle. The timing runner refuses to overwrite a
frozen pair. The independent auditor can be rerun without repeating the
experiment:

```sh
python3 scripts/pdc_service_audit.py --artifacts experiments/pdc --model-bundle bundles/gpt2
python3 -m unittest discover -s scripts -p 'test_pdc_*.py' -v
```

Thirteen focused tests pass, including selected-row loading, rejection of
accidental full-vocabulary service and protection of the reserved 25
contexts. The reserved slice remains uncaptured. This arm does not close
the original truth-sound R8 gate or criterion 9, and does not establish the
unperformed R6 memory or R7 admission-learning hypotheses.
