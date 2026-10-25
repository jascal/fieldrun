# Final held-out diagnostic audit (2026-10-25)

The unchanged original R8-H policy abstains on **25/25 reserved template-3
contexts for both Viterbi and marginal readouts**. Coverage is zero; host
agreement and hallucination rates among commitments are undefined. The
nonempty-coverage gate fails, so this audit does not satisfy R8 or criterion 9.
No theory-soundness claim is made. The reserved slice is now consumed and
must not be described as untouched or used for a new confirmatory policy.

The audit was frozen in FINAL_AUDIT_PREREG.md before native execution.
final_audit/manifest.json pins evaluator, checker, probe, domain and model
hashes. Original output weights were used because neither R7 endpoint
passed its advance gate. The ordinary service CLI still rejects reserved
contexts. The separate final-audit runner uses an explicit heldout scope,
with no domain relabeling. Every response retains readout, margin, fragile
bit, full body tables, explicit abstention and external addition judgment;
memory provenance is none. There is no admitted or appended memory in this
service policy. R6's memory chart remains separate evidence.

| Quantity | Result |
|---|---:|
| Contexts / body-probe annotations checked | 25 / 127,725 |
| Body score / native probe score discrepancy | 0 / 0 |
| Paired source-write discrepancy | 0 |
| Theory misses / fragile truths | 18 / 1 |
| Full-baseline native output-row calls | 1,256,425 |
| Subset-service native output-row calls | 42,575 |
| Native output rows skipped | 1,213,850 (96.61%) |
| Body-probe dots, each arm | 127,725 |
| Combined output/body dot reduction | 87.70% |
| Layers / positions skipped | 0 / 0 |

One ordered baseline/service pair took 3.476 / 3.257 seconds for the native
process, including model loading and capture serialization. These are
descriptive timings, not evidence for a general latency improvement.
The independent NumPy model-reference check matches all 25/25 full-host
winners; its result is retained separately in
final_audit/reference_verification.json. The independent certificate checker
is scripts/pdc_final_audit_check.py, with results in final_audit/results.json.
Both paired captures, bundles, selections and timing logs are retained.
No quality-based rerun or policy adjustment occurred.

The original R5 epistemic stop, R4's unachieved intended negative-domain
claim, R6's empty admitted stratum and R7's negative learning outcome remain
in force. This is stopped-stage evidence, not the full PDC north star.
