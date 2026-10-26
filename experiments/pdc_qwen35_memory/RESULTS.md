# Qwen two-hop memory: refused-stratum observation

The protocol and exact input construction were published in commit af3c5c6
before inference. All 27 triples have a singleton-refused first hop. There are
no admitted cases and no actual admitted residual replays.

| Arm | Correct full-host first-token answers | Both readouts: affirmed / hallucination / abstain |
|---|---:|---:|
| Rederive | 12/27 | 11 / 14 / 2 |
| Crystallize (refusal fallback) | 12/27 | 11 / 14 / 2 |
| Generated intermediate | 27/27 | 27 / 0 / 0 |
| Tool intermediate | 27/27 | 27 / 0 / 0 |

Appending the correct intermediate improved this fixed slice by 55.56 percentage
points, exceeding the preregistered five-point bar. Generated and tool inputs
are identical because all first-hop answers were correct. Crystallize and
rederive inputs are identical because every first hop was refused.

This intervention also changes an empty prior assistant turn into a nonempty
one. There is no neutral-token or wrong-token control. Consequently these
results do not isolate semantic memory use from conversation formatting, or
establish general necessity of intermediate tokens. Redundancy where admission
succeeds remains untested.

All committed answers agree with the host on the covered candidates, including
14 wrong commitments among 25 commitments in each baseline readout (56%). The
certificate preserves score decisions; it does not certify arithmetic truth.
The previous perfect single-step Qwen result therefore does not restore the
retired epistemic claim.

Exact input deduplication yields 54 unique second-hop forwards for 108 named
paths, plus nine reused first-hop captures. Independent Transformers reference
matches 54/54. The independent audit checks 366282 body/probe annotations and
104 unique readout commitments, with maximum body-score error 2.14e-14. Native
and reference calls are accounted separately. No original consumed final
25-context inputs were used.

`results.json`, `verification.json` and `run_manifest.json` record the numerical
results and hashes. Large raw captures, reference output, full bundle and the
learning NPZ must accompany the supplemental release; they are not replaced by
this report. The subsequent frozen 200-proposal learning comparison produced
zero admissions in both arms. `learning_chart_audit.json` grades both endpoints
with unchanged fieldrun rules and records that the joint advance gate failed.
