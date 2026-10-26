# Qwen memory-content controls: supplied-digit sensitivity

Protocol and all exact inputs were published in efa57e0 before new model
inference. The 27 triples and prior empty/correct-digit outcomes were already
known. These are paired development controls, not fresh confirmation.

| Prior assistant answer | Correct final answers | Follows supplied wrong sum |
|---|---:|---:|
| Empty (reused baseline) | 12/27 | — |
| Correct digit (reused baseline) | 27/27 | — |
| Period `.` | 9/27 | — |
| Wrong digit w=(a+b+1) mod 5 | 0/27 | 27/27 |

Every wrong-digit result equals w+c, rather than the correct a+b+c.
The predeclared content-sensitivity bar passes (correct-minus-wrong ≥6
and following-wrong ≥14); the neutral-minus-empty improvement bar fails
(9−12=−3, below +2). Correct and wrong digits occupy the same one-token
answer slot with identical remaining context. Thus on this slice the
model's behavior depends on the intermediate digit's content, and a
nonempty period does not reproduce the correct-digit benefit. The
punctuation control is not semantically or distributionally equivalent to
a number. No exclusive internal mechanism, general causal necessity,
residual crystallization or held-out generalization is established.

The result also exposes error propagation: an unverified supplied
intermediate leads to a wrong final answer on every tested triple.
It does not rehabilitate the retired certificate-to-truth claim. No
actual admitted replay, admission learning, body search, served chart,
threshold change or measured compute saving occurs in this diagnostic.

Native and independent Transformers f16 reference agree54/54. The independent
standard-library audit verifies all54 input assemblies, arithmetic labels,
reference predictions, counts, source-capture cardinality and23 frozen
input/code/model/runtime hashes. Native vocabulary rows dispatched total
13409280. The budget is54 new native plus54 reference forwards; prior
baseline results are reused. No extra controls or model runs follow.
`results.json`, `verification.json`, `run_manifest.json` and the exact
selection files are the directly reviewable evidence. Raw captures and
reference are distributed in a supplemental archive, not ordinary Git.
