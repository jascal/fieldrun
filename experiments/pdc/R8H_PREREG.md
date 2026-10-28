# R8-H preregistration: host-agreement-only bundle service, 2026-10-25

This is the continuation explicitly permitted by the R5 stop rule. The
original truth-sound R8 gate remains unmet; no theory-soundness claim is
revived. R6/R7 remain separate unperformed experiments. Preserve the R5
75-context results and the 25 uncaptured held-out contexts without retuning.

## Fixed policy and data

Use the same pinned GPT-2 bundle and the 75 R5 contexts only. Keep the nine
answer atoms, three source-coalition body generators, T=1, readout/body
margin gamma=0.5, and both readouts. No body or margin is selected from R8-H
host outputs. The full-vocabulary R5 predictions are the frozen evaluation
reference, never a runtime selection input.

Add a fixed lexical probe set to the certificate coverage, chosen by this
rule before its model scores are inspected. Decode each vocabulary token
with the pinned tokenizer. Include all nonempty decoded strings of at most
three ASCII characters which contain no ASCII letters; also include a
single ASCII letter with zero or one leading space, all single ASCII
characters, every tokenizer-added special token, and the nine answer tokens.
Sort/deduplicate token IDs and freeze the resulting list and hash. The rule
is based on token text, not observed winners. Do not expand it after a miss.

The derive query still compares the nine answer atoms. The certificate
compares its chosen body's target against *every* probe token, using the
same finite pairwise gap/drift guard. Coverage explicitly contains the
derive rivals and the additional probes; it is not the entire vocabulary.
The chosen body must strictly decide the target on that expanded set. A
failed guard abstains; an out-of-probe vocabulary winner remains a possible
failure of full-host agreement and must be counted as such, not certified
away. Record the two scopes separately.

## Actual runtime and accounting

The service runs the native frozen forward and retains the complete source
program. It evaluates only probe unembedding rows with
`Bundle::rowdot_f32_subset`; it must never invoke a full-vocabulary scorer
or the independent reference forward on the service path. Python body
readouts load only the selected unembedding rows and final-LN parameters.
No cached answer, theory answer, or full-vocabulary winner may select the
returned atom. Every response contains the bundle or an explicit abstention.

Export two aligned native arms over identical contexts: full-vocabulary
audit baseline and probe-only service. Each uses one forward per context
without the second observer-validation forward. Validate that the service's
probe scores and source writes equal the matching baseline slice. The
existing observer/reference gates remain their own earlier evidence.
Measure one fixed batch per arm, full baseline first then probe service;
report per-context native forward-plus-decode timing and total wall time,
with loading/serialization caveats. Do not claim a speedup from a single
noisy run. The required compute measurement is actual output-row dispatch,
not an extrapolated FLOP count or a count obtained from an offline capture.

Report distinct output vocabulary rows and native output-row dot calls per
context, plus the three body/probe score tables (3 times the probe count)
separately. Body projection work is shared by both arms. Do not count
model loading or embedding lookups as skipped computation. No transformer
layer or context position is skipped in this arm.

## Advance gates and retained negative outcomes

For each readout separately:

- Non-abstaining full-vocabulary host agreement must be at least 95%.
- There must be at least one commitment; zero coverage is untested, not pass.
- Native output-row calls must be reduced, and score equality must hold.
- Retain all R5 theory verdicts and separately grade the new commitments;
  hallucination is reported with no success bar or epistemic claim here.

A miss retires this service policy at these parameters. This host-only arm
cannot close the original below-10%-hallucination gate by changing its
name. The 25 reserved contexts remain untouched until a final policy and
learning/memory disposition are frozen; this is not the criterion-9 audit.
