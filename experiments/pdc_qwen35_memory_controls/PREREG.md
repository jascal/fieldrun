# Qwen memory-content controls, 2026-10-26

This is a new, bounded development diagnostic revising the interpretation of
the completed R6 follow-up. It does not rerun or extend the stopped R6/R7
arms, change their results, select a better suffix/body/frame, or revive an
epistemic claim. The existing 27 triples and their empty-history (12/27) and
correct-intermediate (27/27) outcomes are already known. Those comparisons
confound content with assistant-turn nonemptiness. The new control outcomes
have not been inspected. Publish this protocol, exact input IDs and script
before native inference. This is not preregistration of fresh confirmation.

## Fixed intervention and budget

Reuse exactly the same 27 triples a,b,c in {0,1,2}^3, nine first-hop prefixes,
second user question, template, thinking-disabled assistant prefix and
2261 probes from the completed Qwen R6 run. Do not alter any prefix/suffix.
For each triple insert exactly one token into the prior assistant answer:

- Neutral control: literal period `.`. Validate it is one token. This gives
  a nonempty assistant turn without naming an arithmetic intermediate.
- Wrong-digit control: decimal digit w=(a+b+1) mod 5, which always differs
  from the correct intermediate and is one token. Keep original user facts.
  Its apparent downstream sum is w+c, independently computed for grading.

Correct and wrong numeric answers have identical token counts and chat
structure. Neutral punctuation is a separate imperfect format control;
its distribution and semantics differ from a numeric answer. No control
proves causal necessity or identifies a residual memory mechanism.
Check canonical tokenize(text) equality, exact prefix/suffix preservation,
54 unique new inputs, and no identity with prior R6 inputs before execution.
No fresh prompt, operand, token, suffix, body or threshold search is allowed.

Reuse pinned Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a,
original f16 bundle and native runtime, frozen backbone/output frame, no
learning. Native budget54 full forwards, capturing the unchanged65 sources
and full-vocabulary winner. Reference budget54 independent Transformers
f16 forwards on identical IDs, run after native model exits. Require54/54
reference agreement for scientific interpretation. Engineering failure is
reported, not hidden by changing the protocol. Source captures are retained
for reconstruction, not used for body selection. No served chart or compute
saving is claimed by this diagnostic.

## Metrics and decision rule

Report paired counts over27 triples, not108 independent observations:
full-host theory accuracy for original empty and correct-digit arms and new
neutral and wrong-digit arms; on wrong-digit inputs also report fraction
whose winner equals w+c; preserve every token/ID and reference prediction.
All correctness uses integer arithmetic on original a,b,c, not the supplied
wrong digit. Predictions outside0..8 count wrong rather than being remapped.

Evidence for content sensitivity requires BOTH correct-digit accuracy minus
wrong-digit accuracy at least0.20 (at least6 paired-count difference here)
and following-wrong count at least14/27. If either fails, the content-use
hypothesis is not supported on this diagnostic; do not infer content is
irrelevant generally. Separately report neutral-minus-empty accuracy: an
increase at least0.05 (at least2 cases) demonstrates that nonnumeric context
also changes task success, without identifying a pure formatting mechanism.
All regions/counts are published regardless of the bars; no significance,
held-out generalization or exclusive causal attribution is claimed.

Stop after54 new native and54 reference forwards and an independent count /
input/hash audit. No additional controls, alternate wrong values, checkpoint
selection, thresholds, learning or follow-up runs are authorized by this
protocol. No consumed original final25 inputs are loaded. No actual admitted
residual replay or admitted-stratum redundancy is tested. The previous full
PDC goal remains incomplete and no later positive gate is inferred here.
