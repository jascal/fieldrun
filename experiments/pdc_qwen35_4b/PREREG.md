# Qwen3.5-4B frozen-model comparison, 2026-10-25

User requests testing the PDC approach on the stronger Qwen3.5-4B model.
This is a new model/domain arm, not a rescue or retuning of GPT-2 results.
Original R4–R8 negatives, v0.10 archive and consumed final-25 prohibition
remain intact. No outputs of this checkpoint on these cases have been read.

## Model, computation and engineering gate

Official Qwen/Qwen3.5-4B revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, text-only native inference,
float16 converted weights, no neuron routing/pruning or quantized KV.
Tokenizer/chat template from the same revision, explicit thinking disabled,
greedy first answer token. No generated reasoning, forced answer tokens,
correctness filtering or feedback-conditioned prompts. Use CPU within the
available machine; timing is descriptive, not a speedup claim.

Extend fieldrun's native Qwen3.5 hybrid walk to observe the final-query
embedding plus every attention-block and MLP residual write. The hybrid
linear-attention gates remain uninterpreted; do not claim R2's ordinary
softmax-key identification for DeltaNet. Existing full attention and
DeltaNet computations stay unchanged. Record actual final residual and
post-final-RMSNorm hidden. RMS folding is d_j*(1+w_norm)/sqrt(mean(r^2)+eps),
with no mean subtraction or affine beta. Preserve numerical source-sum
closure explicitly; it is not an eligible native singleton or edited write.

Before scientific capture, validate on a deterministic tiny Qwen3.5 hybrid
against the independent transformers reference: top-1 agreement 100%,
observed/unobserved native hidden bit identity, complete ordered source
program and bounded independently reported reconstruction error. Then
check a disjoint fixed engineering subset (five literal arithmetic prompts
0+0 through 4+4) on the real checkpoint against transformers using the
same model weights/precision convention. Any top-1 mismatch or observer
effect blocks scientific interpretation; investigate and retain the failed
check instead of dropping a context or selecting another precision for quality.
Engineering prompts never enter the statistical evaluation or template choice.
Engineering numerical tolerances: source-sum and RMS-fold max errors each
<1e-4*(1+max absolute corresponding native vector); full-body/probe score
error <1e-3*(1+max absolute native probe score). Native/reference logit error
is reported separately because transformers uses float16 activations while
the native kernel accumulates in float32 from identical float16 weights.
The top-1 engineering gate remains exact, not a tolerance on winner IDs.

## Fresh inputs and fixed split

Generate 100 cases before any model inference: four fixed instruction
wordings, each with all 25 ordered a,b pairs in 0..4. Wording templates 0–1
are development (50); 2–3 heldout (50), opened only after input/code/policy
hashes are frozen. These are new instruction/chat inputs; no old GPT-2
final-25 prompt or native capture is loaded. Repeated arithmetic facts across
wordings are deliberate: the holdout is template-level, not an unseen-fact
claim. Exact templates, IDs and tokenizer outputs are stored in domain.json.
No learning, threshold selection, template revision or best-body selection
from development outcomes is allowed in this comparison. Run each case once.

User asks for only the single digit answer, no explanation. Leading-space
GPT-2 token IDs are never reused; validate the nine Qwen digit token IDs
from its tokenizer before capture. If a digit is not one token the protocol
is inapplicable; do not silently teacher-force a prefix to make it one.
A non-answer first output token counts as a host accuracy failure. There is
no post-hoc extraction of a later correct digit.

## PDC policy

Keep T=1, gamma floor 0.5, Viterbi AND marginal, and the same three
pre-named bodies: complete source program; all attention-block writes
(including DeltaNet); all MLP writes. Count each body once and bias zero
for this model. Query rivals are the nine answer atoms. Freeze expanded
certificate coverage by the R8-H tokenizer-text lexical rule, plus those
nine tokens. No probe expansion based on observed winners. Keep query and
certificate coverage distinct from the unrestricted full-vocabulary host.

Use the existing exact pairwise source-to-host gap-drift guard and readout/
body margins at gamma 0.5 (`pdc_certificate.certify`), unchanged. A certificate
preserves only covered score decisions. External integer addition supplies
truth independently, after readout/selection. Full-host agreement is checked
against the native all-vocabulary decode, never used to choose the chart atom.
All non-abstentions retain readout/body/margin/fragility/memory=none and theory
verdict, with complete source/body tables. Explicit abstention stays explicit.
This single-position comparison does not establish two-hop memory transfer.

## Metrics and frozen interpretation

Report development and heldout separately, and both readouts separately:
full-host first-token arithmetic accuracy; commitments/abstentions; affirmed/
hallucinated commitments; useful coverage = affirmed commitments divided by
ALL cases; full-host agreement among commitments; candidate-covered agreement;
misses and fragile truths. Undefined conditional rates at zero commitments
remain undefined. Report errors/argmax agreement for engineering gates, raw
and RMS-folded reconstruction error and full-body/native score discrepancy.
Record all dispatched native output-row calls and body/probe annotations,
forwards and wall time. This baseline scores the full vocabulary; do not
claim subset service savings or close original R8 from offline projections.

Each heldout readout is a local positive only if host accuracy >=95%, useful
coverage >=20%, commitment full-host agreement >=95%, and hallucination among
commitments <10%. Negative if useful coverage <10% OR defined commitment
theory accuracy <70%; otherwise inconclusive. Record every outcome, not the
better readout. This finite sample is not a generalization theorem. A positive
would motivate a separately preregistered real subset service/two-hop arm;
a negative retires this fixed model/body/policy combination on these templates.
Neither outcome revives GPT-2's retired epistemic claim or proves impossibility.
Stop after this comparison; no prompt/body/margin sweeps or new training.


## Engineering timing disclosure

The explicit numerical reconstruction tolerances above were recorded after
seeing the tiny engineering fixture, before comparing the real-checkpoint
engineering scores and before any 100-case scientific capture. They are
engineering checks, not tuned scientific thresholds. The fixed scientific
policy remains gamma=0.5 and T=1 throughout. Ten native capture batches of
ten cases are an artifact-persistence choice, not adaptive selection: all
100 cases are frozen before the first batch, with no intermediate policy
or prompt changes. Each scientific case still has exactly one native forward.


## Independent scientific-input faithfulness budget, frozen before capture

In addition to the 100 native scientific forwards, run 100 independent
Transformers reference forwards on the identical frozen inputs, sequentially
after native capture so two full models do not compete for RAM. These are
verification calls, never selection inputs or retries. Count native and
reference calls separately. Require 100/100 top-1 agreement for scientific
interpretation; otherwise retain all cases and report a failed faithfulness
gate. This expands verification, not the native policy's compute budget or
its allowed prompts. Report reference/native logit error separately because
reference float16 activation arithmetic differs from native float32 accumulation.
No outputs of the 100 scientific cases have been read before this addendum.
