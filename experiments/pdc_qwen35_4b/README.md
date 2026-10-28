# Qwen3.5-4B frozen PDC comparison

This follow-up was requested after the GPT-2 negative results and the saved-
frame R7 diagnostic. It uses Qwen/Qwen3.5-4B revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, converted to float16, with a fixed
100-case instruction/chat domain. It is not an intervention isolating
parameter count: architecture, instruction training, tokenizer and chat
format also differ from GPT-2. Read PREREG.md for exact scope and gates.

`domain.json` fixes four instruction templates, 25 arithmetic facts per
template, development/heldout allocation and rendered chat inputs before
any scientific forward. The holdout is over wordings, not unseen facts.
The old GPT-2 final-25 slice is not loaded or reused. No learning or tuning
occurs. The first output token is graded, so format failures count as such;
this is not an unconstrained multi-token arithmetic competence benchmark.

`probe.json` freezes 2261 tokenizer-text probes. The derive query compares
nine digit atoms; the certificate covers all probes. Neither is the whole
248320-row vocabulary. Native full-vocabulary output scores provide the
independent host label for agreement measurement. The certificate is not a
truth theorem; a separate integer-addition checker supplies the grade.

## Native source interface

The dense-Qwen-only `--pdc-qwen-source-dump` flag and explicit JSON selection
observe the same forward used by the default native decoder. The source
program is embedding plus the attention-block output and MLP output for
each layer: 65 vectors on the 4B checkpoint. Attention includes both gated
DeltaNet and gated full-attention blocks; no ordinary softmax read-PIC
identification is claimed for DeltaNet. Gate values and the final RMS scale
are the observed full-host values, not recomputed causal deletions.

Schema `pdc.qwen35-source-capture.v1` retains input IDs, ordered raw source
vectors, final residual, final RMS-normalized hidden, selected probe scores,
full-vocabulary winner, actual output-row dispatch count, optional observer
check and descriptive timing. RMS folding uses `(1+norm_weight)` and the
actual final residual's RMS; there is no LayerNorm mean subtraction or beta.
Numerical source closure is preserved through host score drift, never
promoted to another native source. Default generation has no source capture.

The engineering gate checks observer identity, complete source ordering,
source/RMS/score reconstruction and exact top-1 reference agreement before
scientific capture. Tiny default prediction also matches the independent
Transformers reference on 60/60 positions. All 11 Qwen unit tests and 13
existing PDC tests pass. The real-checkpoint engineering gate is 5/5, with
zero observer effect; its numeric discrepancies are explicitly reported.

## Files and review

- `run_manifest.json` freezes scientific input, code, model and binary hashes.
- Ten `batch_*_capture.json` files retain one native forward per scientific
  case; batching only persists progress and reloads the model.
- `scientific_reference.json` is a separate 100-forward verification budget,
  sequential after native execution to keep memory bounded.
- `bundle.json` retains typed readout, body tables, certificates/abstentions,
  fragility, memory=none and separate theory judgments, linked to capture hashes.
- `results.json` reports both readouts and both splits, with undefined
  conditional rates at zero coverage left undefined.
- `verification.json` independently reconstructs all body/probe tables and
  exact pairwise guards and checks the 100 native/reference host winners.

The baseline scores the whole vocabulary. Offline probe tables do not save
native scoring work; no subset-service speedup or R8 success is claimed.
Model weights and environments are external prerequisites, not Git content.
Large captures will be linked as evidence assets rather than committed to git.
The original v0.10 evidence archive and all stopped-stage results are immutable.
