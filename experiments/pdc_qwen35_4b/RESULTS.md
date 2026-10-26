# Qwen3.5-4B PDC comparison: positive full-program coverage (2026-10-25)

The fixed stronger-model comparison passes its preregistered local gates.
Both readouts commit correctly on all 50 heldout cases. **Every witness is
the full source program**. This is the first tested nonzero-coverage regime
in this PDC sequence, but it does not establish a smaller sufficient circuit,
crystallization, useful learned admission, multi-hop reasoning or compute saving.

| Split | Host correct | Viterbi committed / affirmed | Marginal committed / affirmed | Hallucinations | Full-host agreement |
|---|---:|---:|---:|---:|---:|
| Development | 50/50 | 50 / 50 | 50 / 50 | 0 both | 100% both |
| Heldout | 50/50 | 50 / 50 | 50 / 50 | 0 both | 100% both |

There are no abstentions, misses or fragile truths in either split. Useful
coverage (affirmed commitments / all cases) is 100% for both readouts.
These are 100 contexts with two readouts, not 200 independent test examples.
The holdout is over two new instruction wordings with the same 25 arithmetic
facts, not unseen facts or a broad mathematics benchmark. The old GPT-2
final-25 slice is not reused. No prompt, body, margin or model selection
changed after seeing scientific outputs.

## What improved, and what did not

The new host answers this first-token task correctly. Its complete source
body wins every chart and supports a valid finite-probe certificate. This
removes the wrong-host-answer failure on these inputs. It also makes much
of the observed certificate success close to a reconstruction check of the
full host score: this is NOT evidence that PDC discovered a compact explanation.
The attention-only and MLP-only bodies never win. Both raw host and committed
answers already have zero hallucinations, so no reduction over an uncertified
host baseline is demonstrated. No layers, positions or native output rows
are skipped. Original R8 and the full north star are not closed by this run.

The model, architecture, instruction training, tokenizer and chat format
all differ from GPT-2; the experiment does not isolate parameter count as a
causal treatment. The retired GPT-2 epistemic claim and original negative
R4/R6/R7/R8 dispositions remain unchanged. Passing this local empirical gate
does not turn the score-preservation theorem into a truth theorem.

## Model and gates

Official Qwen/Qwen3.5-4B revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, dense text path, float16 weight bundle,
CPU, no routing/pruning, explicit thinking disabled. Four fixed instructions
*25 ordered operand pairs in 0..4 generate 100 chat inputs before inference;
50 development and 50 heldout by wording. The first output token is graded,
with no forced digit, multi-token extraction, correctness filtering or learning.
Nine digit query atoms, 2261 fixed tokenizer-text certificate probes, T=1,
gamma=0.5 and the original exact pairwise guard are unchanged during the run.
Full-vocabulary native output has 248320 rows; covered score decisions and
unrestricted host comparison remain separate. External addition grades truth.

Native observation captures embedding and every attention/MLP residual write
(65 sources). DeltaNet gates remain uninterpreted; ordinary attention-key PIC
identification is not claimed for linear attention. Final RMS folding uses
1+norm_weight, no mean subtraction or affine beta; closure stays in the
body/host drift, not a fabricated native source.

- Eleven Qwen unit tests, including observer/source identity, pass.
- Thirteen existing PDC tests pass; tiny default native/reference 60/60.
- Tiny observed reference sample 3/3 and real engineering sample 5/5 match,
  with zero observer effect and recorded precision/reconstruction errors.
- Independent scientific reference matches **100/100 full-host winners**.
- Independent certificate audit checks **678300 body/probe annotations** and
  **200 readout commitments**, maximum independent score discrepancy
  **1.60e-14**; all frozen input/code/model/binary hashes match.
- Largest raw source-sum error 3.621e-6, RMS-fold error 2.443e-5, and full-body
  native-score error 6.230e-5. Those numeric errors are separate from agreement.

Native scientific forwards: 100. Independent scientific reference forwards:
100 more, verification-only and never selection inputs. Engineering and
tiny fixtures are separate budgets. Ten native batches take 562.52 seconds
including model loading and capture serialization. This is descriptive
CPU timing, not a general latency claim. Native dispatch scores 24832000
output rows; body/probe annotations are additional offline work. No native
subset-scoring savings are claimed.

## Evidence and reconstruction

PREREG.md fixes policy, exact arm semantics, timing disclosures and gates.
run_manifest.json freezes checkpoint/source/input/binary hashes before the
first scientific batch. domain.json, probe.json and selection.json retain
all inputs. Ten raw capture files, scientific_reference.json and bundle.json
retain native writes/scores, the independent reference and both readouts with
body, margin, fragility, memory=none and separate external-theory verdicts.
results.json is the complete numerical summary. verification.json is the
independent audit report. Witness distribution is retained separately.

Large evidence files and the source snapshot are in the supplemental GitHub
release: https://github.com/jascal/pdc/releases/tag/pdc-qwen35-4b-v1
They are separate from the immutable original v0.10 archive. Model weights
are external prerequisites; the pinned checkpoint and converted model hashes
are in run_manifest.json. Reproduction instructions are in README.md.
Stop after this frozen comparison: a subset service or more demanding
reasoning/memory task requires a separately fixed protocol and appropriate
fresh evaluation, not a post-hoc claim that this run already established it.
