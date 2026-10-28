# Qwen native-singleton diagnostic: matching decisions, no admissions

This separately frozen saved-data diagnostic scores all65 actual native
sources on each of the100 previously observed Qwen cases. **Every case has
at least one singleton whose strict candidate argmax matches the host, but
none is admitted by either the exact pairwise or centered-tail procedure.**
No model forward, optimizer step, new prompt, larger coalition or threshold
sweep was run. This is not fresh heldout confirmation.

| Original split (now saved data) | Any matching singleton | Any singleton margin >0.5 | Exact-pairwise admission | Centered-tail admission |
|---|---:|---:|---:|---:|
| Development | 50/50 | 48/50 | 0/50 | 0/50 |
| Heldout | 50/50 | 49/50 | 0/50 | 0/50 |
| All | 100/100 | 97/100 | 0/100 | 0/100 |

No singleton has positive minimum pairwise slack on any case. Conditional
agreement among admitted singletons is undefined, not100%. The original
host remains correct on all100, with its frame and outputs fixed. These
unchanged outputs are an input invariant, not an intervention result.

## What the guard actually asks

Let h be a full-host score gap to one rival and p the singleton's gap.
The exact sufficient condition is abs(h-p)<p. It is equivalent to
0<h AND h<2p: the source gap must exceed half the host gap for EVERY rival.
The strict Isabelle lemma `pairwise_gap_guard_iff`, added to
`i-orca/examples/pdc/PDC_Admission.thy`, checks this algebra with no holes.
It explains the difference between choosing the same atom and satisfying
this particular sufficient certificate. It does not prove causal adequacy.

Here positive singleton decisions exist in every case; in97 cases the
absolute0.5 floor also can be met. Refusal therefore cannot be explained
as a universal absence of matching native singleton decisions, and the
absolute floor alone is not the complete explanation. The pairwise drift
condition is not necessary for two already-observed scores to have the
same argmax. Even the less conservative exact guard fails on these frames.
No claim is made that a stronger model always worsens the guard, that
matched singleton scores are causal substitutes, or that admission is
impossible. Individual-source margins can be insufficient relative to
full-host margins even when both select the correct answer.

The centered comparator is a subset of exact admission as expected. The
predeclared local positive bar (>=10% exact admission) is not met. The fixed
singleton procedures are negative on these frames. Size-2/larger bodies,
source rescaling, alternative certificates, learning and residual replay
are untested and were NOT added after seeing the result. Source rescaling
would change the measured write and cannot silently count as this experiment.
This is not the original free-frame negative-domain Farkas result.

## Evidence and verification

PREREG.md, the executable, raw captures, original manifests and converted
frame hashes were frozen in manifest.json before source scoring. The frame
identity matches the original Qwen run. `results.json` stores all6500 source
verdicts with margins, minimum pairwise slack, centered-tail range and
covered target, plus first-witness rules. There are no passing witnesses.
The independent checker scales frame rows before multiplication, uses torch
float64 instead of NumPy and checks the alternative 0<h<2p formulation.
All6500 verdicts agree; maximum numerical discrepancy is5.69e-14. Frozen
input/source/model hashes and original-frame identity pass verification.

Large input captures remain in the immutable Qwen evidence release:
https://github.com/jascal/pdc/releases/tag/pdc-qwen35-4b-v1.
The new protocol, scripts and complete small diagnostic outputs are directly
in fieldrun PR #148. They are not retroactively inserted into that archive.
Full-program Qwen coverage100/100 remains true; compact certified singleton
coverage is0/100. These are different measurements and neither erases the other.
