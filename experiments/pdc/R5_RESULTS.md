# R5: bounded addition theory, stop rule triggered, 2026-10-24

**Empirical negative result.** The preregistered claim that high-margin,
non-abstaining commitments have hallucination below 10% is retired for both
readouts on this GPT-2 domain. The fixed wide-miss threshold was 25%; both
readouts exceed it. R5's epistemic stop restriction applies. A later R8 arm
may serve measured host agreement but may not advertise theory soundness on
the strength of this certificate.

Frozen preregistration `R5_PREREG.md`, SHA256
`7e9336b016c679d1b340d9b24efaa9ea08336455d1c467d5838e713cea0452c7`.
Domain SHA256 `3dc65f9f9483b52dc803de69dd6bbae77810319f513edf927ff2539d843c39a3`.
Selection SHA256 `8f16be9fcb6e0a2cd9c392ade22005d7481dea024fff7cc25d2c4b89b681a78d`.
No threshold or body generator was changed after observing outputs.

| Quantity | Viterbi | Marginal, T=1 |
|---|---:|---:|
| Evaluation contexts | 75 | 75 |
| Certified commitments | 8 | 10 |
| Abstentions | 67 | 65 |
| Theory-affirmed commitments | 1 | 1 |
| Hallucinations among commitments | 7 | 9 |
| Hallucination rate | 87.5% | 90% |
| Raw readout/theory agreement | 12/75 = 16% | 15/75 = 20% |
| Committed/full-vocabulary host agreement | 5/8 = 62.5% | 6/10 = 60% |
| Committed/covered-candidate host agreement | 8/8 = 100% | 10/10 = 100% |

There are 47 misses (no recorded body strictly decides the true atom), three
fragile contexts, and three fragile truths. Misses are counted independently
of readout choice or abstention. Viterbi abstains 34 times on pairwise fidelity
and 33 times on its readout margin. Marginal abstains 27 times on fidelity,
37 times on readout margin, and once on body margin. These reason counts
follow the preregistered checking order.

## What was measured and checked

The external theory is ordinary bounded integer addition with operands 0–4
and answers 0–8. Nine distinct leading-space answer tokens form the covered
candidate set. Each context has three fixed source coalitions per atom:
all sources, all attention-block writes, and all MLP-block writes. Thus the
75-context bundle contains 2,025 bodies, all generated before the host token
is considered. Both readouts and the whole body table are retained.

The native observer records actual block writes inside GPT-2's forward path.
Observed and unobserved hidden states are identical. An independent NumPy
forward agrees with all 75 final-position full-vocabulary predictions.
Maximum raw source-sum error is 1.682e-5; measured final-LN folding error is
1.702e-4; maximum full-coalition candidate-score error is 2.399e-4. These
floating-point effects are reported, not represented as algebraic equality.
Actual native candidate scores enter the finite pairwise drift check.

The independent bundle auditor checks all 2,025 annotations from the stored
source vectors and model frame, recomputes both readouts, verifies every
commit/abstain decision, and checks the candidate-wise host consequence.
Maximum stored annotation discrepancy is 0. The certificate's scope is the
nine named candidates. It does not certify agreement with an unrestricted
vocabulary argmax; that separate agreement is much lower, as the table shows.

The symbolic pairwise guard is **proved** in i-orca `PDC_Admission.thy`,
`pairwise_source_preserves_decision`, strict session `PDC`, zero holes.
That theorem concerns relative scores, not integer addition or truth.
Concrete guards are numerical artifact checks, not kernel proofs about all
floating-point operations.

An explicit error is case `t2-a0-b0`: theory says 0+0=0, both readouts commit
atom 1 using the MLP coalition, native full-vocabulary token is also 1.
Body margin is 3.040224 and minimum pairwise guard slack is 1.040681. The
certificate survives while the answer is false. An attention coalition in
the same chart decides the true atom, so this context is not a miss. This
distinguishes a theory error from a missing recorded explanation.

## Artifacts and untouched slice

- `r5_domain.json`: all 100 fixed prompt/theory manifests; template 3's 25
  contexts remain uncaptured and ungraded for the final held-out audit.
- `r5_selection.json`, `r5_reads.json`, `r5_capture.log`: only the 75 R5 contexts.
- `r5_bundle.json`, `r5_bundle.log`: source/body program and two certified queries.
- `r5_audit.json`: independent source/body/readout/guard audit and input hashes.
- `r5_theory_results.json`: separate external-theory judgments and counts.

The bundle contains memory tag `none`; no memory or learned-admission result
is implied. It is a finite exported artifact, not an online service or a
demonstration of a vocabulary-row compute saving. R6–R8 remain unperformed.
The negative result is scoped to the frozen model, prompts, coalitions,
candidate set and gamma; it is not an impossibility theorem for all models
or all certificates. Changing the domain is a new preregistered arm, not a
repair to this result.

Reproduction from fieldrun root (the domain builder additionally needs
`tokenizers`; its frozen outputs are already retained):

```sh
cargo build --release --no-default-features
target/release/fieldrun --bundle bundles/gpt2 --pdc-read-selection experiments/pdc/r5_selection.json --pdc-read-dump experiments/pdc/r5_reads.json
python3 scripts/pdc_add_bundle.py --dump experiments/pdc/r5_reads.json --domain experiments/pdc/r5_domain.json --selection experiments/pdc/r5_selection.json --bundle bundles/gpt2 --out experiments/pdc/r5_bundle.json
python3 scripts/pdc_bundle_audit.py --bundle experiments/pdc/r5_bundle.json --domain experiments/pdc/r5_domain.json --capture experiments/pdc/r5_reads.json --model-bundle bundles/gpt2 --out experiments/pdc/r5_audit.json
python3 scripts/pdc_theory_check.py --bundle experiments/pdc/r5_bundle.json --domain experiments/pdc/r5_domain.json --out experiments/pdc/r5_theory_results.json
python3 -m unittest discover -s scripts -p 'test_pdc_*.py' -v
```

Ten focused tests pass, including coverage mismatch, target drift, strict
boundary rejection, independent per-rival guards, the distinction between
abstention and a miss, rejection of a commitment without a body decision,
and verification that theory labels follow operands rather than host outputs.
