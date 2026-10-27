# Replay review fixes

The NumPy Qwen DeltaNet wrapper, its generic key-normalization path and the
cross-check helper now use x/sqrt(sum(x*x)+1e-6), matching the Rust kernel and
the pinned Transformers low-norm golden. The Rust numerical kernel and golden
are unchanged. Its comments now distinguish the loose historical O(1) fixture
from the discriminating low-norm normalization pin.

A NumPy test against that same low-norm fixture passes at absolute tolerance
1e-7. The live normalized-kernel cross-check against Transformers has maximum
absolute error 8.60e-8 across four seeds. All four NumPy recurrence properties
pass. This is a kernel check, not a new full-model or scientific comparison.

The saved-replay checker now requires baseline/zero/nonzero arm semantics,
aligned contexts, embedding-width finite payloads, actual zeros in the zero
arm, and actual nonzeros in the nonzero arm. Regression tests reject fake-zero,
fake-nonzero, nonfinite, wrong-width and crossed-context controls.

The saved 9-case tiny and 5-case real packets pass. Their reports explicitly
identify saved-artifact comparison and deny execution of Transformers or the
normalization golden. A green artifact report cannot certify a current kernel's
epsilon placement. The real packet retains hidden-state differences up to
0.03407; these are reported differences, not exact full-model hidden parity.
The low-norm Rust test remains the separate executable normalization gate.

No capture, filler, threshold, model or scientific contrast was changed. No
additional full-model forward was requested for this review repair.
