# R5 preregistration: finite addition theory, 2026-10-24

Frozen before inspecting any host predictions on this domain. Keep the R2
GPT-2 bundle (manifest SHA256 322d79629f2e121c446560f57b71c5199e0d8a2a912c3a396137ca7eb207692c,
weights SHA256 4edbcffed3bc19d571e679896ab3efebaa3802f8d7c9db3bdad31ff2416490f1).
No gates, weights, temperature, body generators or thresholds may be tuned on
this run. The domain is separate from R3's token-ID copy contexts and R4's
English source sample.

## Theory, domain and untouched slice

External theory is bounded integer addition: for a,b in {0,1,2,3,4}, theory
affirms exactly the atom n=a+b in the nine-atom candidate set {0,...,8}.
No world knowledge or host agreement is part of the theory. Atom n denotes
one next-token answer with leading space, tokenized from `" "+str(n)` by the
pinned GPT-2 tokenizer. Require each to be a distinct single token before
capture; otherwise stop to repair this representation, without inspecting
host tokens. Retain the exact prompt text, token IDs, integer operands,
theory answer and candidate-token map in an immutable domain JSON.

100 contexts = 25 ordered operand pairs times four templates. Fixed examples:
`0 + 0 = 0\n1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n4 + 4 = 8\n`.

Templates, each followed by the query `{a} + {b} =`:

0. Examples only, then query.
1. `Complete the addition equations.\n` then examples and query.
2. `Use ordinary integer addition.\n` then examples and query.
3. `Solve the final equation.\n` then examples and query.

Templates 0–2 (75 contexts) are the R5 development/evaluation domain. Template
3 (25 contexts) is the untouched criterion-9 slice: write its manifest now,
but do not capture, grade or train on its host outputs before the final R8/9
audit. All numeric thresholds are fixed here, so none are selected on either
slice. A later change of domain/threshold is a new preregistered arm.

## Bodies and certificates

At the final prompt position capture the actual native additive residual
writes: embedding, every attention block (including its output bias), and
every MLP block (including bias). Observe them inside the same forward path;
verify observed/unobserved hidden outputs are identical. Retain raw source
writes, final pre-LN residual and final hidden vector. Independently compare
the native full-vocabulary token at that position with the NumPy reference.
Keep numerical summation/LN-folding error separate from decision agreement.

Fold each source by its *measured* final-LN centering, scale and gamma. Treat
the final-LN beta read as the once-per-body bias. This is an attribution frame,
not an intervened forward. Each candidate has exactly three fixed bodies:
the full source coalition, all attention-block sources, and all MLP-block
sources. No body is generated, deleted or chosen after seeing a host token.
Its bracket is the coalition's candidate incidence plus the bias. Retain all
27 bodies and their source sets. Use Viterbi and marginal T=1 from the R1
executor. A marginal query records its complete body set and the best-bracket
body witness separately; one witness alone is not its aggregate explanation.

For each readout, ties abstain. Select the best-bracket body for the unique
winning atom, breaking a body tie by lexicographic body ID. The recorded body
must itself strictly decide that atom against all nine candidates. Fix
gamma=0.5. Both the readout margin and the body margin must exceed gamma.
Certificate coverage must equal the nine fixed candidate tokens, including
the target; a mismatch abstains.

Use exact finite pairwise predicates, not a uniform bound over the vocabulary.
For each rival v compute body gap g_v and measured source-to-host gap drift
e_v = (L_host(t)-L_body(t)) - (L_host(v)-L_body(v)). Require
`g_v > abs(e_v)` for every rival. This is the margin > 2delta guard on each
two-candidate set with delta_v=abs(e_v)/2; store all gaps, drifts and positive
slacks. The bound includes actual native f32 score/summation discrepancy via
the measured host scores, not a guessed global epsilon. It certifies host
agreement on exactly the nine-atom domain, not the entire vocabulary. Report
full-vocabulary host agreement independently. No theory answer enters this
certificate or the readout selection. Reject nonfinite data.

## Counts, hypothesis and stop rule

Per readout report committed, abstained (with reasons), affirmed,
hallucination, and hallucination rate among non-abstaining commitments.
Report misses per context when no recorded body strictly decides the true
atom, independently of whether a readout chose it. Report fragile truth when
marginal selects the theory atom and Viterbi does not; also report the
unconditional fragility rate, theory accuracy and full-host agreement.

The advance bar is hallucination rate **strictly below 10%**, at gamma=0.5,
for each readout separately. Zero commitments is untested/vacuous, never a
passing bar. A rate at or above 10% retires the claim for that readout/domain.
A rate at or above 25% (the fixed meaning of a wide miss) retires the claim
that high margin is an epistemic instrument on this domain and activates
the R5 stop restriction: R8 may serve host agreement but may not advertise
theory soundness. Report empirical counts; no statistical or universal
truth guarantee follows even if a finite rate passes.

R6 will compose two applications of this same finite addition theory, with
operands chosen so each hop remains inside the theory. Its memory arms and
admission split require their own preregistration; this document does not
authorize assuming a favorable outcome or treating a refusal as impossible
over all faithful frames. R7 must keep these measured writes/gates frozen.
