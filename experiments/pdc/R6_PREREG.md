# R6 preregistration: two-hop finite addition memories, 2026-10-25

This is an empirical memory comparison under the R5 stop restriction. It
does not revive margin-to-truth soundness. R8-H's zero-coverage result is
retained; neither it nor R5 confirms or retires a two-hop memory hypothesis.
No R6 hop-2 output has been captured or inspected before this registration.

## Task, contexts and shared host

Use the pinned GPT-2 bundle and tokenizer from R5, with frozen weights and
full computation. All 27 triples (a,b,c) in {0,1,2} cubed are included, in
lexicographic order. The finite theory affirms the final atom a+b+c, which
lies in {0,...,6}, and intermediate a+b in {0,...,4}. There is no threshold
selection from the 25 reserved R5 template-3 contexts; they stay untouched.

Hop 1 uses exactly R5 template 0's prompt for a+b. These nine first-hop
prompts and full-host outputs already exist in the frozen R5 capture; they
are not represented as fresh held-out measurements. Reuse their exact raw
source writes, hidden states, and global greedy token. A generated memory
is that actual next token, even when it is not the theory answer or is not
one of the nine digit-answer tokens. Report such errors rather than replacing
the token by a candidate-set winner or the correct answer.

Token assembly is operational and never retokenizes a generated position.
Let H be the exact hop-1 prompt token sequence. Encode the shared suffix
`\n{a} + {b} + {c} =` separately with the pinned tokenizer. The four hop-2 arms:

1. **re-derive:** H followed by the suffix; no intermediate memory write.
2. **crystallize:** the identical token sequence as re-derive. Inject a
   residual source only when the admission procedure below admitted it.
   A refused case has exactly the re-derive computation and token sequence.
3. **append/generated:** H, then the one actual hop-1 greedy token as a new
   position tagged generated, then the same suffix.
4. **append/tool:** H, then the one leading-space token for the theory's
   a+b as a new position tagged tool, then the same suffix. Verify the finite
   theory affirms every appended tool atom before execution.

All arms share the same host parameters, original H, operands and suffix.
The context difference of an appended arm is exactly one memory position.
If the raw generated token is special or not a digit, retain and label it;
this is a controlled one-token context experiment, not a natural EOS-stopped
chat trajectory. The tool arm is the intended comparison upper bound;
if the measured host does worse with it, report that instead of clamping rates.

## Admission: an actual R4 witness, with its scope stated

Hold every first-hop measured source write fixed. Use the native decoder
as a faithful frame witness on the first-hop candidate set: the nine R5
digit tokens plus the actual global host token if absent. The target is the
host token, not the theory answer. Verify the full-source frame still
strictly selects that target on this set after numerical decomposition.
No feasible frame is inferred from a solver status or a norm heuristic.

Try all 25 native singleton blocks in their recorded order (embedding,
L0.attn, L0.mlp, ..., L11.mlp); use the first that satisfies the same R4
centered-tail admission inequality. Let m be its actual coalition margin,
tail scores include both target and all covered rivals, kappa be their
midrange, and delta half their range. Require m>max(0.5,2delta), and store
gamma=(m+max(0.5,2delta))/2. Thus m>gamma>2delta is checked. Store every
attempt, the admitted source vector, candidate set, actual frame/bias,
margin and guard slack. A failed faithful-frame check is unknown, not refusal.

This is admission in a particular stored native frame. A refusal means
no singleton passed this *procedure*; it is not an infeasibility proof over
all faithful frames. R4's historical Qwen 0/40 is not used as a supposed
negative split. The split is whatever these nine actual first-hop witnesses
produce, repeated across their three c values; report unique and expanded counts.

In raw residual coordinates, the admitted write is exactly the measured
singleton's raw write. The equivalent first-hop certificate uses the
measured final-LN-folded frame. For hop 2, replay that raw write once at
the same source site: after embedding or after the named attention/MLP
block, at the final query position. Later blocks (if any) see it. Record it
as an extra source with residual-memory provenance. Never replay a refused
or unknown source. No adapter, gain, layer or vector is trained or swept.

Admission proves the first-hop score statement on its measured frame.
It does not prove transfer through another position's normalization or
later computation. Raw replay at the homologous site is the concrete
memory operation under test; an adverse outcome narrows/retire this arm,
not every conceivable crystallization implementation.

## Scores, provenance and advance decisions

Score hop-2 full-vocabulary greedy agreement with the finite theory, not
only a restricted digit argmax. Keep wrong non-digit tokens as failures.
Export all actual block writes and the existing R2 set of 12 read heads at
the final query. In appended arms record each head's actual probability on
the new memory position. Retain the fixed three-coalition body/readout
diagnostics and their candidate coverage separately from the global host.
There is no theory-soundness or service-coverage claim in this experiment.

For each measured admission stratum, publish N, all four accuracies,
paired correct/incorrect transitions, and changes against re-derive.
The advance memory hypothesis is operationalized as:

- On refused cases, generated append improves theory accuracy by at least
  0.05 over re-derive (if less, this improvement claim fails).
- On admitted cases, generated append is redundant when its absolute
  difference from crystallize is at most 0.05, and crystallize does not
  reduce accuracy versus re-derive.

An empty stratum is untested, never confirmed. Unknown admissions form a
separate reported group and cannot support either hypothesis. If both
nonempty strata show identical append/crystallize correctness outcomes,
retire the proposed split for this arm as required by R6's stop rule.
If only one stratum is observed, narrow the conclusion to that stratum.
Report the tool comparison and generated-memory correctness independently.
No statistical population claim follows from these finite, correlated triples.

Validate replay implementation separately with a zero write (identity)
and one fixed nonzero engineering write against an independent NumPy
forward at the same site. These are engineering probes, not admitted
memory cases and not extra R6 observations. The real crystallize arm must
still use only the stored R4 admissions. No held-out prompt is used for
these probes. Freeze all exact request IDs, vectors and hashes before
the first hop-2 forward call.
