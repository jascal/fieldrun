# Qwen replay-hook engineering validation, 2026-10-26

This validates an explicit experimental hook, not an R4 admission or R6
transfer result. It introduces no new arithmetic study inputs. Reuse the
existing tiny Qwen fixture and old real-model engineering contexts. Do not
change future study parameters based on these outcomes.

Operator: add one finite raw residual-width vector at the final query's
embedding row, before every transformer layer. Other embedding rows stay
unchanged. The public Qwen capture API accepts memory_writes aligned with
contexts, site=embed only, and explicitly check_observer=false when a write
is present. Missing/empty memory_writes means no intervention. Invalid sites,
widths, nonfinite vectors, count mismatches or unmatched observer checks
must be rejected. Capture the extra memory@embed source and exact write.
The model weights are unchanged; downstream gates naturally recompute in
response to an intervention. No theorem guarantees downstream semantics.

Tiny fixture: three old token windows, each baseline/zero/nonzero replay,
nine native and nine independent Transformers float32 forwards. Nonzero
vector is fixed0.01 times alternating signs, not a theory-selected source.
Real engineering: first old engineering context baseline/zero/nonzero,
plus two further old plain contexts: five native and five independent
Transformers float16 forwards. Nonzero vector is the same alternating0.01
pattern at width2560. Reference uses an independent embedding forward hook,
adding in float32 and casting back to the original embedding output dtype.
No native/reference model processes run concurrently.

Gate: all per-case native/reference global argmaxes agree; zero replay is
bit-identical to its native baseline and exactly identical in the reference;
nonzero replay changes final hidden state in both. Raw source accounting
must close within1e-4*(1+max_abs_residual). Report all hidden/logit discrepancies.
Tiny float32 final-hidden max error must be<1e-4. Real float16 hidden errors
are descriptive, not a claim of equal intermediate arithmetic. Default
unmodified60-position tiny reference and all Qwen native tests must also pass.
Unit tests separately verify earlier final positions are unchanged and bad
requests are rejected. No real-model semantic success claim is licensed.

Build in a separate CARGO_TARGET_DIR and retain the archived old binaries.
Record source/runtime/input hashes and every result. Failures remain recorded;
no engineering input or hook-site shopping. Only after validation can a new
scientific protocol freeze its body, transport operator, fresh inputs,
readouts, guards, budgets, reference and theory grading.
