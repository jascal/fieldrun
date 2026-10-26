# Replay-hook engineering and baseline normalization correction

This is engineering validation, not an admitted-memory transfer result.
No new scientific prompts were run. PROTOCOL.md fixed the intervention,
old inputs, vectors and thresholds before engineering execution.

Initial unmodified-kernel run: tiny9/9 global predictions and ordinary60/60
reference predictions agree, but baseline hidden error0.0017129183 fails
the fixed1e-4 engineering threshold. Zero replay is identical in all three
pairs. Retain the failed gate, original capture/reference and metrics.
Investigation identifies sqrt(sum(q*q))+epsilon versus the official
sqrt(sum(q*q)+epsilon). An independent low-norm official-kernel golden
fails before the fix with0.08847154 error. The correction changes the
native default DeltaNet normalization; it is not a claim that the replay
observer alone preserved the old implementation.

After the correction, SAME inputs, reference, vectors and thresholds:

- Tiny9/9 native/reference global predictions match; final-hidden maximum
  error7.153e-7, below the original1e-4 threshold. Zero replay bit-identical;
  nonzero replay changes both implementations. Raw source accounting passes.
- Ordinary60-position tiny reference still matches60/60.
- Real engineering5/5 global predictions match; zero replay bit-identical;
  nonzero replay changes native/reference hidden by0.54968/0.55469. Maximum
  absolute hidden discrepancy0.03407 is descriptive: native f32 arithmetic
  with f16 weights is not equal to the Transformers f16 intermediate path.
- All194 native unit tests pass:14 library and180 binary tests, including
  independent low-norm golden, state/chunk invariants and replay validation.
  Earlier-position invariance is checked by the native unit fixture only.

The new experimental API accepts one finite residual-width vector per
context at site=embed, added ONLY to the final query's embedding before
all layers. Explicit check_observer=false is required for replay; invalid
sites/widths/counts/nonfinite writes are rejected. Missing writes follow
the ordinary path. Capture includes the extra memory@embed source. This
operator has no theorem guaranteeing downstream semantic preservation.

Engineering accounting: initial tiny9 + corrected tiny9 + real5 native
capture forwards; independent tiny9 + real5 reference forwards; ordinary
reference validation adds60 native forwards before and60 after correction.
These are engineering budgets, not arithmetic study observations. No
admission, body selection, semantic transfer success or compute saving is
inferred. No first-hop vector from the old runtime may be silently pooled
with a corrected-runtime capture. Future transfer inputs and policy must
be preregistered, recertifying their actual float32 carrier on that runtime.

The archived scientific runtime remains hash-identical:
90cd5c8dda5b1f5c16888387baa92b78205c82a99b0eb04a98679ca324c1928d.
Old archives/results remain unchanged and describe that implementation.
NORMALIZATION_CORRECTION.md and initial_engineering_record.json retain
provenance and the failed test; numeric corrected verification files name
the narrow successful engineering checks. Large real captures/reference
and executable snapshots are distributed separately from ordinary Git.
