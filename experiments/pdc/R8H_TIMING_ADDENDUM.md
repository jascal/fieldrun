# R8-H timing instrumentation addendum, 2026-10-25

Written after the initial native full-baseline capture completed, but before
its scores were inspected, any service forward ran, or any service response
was evaluated. The initial capture has native per-context forward/decode
timings and output-row counters, but its outer process was not wrapped by
an end-to-end wall timer. Retain it as the initial validation capture.

To measure the missing quantity, run exactly one additional, explicitly
diagnostic full-vocabulary baseline followed by the first probe-only live
service batch, both through the same end-to-end wrapper. An outer runner
times each subprocess through model loading, native capture, Python body
checking, optional external grading, and final bundle serialization. Retain
all three native captures, label which baseline is paired for timing, and
include the initial validation run in provenance rather than concealing it.

No policy, lexical probe, context, body, threshold or success bar changes.
No repetition is selected for a favorable latency. One pair supports
descriptive timings only; it does not establish a general speedup. The
host-agreement and dispatch-count gates remain those of R8H_PREREG.md.
