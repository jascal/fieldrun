# R2 preregistration, 2026-10-24

Frozen before the first PDC read capture or its evaluation. No model-output
selection, head selection, or tolerance tuning after the capture is allowed.

- Architecture: native CPU GPT-2, full gates, no KV quantization or routing.
- Bundle stem: `bundles/gpt2` (f16 stored weights, f32 native arithmetic).
- Manifest SHA256: `322d79629f2e121c446560f57b71c5199e0d8a2a912c3a396137ca7eb207692c`.
- Weight SHA256: `4edbcffed3bc19d571e679896ab3efebaa3802f8d7c9db3bdad31ff2416490f1`.
- Heads: layers 0, 5, 11 crossed with heads 0, 3, 7, 11.
- Query positions: 3, 7, 15 (zero based). Each candidate set is keys 0 through i.
- Three 16-token contexts: ascending IDs 100–115; alternating IDs 100, 200;
  repeated four-token motif 100, 200, 300, 400. These are token-ID tasks, not
  claims about natural language. Total 108 head/query instances.

Convention: incidence is q dot k / sqrt(64), equivalently use q/sqrt(64) as
the query frame at T=1. Keep unscaled q and k in the dump. Native causal mask
is -1e10 outside j<=i, which underflows to zero probability on these instances;
explicitly check the forbidden-key mass. Measure the pre-output-projection
head write (64 dimensions). The residual write is its linear output projection;
shared projection bias belongs to the attention block, not to each head.

Capture inside `Gpt2::hidden`'s actual forward execution, with an optional
observer only; verify observed and unobserved predictions agree. Record q, k,
v, native masked logits, native probabilities, and the actual assigned head
mixture. Independently compute f64 incidences/softmax/mixture from the dump.

Report separately:

1. Algebraic identity: the host writes softmax(q k transpose / sqrt(64)) times
   v by construction under this convention, with causal-key restriction checked.
2. Numerical error: max absolute scaled-incidence, probability, and head-write
   error against the f64 checker; report RMS head-write error too. A tolerance
   of 1e-4 for mixture is an engineering diagnostic, not an algebraic proof.
3. Decision agreement: argmax sets of independently computed incidences versus
   native probability argmax sets, including ties. Report all 108 instances and
   per-head aggregates. Only families with 1.00 receive the read identification.

Faithfulness: a self-contained NumPy reference over the exact same flat bundle
must agree with native full-vocabulary predictions on all three contexts. Record
top-1 agreement separately from read-key agreement. This is validation of the
observer; do not transfer it to all prompts or architectures. Any failed head is
named and excluded. If none meet the mixture identity, apply the R2 stop rule.
