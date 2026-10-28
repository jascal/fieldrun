# PDC artifacts in fieldrun

The specifications live in the separate PDC repo. This directory holds
preregistered measurements and exported interfaces, with no local imports of
that repo, i-orca, or pil. Results distinguish kernel statements, arithmetic
instance checks, and frozen-model measurements.

- `R1_RESULTS.md`: finite executor, the kernel witness, and 1,280 one-body checks.
- `R2_PREREG.md`, `R2_RESULTS.md`: 108 frozen read-head/query instances.
- `R3_PREREG.md`, `R3_RESULTS.md`: 16 competing-copy contexts and 576 bodies.
- `r4_qwen_sources.jsonl`, `r4_capture.log`: the 80-position R4 source capture;
  the oracle preregistration and results belong to pil. This is a new sample,
  not the unavailable historical 40-position dump.
- `R5_PREREG.md`, `R5_RESULTS.md`: bounded addition theory and the triggered
  epistemic stop rule. Seventy-five evaluation contexts were captured; the
  other 25 contexts were subsequently opened once by the final diagnostic audit.
- `R8H_PREREG.md`, its timing addendum and `R8H_RESULTS.md`: actual native
  subset service with 1,703 lexical probes. It measures output-row skips
  but abstains everywhere; zero coverage is not a successful service gate.
- `R6_PREREG.md`, `R6_RESULTS.md`: 27 refused-stratum two-hop triples.
  Append improves full-host theory accuracy from 3/27 to 9/27; the admitted
  stratum is empty and its redundancy claim remains untested.

## `pdc.finite-derive.v1`

`scripts/pdc_derive.py` reads JSON `atom -> [{id, score}, ...]`. Body IDs must be
distinct within an atom even when scores agree. A missing body is represented
by an empty list and a null score, not zero. Both named readouts are always
returned. Atom winner sets retain ties, which have status `tie`, not `selected`.
The result retains the complete body table, candidates, temperature, scores,
margin, best-body witnesses and fragility. For marginal queries the entire
body set contributes; a best-body witness is not the entire explanation.

## `pdc.read-incidence.v1`

```sh
target/release/fieldrun --bundle bundles/gpt2 --pdc-read-selection selection.json --pdc-read-dump reads.json
```

Selection JSON has `layers`, `heads`, `positions`, and `contexts` (token-ID
arrays). This observer supports CPU GPT-2 with full gates. Each case retains
the context, actual native query/key/value vectors, masked scaled logits,
probabilities, assigned head mixture, per-position vocabulary predictions,
and the observer-effect comparison. The current additive schema also contains
`final_residual` per read (pre final LN) and `final_hidden` per case; older R2
captures omit those optional chart-building fields.

R5 adds optional selection fields: `last_position` chooses the final token
separately for each variable-length context; `capture_sources` retains every
actual additive embedding/attention/MLP write; `last_prediction_only` limits
the full-vocabulary audit to the final position; and `candidate_tokens`
retains native scores on a fixed named subset. All default to the older
behavior when omitted. Captured source records are case-level, with source
ID and position; they are not recomputed by a replacement kernel.

Values are pre-output-projection head coordinates; q/k are unscaled, with
the scale recorded separately. Future causal keys are retained for auditing
but excluded from the finite candidate set. Shared output-projection bias is
not assigned to every head. `scripts/pdc_read_check.py` independently checks
the incidences, softmax and mixtures in f64, and compares native vocabulary
decisions with a separate NumPy forward over the bundle.

## `pdc.copy-chart.v1`

`scripts/pdc_copy_chart.py` consumes a native capture and the exact bundle.
It runs the fixed R3 trigram-match body generator and emits the eight-row
candidate unembedding, effective bias, contexts, complete body tables,
source-write vectors, an additive complement, readout judgments and host/task
comparisons. Source writes use the fixed measured final-LayerNorm folding;
they are additive attribution vectors, not outputs of intervened forwards.
Per-body metadata records its singleton coalition, route, probability,
incidence, bias and bracket so consumers can reconstruct every annotation.
The body table includes no routes selected using host tokens.

These charts are not yet served DerivationBundles. Memory and theory are
explicitly `none`/`not_loaded`, and certificate is `not_evaluated`. A consumer
must not treat those tags as admission, truth, or a passed margin guard.

## `pdc.derivation-bundle.v1` and the external theory grade

`scripts/pdc_add_bundle.py` exports the frozen R5 source programs and all
27 atom/body brackets per context. Each query keeps a readout name, body
witness, margin, fragility, memory tag, covered candidates, and individual
pairwise gap/drift/slack checks. Failed guards emit `abstain` with `atom=null`.
The source frame, bias, full body set and host comparison are retained, so
the certificate can be checked without a token-only answer path.

The exporter does not use the theory answer to select or certify an atom.
`scripts/pdc_theory_check.py` is a separate bounded-addition grader that
emits `pdc.theory-grade.v1`, with affirmed/hallucination/abstain judgments,
misses and fragile truths. `scripts/pdc_bundle_audit.py` independently checks
the numeric source/body/readout/guard artifact against the capture and model.

R5 failed its advance hallucination bar widely: 7/8 and 9/10 certified
commitments are false in the theory. The finite certificate supports host
agreement on nine candidates; it does not establish truth or unrestricted
vocabulary agreement. These are exported offline artifacts, not a measured
online service or compute saving. The R5 stop restriction applies to later
work; see the result document for the exact scope.

## `pdc.served-bundle.v1`: host-only runtime arm

`scripts/pdc_service.py --runtime ...` runs a fresh native forward and
returns named queries with source/body artifacts or atom-null abstentions.
It uses only the lexical probe's output rows for native decoding and Python
body projections. It does not run the independent reference or choose a
cached/theory answer. `--case-id` serves a single named non-held-out request;
the frozen 75-request batch is also supported. External theory grading is
optional and occurs after atom selection. Claims set theory soundness false.

The native selection fields `score_candidates_only` and `check_observer`
default to false and true respectively, preserving older captures. A timed
service request explicitly enables candidate-only scoring and disables the
duplicate observer-validation forward. It reports actual row calls, timing,
and the scoring scope. An absent observer check is null, never claimed zero.
The full diagnostic baseline is explicitly labeled and requires an override.

`scripts/pdc_run_service_experiment.py` executes the frozen timing pair once
and refuses overwrites. `scripts/pdc_service_audit.py` independently checks
both native arms, all body/probe annotations, guard/refusal decisions,
frozen R5 baseline agreement, and dispatch counts. See `R8H_RESULTS.md`:
96.61% of native output-row calls are skipped, but both readouts abstain
on all 75 inputs. The zero denominator is untested, not a passing host or
truth rate. No R6/R7 outcome or held-out result is implied.

## `pdc.two-hop-bundle.v1`: memory trace and observed outcomes

`scripts/pdc_memory_inputs.py` freezes R6 admission witnesses, exact request
tokens and permitted writes. The native `memory_writes` selection field is
an aligned list of optional `{site, write}` records. Replay adds the raw
write once at the named source site and final query position, with a
`memory@site` provenance source. Invalid sites, widths, nonfinite vectors
and unmatched observer comparisons are rejected. Existing captures omit
this field and retain their original computation.

The actual R6 cohort admits no singleton, so no real arm replays a source.
The separately labeled engineering probes validate a zero and fixed nonzero
write against NumPy without counting them as memory observations. All 108
real-arm predictions agree with the independent reference. Exact context
assembly and all tool-theory atoms are checked separately.

`scripts/pdc_memory_bundle.py` exports nine first-hop nodes and 108 second-hop
paths with decode readout, body/source identities, finite covered margin,
derive fragility, memory edge and external judgment. The tool edge can differ
from the host intermediate and is explicitly sourced from the theory.
`scripts/pdc_memory_bundle_check.py` checks 117 affine node judgments and all
provenance edges; source capture references remain attached. Full-vocabulary
reference agreement is an empirical check, separate from the packet's
finite candidate margin. See `R6_RESULTS.md` for the narrowed conclusion.


## Stopped-stage review, v0.10

`FINAL_AUDIT_PREREG.md` and `FINAL_AUDIT_RESULTS.md` document the once-only
25-context held-out audit. Both readouts abstain on every context; the
reservation is consumed. R8 and criterion 9 remain unmet. R7's fixed-budget
negative result belongs to pil; its immutable input NPZ is here.
`release_inventory.json` records local artifact hashes, including sources,
proofs and write-up. It is not a claim that changes have been published.
