# R6: append helps the observed refused slice, 2026-10-25

**Empirical, narrowed outcome.** On all 27 preregistered triples, the
recorded native-frame singleton procedure refuses crystallization. Generated
append improves full-vocabulary theory accuracy from 3/27 to 9/27, a gain of
22.22 percentage points, above the advance five-point bar. This confirms
that operational comparison on the observed refused slice. There is no
admitted stratum, so append redundancy and actual admitted-source transfer
remain untested. No universal memory or crystallization claim follows.

Preregistration `R6_PREREG.md`, SHA256
`c03c9e676af396733f27bb47840f5df8a178392a70cd153a3ab79bc935e5c071`.
Exact requests, memory vectors and hashes were frozen before any hop-2
forward. First-hop template-0 prompts reuse nine existing R5 captures;
they are not presented as held-out measurements. All 25 reserved R5
template-3 contexts remain uncaptured.

| Admission stratum | Triples | Re-derive | Crystallize | Generated append | Tool append |
|---|---:|---:|---:|---:|---:|
| Refused in stored native frame | 27 | 3/27 (11.11%) | 3/27 (11.11%) | 9/27 (33.33%) | 9/27 (33.33%) |
| Admitted | 0 | untested | untested | untested | untested |
| Unknown | 0 | untested | untested | untested | untested |

Generated append retains the three correct re-derive cases and changes six
incorrect cases to correct; 18 remain wrong. The tool arm has the same
aggregate transitions against re-derive, but generated/tool correctness
coincides on only 23/27 triples and their raw output tokens coincide on
21/27. The theory-correct tool memory is not demonstrated to be a pointwise
upper bound. The generated intermediate is correct on only 4/9 unique
first-hop prompts. A benefit from adding a position is not a proof that
the model uses its content as a faithful arithmetic intermediate; no
semantic-content or causal-necessity claim is made.

## Admission, replay and provenance

Every singleton attempt uses the same recorded native first-hop frame and
fixed writes, covering the nine digit tokens plus the actual global host
token if needed. The full-source reconstruction strictly selects each
host token on that covered set. All 25 singleton attempts are retained
for each of the nine first hops; none satisfies
`m > gamma > max(0.5,2delta)` with the preregistered gamma construction.
Refusal is relative to this procedure and frame, not a Farkas impossibility
proof over every faithful frame. The old Qwen 0/40 is not used as an
assumed negative split.

Because all first hops are refused, the real crystallize arm injects
**no** residual memory. Its token sequence, sources, hidden states and
predictions are exactly identical to re-derive on all 27 triples. The
native replay hook is nevertheless independently checked as an engineering
operation; those probes are not admitted memory observations.

Replay adds one recorded raw source at the final query's homologous source
site and retains it as an additional provenance-labeled write. The native
API validates site, vector width, finiteness and context alignment. Its
unmatched no-memory observer comparison must be explicitly disabled when
a replay is present. The engineering checks use L0.mlp, a zero vector and
a fixed alternating +/-0.125 vector; no site, gain or vector was swept.

The zero write leaves every native hidden state exactly unchanged. The
nonzero write changes the hidden state by max 0.537842, leaves all earlier
causal positions unchanged, and its induced change matches an independent
NumPy forward within max 6.390e-5. All three query top-1 predictions agree
with NumPy. Absolute hidden-state differences are also retained: the
unmodified baseline already differs by max 0.003174 at an earlier position,
and the query-position error is 0.0002213 in all three cases. An initial
1e-3 absolute all-state diagnostic therefore failed; it is not reported
as passed. Replay-effect and baseline errors are separate measurements.
This engineering evidence does not prove transfer of an admitted atom
through another position's normalization or later blocks.

## Independent checks and readout artifacts

The main run contains 108 full native forwards, exactly four arms for
every triple in {0,1,2} cubed. Independent NumPy final-position predictions
agree 108/108. The assembly checker independently re-encodes the suffixes,
checks every exact prefix/memory/suffix sequence, verifies all 27 tool
atoms against addition, and verifies complete triple/arm coverage.

The result retains fixed full/attention/MLP coalition tables, both named
derive readouts and their candidate coverage separately from the primary
global greedy host token. Source-sum maximum error is 2.736e-5. All 12
preregistered heads' actual attention probabilities on appended positions
are retained. There are 324 such head reads per appended arm; all are
nonzero. Mean recorded mass is 0.02513 for generated memory and 0.02462 for
tool memory. Attention on a position is not itself evidence of necessity
or correct semantic use.

Artifacts: `r6_requests.json`, `r6_selection.json`, `r6_input_hashes.json`,
`r6_capture.json`, `r6_results.json`, `r6_assembly_verification.json`, and
the engineering selection/capture/results files. The symbolic admission
rule remains the named strict i-orca theorem; these concrete numeric checks
and outcome counts are empirical evidence. R5's epistemic stop remains
active. Neither this narrowed memory result nor zero R8-H coverage
establishes a useful truth-sound service or completes criterion 9.

The additional `r6_chart_bundle.json` exports the actual two-hop provenance
graph without another model call. It separates one-body PIC decode nodes
from the multibody derive diagnostics, records all covered-candidate margins
and external-theory bits, and distinguishes a theory-supplied tool edge from
the host's possibly wrong intermediate. `r6_chart_verification.json` checks
117 affine nodes and 108 paths, with score discrepancy zero: 27 generated
edges, 27 tool edges, no residual-memory edges and 54 no-memory edges. It
does not turn a finite candidate margin into a global vocabulary certificate.

Reproduce the already-frozen inputs with the tokenizer environment, then
engineering validation, then the actual experiment and independent audit:

```sh
../pil/.venv/bin/python scripts/pdc_memory_inputs.py --domain experiments/pdc/r5_domain.json --first-capture experiments/pdc/r5_reads.json --tokenizer bundles/gpt2.tokenizer.json --bundle bundles/gpt2 --out-dir experiments/pdc
cargo build --release --no-default-features
target/release/fieldrun --bundle bundles/gpt2 --pdc-read-selection experiments/pdc/r6_engineering_selection.json --pdc-read-dump experiments/pdc/r6_engineering_capture.json
python3 scripts/pdc_memory_engineering_check.py --capture experiments/pdc/r6_engineering_capture.json --selection experiments/pdc/r6_engineering_selection.json --bundle bundles/gpt2 --out experiments/pdc/r6_engineering_results.json
target/release/fieldrun --bundle bundles/gpt2 --pdc-read-selection experiments/pdc/r6_selection.json --pdc-read-dump experiments/pdc/r6_capture.json
../pil/.venv/bin/python scripts/pdc_memory_assembly_check.py --requests experiments/pdc/r6_requests.json --domain experiments/pdc/r5_domain.json --tokenizer bundles/gpt2.tokenizer.json --out experiments/pdc/r6_assembly_verification.json
python3 scripts/pdc_memory_audit.py --capture experiments/pdc/r6_capture.json --requests experiments/pdc/r6_requests.json --selection experiments/pdc/r6_selection.json --first-capture experiments/pdc/r5_reads.json --domain experiments/pdc/r5_domain.json --bundle bundles/gpt2 --out experiments/pdc/r6_results.json
```

The interpreter environment supplies packages; scripts do not import code
from pil or another repo. The frame, source vectors and metadata are
exchanged through the documented artifacts. No host weights or gates were
trained in this experiment. Admission learning remains R7's open task.
