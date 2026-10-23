# Native MLP observations and interventions

`POST /probe/mlp` is a build-time research endpoint for the dense RoPE family.
It is available in token-ID-only builds (`--no-default-features`). Other model
implementations explicitly decline this probe. Normal inference is unchanged.

```json
{
  "ids": [200, 300, 400],
  "probe": {
    "layers": [7, 15, 23],
    "units": [],
    "patches": [],
    "readouts": [200, 300]
  }
}
```

`layers` observes every MLP neuron at those layers at the last input position.
For smaller responses, use explicit `(layer, neuron)` pairs in `units` instead.
The union is returned in layer/neuron order without duplicates. Each observation
contains zero-based `layer`, `position`, `neuron`, raw gate preactivation `gate`,
up-projection `value`, and `product = SiLU(gate) * value`. A positive product
alone does not distinguish the positive/positive and negative/negative routes.
The response also includes greedy `next`, its `next_logit`, and the requested
`(token, logit)` readouts. Empty observation/readout lists are allowed.

To intervene, supply a product replacement:

```json
"patches": [{"layer": 15, "position": 2, "neuron": 17, "value": 0.0}]
```

Only that position and neuron change, after SwiGLU and before down-projection.
Later layers recompute on the changed residual. Observations at the patched
layer record the product **before** that layer's patches; observations at later
layers include their propagated effects. Setting the product to its observed
value is an identity control. Setting it to zero is an ablation; replacing it
with another context's value is a patch. None of these interventions mutate
model state between requests.

Empty/invalid token IDs, out-of-range layers/neurons/positions/readouts,
nonfinite patch values, and duplicate writes to one coordinate are rejected.
Unknown JSON fields are rejected rather than silently ignored. Invalid or
unsupported requests return the API's ordinary JSON `error` response.

## Validation

The unit tests check unmodified logits against the ordinary forward, products
against the existing activation probe, identity patches, position specificity,
propagation to later layers, clean subsequent requests, and all-position zeroing
against the independent existing ablation path. Invalid coordinates and duplicate
patches are rejected.

`scripts/validate_mlp_probe.py` separates upstream reference collection from
fieldrun checking so both models need not remain in memory together:

```bash
python scripts/validate_mlp_probe.py prepare /tmp/probe-gate --checkpoint /path/to/pinned/hf
python scripts/validate_mlp_probe.py reference /tmp/probe-gate --checkpoint /path/to/pinned/hf
fieldrun --bundle /path/to/f32/bundle --serve 8391
python scripts/validate_mlp_probe.py check /tmp/probe-gate --port 8391 \
  --bundle /path/to/f32/bundle --binary /path/to/fieldrun
```

The concrete gate uses Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, FP32 weights and eager upstream attention.
On its 12 frozen contexts, all 525,312 gate/value/product coordinates at layers
7/15/23 satisfy `abs(error) <= 0.002 + 0.002*abs(reference)`; maximum observed
absolute error is 0.000236786. Baseline and all 108 zero/replacement/identity
interventions match upstream top-1, and selected logits pass that same tolerance.
Fieldrun identity patches preserve selected logits exactly. These are empirical
results on that finite suite, not general activation equivalence guarantees.

Retained evidence is in rosetta's
`reference/benchmarks/native_trace_preflight/` on `research/native-trace-discovery`:
frozen token IDs and protocol, checkpoint hashes, raw upstream and fieldrun NPZ
arrays, intervention references, every comparison, and binary/bundle hashes.
Reference dependencies for this run: Torch 2.12.1+cpu, Transformers 5.12.1.
No Python or Transformers dependency is added to the fieldrun runtime.
