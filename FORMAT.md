# The fieldrun bundle format (v1)

A **fieldrun bundle** is a decompiled LLM in a flat, mmap-friendly layout — the contract between the build side
(`lm-sae`'s `pylm/export_bundle.py`, which has Hugging Face) and the pure-Rust runtime (`fieldrun`). The goal is that
the runtime needs **no zip, no .npy parser, no ML framework** — read a JSON manifest, slurp one raw blob, and view each
array by offset.

A bundle is two files that share a stem:

- `<stem>.fieldrun.json` — the **manifest** (UTF-8 JSON).
- `<stem>.fieldrun.bin` — the **blob**: raw little-endian arrays, concatenated.

## Manifest

```jsonc
{
  "format": "fieldrun-bundle",   // magic; readers must reject anything else
  "version": 1,                  // bumped on any breaking layout change
  "arch": "gpt2",                // which runtime kernel to use: "gpt2" | "rope" | "gemma"
  "config": [12, 12, 768, 1024, 50257],   // kernel-specific int vector (see below)
  "config_f": [10000.0, 1e-6],   // optional float vector (RoPE/Gemma: theta, eps, caps, scale…)
  "store": { ... },              // optional: the Tier-A retrieval tables, embedded inline (see below)
  "arrays": [                    // every weight array, in blob order
    { "name": "wte", "dtype": "f32", "shape": [50257, 768], "offset": 0, "bytes": 154389504 },
    { "name": "h0.ln_1.weight", "dtype": "f32", "shape": [768], "offset": 154389504, "bytes": 3072 },
    ...
  ]
}
```

- **`offset`/`bytes`** index into the blob; `bytes == prod(shape) * sizeof(dtype)` — except `i4`, which is bit-packed
  (`bytes == shape[0] * ceil(shape[1]/2)`).
- **`dtype`** is `"f32"` in v1 (little-endian IEEE-754). `"f16"`/`"i8"`/`"i4"` are the in-RAM/on-disk-precision path; each
  quantised array carries a sibling `"<name>__scale"` `f16` array, and the runtime keeps weights low-precision and
  dequantises per matmul:
  - **`i8`** — per-output-column symmetric int8, stored `(in, out)` row-major; `__scale` shape `[out]`.
  - **`i4`** — group-wise symmetric int4 (default group 32), stored `(out, in)` output-column-major with two
    two's-complement nibbles per byte along `in`; `__scale` shape `[out, ceil(in/group)]`, and the array carries an
    extra `"group"` field. Half the bytes of `i8`; dequantised to f32 on read. For MoE this halves the bytes paged in
    per token (the expert-offload lever).
- Array **names and shapes** match the pylm numpy kernels exactly (`numpy_lm.py` / `numpy_rope.py` / `numpy_gemma.py`),
  so a bundle is just those weights relaid; the Rust kernel mirrors the numpy one.

### `config` by arch

| arch | `config` (ints) | `config_f` (floats) |
|------|-----------------|---------------------|
| `gpt2` | `[n_layer, n_head, n_embd, n_positions, vocab]` | — |
| `rope` | `[n_layer, n_head, n_kv, head_dim, d, ffn, vocab, tied]` | `[rope_theta, rms_eps]` |
| `gemma` | same as `rope` | `[rope_theta, rms_eps, attn_softcap, final_softcap, query_pre_attn_scalar, embed_scale]` |

### `store` (optional, makes the bundle whole)

The Tier-A retrieval tables, embedded so one bundle is the *entire* decompiled model (retrieval + composition):
the n-gram successor tables (`quad`/`tri`/`bi`/`uni`), the induction thresholds, and the optional grammar
skeleton (`skel` + `closed_ids`). Same schema as pylm's `store.json`. If absent, the runtime loads a `store.json`
separately for Tier A.

## Versioning

`format` must equal `fieldrun-bundle` and `version` must be understood by the reader, else it refuses to load. Any
change to the blob layout, dtype encoding, or config vectors bumps `version`. The format is **0.x-era** and may change
until the runtime stabilises.
