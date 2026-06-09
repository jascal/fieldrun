# fieldrun

Run a decompiled LLM as a single native binary — the three [`pylm`](../lm-sae/pylm) tiers in pure Rust, no
deep-learning framework at runtime.

`pylm` (in the `lm-sae` repo) decompiles a small LLM into two halves: a flat-file **retrieval** store (n-gram /
induction / grammar / knowledge) that reproduces ~half the model with stdlib-only Python, and a **composition** kernel
(attention + MLP) that runs the rest as plain numpy matmuls over flat weight arrays — no torch. `fieldrun` is the
distribution form of that result: the same tiers, ported to Rust, built into one static binary you can hand someone.

## Tiers

| Tier | What it adds | Status |
|------|--------------|--------|
| **A · retrieval** | induction + n-gram backoff + grammar skeleton over the flat store | **done** — bit-for-bit faithful to Python `lm.py` |
| **B · composition** | the attention + MLP forward pass as Rust matmuls over flat weights | **done (GPT-2)** — exact vs `numpy_lm.py` (= torch); Llama/Qwen/Gemma-2 next |
| **C · router** | compute only the top fraction of MLP neurons/token (budget known from architecture) | planned |
| `explain` | per-token "explain this prediction": idiom + live circuits + named features | planned |
| API | `/predict` + `/explain` over a thin HTTP server | planned |

The weights + store load from a **fieldrun bundle** ([`FORMAT.md`](FORMAT.md)) — a flat manifest + raw f32 blob that the
build side (`lm-sae`'s `pylm/export_bundle.py`, the one-time Hugging Face step) writes and the runtime mmaps. **Runtime
is pure Rust, no framework.**

## The faithfulness gate

Every tier is validated by **top-1 agreement against the Python/torch reference** on the same inputs:
- Tier A — 0 per-position mismatches vs `lm.py` over 500 held-out positions, on both a no-grammar store (GPT-2) and a
  grammar store (Qwen): same idioms, same arbitration, same token space.
- Tier B — 0 per-position mismatches vs `numpy_lm.py` (itself 100% top-1 vs torch GPT-2); 50.0% next-token top-1, the
  model's own number.

## Concurrency

The scoring loops fan out across cores with rayon — each next-token prediction is an independent, read-only forward.
On a 16-core box, Tier-B scoring of 200 positions drops from ~48 s (one core) to ~11 s. (Single-forward latency is
limited by per-matmul size; a pure-Rust threaded GEMM is the next lever there, kept framework-free.)

## Build & run

```bash
cargo build --release
# Tier A — retrieval over the flat store
./target/release/fieldrun --store ../lm-sae/pylm/store_gpt2.json --ids ../lm-sae/pylm/holdout_gpt2.json
# Tier B — the real GPT-2 forward pass over a fieldrun bundle
./target/release/fieldrun --bundle ../lm-sae/pylm/gpt2 --ids ../lm-sae/pylm/holdout_gpt2.json --n-eval 200
# --dump preds.txt  writes one prediction per line for the faithfulness diff
```

## License

Apache-2.0. This is a 0.x prototype — interfaces and the on-disk bundle format are not yet stable.
