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
| **A · retrieval** | induction + n-gram backoff + grammar skeleton over the flat `store.json` | **done** — bit-for-bit faithful to Python `lm.py` |
| **B · composition** | the attention + MLP forward pass as Rust matmuls over flat weights (GPT-2 / Llama / Qwen / Gemma-2) | planned |
| **C · router** | compute only the top fraction of MLP neurons/token (budget known from architecture) | planned |
| `explain` | per-token "explain this prediction": idiom + live circuits + named features | planned |
| API | `/predict` + `/explain` over a thin HTTP server | planned |

Build-time (one-time) uses `lm-sae` + Hugging Face to export the flat bundle (weights + store); **runtime is pure Rust.**

## The faithfulness gate

Every tier is validated by **top-1 agreement against the Python/torch reference** on the same inputs. Tier A already
passes exactly: 0 per-position mismatches vs `lm.py` over 500 held-out positions, on both a no-grammar store (GPT-2) and
a grammar store (Qwen) — same idioms, same arbitration, same token space.

## Build & run

```bash
cargo build --release
# score Tier A over a held-out token-id stream (defaults point at ../lm-sae/pylm)
./target/release/fieldrun --store ../lm-sae/pylm/store_gpt2.json --ids ../lm-sae/pylm/holdout_gpt2.json
# --dump preds.txt  writes one prediction per line for the faithfulness diff
```

## License

Apache-2.0. This is a 0.x prototype — interfaces and the on-disk bundle format are not yet stable.
