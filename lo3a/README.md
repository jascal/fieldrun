# LO3a — the context-free whole-model emit (demonstration)

This directory demonstrates **LO3a** (`LOGIC_EXPORT.md`): emitting a transformer's *entire forward
pass* as ONE Soufflé Datalog program whose only input is `token(pos, id)`. Unlike `export --logic`
(one decision, baked to a context) or `stitch` (a trace of one reply), this program is
**context-free**: change the token facts and Soufflé recomputes the next token from scratch — it
answers contexts the exporter never saw.

The emitter ships in fieldrun: `fieldrun --bundle <rope-bundle> export --logic-whole`. The files here
exist to *verify* it against an independent reference at a scale Soufflé can actually run.

## Why it's plain Datalog (no FFI)

Soufflé has only `+ - * / ^` and `sum`/`max` — no `exp`/`sqrt`/`sin`/`cos`. Sufficient because:
- `sqrt(x) = x ^ 0.5` (RMSNorm), `exp(x) = E ^ x` (softmax, SiLU) — `^` does real powers.
- RoPE `sin`/`cos` depend only on **position**, never token content → precomputed model-constant facts.
- matmul = `sum`-aggregate; softmax = `max` then `^(s-m)` then `/Z`; argmax = `max`-witness rule.

## Files

| File | Role |
|------|------|
| `mint_and_emit.py` | mints a TINY real rope bundle (fieldrun-loadable), a numpy reference forward mirroring `src/rope.rs` (f32), and (for the base variant) a reference Datalog emit. Knobs: `BIAS=1`, `UNTIE=1`. |
| `verify_all.py` | the verifier: for base / +bias / +untied / +bias+untied, mints a bundle, has **fieldrun** emit the whole-model `.dl`, and checks `souffle(decide) == numpy == fieldrun` on a battery of held-out contexts. |
| `bench.sh` | provable-optimization anchor: compiles the program (`souffle -o`, native C++), checks the decode is identical + logits agree to ~1 ULP (lossless), and times interpreter vs compiled (**~190× faster**, semantics-preserving). See `../PROVABLE_OPT_PROPOSAL.md` §2.1. Needs the local compiled-mode setup (`../SOUFFLE.md` §1.1). |
| `bundle_io.py` | read/write fieldrun rope bundles (f32) + a parametric numpy forward mirroring `src/rope.rs`. Shared by the reducer/exporter. |
| `reduce.py` | **certified Π → smaller bundle reducer**: scores FFN neurons over a calibration set, drops the provably-dead (zero `down_proj` row ⇒ δ=0, exact on every input) and margin-dominated ones, writes a structurally smaller bundle, and certifies decode preservation against fieldrun. |
| `to_safetensors.py` | **HF export + complete round trip**: reduced bundle → Hugging-Face `safetensors` + `config.json` (`LlamaForCausalLM`) → `fieldrun convert` → bundle′ → decode-compare. Closes the loop bundle ↔ HF. |

The full pipeline these demonstrate: **fieldrun model → LO3a Datalog (`export --logic-whole`) → lossless optimize (`bench.sh`, ~190×) → certified reduce (`reduce.py`, smaller bundle) → HF safetensors (`to_safetensors.py`, publishable) → round-trips back to fieldrun losslessly.**
| `tiny*/` | the minted bundles (gitignored). |
| `whole*.dl`, `ctx*/`, `*.facts` | generated programs and context inputs (gitignored). |

## Reproduce

```bash
# from the repo root: build fieldrun and install souffle (see ../SOUFFLE.md §1)
cargo build --release

cd lo3a
python3 verify_all.py
# -> [base/+bias/+untied/+bias+untied] 12/12 held-out contexts agree (souffle == numpy == fieldrun)
#    ==> ALL VARIANTS VERIFIED
```

Or by hand, on the base variant:

```bash
python3 mint_and_emit.py                                  # mint tiny/ + whole.dl + numpy ref
../target/release/fieldrun --bundle tiny/tiny \
    export --logic-whole --out cf.dl --maxpos 16          # fieldrun emits the context-free program
printf '0\t3\n1\t14\n2\t7\n3\t2\n4\t29\n' > ctx/token.facts
souffle cf.dl -F ctx -D -                                  # -> decide(29), computed from scratch
souffle -t explain cf.dl -F ctx                            # interactive: why decide(29)? (proof tree)
```

## The result, and the honest limit

At small scale the program computes the next token for arbitrary inputs, in a neutral engine, exactly
matching the model — LO3a's "possible?" is **yes**. What stays open is **LE-T2/LE-T4**: the dense
`embed`/`unembed` fragment costs `vocab × d` facts, so the program is correct for any model but not
*compact* at full scale (Qwen2.5-0.5B ≈ 136M embed facts — `export --logic-whole` refuses it without
`--force`, naming the wall). The frontier moved from *can you emit a context-free program?* to *can the
dense fragment be emitted compactly?* — see `../SOUFFLE.md` §8 and `../LOGIC_EXPORT.md` LO3a.
