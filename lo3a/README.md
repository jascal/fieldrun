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

| `run_smollm.py` | the **whole pipeline on a REAL small Llama** (SmolLM-135M): `fieldrun convert` → certified FFN reduce → HF safetensors → `fieldrun convert` → bundle′. Uses real high-margin contexts from a fieldrun greedy trace. |
| `pythia_grok.py` | **PO-T7 grokking order-parameter experiment** — converts 28 Pythia-70m checkpoints (step0→143k via `@stepN`), runs the new `--probe-margin`, and plots the certifiable-compressible fraction / margin / PR / accuracy across training (`pythia_grok.png`). Finding: cert fraction *saturates* with accuracy, but PR (circuit concentration) consolidates in **two events** — including a **discrete late one (~step 70k) invisible to accuracy/margin/cert**. The dissociation is the certificate's confidence-boundedness. |
| `lo1_matrix.py` | **LO1 attack** (Grok's matrix/operator-semiring valuation): the descriptive escape's width = the *effective rank* of the dense fragment's Gram, so escape ⟺ low-rank. Synthetic part proves the mechanism; real part (SmolLM) finds the **token-coupling** Gram is low-rank (effrank/K≈0.34) **but margin-invariant** → the forge tax is *not* there, it's in the **circuit-coupling** axis. See `../LOGIC_EXPORT.md` LO1. |

The full pipeline these demonstrate: **fieldrun model → LO3a Datalog (`export --logic-whole`) → lossless optimize (`bench.sh`, ~190×) → certified reduce (`reduce.py`, smaller bundle) → HF safetensors (`to_safetensors.py`, publishable) → round-trips back to fieldrun losslessly.**

### Real-model run (SmolLM-135M, `run_smollm.py`)

A real pretrained Llama (d=576, 30 layers, GQA 9/3, vocab 49152), end to end:
- **convert** → 513 MB f32 bundle; decodes faithfully (`export --logic` FAITHFUL ✓).
- **whole-model Soufflé emit REFUSES** — `vocab×d = 28.3M` facts, the LE-T4 wall. The single-decision `export --logic` still works; only the *context-free whole-model* emit hits the dense wall, exactly as the proposal predicts.
- **certified FFN reduce**: preserves decode 15/18 at 1–2% smaller, 12/18 at 4–6%. The honest result — a *trained* dense FFN has ~0 exactly-dead neurons, so the **losslessly**-removable set is ≈0 and approximate pruning trades fidelity. That **is** the forge tax (PO-T2) measured on a real model: the dense computed fragment does not compress losslessly.
- **HF safetensors round trip**: reduced model → `LlamaForCausalLM` safetensors (509 MB, publishable) → `fieldrun convert` → bundle′ — **weights bit-identical (Δ=0), decode 18/18 ✓**.

(Generate the contexts first: `printf '/export-logic /tmp/smtr.dl <prompt>\n/exit\n' | fieldrun --bundle lo3a/smollm/smollm --chat`.)
| `tiny*/` | the minted bundles (gitignored). |
| `whole*.dl`, `ctx*/`, `*.facts` | generated programs and context inputs (gitignored). |

### Shipping the lever: the two-knob PR-core head (`pr_core_export.py`)

The LE-T4 wall above is on the *lossless* path. `LOGIC_EXPORT.md` LO1 and `PROVABLE_OPT_PROPOSAL.md`
§7 establish the **lossy** escape: the readout argmax factors through a rank-r readout-aligned decision
basis `S`, so the `vocab × d` unembedding becomes `S` (`r × d`) + `A = S·(gain⊙U)ᵀ` (`r × vocab`) — a
**tunable lossy size dial** at `r(d+vocab)` floats with *known, measured* decode preservation. The heavy
tail is intrinsic (τ\*): three cheap routers all fail to make the core decode-exact (`pr_core_v2.py`), so
PR-core ships as a labeled-lossy storage mode, not a free win. The full readout stays the exact default.

`pr_core_export.py` is the shipped artifact:
- **export** — fits `S` on calibration decisions, writes `<out>.prcore.npz` (`S`, `A`) + `.json` manifest
  (rank, sizes, compression, `decode_kept`, lossy flag, provenance), and **verifies** preservation on a
  *fresh* held-out battery (re-runs the real rope forward, compares PR-core argmax to the model's argmax).
- **`--datalog`** — emits a self-contained, souffle-runnable **factored readout** `.dl`:
  `proj(i)=Σ_j xraw(j)·sbasis(i,j)`, `corelogit(v)=Σ_i proj(i)·acore(i,v)`, `best = argmax`. This is the
  LO1 lever applied *to the logic export itself* — the dense `vocab×d` embed facts shrink to the factored
  `r(d+vocab)`. Shortlisted to stay runnable; the manifest records the true full-vocab fact count.

```bash
python3 pr_core_export.py --datalog          # SmolLM-135M, r=92
# -> head 4.6M floats vs 28.3M = 6.2× smaller (LOSSY); decode kept 67% on 450 fresh held-out decisions
# -> factored-readout .dl: PR-core argmax == full argmax (MATCH)
souffle -D- prcore_head/smollm.prcore.dl     # -> best(28), the factored readout run in a neutral engine
```

| `pr_core.py` | the two-knob operating table (PR-core / span90 / wide) + the margin-gated hybrid, on the readout-aligned basis. |
| `pr_core_v2.py` | the router-salvage attempts (Q1 second-stage rank-agreement gate, Q3 activation-covariance whitening) — **both fail**; the decode floor is τ\*. |
| `pr_core_export.py` | **ships the lever**: re-loadable lossy head (`.prcore.npz`+`.json`), held-out verification, and the souffle-runnable factored-readout `.dl`. |
| `pr_core_spec.py` | **speculative-decoding / shortlist evaluation**: no multi-token speedup (the draft shares the whole stack); the single-position shortlist's top-K recall plateaus at **~80%** (r=92), *flat across margin* and two synthetic prompt distributions, **0%** certifiable — a fourth τ\* confirmation. A compute-mode quality bump, not an exactness recovery. |
| `bpe.py` | self-contained byte-level BPE encoder for `smollm.tokenizer.json` (no `torch`/`tokenizers`/`regex` dep) — gives real in-distribution token sequences for `real_recall.py`. |
| `real_recall.py` | **the forge/retrievable split is SEMANTIC**: teacher-forces 21 real passages (1190 decisions), recall vs margin **and** token type. Content-word prediction = forge tax (R@32 **56%** at r=92, 77% at r=256); format tokens (punct/space/digit) = retrievable (R@32 **94–100%**). Margin plateaus ~70% (refutes the high-margin-rescue hypothesis); real text is *harder* than synthetic (random prompts inflate recall via format-token fallback). |
| `pr_core_gate.py` | **position-adaptive codec + the cheap gate**: the free core-class gate fails (lossy core over-predicts format tokens), but a linear probe decodes content-vs-syntax at **~83% bal-acc from the free r-dim `Sx`**. Probe-gated codec compute win scales with format share — prose 1.0× / code 1.4× / **Lisp 1.7× @ 98% exact** (paren-heavy = the favorable extreme). A compute lever, not an exactness escape (τ\* stands). |
| `lo1_localize.py` | **Q1 mechanistic localization**: decomposes the content-probe coordinate `ŵ·x` into each head/neuron's exact discriminative contribution. The content/syntax distinction is **83% MLP-neuron, 17% head, 69% in late layers (L20–29)**, semi-localized — one last-layer neuron (**L29 #906**) carries 13% of the whole separation, top 10% of neurons carry 90%. A late-layer MLP feature with a dominant detector neuron. |
| `grammar_recall.py` | **the axis is open- vs closed-class LEXIS**: closed-class function words recover like punctuation (R@32 **94%**); open-class content collapses (R@32 **17%**). Controls rule out alternatives — Lisp's positionally-pinned verb-first operator slot is still at the floor (R@32 23%), and the highest-margin decisions (Lisp args, margin 1.61) recover at 27%. The forge tax = the cost of selecting from the open lexicon. |
| `info_rank.py` | **τ\* is the text's ENTROPY**: per-token *recoverable rank* (min r for core top-1 == full argmax) rises monotonically with self-information — Spearman **+0.83** vs baseline-logit rarity, **+0.67** vs −log₂ frequency; median rank 16→256→512 by frequency band. The optimal lens allocates rank by frequency, so the heavy-tailed decision spectrum (α≈0.97) *is* the Zipfian tail of language. Plots `info_rank.png`. |
| `lo1_ladder.py` | **scale test (135M→360M→1.7B)**: the recoverable-rank↔entropy law *strengthens* with scale (Spearman 0.67→0.72→0.76); the relative tax shrinks (content ρ/d 1.00→0.75) but absolute rank grows (576→1536 dims) — scale changes the constant, not the floor. Plots `lo1_ladder.png`. |
| `tokipona_recall.py` | **cross-linguistic journey (Toki Pona · Laundry · Lisp · English · Finnish)**: perplexity and recoverable rank *dissociate* — English (best-modeled, 2.4 bits/tok) is *least* compressible (ρ/d 0.33), Toki Pona/Finnish (model's hardest) *most* compressible (0.03/0.01). Driver is **distinct a\*** (output diversity). **Laundry** = the no-OOD-confound control: in-distribution English (4.2 bits/tok) but constrained (86 outputs) ⇒ *genuinely* compressible (ρ/d 0.08, R@32 85%). Cross-lens (Toki/Finn @Eng) jumps to ρ/d=1.00. τ\* = effective rank of the *output* distribution. Plots `entropy_spectrum.png`. |
| `worst_case.py` | **the theoretical worst case** (pure readout geometry, no forward): `recoverable rank ≈ min(exp(H_output), d)`. Zipf-skew sweep — eff-vocab 2048→10 as s:0→1.7, ρ/d 0.89→0.01 (Zipf *saves* language). Uniform-over-m sweep — rank = min(m,d) exactly. The worst-case conlang is **flat anti-Zipf frequency over ≥d forms**, not "many cases". Plots `worst_case.png`. |
| `prcore_head/` | the exported heads + emitted `.dl` (gitignored). |

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
