# TurboQuant KV-Cache Mode and the Margin–Distortion Probe

**An optional, unbiased KV-cache quantizer — and the facet-margin-vs-distortion measurement that turns the
tropical proposal's quantization hypotheses (TO7/E7) into measured-with-a-bound results**

*Status: research proposal. The constructive/quantization companion to
[`TROPICAL_PROPOSAL.md`](./TROPICAL_PROPOSAL.md) (§8 TO7, §12 E7) and
[`PROVABLE_OPT_PROPOSAL.md`](./PROVABLE_OPT_PROPOSAL.md). Source: Zandieh, Daliri, Hadian & Mirrokni,
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," Google Research / DeepMind /
NYU, arXiv:2504.19874, 2025; with the distortion-rate algebra kernel-checked in the i-orca corpus
`examples/turboquant/` (11 theorems, Isabelle2025-2, zero `sorry`). Two deliverables, both **opt-in**:
(A) a TurboQuant KV-cache mode that leaves the exact f32 path the default, and (B) a margin–distortion
probe. Nothing here changes existing behaviour unless a TurboQuant flag is passed.*

---

## Abstract

fieldrun's decision and its entire explain/probe surface are **inner products**: the next token is
`argmax_v ⟨r, U_v⟩`, DLA is `⟨U_v, contribution⟩`, the power-diagram geometry is `⟨U_t, U_v⟩` and
`‖U_t − U_v‖²`. **TurboQuant** is the one quantizer with a *proved unbiased inner-product estimator* and
near-optimal distortion at **every** bit-width and dimension, and it is **data-free** (no calibration
set) — matching fieldrun's pure-Rust, no-GPU `convert` stance. This proposal adds (A) an **optional
TurboQuant KV-cache mode** — fieldrun's KV cache is f32 today (`rope::forward_block_capture` holds
`Array2<f32>` k/v), so this is a *new capability*, not a replacement for the exact int8/int4 **weight**
path — and (B) a **margin–distortion probe** built on the existing tropical facet kernel
(`tropical::nearest_facet`). The intellectual core (§4): TurboQuant's **random rotation makes its
distortion isotropic**, so the tropical **facet margin** (TT2, already the normalized distance to the
decision boundary) is *exactly* the quantity that predicts flip stability, with a **closed-form
threshold `ρ(b,d) ≈ c·2⁻ᵇ/√d`** in which the `‖U_t − U_v‖` cancels. That converts TROPICAL TO7/E7 from
empirical hypotheses into a measurement against a bound. The cost is a deliberate, opt-in shift in the
explain contract from **exact** to **unbiased-in-expectation** (§6).

---

## 1. Background — TurboQuant

TurboQuant compresses a high-dimensional vector to `b` bits/coordinate while preserving its geometry. It
(i) **randomly rotates** the input — making each coordinate a concentrated **Beta** distribution that
converges to `N(1, 1/d)` in high dimension — then (ii) applies **optimal scalar Lloyd–Max** quantizers
per coordinate (the rotation near-decorrelates coordinates, so scalar quantizers are near-optimal), and
(iii) for **unbiased inner products** composes the MSE quantizer with a **1-bit Quantized-JL** transform
on the residual. For any worst-case unit vector, at every `b` and `d`:

| | upper bound (achievable) | lower bound (Shannon + Yao) |
|---|---|---|
| **MSE** | `D_mse ≤ (√3π/2)·4⁻ᵇ` | `≥ 4⁻ᵇ` |
| **inner product** | `D_prod ≤ (√3π²·‖y‖²/d)·4⁻ᵇ` | `≥ (‖y‖²/d)·4⁻ᵇ` |

with `E[⟨y, dequant(x)⟩] = ⟨y, x⟩` (**unbiased**) and the inner-product distortion **decaying in `d`**
(high-dimensional advantage). The constant gap to the information floor is `√3π/2 ≈ 2.7` (`2.7 < √3π/2 <
2.73`); a 4-bit MSE distortion is `< 0.011`.

**Formal status.** The i-orca corpus kernel-checks the **rate algebra** the paper highlights — the ratio
is the constant `√3π/2`, the geometric `4⁻ᵇ` decay, the `≈2.7` constant, the dimensional decay, and the
inner-product unbiasedness *given* coordinate unbiasedness. The **achievability** (random rotation, Beta
concentration, Lloyd–Max optimality) and the **Shannon + Yao lower bound** are explicitly *meta* (not
formalised). So a fieldrun "provable stability" claim **assumes** the paper's achievability and *measures*
the consequence; it does not re-derive the source-coding engine. This proposal is honest about that line.

---

## 2. Why fieldrun — the inner-product surface, and the gap

The whole program is inner products against fixed frames, and fieldrun already has a mature *weight*
quantization surface (`src/bundle.rs`): `I8` (per-output-column symmetric int8), `I4` (group-wise
symmetric int4), `Q4A` (group-wise affine int4), `RowI8` (per-row int8 for embed/unembed). **All of these
quantize weights; the KV cache is f32.** That is the gap TurboQuant fills, and it fills it where the
algorithm is strongest:

- **KV cache** = streaming, per-vector, read-many-times — exactly TurboQuant's online setting; its
  unbiased inner products preserve attention scores `⟨q, k⟩`.
- **The decision is inner products** — so an unbiased quantizer keeps the *logits* unbiased, hence the
  DLA, the contrib decode (LE-T5), and the facet geometry unbiased *in expectation* (§6).
- **fieldrun already measures the geometry** — `--probe-facet`/`--probe-tropical` compute the exact facet
  distances/angles, so fieldrun is the instrument that can *measure* TurboQuant's distortion-vs-margin
  tradeoff (§4, deliverable B). The marriage is two-way: TurboQuant gives the tropical paper its
  quantization theorem; the tropical probe gives TurboQuant its empirical validation in a real model.

---

## 3. Deliverable A — the optional KV-cache TurboQuant mode

**Default off.** A new flag `--kv-quant turbo[:b][,mode=mse|prod]` (b default 8) switches the K/V cache
from f32 to TurboQuant; absent the flag, the exact f32 path and `--verify-cache` are byte-for-byte
unchanged. New module `src/turboquant.rs` (pure geometry + codec; no GPU, no forward pass):

1. **Rotation `R` — structured, `O(d log d)`.** A random `±1` diagonal sign flip composed with a
   fast Walsh–Hadamard transform (`SRHT`-style), seeded **per (layer, head)** so it is *reproducible*
   run-to-run (not byte-identical to f32 — see §6). Orthogonal ⇒ exactly invertible up to the scalar
   quantization; preserves `⟨q, k⟩` because `⟨R q, R k⟩ = ⟨q, k⟩`.
2. **Data-free Lloyd–Max levels.** The post-rotation per-coordinate marginal is *known* (the
   `Beta((d−1)/2, (d−1)/2)`-type law → `N(0, 1/d)` after centering), so the `2ᵇ` Lloyd–Max levels are
   **precomputed once from the analytic density**, not fit to data — preserving `convert`'s
   no-calibration property. Store per vector: the scale (its norm), and `b`-bit codes.
3. **Unbiased-dot mode (`prod`).** Append the 1-bit Quantized-JL residual so `E[⟨q, dq(k)⟩] = ⟨q, k⟩`
   exactly (the attention-score-preserving mode); `mse` mode skips it (cheaper, biased-but-tiny).
4. **Memory.** f32 → `b`-bit + scale: ≈ 4× at 8 bits, ≈ 8× at 4 bits, on the KV cache (the dominant
   long-context memory cost). The compression is reported alongside the distortion.

**Integration.** Encode K/V at write in `forward_block_capture`; decode on read in the attention dot.
Behind the flag the f32 buffers become a `KvStore { codes, scales, rot_seed }`; the exact path stays the
default branch. `explanation_stream` (the KV-cached explain) runs unchanged on the f32 path; on the
TurboQuant path it produces the **unbiased** explain of §6.

**Two honest interaction caveats (load-bearing, see TQ-O5/O6):**
- **RoPE.** K is RoPE-rotated before the dot; quantize the **post-RoPE** K (the thing actually stored and
  dotted) so `⟨q, k_roped⟩` is the preserved quantity.
- **Head dimension is small.** `head_dim` (e.g. 64 on Qwen2.5-0.5B) is *not* high-dimensional, so the
  per-head `1/d` advantage is weak. Option: rotate/quantize across the **full hidden `d`** (concatenated
  heads) before the head split, recovering the high-`d` regime — at the cost of doing the rotation before
  RoPE. This is the main design fork to resolve in a prototype.

---

## 4. Deliverable B — the margin–distortion probe (the TO7/E7 settle)

**The centerpiece.** Write `L_v = ⟨U_v, r⟩`. Under TurboQuant the residual (or unembedding) becomes
`r̂` with `E[r̂] = r` and **isotropic** per-coordinate distortion (isotropy is *exactly* what the random
rotation buys). The signed distance from `r` to the `t`–`v` facet is the normalized margin
`m = (L_t − L_v)/‖U_t − U_v‖` (TT2, already computed by `tropical::nearest_facet`). The perturbation of
that signed distance is the projection of `r̂ − r` onto the **unit** facet normal `(U_t − U_v)/‖U_t − U_v‖`;
because the distortion is isotropic, its variance is **direction-independent** — call its RMS `ρ(b, d)`.
Therefore the decision flips across the facet roughly when

> **`m < z · ρ(b, d)`,  with  `ρ(b, d) ≈ c · 2⁻ᵇ / √d`  (`c` a small constant from `√3π²`).**

The `‖U_t − U_v‖` **cancels** — the tropical *normalized* margin is precisely the right quantity, and the
threshold is a closed form in `(b, d)`. This is why TurboQuant and the power-diagram view fit so cleanly:
the rotation isotropizes the noise, and the facet margin is the isotropic-noise stability radius.

**Probe `--probe-distortion [--kv-bits b] [--mode mse|prod]`** (or `--probe-tropical --distortion`):
per position compute (i) the facet margin `m` and the closed-form `ρ(b,d)` (free, from the existing
kernel), and (ii) the **actual** flip when the decision is recomputed under TurboQuant-compressed inputs.
Report, grouped by route (RETRIEVED/SELECTED/COMPOSED) and by `m/ρ` bucket:
- `flip%` vs `m/ρ` — expected to fall through ~1 (the predicted threshold), monotone across routes
  (COMPOSED, small `m`, flips first — the E7 gradient with a *bound* attached);
- the **unbiasedness check**: `mean(L̂_v − L_v) ≈ 0` with empirical variance `≈ D_prod`;
- the **stability frontier**: the fraction of the vocabulary provably-stable at `b` bits
  (`m > z·ρ`), the retrievable-under-compression set.

This reuses `tropical::nearest_facet` (shared with `--probe-facet`/`--probe-tropical`) plus the
`turboquant` codec, so the facet numbers are identical-by-construction to the other probes.

---

## 5. Claims by status

| Claim | Content | Status |
|---|---|---|
| **TQ-T1** | KV TurboQuant preserves `⟨q, k⟩` unbiasedly (`prod` mode) | Inherited (paper Thm 2, given achievability) |
| **TQ-T2** | `m > z·ρ(b,d)` ⇒ decision stable under `b`-bit TurboQuant; `ρ ≈ c·2⁻ᵇ/√d`, `‖U_t−U_v‖` cancels | **The contribution** — closed-form, falsifiable (§4, E-TQ2) |
| **TQ-T3** | explain (DLA / logits / contrib / LE-T5 `Σcontrib==logit`) is unbiased-in-expectation under TurboQuant | Structural (linearity + Thm 2); variance `D_prod` |
| **TQ-T4** | margin-adaptive bit allocation (bits ∝ `−log m`) minimises flips at a fixed bit budget — the forge-tax-aware allocator | **Conjecture** (TO5/forge-tax tie) |
| **TQ-T5** | concrete operating points (4-bit MSE `< 0.011`, etc.) | Inherited (i-orca kernel-checked algebra) |

---

## 6. Implications for the existing explain features

This is the deliberate trade, and it is **opt-in** (default-off): TurboQuant moves the explain contract
from **exact + deterministic** to **unbiased + bounded-distortion**.

- **DLA / logits / contrib decode (LE-T5 `Σcontrib == logit`; the 12/12-faithful decode).** Under
  TurboQuant these are **unbiased estimators**, not exact identities: `Σcontrib == logit` holds *in
  expectation*, variance `D_prod`. Attribution *ordering* is preserved in expectation; averaging over
  positions/runs recovers the truth. A single explain trace becomes a *sample*, not a certificate — so
  the faithful-by-construction language of `LOGIC_EXPORT` (LE-T5) needs an "in expectation" qualifier on
  the TurboQuant path.
- **`--verify-cache` (byte-identical KV).** **Breaks** — the random rotation is not bit-reproducible
  against f32. Replace, on this path only, with a **reproducible-seed + distortion-bounded** check:
  fixed seed ⇒ run-to-run bit-reproducible; validate `‖dq(k) − k‖² ≤ D_mse·‖k‖²` and
  `mean(L̂ − L) ≈ 0` instead of byte-identity. The f32 path keeps `--verify-cache` exactly.
- **`decompose_descent` / the irreducible atom / E2.** The descent thresholds on `m_j^v > 0`; the noise
  can flip marginal survivors, so atom membership (`σ(t)`, `interior%`, `necessary%`) acquires variance
  **near the boundary** — but `DescentResult::min_slack` and the facet margin already *measure* which
  atoms are fragile, so fieldrun can **predict** which explanations survive compression (E-TQ4). This is
  a synergy, not just a hazard.
- **`--probe-facet`/`--probe-tropical`/`headgate.rs`.** The Gram/facet quantities become unbiased
  estimates with known variance — which *is* the E7 measurement. Head-gating should gate only when the
  margin exceeds the distortion (`m > z·ρ`), i.e. TQ-T2 is also the correct gate condition under
  compression.

---

## 7. Implementation surface (sketch)

```rust
// src/turboquant.rs — pure codec + geometry; no forward pass, no I/O.
pub struct Rotation { seed: u64, d: usize }            // ±1 diagonal ∘ FWHT (SRHT), O(d log d), orthogonal
impl Rotation { pub fn apply(&self, x: &mut [f32]); pub fn invert(&self, x: &mut [f32]); }

pub struct Codec { bits: u8, levels: Vec<f32>, mode: Mode }  // Mode::{Mse, Prod}; levels precomputed from N(0,1/d)
impl Codec {
    pub fn encode(&self, rot: &Rotation, x: &[f32]) -> (Vec<u8>, f32);   // (b-bit codes, scale=‖x‖)
    pub fn decode(&self, rot: &Rotation, codes: &[u8], scale: f32) -> Vec<f32>; // E[decode] = x
}

/// Closed-form per-coordinate RMS distortion ρ(b,d) and the inner-product distortion D_prod(b,d,‖y‖²).
pub fn rho(bits: u8, d: usize) -> f32;                 // ≈ c · 2^-bits / sqrt(d)
pub fn d_prod(bits: u8, d: usize, ynorm2: f32) -> f32; // (√3 π² ‖y‖² / d) · 4^-bits
```

New CLI: `--kv-quant turbo[:b][,mode=mse|prod]` (deliverable A), `--probe-distortion [--kv-bits b]
[--mode …]` (deliverable B). Unit tests: rotation is orthogonal/invertible to `1e-5`; `encode∘decode`
is unbiased on synthetic `N(0,1/d)` (mean error ≈ 0); `rho`/`d_prod` scale as `2⁻ᵇ`/`4⁻ᵇ` and `1/√d`;
Lloyd–Max levels monotone and symmetric. (Pure-geometry tests, no model — same discipline as
`src/tropical.rs`.)

---

## 8. Experiment plan

| # | Experiment | Method | Success criterion |
|---|---|---|---|
| **E-TQ1** | KV memory vs quality | `--kv-quant turbo:{8,4,2}`, held-out perplexity | ≈4×/8× KV shrink at small Δperplexity; matches `D_mse` scaling |
| **E-TQ2** | Margin–distortion flip curve (**TO7/E7**) | `--probe-distortion` | `flip%` falls through `m/ρ ≈ 1`; threshold ~constant across `b`; COMPOSED flips first |
| **E-TQ3** | Unbiasedness | `mean(L̂_v − L_v)` and its variance | mean ≈ 0; variance ≈ `D_prod` (TQ-T1/T3) |
| **E-TQ4** | Explain stability under compression | atom `σ(t)`/`interior%` under KV-TQ vs f32 | divergence concentrated where `min_slack`/margin is small (predicted, not surprising) |
| **E-TQ5** | Margin-adaptive vs uniform bits | bits ∝ `−log m` at matched budget | fewer flips than uniform at equal mean bits (TQ-T4) |

Priority: **E-TQ2 first** (it is the TO7/E7 payoff and needs only the probe + codec, no full-model
re-quant), then E-TQ1/E-TQ3, then E-TQ4/E-TQ5. All run on a single 0.5B rope model. **Heavy runs deferred
until the box frees up** (a long job is currently resident).

---

## 9. Open problems

- **TQ-O1 (achievability is meta).** fieldrun *measures* TurboQuant's behaviour and *assumes* the paper's
  achievability (rotation/Beta/Lloyd–Max) and Shannon bound; it does not re-derive them. TQ-T2's "provable"
  is "provable *given* the paper" + the isotropy/projection argument of §4.
- **TQ-O2 (hot-path cost).** KV amortises the `O(d log d)` rotation (encode once, read many times);
  **activations** in the int8 decode loop likely do not — so this proposal targets KV (and optionally the
  unembedding), not the per-matmul activation quant.
- **TQ-O3 (`prod` vs `mse`).** When is the extra 1-bit QJL residual (unbiased dot) worth it vs plain MSE?
  Likely: `prod` for the *unembedding* (logit fidelity), `mse` for the *KV* (attention is more forgiving).
- **TQ-O4 (validation notion).** Formalise the "reproducible-seed + distortion-bounded" replacement for
  byte-identity on the TurboQuant path (the §6 contract change); keep the exact path's `--verify-cache`.
- **TQ-O5 (RoPE placement).** Quantize post-RoPE K (preserves `⟨q, k_roped⟩`) vs pre-RoPE + rotate-on-
  decode — interacts with TQ-O6.
- **TQ-O6 (per-head vs full-`d`).** `head_dim` is small, weakening the `1/d` advantage; rotating across the
  full hidden `d` recovers it but moves the rotation before the head split / RoPE. The central prototype
  fork.

---

## 10. Related work & provenance

- **TurboQuant** — Zandieh, Daliri, Hadian & Mirrokni, *"TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate,"* arXiv:2504.19874, 2025. The source algorithm and bounds.
- **i-orca `examples/turboquant/`** — a kernel-checked formalisation of the distortion-rate algebra (11
  theorems under Isabelle2025-2, zero `sorry`): the rate is geometric `4⁻ᵇ`, the near-optimality ratio is
  the constant `√3π/2 ∈ (2.7, 2.73)`, the inner-product bound decays in `d`, and the inner product is
  unbiased given coordinate unbiasedness. The *achievability* and Shannon lower bound are meta there too.
- **fieldrun companions:** [`TROPICAL_PROPOSAL.md`](./TROPICAL_PROPOSAL.md) (TT1/TT2 facet geometry; TO7/E7
  — this proposal supplies their bound), [`PROVABLE_OPT_PROPOSAL.md`](./PROVABLE_OPT_PROPOSAL.md) (the
  forge tax / bit-allocation tie, TQ-T4), [`FINDINGS.md`](./FINDINGS.md) §5b (the measured facet anchors),
  and the existing `src/bundle.rs` weight dtypes (I8/I4/Q4A/RowI8) this sits beside.

The stake: **the KV cache compressed by a data-free, unbiased, near-optimal quantizer — with the tropical
facet margin as the exact, closed-form predictor of which decisions (and which explanations) survive the
compression.** Opt-in, default-off; the exact path and its byte-identity guarantees are untouched.
