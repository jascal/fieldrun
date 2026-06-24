# Certified mixed-precision quantization (Step 1)

**Margin-certified per-tensor bit allocation: as small as int4 where the decode tolerates it, int8/f16
where it doesn't — with a kernel-checked guarantee the decode is preserved within tolerance.**

*Status: build scope. The constructive follow-up to `experiments/certified_prune_step0/` (which measured
that **quantization, not pruning, is the cashable lever**: ~3–6 certified bits/position, ~7–8-bit
ship-able static, scale-stable). Certificate: i-orca `examples/pic_krein/PIC_Quant.thy`
(`margin_certified`, `frame_quant_logit_bound`, `quant_decode_preserved`). Distortion bound: the
TurboQuant `ρ(b,d) ≈ c·2⁻ᵇ/√d` of [`TURBOQUANT.md`](./TURBOQUANT.md) (kernel-checked,
`examples/turboquant/`). Opt-in: nothing changes unless a certified-quant bundle is built/loaded.*

---

## 0. Why this is small

The expensive parts already exist:

- **The loader/matmul already dispatch per-array dtype** (`bundle.rs:317` matches `a.dtype.as_str()` →
  `int8`/`int4`/`q4a`/`f16`/`rowi8`, mmap'd). A bundle with *different dtypes per tensor already loads and
  runs today.* No kernel, matmul, or loader change.
- **The quantizers exist** (`convert.rs` `put_i8`/`put_i4`/`put_q4a`/`put_f16`, with per-tensor/per-row
  scales and group-wise int4).
- **The calibration data exists** (`recursion_probe.rs:run_pil_dump` → per-block incidences + margins;
  the `experiments/certified_prune_step0` probe already turns it into per-tensor sensitivity + the
  per-position certified bit-width).

The only gap: `convert.rs` applies **one global `dtype`** to every `put_lin` (`convert.rs:123`). Step 1 =
(a) an offline **allocator** that chooses a dtype per tensor under the margin certificate, and (b) a small
convert change to read that allocation. Everything else is measurement.

## 1. The certificate → allocation

PIC logit `L(v) = Σ_t c_t(v) + b_v`, `c_t(v)` = tensor/block `t`'s incidence (what `--pil-dump` emits).
Quantizing tensor `t` to `b_t` bits perturbs its incidence by a bounded `δ_t(b_t)`:

> `δ_t(b_t) ≤ s_t · ρ(b_t, d)`,  `ρ(b,d) ≈ c·2⁻ᵇ/√d` (TurboQuant, data-free, unbiased)

where `s_t` is tensor `t`'s contribution magnitude (the per-block `β_t = max_v|c_t(v)|` from the Step 0
probe, or the L2 sensitivity for the RMS variant). By `margin_certified` (triangle over tensors), the
decode at position `x` is **preserved** when

> `2 · Σ_t δ_t(b_t) < margin(x)`.

A shippable (static) bundle must satisfy this for the calibration corpus `D`. Per the Step 0 finding that
the smallest-margin positions cap everything, allocation is **corpus-relative with a residue policy**: the
bottom-`q%` smallest-margin (forge-tax) positions are excused (kept at a high-precision floor or excluded
from the constraint), and the rest must hold. The Step 0-quant numbers predict the operating point:
**~7–8-bit-equivalent global at 10% residue, scale-stable** — i.e. mostly int8, int4 on the insensitive
tensors, f16 on the few margin-driving ones.

## 2. The allocator (offline)

Minimize bundle **bytes** (∝ inference bandwidth, the CPU bottleneck) over a per-tensor bit menu
`b_t ∈ {f16=16, int8=8, int4≈4}` (the existing dtypes; finer in v2) subject to the certificate:

```
for each position x in D (after dropping the residue%):  2·Σ_t s_t·ρ(b_t,d) < margin(x)
minimize  Σ_t bytes_t(b_t)
```

This is a **rate–distortion / Lagrangian bit-allocation** (reverse water-filling): spend bits where
`s_t` is large (sensitive tensors → high precision), coarsen where `s_t` is small. With a 3-level menu it
is a small ILP / greedy: start everything at int4, raise the tensors that violate the tightest
(min-margin-after-residue) constraint to int8, then f16, cheapest-sensitivity-per-byte first, until
feasible. Greedy-by-`Δbytes/Δslack` is near-optimal here; exact ILP is trivial at this size (≈ #tensors).
Output: a JSON map `tensor_name → dtype`.

## 3. Pipeline & hooks

1. **Calibrate** — `fieldrun --recursion-explain --pil-dump calib.jsonl` on the target corpus (exists).
   Reuse `experiments/certified_prune_step0/` to emit per-tensor `s_t` and per-position margins. *(The
   `--pil-dump` block index ↔ tensor mapping is the one thing to pin down: each DLA block is one
   attention or MLP write = one `l{L}.{attn/mlp}` linear group; map block→tensor names.)*
2. **Allocate** — offline solver (Python, ~100 lines) → `alloc.json` (`name → {f16|int8|int4}`).
3. **Emit** — `fieldrun convert … --dtype-map alloc.json` (**the one Rust change**): thread a
   `name → dtype` lookup into `put_lin` (`convert.rs:123`), defaulting to the global `--dtype` when a name
   is absent. ~20 lines; no new quantizer, no format change (per-array dtype tags already exist).
4. **Run & measure** — load the mixed bundle (works unchanged) and report:
   - **bundle bytes** & **bytes/token** (bandwidth — the cashable metric on CPU);
   - **tokens/sec** (CPU, the 96-core/251 GB regime) via existing bench;
   - **decode fidelity**: 0 flips on `D` by construction; **held-out flip rate** (the real test) and
     next-token top-1 vs the f16 reference (existing `scripts/bench.sh` / `validate_all.sh`).

## 4. Value proposition (vs what fieldrun ships today)

fieldrun already ships uniform int8 (90.8–100% quality) and uniform int4 (lossy; Step 0's control showed
unchecked aggressive quant flips decodes). Certified-mixed is the principled middle:

| | size | decode guarantee |
|---|---|---|
| uniform int8 | baseline | none (empirically good) |
| uniform int4 | smaller | none — flips on sharp/low-margin tensors |
| **certified-mixed** | **between, near int4** | **certified** (corpus, within residue): int4 only where the margin permits, int8/f16 where it doesn't |

So the win is **smaller-than-int8 with a certificate uniform-int4 cannot offer** — and the bytes saved are
real bandwidth on every token.

## 5. Phasing

- **v1 (this scope):** per-tensor allocation over `{f16,int8,int4}`, corpus-relative + residue,
  worst-case (triangle) δ. The minimal end-to-end: calib → allocate → `--dtype-map` convert → measure.
- **v2:** finer bit menu (5/6/7-bit kernels) for tighter allocation; per-output-row granularity (int4 is
  already group-wise); the RMS/signed δ (exploits cancellation, like Step 0's signed≫budget — allocates
  fewer bits); fold in the TurboQuant *unbiased* estimator for a tighter, data-free `δ_t`.
- **v3 (adaptive):** per-position bit-width at inference (the ~3–6-bit adaptive ceiling) — needs a runtime
  precision switch, more invasive; only if v1/v2 bandwidth wins justify it.

## 6. Certified vs heuristic vs measured (honesty)

- **Certified (kernel):** `2·Σδ < margin ⇒ decode preserved` (`PIC_Quant`); the TurboQuant distortion
  bound (`examples/turboquant`). The bundle is certified **on `D` within the residue** — corpus-relative,
  not global (the §7 activation-relative result; Step 0 confirmed static-global is margin-capped).
- **Heuristic:** the allocator's greedy/ILP search (standard rate-distortion; any feasible allocation is
  certified regardless of search quality — correctness decoupled from optimization, as in `PRUNE.md`).
- **Measured:** bandwidth, tokens/sec, held-out flip rate, top-1 vs reference.
- **Caveats:** δ is contribution-relative (a *favorable* proxy for weight-bits — a weight error spreads
  over the hidden dim); the triangle δ is conservative (v2 signed is tighter); argmax-lossless, not
  softmax-lossless; needs the block→tensor name map; calibration-corpus-relative.

## 7. Effort & risk

- **Rust:** ~20 lines (`convert.rs` per-tensor dtype map). **Low.**
- **Python:** the allocator + block→tensor mapping + measurement glue. ~1–2 days. **Low–medium.**
- **No** kernel/matmul/loader/format change (the de-risking finding of §0). **Main risk** is empirical:
  whether certified-mixed beats uniform int8 on bytes at acceptable held-out fidelity — which Step 0-quant
  already predicts (~7–8-bit static), so the prior is favorable.

## 8. Success criteria (go/no-go for v2)

Build v1; ship-or-stop on: **(a)** certified-mixed bundle is meaningfully smaller than uniform int8
(target ≥1.3× toward int4) at **(b)** held-out decode-flip rate ≈ the uniform-int8 baseline (certificate
holds out-of-calibration), with **(c)** a measured tokens/sec gain tracking the bandwidth reduction. If
(a)–(c) hold → v2 (finer bits, signed δ, per-row). If the residue needed to get below int8 is large
(forge-tax dominates) → the lever is upstream margins (pil), per the Step 0 verdict.

## 9. v1 build status & the embed finding (`experiments/certified_quant/`)

**Built:** the allocator (`step1_allocate.py`, runs on real data) + the convert `--dtype-map`
(`src/convert.rs` `DTYPE_MAP`/`dmap_dtype`, `src/main.rs`; `cargo check` clean). Loader unchanged (already
per-array-dtype). The only untested step is the end-to-end mixed bundle + tokens/sec, which needs source
safetensors (`convert` reads HF), not the local pre-converted `.fieldrun.bin`.

**v1 result (Qwen2.5-0.5B):** certified int8→int4 on **write tensors** is **modest** — 4/48 blocks at 10%
residue, `630→604 MB` (~7% of writes); mostly MLP. Consistent with Step 0's margin cap.

**The finding that reshapes the priority:** building v1 surfaced that the **`embed`/unembed read-out is
272 MB = 43% of the bundle** and the per-block write proxy **cannot** certify it (quantizing `embed` shifts
every logit via the tied `U_v`). The cashable MB are there, addressable by the **frame-quant bound**
(`PIC_Quant.frame_quant_logit_bound`), not the write-tensor downgrade.

**Revised next step → v1.5:** certify the embed/read-out via frame-quant — a `--source-dump`-style probe
emitting the raw `U_v` (already proposed for the forge-tax cert), or a direct embed-requant-and-measure.
v1's write-tensor path stays as the small complementary win; the elephant is the read-out.
