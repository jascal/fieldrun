# Router-oracle ceiling — a Stage-0 probe that killed the pageable-MLP-MoE direction

**Verdict: NEGATIVE. Do not build a learned router + pageable MLP "experts" on the density-bucketed partition.**
This is a deliberately cheap decision gate (~3 CPU-hours) run *before* any router work. It measures the *ceiling* —
what a perfect (oracle) router could achieve — and the ceiling is not worth chasing.

Branch: `research/oracle-router-probe`. Not intended for merge; kept for the record.

## Why this was run

The "optimization" roadmap (README Tier C, `DENSITY_BUCKETING.md`, `SWEEP_RESULTS.md`) proposes turning the offline
density-bucketed neuron partition into a **runtime MoE over the dense MLP**: carve each layer's `ffn` neurons into `G`
groups ("experts"), keep a learned router that predicts which groups a token needs *before* gate/up, and page group
weights on demand via the existing expert-offload path (`bundle.rs` `ExpertSpec`/`expert_mm`/`prefetch`, already used by
Qwen3-MoE / Gemma-4 / MiniMax). `--route-frac` today is accuracy-only (gate/up still run in full, scalar down path), so
this would be the first real wall-clock/memory win.

Before committing to the router (Stages 1–4: bundle format, distilled router, decode wiring, measurement), Stage 0 asks
the only question that gates all of it: **is there a peaky, reusable, group-roundable structure a router could exploit?**

## What the probe measures

`--mlp-oracle "G,..."` (added to the `--corpus-decompose` path) reuses the validated density-min descent
(`bucketing::atom_and_pred_at` → `explain::decompose_descent`). For each eval token it takes the **certificate** = the
minimal DLA-sufficient coalition of circuits (the set whose contributions alone keep top-1, to the linear-DLA
approximation `--probe-ablate` validated), restricts it to MLP-neuron circuits (attention stays dense), then for each `G`:

- clusters the neurons into `G` groups with the existing co-firing partition (`cluster_atoms`);
- routes each token to the groups its certificate touches and charges the **full neuron count of those groups**
  (whole groups page in / compute — the grouping tax is included);
- compares against a deterministic **random hash-grouping** at the same `G` (the chance floor);
- reports it all in runtime units (every MLP neuron = `3·d` MACs = `3·d` bytes int8).

It also emits a near-free **working-set growth curve** (cumulative distinct certificate-neurons vs corpus tokens) — the
expensive part is the descent, done once; the curve is a cumulative-distinct walk over the atoms it already computed.

Model: `Qwen2.5-0.5B-Instruct` (rope; 24 layers × 4864 ffn = 116,736 MLP neurons), `d=896`, ctx 48.

## Results

### The certificate is genuinely sparse (and stable)

| corpus | per-neuron floor (atom size) | MLP-free tokens |
|---|---|---|
| 300 tokens | 2.65 neurons/tok | 38% |
| 10,000 tokens | 2.79 neurons/tok | 38% |

So ~2.8 MLP neurons decide each token's argmax and **38% of tokens need zero MLP** (attention alone). An *idealized
per-neuron* oracle would skip ~99.99% of MLP FLOPs. This is a real interpretability asset.

### …but there is no compact recurring core (Heaps'-law growth, no plateau)

Distinct certificate-neurons vs corpus tokens (10k run):

| tokens | distinct | marginal new/tok |
|---:|---:|---:|
| 100 | 94 | 0.94 |
| 500 | 438 | 0.80 |
| 1,200 | 850 | 0.63 |
| 2,000 | 1,249 | 0.50 |
| 5,000 | 2,424 | 0.37 |
| 8,000 | 3,238 | 0.27 |
| 10,000 | 3,719 | 0.24 |

The marginal rate decays but never flattens — fitting `distinct ≈ Tᵝ` over 1k–10k gives **β ≈ 0.70**, the classic
vocabulary-growth signature. Extrapolating: ~16% of the MLP at 100k tokens, ~80% at 1M tokens. At any deployment scale
the "hot set" is most of the model. **The pageable-experts assumption (a small resident hot set) does not hold.**

### …and whole-group rounding is fatal, and co-firing ≈/< random

Grouping tax = (whole-group neurons/tok) ÷ per-neuron floor. At 10,000 tokens (floor 2.79):

| G | real n/tok (tax×) | random n/tok (tax×) |
|---:|---:|---:|
| 8 | 904 (324×) | 1038 (372×) |
| 16 | 706 (253×) | 573 (205×) |
| 32 | 536 (192×) | 306 (110×) |
| 64 | 434 (155×) | 159 (57×) |
| 128 | 338 (121×) | 82 (30×) |

A ~2.8-neuron certificate scatters across the (large) groups, so you pay 100–320× the floor. The density-bucketing
co-firing partition **loses to a hash** at every `G ≥ 16` — it concentrates mass in big hub experts + a large residual
catch-all, so touching a hot group is expensive. To approach the floor you need near-per-neuron granularity, which is the
unstructured-sparsity regime `--route-frac` already showed isn't a wall-clock win. The tax got *worse* as the corpus grew.

## Why the decision is robust

The three failures compound, they don't trade off: the sparsity is **tiny** (good), but **unstructured**
(group-rounding-hostile) **and non-recurring** (Heaps growth). And this used the *certificate* — the most optimistic
"needed" set. A faithful-logits requirement (preserve values, not just argmax) needs strictly more neurons, making
structured routing look only worse. Even a partition that beats random can't fix unbounded working-set growth or the
floor-vs-group-tax gap, so the optional "smarter partition" probe was also skipped.

**Where the certificate sparsity is actually useful:** the margin-gated retrieval-pruned *output head*
(`--pruned-head`, already shipped) — not the MLP.

## Caveats

- One model (Qwen2.5-0.5B), one corpus genre, ctx 48. But the mechanism (tiny certificate + Heaps working-set growth +
  whole-group tax) is architecture-general, not a Qwen quirk.
- The certificate is a *linear-DLA* sufficiency set, not a re-run forward; `--probe-ablate` validated that approximation
  against true causal ablation, and the conclusion is conservative regardless (faithful compute ⇒ worse).

## Reproduce

```bash
# small built-in corpus (no network):
.venv/bin/python scripts/make_holdout_qwen.py                       # -> holdout_qwen.json (385 tok)
./target/release/fieldrun --bundle Qwen2.5-0.5B-Instruct --ids holdout_qwen.json \
    --ctx 48 --n-eval 300 --corpus-decompose --experts 8 --decomp-k 4 --mlp-oracle "8,16,32,64,128"

# 10k-token growth curve (Project Gutenberg, public domain):
curl -s https://www.gutenberg.org/cache/epub/1342/pg1342.txt -o /tmp/book.txt
.venv/bin/python scripts/make_holdout_qwen.py /tmp/book.txt 12000 holdout_qwen_big.json
./target/release/fieldrun --bundle Qwen2.5-0.5B-Instruct --ids holdout_qwen_big.json \
    --ctx 48 --n-eval 10000 --corpus-decompose --experts 8 --decomp-k 4 --mlp-oracle "8,16,32,64,128"
```

Runtime ≈ 0.63 s/token on a 16-core x86_64 box (10k tokens ≈ 1h45m).
