#!/usr/bin/env python3
"""Tropical-rank of a real model's decode — does a COMPACT tropical (max-plus) unembed exist?

The decode argmax_v ⟨x,U_v⟩ is a max-plus polynomial whose monomials are the unembed rows; the paper says the
irreducible computation is a TROPICAL-rank floor that linear/SVD rank can't see (which is why the SVD/Cauchy–Schwarz
compactions failed on real models). The tropical analog of "rank" here is the **support of the decision surface** — the
set of tokens that EVER win the argmax over the input distribution. If that support saturates at K ≪ vocab, a compact
tropical decode keeps only those K monomials (and reproduces the decode on that distribution); if it keeps growing
toward vocab, the unembed is genuinely tropical-high-rank ("no compact extension").

This generates DIVERSE in-distribution contexts by SAMPLING (greedy-decode degenerates), measures the saturation curve
(distinct winners vs #positions) and the held-out coverage (does a train-built winner support contain held-out winners,
vs its size). Reports whether a compact tropical support exists.

Run from lo3a/: python tropical_rank.py [smollm|smollm360]  [--steps N] [--seqs M]
"""
import os, sys, collections
import numpy as np
import bundle_io as bio

MODEL = next((a for a in sys.argv[1:] if not a.startswith("-")), "smollm")
SEQS = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seqs=")), 40)
STEPS = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--steps=")), 30)
T = 1.0


def sample_seqs(W, cfg, cfg_f, vocab, n_seqs, steps, rng):
    """Generate diverse in-distribution sequences by temperature sampling; record the model's argmax winner per step."""
    wins, seqs = [], []
    for _ in range(n_seqs):
        ids = rng.integers(5, vocab - 1, size=2).tolist()
        ws = []
        for _ in range(steps):
            lg = bio.forward(W, cfg, cfg_f, ids)
            ws.append(int(lg.argmax()))                         # the DECODE winner (tropical monomial that wins)
            p = np.exp((lg - lg.max()) / T); p /= p.sum()
            ids.append(int(rng.choice(len(p), p=p)))            # SAMPLE the continuation → diverse contexts
            if len(ids) > 40: ids = ids[-40:]
        wins.extend(ws); seqs.append(ws)
    return wins, seqs


def main():
    stem = f"{MODEL}/{MODEL}"
    man, W = bio.read_bundle(stem); cfg, cfg_f = man["config"], man["config_f"]
    vocab = int(cfg[6])
    rng = np.random.default_rng(0)
    train, _ = sample_seqs(W, cfg, cfg_f, vocab, SEQS, STEPS, rng)
    test, _ = sample_seqs(W, cfg, cfg_f, vocab, SEQS, STEPS, rng)
    print(f"== tropical-rank of the decode · {MODEL} · vocab={vocab} · {len(train)} train + {len(test)} test positions (sampled) ==\n")
    # saturation: distinct winners as we scan more train positions
    seen, curve = set(), []
    for i, w in enumerate(train, 1):
        seen.add(w)
        if i in (50, 100, 200, 400, len(train)) or i == len(train):
            curve.append((i, len(seen)))
    print("#   saturation (distinct winners vs positions scanned):")
    for i, dis in curve:
        print(f"#     {i:>5} positions → {dis:>5} distinct winners ({100*dis/vocab:.1f}% of vocab)")
    new_rate = (curve[-1][1] - curve[-2][1]) / max(1, curve[-1][0] - curve[-2][0]) if len(curve) >= 2 else 1.0
    print(f"#   new-winner rate in the last segment: {new_rate:.2f}/position  (→0 = saturating; ~const = growing)\n")
    # held-out coverage: a support of the top-K frequent train winners — does it hold the held-out winner?
    freq = collections.Counter(train)
    print("#   held-out coverage of a top-K-frequent-winner support (the compact tropical monomial set):")
    print(f"#     {'K':>6}{'% of vocab':>11}{'held-out covered':>18}")
    distinct_train = len(set(train))
    for K in [256, 1024, 4096, distinct_train]:
        if K > vocab: continue
        support = set(t for t, _ in freq.most_common(K))
        cov = sum(1 for w in test if w in support) / len(test)
        print(f"#     {K:>6}{100*K/vocab:>10.1f}%{100*cov:>16.0f}%")
    print(f"#   (distinct train winners = {distinct_train}; if coverage saturates near 100% at small K ⇒ compact tropical decode exists)")


if __name__ == "__main__":
    main()
