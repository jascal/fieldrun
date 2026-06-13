#!/usr/bin/env python3
"""LO1 attack — does a matrix/operator-semiring valuation collapse the dense-Gram forge tax, or does the
required valuation width just track the PR (Grok's self-defeating obstruction)?

Grok's proposal (LOGIC_EXPORT LO1): lift scalar provenance to operators over the frame span{U_v}; the
coupling ⟨U_v,U_w⟩ is realized by matrix product inside the algebra, so a dense scalar clique becomes a
chain of matrix multiplications. The whole thing reduces to ONE measurable quantity:

    matrix-valuation width  =  effective rank of the dense fragment's Gram G.

So the descriptive escape works iff G is LOW-RANK relative to the scalar treewidth/PR. The obstruction is
whether that effective rank just tracks the participation ratio.

Part A (synthetic): make the mechanism exact — scalar clique width = k regardless, matrix width = effrank(G).
Part B (real, SmolLM-135M): is the candidate-token Gram (the literal LE-T2 object G_{vw}=⟨U_v,U_w⟩) low-rank
on FORGE-TAX (thin-margin) decisions vs RETRIEVABLE (thick-margin) ones?
"""
import os, sys
import numpy as np
import bundle_io as bio

HERE = os.path.dirname(os.path.abspath(__file__))

def effrank(G):
    """participation ratio of a PSD matrix's eigenvalues = (Σλ)²/Σλ²  (the operator-valuation width)."""
    w = np.linalg.eigvalsh(G); w = np.clip(w, 0, None)
    s = w.sum()
    return float(s * s / (np.square(w).sum() + 1e-30))

# ---------- Part A: the mechanism is exact ----------
def part_a(k=8):
    print("== Part A (synthetic): matrix-valuation width = effective rank of the fragment's Gram ==")
    print(f"   {k} propositions; scalar provenance treats dense coupling as a CLIQUE (treewidth ~ k-1 = {k-1}).")
    print(f"   {'true rank ρ':>12} {'scalar clique width':>20} {'matrix-valuation width = effrank(G)':>38}")
    rng = np.random.default_rng(0)
    for rho in [1, 2, 4, 6, 8]:
        B = rng.standard_normal((k, rho))
        G = B @ B.T                                  # dense Gram of TRUE rank ρ
        print(f"   {rho:>12} {k-1:>20} {effrank(G):>38.2f}")
    print("   => the operator valuation collapses the clique to width≈ρ; the escape works IFF the dense")
    print("      fragment's Gram is LOW-RANK. The empirical question is whether real forge-tax Grams are.\n")

# ---------- Part B: is the real dense-Gram coupling low-rank where it matters? ----------
def part_b(stem, n_pos=150, K=32, seed=1):
    man, W = bio.read_bundle(stem)
    cfg, cfg_f = man["config"], man["config_f"]
    U = W["lm_head"] if cfg[7] == 0 else W["embed"]      # unembedding directions [vocab, d]
    rng = np.random.default_rng(seed); V = int(cfg[6])
    print(f"== Part B (SmolLM-135M): effective rank of the candidate Gram G_cand = U_cand U_candᵀ (K={K}) ==")
    recs = []
    for _ in range(n_pos):
        ids = [int(t) for t in rng.integers(0, V, size=int(rng.integers(6, 14)))]
        lg = bio.forward(W, cfg, cfg_f, ids)
        order = np.argsort(lg)[::-1]
        margin = float(lg[order[0]] - lg[order[1]])
        cand = order[:K]
        Uc = U[cand].astype(np.float64)
        G = Uc @ Uc.T
        recs.append((margin, effrank(G)))
    recs.sort(key=lambda r: r[0])
    n = len(recs); t = n // 3
    strata = [("forge-tax  (thin margin)", recs[:t]),
              ("middle",                    recs[t:2*t]),
              ("retrievable (thick margin)",recs[2*t:])]
    print(f"   {n} held-out decisions; scalar clique width over candidates = K = {K}.")
    print(f"   {'stratum':<28}{'n':>4}{'mean margin':>13}{'mean effrank(G_cand)':>22}{'effrank / K':>13}")
    for lbl, g in strata:
        mm = np.mean([r[0] for r in g]); er = np.mean([r[1] for r in g])
        print(f"   {lbl:<28}{len(g):>4}{mm:>13.3f}{er:>22.2f}{er/K:>13.3f}")
    thin = np.mean([r[1] for r in recs[:t]]); thick = np.mean([r[1] for r in recs[2*t:]])
    print(f"\n   READING: matrix-valuation width (effrank) is {'LOWER' if thin<thick else 'HIGHER'} on forge-tax than "
          f"retrievable decisions ({thin:.1f} vs {thick:.1f}).")
    print("   effrank/K << 1  => the token-coupling Gram is low-rank => the DESCRIPTIVE escape (LO1) has traction")
    print("   on that axis: the matrix valuation carries the dense coupling at width far below the scalar clique.")
    print("   (Caveat: this is the TOKEN-coupling Gram — the literal LE-T2 object. The CIRCUIT-coupling axis")
    print("    (within-block PR≈45) needs per-component contribution vectors — a probe to add. That is where")
    print("    Grok's rank-tracks-PR obstruction would bite if it bites.)")

if __name__ == "__main__":
    part_a()
    part_b(os.path.join(HERE, "smollm", "smollm"))
