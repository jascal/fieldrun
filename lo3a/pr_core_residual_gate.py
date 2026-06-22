#!/usr/bin/env python3
"""Can a cheap gate make the lossy PR-core decode-EXACT? (advancing the lossy unembed path)

pr_core.py shows the readout-aligned rank-r core keeps ~67% of decodes at ~6× compression, and that its MARGIN GATE
(route thin-core-margin to the full readout) does NOT reach exactness — overall-kept stays ~70%, because the failures
are tokens where the core is *confidently wrong* (wide core-margin), which the margin gate accepts. core-margin ≠
core-correctness. This tests a better gate: the **projection residual** ρ = ‖(I−SS^T)x‖/‖x‖ — the fraction of x the
rank-r basis missed. If correct ⟺ small ρ, gating on ρ (accept core when ρ < τ, else full) makes the hybrid reach
~exactness at a real average compression. Reports, on REAL in-distribution contexts (greedy-decoded), how well each
signal separates correct-vs-wrong core decodes, and the hybrid operating curve for both gates.

Run from lo3a/: python pr_core_residual_gate.py [smollm|smollm360]
"""
import os, sys
import numpy as np
import bundle_io as bio

MODEL = sys.argv[1] if len(sys.argv) > 1 else "smollm"
R = 92


def gen_decisions(W, cfg, cfg_f, vocab, seeds, steps=24):
    out = []
    rng = np.random.default_rng(0)
    starts = [rng.integers(5, vocab - 1, size=2).tolist() for _ in range(64)]
    for s in seeds:
        ids = starts[s]
        for _ in range(steps):
            lg, xf = bio.forward(W, cfg, cfg_f, ids, want_x=True)
            o = np.argsort(lg)[::-1]
            out.append((int(o[0]), o[:9].copy(), xf.astype(np.float64)))
            ids.append(int(o[0]))
            if len(ids) > 40: ids = ids[-40:]
    return out


def main():
    stem = f"{MODEL}/{MODEL}"
    man, W = bio.read_bundle(stem); cfg, cfg_f = man["config"], man["config_f"]
    U = (W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain = W["norm"].astype(np.float64)
    d, V = int(cfg[4]), int(cfg[6])
    tr = gen_decisions(W, cfg, cfg_f, V, range(0, 8))
    te = gen_decisions(W, cfg, cfg_f, V, range(8, 16))
    # readout-aligned basis S (decode-optimal) from TRAIN winner−competitor diffs, normalized. NB bio.forward(want_x)
    # returns xf with the final-norm gain ALREADY applied, so gain must NOT be re-applied to U here (else gain²).
    rows = [(lambda df: df / (np.linalg.norm(df) + 1e-30))(U[p] - U[v]) for p, o, _ in tr for v in o[1:]]
    _, _, Vt = np.linalg.svd(np.array(rows), full_matrices=False)
    S = Vt[:R]                                                # [R, d] basis
    A = S @ U.T                                               # [R, vocab]  (xf already carries gain)
    full = V * d
    # per test decision: core argmax, core-margin, projection residual ρ, and whether the core is correct
    recs = []
    for p, _, x in te:
        cl = (S @ x) @ A                                      # rank-R core logits
        sx = np.sort(cl)[::-1]; cm = float(sx[0] - sx[1])
        proj = S.T @ (S @ x)
        rho = float(np.linalg.norm(x - proj) / (np.linalg.norm(x) + 1e-30))
        recs.append((int(np.argmax(cl)) == p, cm, rho))
    correct = np.array([r[0] for r in recs]); cm = np.array([r[1] for r in recs]); rho = np.array([r[2] for r in recs])
    n = len(recs); kept = correct.mean()
    print(f"== PR-core gate test ({MODEL}; r={R}; {len(tr)} train / {n} test decisions) ==")
    print(f"   compression vocab×d/(vocab×r) ≈ {full/(V*R):.1f}×   core decode-kept = {100*kept:.0f}%")
    # does each signal separate correct from wrong? (AUC that the signal ranks WRONG above correct)
    def auc(sig, lo_is_wrong):
        w, c = sig[~correct], sig[correct]
        if not len(w) or not len(c): return float("nan")
        s = np.concatenate([w, c]); order = s.argsort(); rank = np.empty(len(s)); rank[order] = np.arange(1, len(s) + 1)
        a = (rank[:len(w)].sum() - len(w) * (len(w) + 1) / 2) / (len(w) * len(c))
        return a if not lo_is_wrong else 1 - a
    print(f"   separation of WRONG core decodes (AUC, 0.5=none): core-margin {auc(-cm, False):.2f} (wrong=low margin?) · "
          f"residual ρ {auc(rho, False):.2f} (wrong=high ρ?)")
    # hybrid operating curve: accept core when the gate says reliable, else full readout (exact)
    print(f"\n   {'gate':<12}{'thr':>7}{'→core %':>9}{'overall kept':>14}{'avg compression':>16}")
    for name, sig, accept in [("core-margin", cm, lambda t: cm >= t), ("residual ρ", rho, lambda t: rho <= t)]:
        for t in (np.quantile(sig, [0.3, 0.5, 0.7, 0.9]) if name == "core-margin" else np.quantile(sig, [0.1, 0.3, 0.5, 0.7])):
            acc = accept(t)
            ok = (correct & acc).sum() + (~acc).sum()        # core-correct where accepted, exact where routed to full
            cost = acc.sum() * (V * R) + (~acc).sum() * (V * d)
            print(f"   {name:<12}{t:>7.2f}{100*acc.mean():>8.0f}%{100*ok/n:>13.0f}%{full/(cost/n):>15.1f}×")
    print("   reading: a gate works if overall-kept ~100% while →core (cheap) stays high; compare the two signals.")


if __name__ == "__main__":
    main()
