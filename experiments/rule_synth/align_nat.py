#!/usr/bin/env python3
"""Natural-text alignment (the two-regime check), against the paper's ACTUAL claims.

The paper: the next-token logit is a ~45-way additive sum with **PR ≈ 45, route-invariant** — retrieval and computation
do NOT separate on PR; they separate on the **power-diagram margin** and the **readout multiplicity μ_t**. So this script
tests three things on natural text (`fieldrun --natural-pr-dump`), with track A = a parameter-free INDUCTION baseline
(copy what followed the current token's last occurrence) = retrieved; computed = induction misses:

  1. ABSOLUTE source-PR level — expect ≈ the paper's ~45-way regime, and MUCH higher than the structured-task ~3–8
     (`align.py`); that contrast is the real two-regime result.
  2. ROUTE-INVARIANCE of PR — expect retrieved ≈ computed (AUC ≈ 0.5): PR should NOT separate them.
  3. margin + μ_t SEPARATE — expect computed LOWER (AUC < 0.5): the paper's actual retrieve-vs-compute axes.

Usage:  python align_nat.py /tmp/natpr.jsonl
"""
import json, sys
import numpy as np


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(allv) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/natpr.jsonl"
    rs = [json.loads(l) for l in open(path) if l.strip()]
    rs.sort(key=lambda r: r["pos"])
    ids = [r["cur"] for r in rs]                       # the actual token sequence (cur = ids[pos])
    pos0 = rs[0]["pos"]
    # induction baseline: at record k (position p=pos0+k), predict the token that followed ids[k]'s last earlier
    # occurrence; retrieved = the model's pred matches it. (Parameter-free; the paper's "one clean recursive idiom".)
    retr = np.zeros(len(rs), dtype=bool)
    for k, r in enumerate(rs):
        tok = ids[k]
        j = next((m for m in range(k - 1, -1, -1) if ids[m] == tok), None)
        if j is not None and j + 1 < len(ids):
            retr[k] = (r["pred"] == ids[j + 1])
    PR = np.array([r["pr"] for r in rs]); PRM = np.array([r["prmag"] for r in rs])
    MG = np.array([r["margin"] for r in rs]); MU = np.array([r["mu"] for r in rs])
    nb = rs[0]["nb"]; comp = ~retr
    print(f"# natural-text alignment · {path} · {len(rs)} tokens · {nb} blocks · "
          f"retrieved(induction) {retr.mean()*100:.0f}% / computed {comp.mean()*100:.0f}%\n")
    print(f"# 1. ABSOLUTE source-PR: mean {PR.mean():.1f} / {nb}   (paper's ~45-way regime; structured tasks were ~3-8)")
    print(f"#    PR-magnitude mean {PRM.mean():.1f} / {nb}\n")
    print("# 2/3. retrieved vs computed —  (paper: PR route-INVARIANT; margin & μ_t SEPARATE, computed lower)")
    for nm, v, expect in [("source-PR", PR, "≈ (invariant)"), ("PR-magnitude", PRM, "≈ (invariant)"),
                          ("margin", MG, "computed lower"), ("μ_t", MU, "computed lower")]:
        au = auc(v[comp], v[retr])     # AUC that COMPUTED ranks above retrieved
        print(f"#   {nm:<12} retrieved {v[retr].mean():7.3f}  computed {v[comp].mean():7.3f}  "
              f"AUC(computed>retr) {au:.2f}   expect {expect}")
    print("\n# reading: PR AUC≈0.5 ⇒ route-invariant (confirms paper); margin/μ_t AUC<0.5 ⇒ they separate (confirms paper).")
    print("#          the high ABSOLUTE PR here vs ~3-8 on structured tasks is the two-regime result.")


if __name__ == "__main__":
    main()
