#!/usr/bin/env python3
"""Alignment check (PIC_LOSSINESS §6, track A ↔ B): does the SURROGATE residue line up with the model's COMPUTED tokens?

Track A: where the best crisp synthesized program disagrees with the model output = "residue" tokens.
Track B: the model's own per-token DLA signals from `fieldrun --source-pr-dump` — source-PR `(Σ_b c_b)²/Σ_b c_b²` over
the ~57 residual-write blocks (the paper's diffuseness / Thm-5 quantity), the decode margin, and μ_t (how many blocks
already argmax to the chosen digit; μ_t=0 = composed).

Hypothesis (the paper's retrieve-vs-compute split): residue tokens are MORE COMPUTED ⇒ higher source-PR, LOWER margin,
LOWER μ_t than the tokens the crisp program captures. CONFIRM ⇒ the two residues are the same object and the export is a
clean crisp-head + PIC-residue join. SURPRISE ⇒ the surrogate residue ≠ the mechanistic computed fragment.

Usage:  python align.py /tmp/srcpr.jsonl [K]
"""
import json, sys
from collections import defaultdict
import numpy as np
import synth


def best_program(lists, target, K):
    levels, _ = synth.enumerate_programs(lists, K=K, enabled={"base", "pos", "hist"})
    progs = [p for p in synth.all_int_programs(levels) if all(v is None or (0 <= v <= 9) for v in p.vals)]
    idx = list(range(len(lists)))
    progs.sort(key=lambda p: (-(synth.faith(p.vals, target, idx) - synth.SIZE_PEN * p.size), p.size))
    return progs[0]


def auc(pos, neg):
    """P(x_pos > x_neg) — Mann-Whitney rank statistic; 0.5 = no separation."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/srcpr.jsonl"
    K = next((int(a) for a in sys.argv[2:] if a.isdigit()), 4)
    by_task = defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line); by_task[r["task"]].append(r)

    rows = []           # per-example: (task, residue?, pr, prmag, margin, mu)
    per_task = []
    for task, rs in by_task.items():
        lists = [r["list"] for r in rs]
        out = [r["out"] for r in rs]
        best = best_program(lists, out, K)
        resid = np.array([best.vals[i] != out[i] for i in range(len(rs))])
        pr = np.array([r["pr"] for r in rs]); prm = np.array([r["prmag"] for r in rs])
        marg = np.array([r["margin"] for r in rs]); mu = np.array([r["mu"] for r in rs])
        for i in range(len(rs)):
            rows.append((task, bool(resid[i]), pr[i], prm[i], marg[i], mu[i]))
        if resid.any() and (~resid).any():
            per_task.append((task, best.rep, resid.mean(),
                             pr[resid].mean() - pr[~resid].mean(),
                             marg[resid].mean() - marg[~resid].mean(),
                             mu[resid].mean() - mu[~resid].mean()))

    R = np.array([r[1] for r in rows])
    PR = np.array([r[2] for r in rows]); PRM = np.array([r[3] for r in rows])
    MG = np.array([r[4] for r in rows]); MU = np.array([r[5] for r in rows])
    print(f"# alignment · {path} · K={K} · {len(rows)} tokens · residue {R.mean()*100:.0f}%\n")
    print("# POOLED (all tokens): residue vs captured —  (paper's computed signature in parens)")
    for nm, v, lo_is_computed in [("source-PR", PR, False), ("PR-magnitude", PRM, False), ("margin", MG, True), ("μ_t", MU, True)]:
        a = v[R].mean(); b = v[~R].mean()
        expect = "lower" if lo_is_computed else "higher"
        au = auc(v[R], v[~R])
        ok = (a < b) if lo_is_computed else (a > b)
        print(f"#   {nm:<10} residue {a:7.3f}  captured {b:7.3f}  Δ {a-b:+7.3f}  AUC {au:.2f}  "
              f"(computed⇒residue {expect}; {'✓ as predicted' if ok else '✗ opposite'})")

    print("\n# WITHIN-TASK (controls for per-task baseline): mean over tasks of [residue − captured]")
    dpr = np.mean([t[3] for t in per_task]); dmg = np.mean([t[4] for t in per_task]); dmu = np.mean([t[5] for t in per_task])
    print(f"#   ΔPR {dpr:+.3f} (computed⇒+)   Δmargin {dmg:+.3f} (computed⇒−)   Δμ_t {dmu:+.3f} (computed⇒−)   over {len(per_task)} tasks")
    print(f"\n{'task':<9}{'best-1':<22}{'resid%':>7}{'ΔPR':>7}{'Δmargin':>9}{'Δμ_t':>7}")
    for t, rep, rm, dp, dg, du in sorted(per_task, key=lambda x: -x[2]):
        print(f"{t:<9}{rep[:22]:<22}{rm*100:>6.0f}%{dp:>7.2f}{dg:>9.2f}{du:>7.2f}")

    pr_ok = dpr > 0; mg_ok = dmg < 0; mu_ok = dmu < 0
    n_ok = sum([pr_ok, mg_ok, mu_ok])
    print(f"\n# reading: {n_ok}/3 signals point the paper's way (residue = more computed: higher PR, lower margin, lower μ_t).")


if __name__ == "__main__":
    main()
