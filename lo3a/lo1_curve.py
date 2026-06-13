#!/usr/bin/env python3
"""LO1 — the clean closer (Grok): the fixed-S decode-preservation CURVE.

Build a genuinely CONTEXT-FREE valuation: fit one fixed basis S = top principal components of the decision
directions ΔU = gain⊙(U_pred − U_runnerup) on a TRAIN split, then on HELD-OUT decisions project the residual
onto the rank-r subspace of S and measure how often the argmax survives. preservation(r) is the operational
test of whether a fixed (architecture-preserving) geometry-valued valuation escapes the forge tax.
  preservation high at small r (≪ PR)  => Δ_descr substantial in the decision basis (escape works)
  needs r ≈ PR / d for the tail/forge-tax => coverage-knee = the residual intrinsic floor τ*
SmolLM-135M. Plots preservation vs r, stratified by margin (forge-tax vs retrievable).
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import bundle_io as bio
from lo1_circuit import forward_capture

HERE = os.path.dirname(os.path.abspath(__file__)); STEM = os.path.join(HERE, "smollm", "smollm")
RS = [1,2,4,8,16,24,32,48,64,96,128,192,256,384,576]

def main():
    man, W = bio.read_bundle(STEM); cfg, cfg_f = man["config"], man["config_f"]
    U = (W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain = W["norm"].astype(np.float64)
    d = int(cfg[4]); eps = float(cfg_f[1]); V = int(cfg[6]); rng = np.random.default_rng(7)
    recs = []
    for _ in range(800):
        ids = [int(t) for t in rng.integers(0, V, size=int(rng.integers(6, 14)))]
        lg, xf, *_ = forward_capture(W, cfg, cfg_f, ids); o = np.argsort(lg)[::-1]
        m = float(lg[o[0]] - lg[o[1]]); dU = gain*(U[o[0]]-U[o[1]])
        recs.append((m, dU/(np.linalg.norm(dU)+1e-30), xf.astype(np.float64), int(o[0])))
    # context-free: fit S on TRAIN decisions, test on HELD-OUT
    n = len(recs); tr = recs[:n//2]; te = recs[n//2:]
    D = np.array([r[1] for r in tr])                                  # unit decision dirs (train)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)                 # rows of Vt = principal directions of S
    def decode(xp):
        xn = (xp/np.sqrt((xp**2).mean()+eps))*gain; return int(np.argmax(xn@U.T))
    te.sort(key=lambda r:r[0]); t=len(te)//3
    strata = [("forge-tax (thin)", te[:t]), ("retrievable (thick)", te[2*t:]), ("ALL", te)]
    pr_dla = 92  # measured scalar circuit PR (lo1_circuit) for reference
    print(f"== fixed-S decode-preservation curve (SmolLM-135M; S fit on {len(tr)} train, tested on {len(te)} held-out) ==")
    print(f"   d={d}, scalar circuit PR≈{pr_dla}. preservation = fraction of held-out decodes kept by rank-r projection onto S.")
    curves = {}
    hdr = "   r:" + "".join(f"{r:>5}" for r in RS); print(hdr)
    for lbl, g in strata:
        row = []
        for r in RS:
            Vr = Vt[:r]; keep = sum(decode(Vr.T@(Vr@x)) == pred for _,_,x,pred in g)/max(1,len(g))
            row.append(keep)
        curves[lbl] = row
        print(f"   {lbl:<20}" + "".join(f"{100*v:>5.0f}" for v in row))
    # knees
    def knee(row, thr):
        for r,v in zip(RS,row):
            if v>=thr: return r
        return RS[-1]
    print("\n   coverage-knee r(η):")
    for lbl in ["forge-tax (thin)","retrievable (thick)","ALL"]:
        c=curves[lbl]; print(f"   {lbl:<20} r(90%)={knee(c,.9):>3}  r(99%)={knee(c,.99):>3}")
    fig,ax=plt.subplots(figsize=(8,5))
    for lbl in strata:
        ax.plot(RS,[100*v for v in curves[lbl[0]]],"o-",label=lbl[0])
    ax.axvline(pr_dla,ls="--",c="gray",alpha=.6); ax.text(pr_dla*1.02,20,f"scalar PR≈{pr_dla}",color="gray",fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("fixed-S valuation rank r"); ax.set_ylabel("% held-out decodes preserved")
    ax.set_title("LO1: context-free fixed-S decode preservation vs rank (SmolLM-135M)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); out=os.path.join(HERE,"lo1_curve.png"); fig.savefig(out,dpi=120); print(f"\n   wrote {out}")

if __name__=="__main__": main()
