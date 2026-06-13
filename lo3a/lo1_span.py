#!/usr/bin/env python3
"""LO1 — THE decisive measurement (Grok): the dimension of the decision-direction span.

A context-free matrix/operator valuation has FIXED operators per proposition; it cannot adapt its basis to
each input's decision direction ΔU(x) = gain⊙(U_pred(x) − U_runnerup(x)) (residual-space, low-energy but
decisive). So the relevant intrinsic object is the SPAN of the margin-normalized decision directions:
    S = span{ ΔU(x) / m(x) : x ~ inputs }            (m = margin = facet distance)
dim(S) is the minimal width ANY fixed (architecture-preserving) description must have to preserve the argmax
across inputs — a clean estimator of the decision-relevant τ*.
  high (≳100s, rising as margin↓)  => Δ_descr small on the decision axis => forge tax intrinsic/Δ_repr (a wall)
  low                               => a context-free decision-adapted valuation could still escape
SmolLM-135M (and Pythia-70m as a 2nd model). Stratified by margin (forge-tax vs retrievable).
"""
import os, sys
import numpy as np
import bundle_io as bio

HERE = os.path.dirname(os.path.abspath(__file__))

def eff_rank(S):                         # participation ratio of squared singular values = effective dimension
    sv = np.linalg.svd(np.asarray(S, float), compute_uv=False); e = sv**2
    return float((e.sum()**2) / (np.square(e).sum() + 1e-30))

def energy_rank(S, frac=0.90):           # # of singular directions to reach `frac` of the energy
    sv = np.linalg.svd(np.asarray(S, float), compute_uv=False); e = np.cumsum(sv**2); e /= e[-1]
    return int(np.searchsorted(e, frac) + 1)

def collect(stem, n=800, seed=2):
    man, W = bio.read_bundle(stem); cfg, cfg_f = man["config"], man["config_f"]
    U = W["embed"] if cfg[7] else W["lm_head"]; gain = W["norm"].astype(np.float64); V = int(cfg[6])
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        ids = [int(t) for t in rng.integers(0, V, size=int(rng.integers(6, 14)))]
        lg = bio.forward(W, cfg, cfg_f, ids); o = np.argsort(lg)[::-1]
        m = float(lg[o[0]] - lg[o[1]])
        dU = gain * (U[o[0]].astype(np.float64) - U[o[1]].astype(np.float64))     # decision dir in residual space
        rows.append((m, dU))
    return cfg, rows

def report(name, cfg, rows):
    d = int(cfg[4]); rows = sorted(rows, key=lambda r: r[0]); n = len(rows); t = n // 3
    unit = lambda R: np.array([dU/ (np.linalg.norm(dU)+1e-30) for _, dU in R])
    mnorm = lambda R: np.array([dU/ (m+1e-9) for m, dU in R])      # margin-normalized (emphasizes forge tax)
    print(f"\n== {name}  (residual dim d={d}, {n} decisions) ==")
    print(f"   {'subset':<26}{'n':>5}{'mean margin':>12}{'effrank(unit ΔU)':>18}{'90%-energy rank':>17}")
    for lbl, g in [("forge-tax (thin margin)", rows[:t]), ("middle", rows[t:2*t]),
                   ("retrievable (thick)", rows[2*t:]), ("ALL", rows)]:
        Uu = unit(g); print(f"   {lbl:<26}{len(g):>5}{np.mean([m for m,_ in g]):>12.3f}{eff_rank(Uu):>18.1f}{energy_rank(Uu):>17}")
    Dm = mnorm(rows)
    print(f"   margin-normalized span  effrank(ΔU/m) = {eff_rank(Dm):.1f}   90%-energy rank = {energy_rank(Dm)}   (of max {d})")

if __name__ == "__main__":
    cfg, rows = collect(os.path.join(HERE, "smollm", "smollm"))
    report("SmolLM-135M", cfg, rows)
    p = os.path.join(HERE, "pythia", "p70m_s143000")
    if os.path.exists(p + ".fieldrun.json"):
        cfg2, rows2 = collect(p); report("Pythia-70m (step143000)", cfg2, rows2)
    print("\n   READING: high effrank / energy-rank (≳100s, higher on forge-tax) => no fixed low-rank valuation")
    print("   preserves the decode across inputs => Δ_descr small on the decision axis => intrinsic/Δ_repr floor.")
