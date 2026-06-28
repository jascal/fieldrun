#!/usr/bin/env python3
"""Idiom discovery on the CAUSAL signature — cracks the depth axis DLA can't see.

DLA (direct logit attribution) is late-biased by construction (early writes are read out through later layers, so their
direct projection is ~0). CAUSAL attribution asks the counterfactual instead: ablate a block, does the prediction FLIP?
A load-bearing EARLY/MID block shows up here even though its logit share is tiny. Signature per (last-position) decision:

  flip_frac   = #load-bearing blocks / (2·n_layer)        # how concentrated the critical mass is
  earliest    = shallowest flipping layer / n_layer        # how EARLY a critical block lives (the dead-axis crack)
  early/mid/late_frac = where the load-bearing blocks sit (fraction of flippers per layer-third)
  attn/mlp_frac = which KIND is load-bearing

Headline question: do EARLY/MID blocks ever flip the answer? (If yes, causal sees depth DLA cannot.)
Usage: python3 discover_causal.py [dumps_causal_dir] [k]
"""
import sys, os, json, glob, random
import numpy as np

random.seed(0); np.random.seed(0)
DUMPS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "dumps_causal")
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
FEATS = ["flip_frac", "earliest", "early_frac", "late_frac", "attn_frac"]


def load(dumps):
    rows = []
    for f in sorted(glob.glob(os.path.join(dumps, "*.jsonl"))):
        pid = os.path.basename(f)[:-6]
        try:
            r = json.loads(open(f).read().strip().splitlines()[0])
        except Exception:
            continue
        nl = r["n_layer"]; flips = r.get("flips", []); n = len(flips)
        ls = [fl["l"] for fl in flips]
        early = sum(1 for l in ls if l < nl / 3); late = sum(1 for l in ls if l >= 2 * nl / 3)
        mid = n - early - late
        attn = sum(1 for fl in flips if fl["kind"] == "attn")
        rows.append({
            "pid": pid, "pred": r.get("pred_s", "?"), "nl": nl, "n_flip": n,
            "earliest_layer": (min(ls) if ls else nl),
            "flip_frac": n / (2 * nl),
            "earliest": (min(ls) / nl if ls else 1.0),
            "early_frac": early / n if n else 0.0,
            "mid_frac": mid / n if n else 0.0,
            "late_frac": late / n if n else 0.0,
            "attn_frac": attn / n if n else 0.0,
            "early_n": early, "mid_n": mid, "late_n": late,
            "blocks": ", ".join(f"L{fl['l']}.{fl['kind']}→{fl['to']}" for fl in sorted(flips, key=lambda x: x["l"])[:6]),
        })
    return rows


def kmeans(X, k, iters=100):
    c = X[np.random.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = ((X[:, None, :] - c[None, :, :]) ** 2).sum(2).argmin(1)
        nc = np.array([X[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return a, c


def main():
    rows = load(DUMPS)
    if not rows:
        print(f"no causal profiles in {DUMPS} — run ./collect_causal.sh first"); return

    # headline: does causal attribution see depth the DLA profile couldn't?
    n = len(rows)
    any_early = sum(1 for r in rows if r["early_n"] > 0)
    any_mid = sum(1 for r in rows if r["mid_n"] > 0)
    redundant = sum(1 for r in rows if r["n_flip"] == 0)
    print(f"\n=== {n} decisions (last-position) — CAUSAL depth crack ===")
    print(f"  decisions with an EARLY-third load-bearing block:  {any_early}/{n}")
    print(f"  decisions with a  MID-third  load-bearing block:   {any_mid}/{n}")
    print(f"  fully redundant (no single block flips):           {redundant}/{n}")
    print(f"  median earliest critical layer:  {np.median([r['earliest_layer'] for r in rows]):.0f} / {rows[0]['nl']}\n")

    X0 = np.array([[r[f] for f in FEATS] for r in rows], float)
    X = (X0 - X0.mean(0)) / (X0.std(0) + 1e-9)
    a, c = kmeans(X, K)
    for j in range(K):
        idx = np.where(a == j)[0]
        if len(idx) == 0:
            continue
        cen = X0[idx].mean(0)
        print(f"--- cluster {j} (n={len(idx)})  " + "  ".join(f"{f}={cen[i]:.2f}" for i, f in enumerate(FEATS)))
        for i in idx[np.argsort(((X[idx] - c[j]) ** 2).sum(1))][:6]:
            r = rows[i]
            print(f"      [{r['pid']}] → {r['pred']!r}  n_flip={r['n_flip']} earliest=L{r['earliest_layer']} "
                  f"(e{r['early_n']}/m{r['mid_n']}/l{r['late_n']})  {r['blocks']}")
        print()


if __name__ == "__main__":
    main()
