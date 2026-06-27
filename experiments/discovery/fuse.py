#!/usr/bin/env python3
"""Fuse the three signatures into ONE decision vector and re-cluster.

The three lenses are blind where each other sees:
  recursion = WHEN  (resolve-timing, fold-depth, copy)        — dumps/        (last in-prompt position per prompt)
  DLA       = WHICH (block-kind: attn/mlp + suppression)      — dumps_dla/    (last-position, rope residual_decomp)
  causal    = WHERE (load-bearing depth: early vs late)       — dumps_causal/ (last-position ablation flips)

All collected on the SAME model (Qwen2.5-0.5B) so the per-prompt decisions align. (Impurity: the recursion signature is
the last IN-PROMPT decision, DLA/causal the continuation — an off-by-one end-of-prompt proxy; see README.) Fused vector
= a few features from each; clusters are then described by all three lenses at once.

Usage: python3 fuse.py
"""
import os, json, glob, random
import numpy as np

random.seed(0); np.random.seed(0)
HERE = os.path.dirname(__file__)


def is_copy(ids, pos, final):
    ctx = ids[: pos + 1]
    for span in range(min(6, len(ctx) - 1), 1, -1):
        tail = ctx[len(ctx) - span:]
        for i in range(len(ctx) - span - 1, -1, -1):
            if ctx[i:i + span] == tail and i + span < len(ctx):
                return 1.0 if ctx[i + span] == final else 0.0
    return 0.0


def load_recursion(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        pid = os.path.basename(f)[:-6]
        lines = open(f).read().splitlines()
        if len(lines) < 2:
            continue
        ids = json.loads(lines[0])["ids"]
        recs = [json.loads(x) for x in lines[1:] if x.strip()]
        if not recs:
            continue
        r = max(recs, key=lambda z: z["pos"])           # last in-prompt decision
        nl = r["n_layer"]; lens = r.get("lens", [])
        churn = sum(1 for a, b in zip(lens, lens[1:]) if a != b) / max(len(lens) - 1, 1)
        out[pid] = {"resolve_frac": r["resolve"] / nl, "reach_norm": (r["pos"] - r["back"]) / max(r["pos"], 1),
                    "copy": is_copy(ids, r["pos"], r["final"]), "churn": churn, "rec_pred": r.get("final_s", "?")}
    return out


def classify(label):
    lo = label.lower()
    return ("embed" if "embed" in lo else "attn" if "attn" in lo else "mlp" if ("mlp" in lo or "ffn" in lo) else "other")


def load_dla(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        pid = os.path.basename(f)[:-6]
        lines = open(f).read().splitlines()
        if len(lines) < 2:
            continue
        labels = json.loads(lines[0])["labels"]; kinds = [classify(l) for l in labels]
        r = json.loads(lines[-1])                        # last-position profile
        c = np.array(r["contrib"], float)
        if c.size != len(labels):
            continue
        a = np.abs(c); tot = a.sum()
        if tot < 1e-6:
            continue
        out[pid] = {"conc": a.max() / tot,
                    "mlp_frac": sum(a[i] for i, k in enumerate(kinds) if k == "mlp") / tot,
                    "neg_frac": sum(-c[i] for i in range(len(c)) if c[i] < 0) / tot}
    return out


def load_causal(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        pid = os.path.basename(f)[:-6]
        try:
            r = json.loads(open(f).read().strip().splitlines()[0])
        except Exception:
            continue
        nl = r["n_layer"]; flips = r.get("flips", []); n = len(flips); ls = [fl["l"] for fl in flips]
        out[pid] = {"flip_frac": n / (2 * nl), "earliest": (min(ls) / nl if ls else 1.0),
                    "early_frac": (sum(1 for l in ls if l < nl / 3) / n if n else 0.0),
                    "caus_pred": r.get("pred_s", "?")}
    return out


def kmeans(X, k, iters=100):
    c = X[np.random.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = ((X[:, None, :] - c[None, :, :]) ** 2).sum(2).argmin(1)
        nc = np.array([X[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return a, c


REC = ["resolve_frac", "reach_norm", "copy", "churn"]
DLA = ["conc", "mlp_frac", "neg_frac"]
CAU = ["flip_frac", "earliest", "early_frac"]


def main():
    rec, dla, cau = load_recursion(os.path.join(HERE, "dumps")), load_dla(os.path.join(HERE, "dumps_dla")), load_causal(os.path.join(HERE, "dumps_causal"))
    pids = sorted(set(rec) & set(dla) & set(cau))
    if not pids:
        print(f"no aligned prompts (rec={len(rec)} dla={len(dla)} causal={len(cau)}) — collect all three on the SAME model"); return
    feats = REC + DLA + CAU
    X0 = np.array([[{**rec[p], **dla[p], **cau[p]}[f] for f in feats] for p in pids], float)
    X = (X0 - X0.mean(0)) / (X0.std(0) + 1e-9)
    K = 4
    a, c = kmeans(X, K)
    print(f"\n=== FUSED signature · {len(pids)} prompts · k={K} ===")
    print(f"  WHEN  {REC}\n  WHICH {DLA}\n  WHERE {CAU}\n")
    for j in range(K):
        idx = [i for i in range(len(pids)) if a[i] == j]
        if not idx:
            continue
        cen = X0[idx].mean(0)
        d = dict(zip(feats, cen))
        print(f"--- cluster {j} (n={len(idx)})")
        print(f"      WHEN : resolve={d['resolve_frac']:.2f} reach={d['reach_norm']:.2f} copy={d['copy']:.2f} churn={d['churn']:.2f}")
        print(f"      WHICH: conc={d['conc']:.2f} mlp={d['mlp_frac']:.2f} neg={d['neg_frac']:.2f}")
        print(f"      WHERE: flip={d['flip_frac']:.2f} earliest={d['earliest']:.2f} early={d['early_frac']:.2f}")
        for i in idx[:6]:
            print(f"        [{pids[i]}] rec→{rec[pids[i]]['rec_pred']!r} cont→{cau[pids[i]]['caus_pred']!r}")
        print()


if __name__ == "__main__":
    main()
