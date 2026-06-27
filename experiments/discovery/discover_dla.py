#!/usr/bin/env python3
"""Idiom discovery on the DLA-PROFILE signature — idioms BEYOND the recursion/binding subspace.

Where discover.py clustered the recursion substrate (binding/fold-timing), this clusters each decision by its per-block
CONTRIBUTION profile (fieldrun --dla-dump): ⟨write_b, U_pred⟩ for every block b, with Σ_b == the predicted logit. That
profile is "which circuit produced the prediction", so clusters = circuit-level idioms (embed-token-identity,
early-attention, late-MLP-recall, distributed/composed, suppression-heavy, …).

Per-decision signature (interpretable aggregates of the block-contribution vector):
  conc       = max_b|c_b| / Σ|c_b|            # peakedness — one block dominates (retrieved) vs spread (composed)
  spread     = participation ratio / nblocks  # PIC O2 support number (low = concentrated)
  embed/attn/mlp_frac                          # which block KIND carries the logit
  early/late_frac                              # depth (layer-third) the contribution comes from
  neg_frac   = Σ_{c<0}|c| / Σ|c|              # share that SUPPRESSES the predicted token

Usage: python3 discover_dla.py [dumps_dla_dir] [k]
"""
import sys, os, json, glob, re, random
import numpy as np

random.seed(0); np.random.seed(0)
DUMPS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "dumps_dla")
K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
FEATS = ["conc", "spread", "embed_frac", "attn_frac", "mlp_frac", "early_frac", "late_frac", "neg_frac"]


def classify(label):
    """(layer_index_or_None, kind) from a block label like 'embed', 'L5.attn', 'l5.mlp'."""
    m = re.search(r"(\d+)", label); layer = int(m.group(1)) if m else None
    lo = label.lower()
    kind = ("embed" if "embed" in lo else "attn" if ("attn" in lo or "attention" in lo)
            else "mlp" if ("mlp" in lo or "ffn" in lo or "feed" in lo) else "ple" if "ple" in lo else "other")
    return layer, kind


def load(dumps):
    rows = []
    for f in sorted(glob.glob(os.path.join(dumps, "*.jsonl"))):
        pid = os.path.basename(f)[:-6]
        lines = open(f).read().splitlines()
        if len(lines) < 2:
            continue
        labels_all = json.loads(lines[0])["labels"]
        meta_all = [classify(l) for l in labels_all]
        # drop gemma PLE blocks (gemma-specific structural, huge magnitude) — study the attn/mlp/embed circuit
        keep = [i for i, (_, k) in enumerate(meta_all) if k != "ple"]
        labels = [labels_all[i] for i in keep]
        meta = [meta_all[i] for i in keep]
        layers = [m[0] for m in meta if m[0] is not None]
        nl = (max(layers) + 1) if layers else 1
        for ln in lines[1:]:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            c = np.array(r["contrib"], float)
            if c.size != len(labels_all):
                continue
            c = c[keep]
            if np.abs(c).sum() < 1e-6:
                continue
            a = np.abs(c); tot = a.sum()
            top = np.argsort(-a)[:3]
            rows.append({
                "pid": pid, "pos": r["pos"], "pred": r.get("pred_s", "?"),
                "topblocks": ", ".join(f"{labels[i]}({c[i]:+.2f})" for i in top),
                "conc": a.max() / tot,
                "spread": (tot ** 2 / (c @ c)) / len(c),
                "embed_frac": sum(a[i] for i, (_, k) in enumerate(meta) if k == "embed") / tot,
                "attn_frac": sum(a[i] for i, (_, k) in enumerate(meta) if k == "attn") / tot,
                "mlp_frac": sum(a[i] for i, (_, k) in enumerate(meta) if k == "mlp") / tot,
                "early_frac": sum(a[i] for i, (L, _) in enumerate(meta) if L is not None and L < nl / 3) / tot,
                "late_frac": sum(a[i] for i, (L, _) in enumerate(meta) if L is not None and L >= 2 * nl / 3) / tot,
                "neg_frac": sum(-c[i] for i in range(len(c)) if c[i] < 0) / tot,
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
        print(f"no DLA profiles in {DUMPS} — run ./collect_dla.sh first"); return
    X0 = np.array([[r[f] for f in FEATS] for r in rows], float)
    X = (X0 - X0.mean(0)) / (X0.std(0) + 1e-9)
    a, c = kmeans(X, K)
    nearest = ((X - c[a]) ** 2).sum(1) ** 0.5
    residual = nearest > nearest.mean() + 1.5 * nearest.std()

    print(f"\n=== {len(rows)} decisions · {len(set(r['pid'] for r in rows))} prompts · k={K} (DLA-profile signature) ===")
    print(f"signature = {FEATS}\n")
    for j in range(K):
        idx = np.where((a == j) & ~residual)[0]
        if len(idx) == 0:
            continue
        cen = X0[idx].mean(0)
        print(f"--- cluster {j} (n={len(idx)})  " + "  ".join(f"{f}={cen[i]:.2f}" for i, f in enumerate(FEATS)))
        for i in idx[np.argsort(((X[idx] - c[j]) ** 2).sum(1))][:6]:
            r = rows[i]
            print(f"      [{r['pid']}:{r['pos']}] → {r['pred']!r}   via {r['topblocks']}")
        print()

    res = np.where(residual)[0]
    print(f"=== RESIDUAL — DLA frontier (n={len(res)}, completeness={100*(1-len(res)/len(rows)):.0f}%) ===")
    for i in res[np.argsort(-nearest[res])][:8]:
        r = rows[i]
        print(f"      [{r['pid']}:{r['pos']}] → {r['pred']!r}   conc={r['conc']:.2f} neg={r['neg_frac']:.2f}  via {r['topblocks']}")


if __name__ == "__main__":
    main()
