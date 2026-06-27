#!/usr/bin/env python3
"""Idiom-DISCOVERY loop (prototype) — find UNNAMED computational idioms, unsupervised.

Pipeline: load the per-decision recursion signatures (fieldrun --recursion-dump) over a diverse corpus → build a
content-AGNOSTIC mechanism-signature vector per decision → cluster in signature space (clusters = candidate idioms,
emergent, not pre-named) → a RESIDUAL critic flags decisions no cluster explains (the frontier of unnamed idioms).

The loop is closed by a frontier model (the agent): read the cluster exemplars, NAME the dense idioms, and take the
residual exemplars as the next round's probe targets. "Discover, don't checklist."

Signature (per decision, all content-agnostic):
  resolve_frac  = resolve_layer / n_layer        # compute depth: high = the answer commits late (deferred compute)
  reach_norm    = (pos - back) / pos              # how far back the dominant fold reaches (0 = local)
  conc          = attention mass on that fold     # binding strength
  copy          = 1 if the prediction is a flat in-context induction copy (longest-suffix match), else 0
  lens_churn    = fraction of layer→layer logit-lens argmax changes (value-stack instability = more "thinking")

Usage: python3 discover.py [dumps_dir] [k]
"""
import sys, os, json, glob, math, random
import numpy as np

random.seed(0); np.random.seed(0)
DUMPS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "dumps")
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
FEATS = ["resolve_frac", "reach_norm", "conc", "copy", "lens_churn"]


def is_copy(ids, pos, final):
    """Faithful port of recursion_spectrum's copy test: the longest suffix ending at pos that recurs earlier, and the
    token AFTER that earlier occurrence equals `final` ⇒ an induction copy."""
    ctx = ids[: pos + 1]
    maxspan = min(6, len(ctx) - 1)
    for span in range(maxspan, 1, -1):
        tail = ctx[len(ctx) - span:]
        for i in range(len(ctx) - span - 1, -1, -1):
            if ctx[i:i + span] == tail and i + span < len(ctx):
                return 1 if ctx[i + span] == final else 0
    return 0


def load(dumps):
    rows = []
    for f in sorted(glob.glob(os.path.join(dumps, "*.jsonl"))):
        pid = os.path.basename(f)[:-6]
        lines = open(f).read().splitlines()
        if not lines:
            continue
        hdr = json.loads(lines[0]); ids = hdr["ids"]; toks = {}
        recs = []
        for ln in lines[1:]:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            recs.append(r); toks[r["pos"]] = r.get("tok_s", "?")
        for r in recs:
            pos, nl = r["pos"], r["n_layer"]
            if pos < 2 or nl == 0:
                continue
            reach = pos - r["back"]
            lens = r.get("lens", [])
            churn = sum(1 for a, b in zip(lens, lens[1:]) if a != b) / max(len(lens) - 1, 1)
            rows.append({
                "pid": pid, "pos": pos, "tok": r.get("tok_s", "?"), "pred": r.get("final_s", "?"),
                "ctx": " ".join(toks.get(p, "·") for p in range(max(0, pos - 5), pos + 1)),
                "resolve_frac": r["resolve"] / nl, "reach_norm": reach / max(pos, 1),
                "conc": r["conc"], "copy": float(is_copy(ids, pos, r["final"])),
                "lens_churn": churn,
            })
    return rows


def kmeans(X, k, iters=100):
    c = X[np.random.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(2)
        a = d.argmin(1)
        nc = np.array([X[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return a, c


def axis_idioms(rows, X0, residual, feats, min_tail=4, max_frac=0.35, min_gap=0.12):
    """Catch idioms that are dense along ONE signature dimension but spread across others — k-means scatters these into
    the residual (e.g. the early-resolution glue-reflex: low resolve_frac, but spread in reach). For each feature, find
    the largest GAP in the sorted values that carves off a minority tail; report tails whose members were mostly
    k-means residual as NEWLY NAMED idioms (the ones clustering missed)."""
    n = len(rows)
    print(f"\n=== AXIS IDIOMS — 1-D tail isolation (catches sparse/spread idioms k-means scatters) ===")
    for fi, f in enumerate(feats):
        v = X0[:, fi]; order = np.argsort(v); sv = v[order]
        for lo in (True, False):
            best_gap, best_cut = 0.0, None
            for cut in range(min_tail, int(max_frac * n) + 1):
                gap = (sv[cut] - sv[cut - 1]) if lo else (sv[n - cut] - sv[n - cut - 1])
                if gap > best_gap:
                    best_gap, best_cut = gap, cut
            if not best_cut or best_gap < min_gap:
                continue
            idx = order[:best_cut] if lo else order[n - best_cut:]
            thr = (sv[best_cut] + sv[best_cut - 1]) / 2 if lo else (sv[n - best_cut] + sv[n - best_cut - 1]) / 2
            res_n = int(residual[idx].sum())
            tag = "NEWLY NAMED (k-means missed it)" if res_n >= 0.5 * len(idx) else "(overlaps a k-means cluster)"
            print(f"--- {f} {'<' if lo else '>'} {thr:.2f}  n={len(idx)}  gap={best_gap:.2f}  "
                  f"residual-overlap={res_n}/{len(idx)}  {tag}")
            for i in idx[:5]:
                r = rows[i]
                print(f"      [{r['pid']}] …{r['ctx']!r} → {r['pred']!r}   "
                      f"rf={r['resolve_frac']:.2f} reach={r['reach_norm']:.2f} conc={r['conc']:.2f} "
                      f"copy={int(r['copy'])} churn={r['lens_churn']:.2f}")


def main():
    rows = load(DUMPS)
    if not rows:
        print(f"no decisions in {DUMPS} — run ./collect.sh first"); return
    X0 = np.array([[r[f] for f in FEATS] for r in rows], float)
    mu, sd = X0.mean(0), X0.std(0) + 1e-9
    X = (X0 - mu) / sd
    a, c = kmeans(X, K)
    nearest = ((X - c[a]) ** 2).sum(1) ** 0.5            # distance to OWN centroid
    thresh = nearest.mean() + 1.5 * nearest.std()
    residual = nearest > thresh

    print(f"\n=== {len(rows)} decisions · {len(set(r['pid'] for r in rows))} prompts · k={K} clusters ===")
    print(f"signature = {FEATS}\n")
    for j in range(K):
        idx = np.where((a == j) & ~residual)[0]
        if len(idx) == 0:
            continue
        cen = X0[idx].mean(0)
        sig = "  ".join(f"{f}={cen[i]:.2f}" for i, f in enumerate(FEATS))
        print(f"--- cluster {j}  (n={len(idx)})   {sig}")
        order = idx[np.argsort(((X[idx] - c[j]) ** 2).sum(1))][:6]   # most central exemplars
        for i in order:
            r = rows[i]
            print(f"      [{r['pid']}] …{r['ctx']!r} → {r['pred']!r}   "
                  f"rf={r['resolve_frac']:.2f} reach={r['reach_norm']:.2f} conc={r['conc']:.2f} "
                  f"copy={int(r['copy'])} churn={r['lens_churn']:.2f}")
        print()

    res_idx = np.where(residual)[0]
    comp = 100 * (1 - len(res_idx) / len(rows))
    print(f"=== RESIDUAL — the unnamed-idiom frontier (n={len(res_idx)}, completeness={comp:.0f}%) ===")
    print("decisions no cluster explains tightly — the next round's probe targets:")
    for i in res_idx[np.argsort(-nearest[res_idx])][:10]:
        r = rows[i]
        print(f"      [{r['pid']}] …{r['ctx']!r} → {r['pred']!r}   "
              f"rf={r['resolve_frac']:.2f} reach={r['reach_norm']:.2f} conc={r['conc']:.2f} "
              f"copy={int(r['copy'])} churn={r['lens_churn']:.2f}")

    axis_idioms(rows, X0, residual, FEATS)


if __name__ == "__main__":
    main()
