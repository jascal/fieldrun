#!/usr/bin/env python3
"""B2 supervised value-probe — a LADDER of decoders, not just a linear map.

Reads the binary dump from `fieldrun --recursion-explain --value-probe-dump <file>` (per-layer residual vectors at
each subtree node's position + that subtree's TRUE value) and asks, per layer, whether the intermediate value is
decodable — and IN WHICH ENCODING. A linear failure does NOT mean the value is absent: LLMs invent non-CS-textbook
formats (e.g. arithmetic-as-a-clock — values as angles, read via cos/sin, invisible to a linear-to-integer probe).
So we run several decoders and report WHICH first reads the value (that names the format):

  linear  — value = w·h            (direction code)
  rff     — ridge on cos(W h + b)  (random Fourier features: approximates periodic + nonlinear kernels — the clock)
  knn     — k-nearest residuals    (nonparametric: value present in ANY smooth local structure, no form assumed)

Honest controls: held-out test split (report TEST acc, never train); a LABEL-SHUFFLE control (a decoder that "reads"
shuffled labels is leaking, not finding). NOTE: probe-presence != causally-used — confirm with a steering/patch test
before claiming the model USES the code. "Not found" only rules out the decoders tried; richer formats stay open.
"""
import struct, sys
import numpy as np

ROLE = {0: "root", 1: "left", 2: "right"}


def load(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"FRVP", "bad magic"
        n, nl, d = struct.unpack("<III", f.read(12))
        rec = 4 + 2 + nl * d * 4
        buf = f.read()
    assert len(buf) == n * rec, f"size mismatch {len(buf)} != {n*rec}"
    vals = np.empty(n, np.int32); roles = np.empty(n, np.uint8); corr = np.empty(n, np.uint8)
    X = np.empty((n, nl, d), np.float32)
    for i in range(n):
        off = i * rec
        vals[i] = struct.unpack_from("<i", buf, off)[0]
        roles[i] = buf[off + 4]; corr[i] = buf[off + 5]
        X[i] = np.frombuffer(buf, np.float32, nl * d, off + 6).reshape(nl, d)
    return vals, roles, corr, X, nl, d


def _prep(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def linear_acc(Xtr, ytr, Xte, yte, lam=10.0):
    Xtr, Xte = _prep(Xtr, Xte)
    ymu = ytr.mean()
    K = Xtr @ Xtr.T
    a = np.linalg.solve(K + lam * np.eye(len(K)), ytr - ymu)
    pred = (Xte @ Xtr.T) @ a + ymu
    return float((np.clip(np.round(pred), 0, None).astype(int) == yte).mean())


def rff_acc(Xtr, ytr, Xte, yte, D=512, lam=1.0, seed=0):
    Xtr, Xte = _prep(Xtr, Xte)
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    best = 0.0
    for gamma in (0.01, 0.03, 0.1):  # bandwidth sweep (median heuristic is ~1/d after standardizing)
        W = rng.normal(0, np.sqrt(gamma), (d, D)); b = rng.uniform(0, 2 * np.pi, D)
        Ftr = np.cos(Xtr @ W + b) * np.sqrt(2.0 / D)
        Fte = np.cos(Xte @ W + b) * np.sqrt(2.0 / D)
        ymu = ytr.mean()
        A = Ftr.T @ Ftr + lam * np.eye(D)
        w = np.linalg.solve(A, Ftr.T @ (ytr - ymu))
        pred = Fte @ w + ymu
        best = max(best, float((np.clip(np.round(pred), 0, None).astype(int) == yte).mean()))
    return best


def knn_acc(Xtr, ytr, Xte, yte, k=7):
    Xtr, Xte = _prep(Xtr, Xte)
    Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-6)
    Xte = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-6)
    sims = Xte @ Xtr.T
    nn = np.argsort(-sims, axis=1)[:, :k]
    pred = np.array([np.bincount(ytr[row]).argmax() for row in nn])
    return float((pred == yte).mean())


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vp.bin"
    only_correct = "--correct-only" in sys.argv
    vals, roles, corr, X, nl, d = load(path)
    print(f"# value-probe LADDER: {len(vals)} samples · {nl} layers · d={d} · model-correct {corr.mean()*100:.0f}%")
    if only_correct:
        keep = corr == 1; vals, roles, X = vals[keep], roles[keep], X[keep]
        print(f"# restricted to model-CORRECT samples: {len(vals)}")
    rng = np.random.default_rng(0)
    decoders = [("linear", linear_acc), ("rff", rff_acc), ("knn", knn_acc)]
    for role in sorted(set(roles.tolist())):
        m = roles == role
        Xr, yr = X[m], vals[m].astype(int)
        if len(yr) < 40:
            print(f"\n[{ROLE[role]}] only {len(yr)} samples — skip"); continue
        idx = rng.permutation(len(yr)); cut = int(0.7 * len(yr)); tr, te = idx[:cut], idx[cut:]
        chance = float((yr[te] == np.bincount(yr[tr]).argmax()).mean())
        print(f"\n[{ROLE[role]}] n={len(yr)} value-range {yr.min()}..{yr.max()} · chance(majority)={chance*100:.0f}% · logit-lens ~5%")
        for name, fn in decoders:
            per = [fn(Xr[tr, l], yr[tr], Xr[te, l], yr[te]) for l in range(nl)]
            bl = int(np.argmax(per))
            # label-shuffle control at the best layer
            ysh = yr[tr].copy(); rng.shuffle(ysh)
            shuf = fn(Xr[tr, bl], ysh, Xr[te, bl], yr[te])
            print(f"  {name:<7} best layer {bl+1:>2}: {per[bl]*100:>5.0f}%   (shuffle-control {shuf*100:>3.0f}%, chance {chance*100:.0f}%)")


if __name__ == "__main__":
    main()
