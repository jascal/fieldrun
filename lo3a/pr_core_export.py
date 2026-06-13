#!/usr/bin/env python3
"""Ship the two-knob PR-core lever as a re-loadable artifact (PROVABLE_OPT §7).

The readout argmax_v ⟨x_f, gain⊙U_v⟩ factors through a rank-r readout-aligned decision basis S:
    logit_v ∝ ⟨S·x_raw, A_v⟩      with  S = top-r SVD of the normalized top-competitor diffs   [r, d]
                                        A = S·(gain⊙U)ᵀ                                          [r, vocab]
Storing only (S, A) is the model's unembedding at r(d+vocab) floats instead of vocab·d — a TUNABLE
LOSSY size dial (6.2×@67% … 2.2×@79% on SmolLM-135M; compression grows with d/PR at scale). The full
readout stays the default decode path; PR-core is for storage / datalog / embedding shrink where a
known-preservation lossy head is acceptable. This script:
  export   → fit the head, write <out>.prcore.npz + .json, verify preservation on FRESH held-out decisions
  --datalog → also emit a self-contained, souffle-runnable factored-readout .dl (shortlisted to stay small;
              the STRUCTURE is exact, the manifest records the true full-vocab fact count)
Lossy by construction and labeled as such. Reuses bundle_io + lo1_circuit (the verified rope forward).
"""
import os, sys, json, argparse
import numpy as np
import bundle_io as bio
from lo1_circuit import forward_capture

HERE = os.path.dirname(os.path.abspath(__file__))

def _norm(v): return v / (np.linalg.norm(v) + 1e-30)

def sample_decisions(W, cfg, cfg_f, n, rng, topk=9):
    """Return [(pred, top_competitors[1:topk], x_raw)] over n random short prompts (the decode reference)."""
    V = int(cfg[6]); out = []
    for _ in range(n):
        ids = [int(t) for t in rng.integers(0, V, size=int(rng.integers(6, 14)))]
        lg, xf, *_ = forward_capture(W, cfg, cfg_f, ids); o = np.argsort(lg)[::-1]
        out.append((int(o[0]), o[1:topk].astype(int), xf.astype(np.float64)))
    return out

def fit_head(W, cfg, cfg_f, r, cal):
    """S = top-r right-singular dirs of the normalized gain-weighted top-competitor diffs; A = S·(gain⊙U)ᵀ."""
    U = (W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64)
    gain = W["norm"].astype(np.float64); gU = gain * U                      # gain folded onto the unembedding
    rows = [_norm(gU[p] - gU[v]) for p, comp, _ in cal for v in comp]
    _, _, Vt = np.linalg.svd(np.array(rows), full_matrices=False)
    S = Vt[:r]                                                              # [r, d]
    A = (S @ gU.T)                                                          # [r, vocab]
    return S, A

def core_argmax(S, A, x_raw):                                              # logit_v ∝ ⟨S·x_raw, A_v⟩ (invn scalar drops)
    return int(np.argmax((S @ x_raw) @ A))

def measure(S, A, test):
    return sum(core_argmax(S, A, x) == p for p, _, x in test) / len(test)

def ff(x):
    """Souffle-safe positional float (no scientific notation, always a decimal point)."""
    s = np.format_float_positional(float(x), trim="0", precision=8)
    if s in ("", "-"): s = "0.0"
    if "." not in s: s += ".0"
    if s.startswith("."): s = "0" + s
    if s.startswith("-."): s = "-0" + s[1:]
    return s

def emit_datalog(S, A, x_raw, pred, shortlist, out_dl):
    """Self-contained factored readout for ONE input over a vocab SHORTLIST (runnable; structure exact).
       proj(i)=Σ_j xraw(j)·sbasis(i,j);  corelogit(v)=Σ_i proj(i)·acore(i,v);  argmax over the shortlist."""
    r, d = S.shape
    L = []
    L.append("// PR-core factored readout (LOSSY) — PROVABLE_OPT §7. argmax_v <S·xraw, A_v>.")
    L.append(f"// rank r={r}, d={d}, shortlist={len(shortlist)} of {A.shape[1]} vocab. Full-vocab facts would be r*(d+vocab).")
    L.append(".decl xraw(j:number, val:float)")
    L.append(".decl sbasis(i:number, j:number, s:float)")
    L.append(".decl acore(i:number, v:number, a:float)")
    L.append(".decl proj(i:number, p:float)")
    L.append(".decl corelogit(v:number, lg:float)")
    L.append(".decl best(v:number)")
    L.append(".output best")
    for j in range(d): L.append(f"xraw({j}, {ff(x_raw[j])}).")
    for i in range(r):
        for j in range(d): L.append(f"sbasis({i}, {j}, {ff(S[i, j])}).")
    for v in shortlist:
        for i in range(r): L.append(f"acore({i}, {int(v)}, {ff(A[i, v])}).")
    L.append("proj(i, s) :- sbasis(i, _, _), s = sum w : { xraw(j, x), sbasis(i, j, b), w = x * b }.")
    L.append("corelogit(v, s) :- acore(_, v, _), s = sum w : { proj(i, p), acore(i, v, a), w = p * a }.")
    L.append("best(v) :- corelogit(v, lg), lg = max l : { corelogit(_, l) }.")
    open(out_dl, "w").write("\n".join(L) + "\n")
    return out_dl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?", default=os.path.join(HERE, "smollm", "smollm"))
    ap.add_argument("-r", "--rank", type=int, default=92)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "prcore_head", "smollm"))
    ap.add_argument("--ncal", type=int, default=450)
    ap.add_argument("--ntest", type=int, default=450)
    ap.add_argument("--datalog", action="store_true", help="also emit a runnable factored-readout .dl (shortlisted)")
    ap.add_argument("--shortlist", type=int, default=64)
    args = ap.parse_args()

    man, W = bio.read_bundle(args.stem); cfg, cfg_f = man["config"], man["config_f"]
    d = int(cfg[4]); V = int(cfg[6]); r = args.rank; rng = np.random.default_rng(7)
    cal = sample_decisions(W, cfg, cfg_f, args.ncal, rng)
    test = sample_decisions(W, cfg, cfg_f, args.ntest, rng)
    S, A = fit_head(W, cfg, cfg_f, r, cal)
    keep = measure(S, A, test)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    full = V * d; head = r * (d + V)
    np.savez(args.out + ".prcore.npz", S=S.astype(np.float32), A=A.astype(np.float32))
    json.dump({
        "format": "fieldrun-prcore-head", "version": 1, "lossy": True,
        "source": os.path.basename(args.stem), "arch": man.get("arch"),
        "rank": r, "d": d, "vocab": V, "tied": int(cfg[7]),
        "full_unembed_floats": full, "head_floats": head,
        "compression": round(full / head, 3), "decode_kept": round(keep, 4),
        "decode": "argmax_v <S @ x_raw, A[:,v]>  (x_raw = final residual, pre-norm; gain folded into A)",
        "note": "LOSSY storage dial. Full readout remains the exact decode path; use PR-core where a "
                "known-preservation lossy unembedding is acceptable (storage / datalog / embedding shrink).",
    }, open(args.out + ".prcore.json", "w"), indent=2)

    print(f"== PR-core head exported: {args.out}.prcore.npz (+ .json) ==")
    print(f"   source {os.path.basename(args.stem)}  vocab×d = {V}×{d} = {full/1e6:.1f}M floats")
    print(f"   rank r={r}  head = r(d+vocab) = {head/1e6:.1f}M floats  →  {full/head:.1f}× smaller (LOSSY)")
    print(f"   decode preservation on {args.ntest} FRESH held-out decisions: {100*keep:.0f}%")

    if args.datalog:
        ex = test[0]; pred, _, x = ex
        cl = (S @ x) @ A; cand = np.argsort(cl)[::-1][:args.shortlist].tolist()
        if pred not in cand: cand.append(pred)
        out_dl = emit_datalog(S, A, x, pred, sorted(set(cand)), args.out + ".prcore.dl")
        core_pick = core_argmax(S, A, x)
        print(f"\n== factored-readout datalog: {out_dl} ==")
        print(f"   one input, shortlist of {len(cand)} vocab. PR-core argmax = {core_pick} (full argmax = {pred}; "
              f"{'MATCH' if core_pick==pred else 'differs — within the lossy floor'}).")
        print(f"   run: souffle -D- {out_dl}   (expect best({core_pick}))")

if __name__ == "__main__":
    main()
