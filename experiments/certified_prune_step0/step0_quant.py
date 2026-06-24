#!/usr/bin/env python3
"""Step 0-quant: certified quantization precision on real --pil-dump data.
For each position, the decode (winner w over the field) survives a relative perturbation tau of the
per-block contributions iff  tau*(L1[w]+L1[v]) < gap(w,v) for every competitor v
(worst-case, |Δc| <= tau*|c|); certified bits b = log2(1/tau).  RMS variant uses L2 (independent
rounding).  Unlike pruning, this cashes in as bits/bandwidth on EVERY weight, every position.
Usage: step0_quant.py LABEL=dump.jsonl ..."""
import json, sys, math, statistics as st

def load(path):
    out=[]
    for l in open(path):
        if not l.strip(): continue
        r=json.loads(l); C=r["contrib"]; nb=len(C); K=len(C[0])
        L=[sum(C[b][k] for b in range(nb)) for k in range(K)]
        w=max(range(K), key=lambda k:L[k])
        L1=[sum(abs(C[b][k]) for b in range(nb)) for k in range(K)]      # worst-case sensitivity
        L2=[math.sqrt(sum(C[b][k]**2 for b in range(nb))) for k in range(K)]  # RMS sensitivity
        out.append(dict(L=L, w=w, K=K, nb=nb, L1=L1, L2=L2))
    return out

def tau_cert(d):   # worst-case relative tolerance
    w=d["w"]; return min((d["L"][w]-d["L"][v])/(d["L1"][w]+d["L1"][v])
                         for v in range(d["K"]) if v!=w and (d["L1"][w]+d["L1"][v])>0)
def tau_rms(d):
    w=d["w"]; return min((d["L"][w]-d["L"][v])/math.hypot(d["L2"][w], d["L2"][v])
                         for v in range(d["K"]) if v!=w and math.hypot(d["L2"][w],d["L2"][v])>0)
def bits(tau): return math.log2(1.0/tau) if tau>0 else float('inf')

def med(xs): return st.median([x for x in xs if math.isfinite(x)])
def p(xs,q): xs=sorted(x for x in xs if math.isfinite(x)); return xs[min(len(xs)-1,int(round(q/100*(len(xs)-1))))]

def static_bits(D, resid):   # global bit-width = worst (max-bits) position after dropping resid% smallest-margin
    bc=sorted((bits(tau_cert(d)), d) for d in D)         # sort by certified bits asc... need by margin
    # static bit-width is set by the WORST position; residue drops the worst ones
    allb=sorted(bits(tau_cert(d)) for d in D)
    keep=allb[:int((1-resid)*len(allb))] if resid>0 else allb   # drop the top-resid hardest
    return keep[-1] if keep else float('inf')

def row(label, D):
    nb=D[0]["nb"]; N=len(D)
    bc=[bits(tau_cert(d)) for d in D]; br=[bits(tau_rms(d)) for d in D]
    return (f"| {label} | {nb} | {N} | {med(bc):.1f} | {med(br):.1f} ({p(br,10):.1f}–{p(br,90):.1f}) | "
            f"{static_bits(D,0.0):.1f} | {static_bits(D,0.10):.1f} | {static_bits(D,0.40):.1f} |")

def main():
    print("| model (corpus) | nb | N | cert bits med (worst-case) | RMS bits med (p10–p90) | "
          "static bits 0%res | static @10%res | static @40%res |")
    print("|---|---|---|---|---|---|---|---|")
    for a in sys.argv[1:]:
        label, path = a.split("=",1)
        D=load(path)
        if D: print(row(label, D))
    print("\nRead: 'bits' = certified RELATIVE precision of the per-block decode contributions "
          "(b bits <-> tolerate 2^-b relative perturbation). Lower = more quantizable. Static = the "
          "single global bit-width (worst position); residue drops the hardest q% (forge-tax). Cashes in "
          "on every weight (bandwidth), unlike pruning. Worst-case is the certified guarantee; RMS is the "
          "realistic (independent-rounding) estimate. Mapping contribution-bits -> weight-bits is a "
          "separate, favorable step (a weight error spreads over the hidden dim).")

if __name__=="__main__": main()
