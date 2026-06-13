#!/usr/bin/env python3
"""LO1 spectral signature (Grok / Spectral-Scaling-Laws framing). On the decision-direction spectrum:
  hard rank  = PR = (Σλ)²/Σλ²                  (energy concentration — the "intrinsic core", flat with scale)
  soft rank  = exp(-Σ p_i ln p_i), p=λ/Σλ      (entropy effective dim — grows ~linearly if the tail spreads)
  α          = power-law slope of λ_k ~ k^{-α}  (fit over ranks 10–200; higher α = front-loaded spectrum)
The hard–soft GAP and α are the asymmetric-scaling signature: widening inflates the low-energy tail, so
hard rank (PR) stays flat while soft rank / span90 grow. Run per model to compare across the ladder.
Usage: python3 lo1_spectrum.py <bundle-stem> [N]
"""
import os, sys
import numpy as np
import bundle_io as bio
from lo1_circuit import forward_capture

def hard_rank(l): s=l.sum(); return float(s*s/(np.square(l).sum()+1e-30))
def soft_rank(l): p=l/(l.sum()+1e-30); return float(np.exp(-(p*np.log(p+1e-30)).sum()))
def alpha(l, lo=10, hi=200):
    l=np.sort(l)[::-1]; hi=min(hi,len(l)); k=np.arange(lo,hi)
    return float(-np.polyfit(np.log(k), np.log(l[lo:hi]+1e-30), 1)[0])

def main(stem, N=800):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64)
    d=int(cfg[4]); V=int(cfg[6]); rng=np.random.default_rng(5); rows=[]
    for _ in range(N):
        ids=[int(t) for t in rng.integers(0,V,size=int(rng.integers(6,14)))]
        lg,xf,*_=forward_capture(W,cfg,cfg_f,ids); o=np.argsort(lg)[::-1]
        dU=gain*(U[o[0]]-U[o[1]]); rows.append(dU/(np.linalg.norm(dU)+1e-30))
    sv=np.linalg.svd(np.array(rows),compute_uv=False); lam=sv**2            # decision-direction eigenvalues
    hr,sr,a=hard_rank(lam),soft_rank(lam),alpha(lam)
    print(f"{os.path.basename(stem):<18} d={d:<5} hard_rank(PR)={hr:6.1f}  soft_rank={sr:7.1f}  soft/hard={sr/hr:5.1f}  alpha={a:.3f}")
    return d,hr,sr,a

if __name__=="__main__":
    if len(sys.argv)>1: main(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 800)
    else: main(os.path.join(os.path.dirname(os.path.abspath(__file__)),"smollm","smollm"))
