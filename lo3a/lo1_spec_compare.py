#!/usr/bin/env python3
"""LO1 capstone — decision-direction spectrum vs raw residual-activation spectrum (Grok follow-up).

Tests whether the decode geometry is harder (heavier-tailed) than the raw activations. Result (SmolLM-135M):
the DECISION spectrum is HEAVIER-tailed (lower α) and spans MORE effective dimensions than the raw residual
— the residual is concentrated in ~8 dominant directions (massive-activation regime), while the decode lives
in a broader heavy-tailed ~18–58-dim structure. The compression difficulty is in the DECODE, not the stream.
"""
import os, sys
import numpy as np
import bundle_io as bio
from lo1_circuit import forward_capture
from lo1_spectrum import hard_rank, soft_rank, alpha

def main(stem, N=800):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64); V=int(cfg[6])
    rng=np.random.default_rng(5); dU=[]; X=[]
    for _ in range(N):
        ids=[int(t) for t in rng.integers(0,V,size=int(rng.integers(6,14)))]
        lg,xf,*_=forward_capture(W,cfg,cfg_f,ids); o=np.argsort(lg)[::-1]
        d=gain*(U[o[0]]-U[o[1]]); dU.append(d/(np.linalg.norm(d)+1e-30)); X.append(xf.astype(np.float64))
    print(f"== decision-direction vs raw-activation spectra ({os.path.basename(stem)}, {N} decisions) ==")
    for M,lab,center in [(dU,"decision-direction (ΔU)",False),(X,"raw residual-activation",True)]:
        M=np.asarray(M); M=(M-M.mean(0)) if center else M
        lam=np.linalg.svd(M,compute_uv=False)**2
        print(f"   {lab:<28} hard_rank={hard_rank(lam):6.1f}  soft_rank={soft_rank(lam):6.1f}  "
              f"soft/hard={soft_rank(lam)/hard_rank(lam):4.1f}  alpha={alpha(lam):.3f}")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),"smollm","smollm"))
