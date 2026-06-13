#!/usr/bin/env python3
"""LO1 nonlinear probe — truncated Volterra/polynomial series on the PR core (Grok's degree-1/2/3 test).

Is the ~30% linear floor recoverable by a COMPACT non-linear valuation, or is it intrinsic τ*?
Project the residual onto the readout-aligned rank≈PR core (coords z), lift z to polynomial features of
degree 1/2/3 (interactions WITHIN the core), ridge-fit features→residual on TRAIN, decode on HELD-OUT.
  big lift with degree (e.g. +10-20% by degree 2)  => floor was lifted-linear slack => compact series works
  flat with degree                                 => floor is intrinsic, even compact non-linearity won't reach it
Closed-form (no torch) — a quick diagnostic; a trained narrow MLP is the rigorous follow-up. SmolLM-135M.
"""
import os, itertools
import numpy as np
import bundle_io as bio
from lo1_circuit import forward_capture

HERE=os.path.dirname(os.path.abspath(__file__)); STEM=os.path.join(HERE,"smollm","smollm")
R_CORE=92          # PR-core rank (linear term spans the full core)
R2, R3 = 20, 10    # interaction terms restricted to the top-R2 / top-R3 STANDARDIZED core coords
NDEC=2400          # plenty of data so degree-2/3 features aren't underdetermined

def poly_feats(Zs, deg):
    """Zs: [n, R_CORE] STANDARDIZED core coords -> linear + (deg≥2/3) interaction columns (no bias here)."""
    cols=[Zs]
    if deg>=2:
        z=Zs[:,:R2]; cols.append(np.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(R2),2)],1))
    if deg>=3:
        z=Zs[:,:R3]; cols.append(np.stack([z[:,i]*z[:,j]*z[:,k] for i,j,k in itertools.combinations_with_replacement(range(R3),3)],1))
    return np.concatenate(cols,1)

def fit_decode(Ftr,Xc,xm,Fte,te,decode,lam):
    B=np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Xc)                  # ridge: [F+1, d]
    return sum(decode(Fte[i]@B + xm)==te[i][1] for i in range(len(te)))/len(te)

def main():
    man,W=bio.read_bundle(STEM); cfg,cfg_f=man["config"],man["config_f"]
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64)
    d=int(cfg[4]); eps=float(cfg_f[1]); V=int(cfg[6]); rng=np.random.default_rng(7)
    decs=[]
    for _ in range(NDEC):
        ids=[int(t) for t in rng.integers(0,V,size=int(rng.integers(6,14)))]
        lg,xf,*_=forward_capture(W,cfg,cfg_f,ids); o=np.argsort(lg)[::-1]
        decs.append((int(o[0]), o[:9], xf.astype(np.float64)))
    n=len(decs); tr=decs[:int(.7*n)]; te=decs[int(.7*n):]
    rows=[(lambda df: df/(np.linalg.norm(df)+1e-30))(gain*(U[p]-U[v])) for p,o,_ in tr for v in o[1:]]
    _,_,Vt=np.linalg.svd(np.array(rows),full_matrices=False); S=Vt[:R_CORE]
    Ztr=np.array([S@x for _,_,x in tr]); Zte=np.array([S@x for _,_,x in te])
    zm,zs=Ztr.mean(0),Ztr.std(0)+1e-8; Ztr=(Ztr-zm)/zs; Zte=(Zte-zm)/zs               # standardize core coords
    Xtr=np.array([x for _,_,x in tr]); xm=Xtr.mean(0); Xc=Xtr-xm                       # centered residual target
    def decode(xh): xn=(xh/np.sqrt((xh**2).mean()+eps))*gain; return int(np.argmax(xn@U.T))
    base=sum(decode(S.T@(S@x))==p for p,_,x in te)/len(te)
    teobj=[(None,p) for p,_,_ in te]                                                    # fit_decode reads te[i][1]==pred
    print(f"== LO1 polynomial-series probe (SmolLM-135M; core rank {R_CORE}; {len(tr)} train / {len(te)} test) ==")
    print(f"   linear PR-core projection (no fit): {100*base:.0f}%   (full residual upper bound: 100%)")
    for deg in [1,2,3]:
        Ftr=poly_feats(Ztr,deg); Fte=poly_feats(Zte,deg)
        fm,fs=Ftr.mean(0),Ftr.std(0)+1e-8; Ftr=(Ftr-fm)/fs; Fte=(Fte-fm)/fs           # standardize features
        Ftr=np.c_[np.ones(len(Ftr)),Ftr]; Fte=np.c_[np.ones(len(Fte)),Fte]            # bias
        best=max(fit_decode(Ftr,Xc,xm,Fte,teobj,decode,lam) for lam in [1.,10.,100.])
        print(f"   degree {deg}  ({Ftr.shape[1]:>5} feats): decode preserved {100*best:.0f}%  (best of λ∈{{1,10,100}})")
    print("   reading: rising with degree => floor is lifted-linear slack (compact series works); flat => intrinsic τ*.")

if __name__=="__main__": main()
