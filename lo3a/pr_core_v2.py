#!/usr/bin/env python3
"""PR-core router salvage attempts (Grok Q1 + Q3).
 Q1 — second-stage self-consistency gate: accept the rank-r core decode iff it agrees with the rank-2r
      decode; else fall back to full. Tests whether cross-rank agreement predicts full-agreement (a cheap
      "is the tail about to flip me?" signal that stays in the readout-aligned basis).
 Q3 — WHITENING: whiten x by the activation covariance C (de-emphasize the massive-activation directions),
      so the decision subspace becomes relatively higher-energy. Redo the PR-core + the discarded-energy
      ratio in whitened coords. Does preservation rise and ‖discarded‖/‖x‖ shrink (⇒ a δ-bound could fire)?
SmolLM-135M.
"""
import os, sys
import numpy as np
import bundle_io as bio
from lo1_circuit import forward_capture

def basis(rows):
    _,_,Vt=np.linalg.svd(np.array(rows),full_matrices=False); return Vt

def main(stem, N=900):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64)
    d=int(cfg[4]); eps=float(cfg_f[1]); V=int(cfg[6]); rng=np.random.default_rng(7)
    decs=[]
    for _ in range(N):
        ids=[int(t) for t in rng.integers(0,V,size=int(rng.integers(6,14)))]
        lg,xf,*_=forward_capture(W,cfg,cfg_f,ids); o=np.argsort(lg)[::-1]
        decs.append((int(o[0]), o[:9], xf.astype(np.float64)))
    tr=decs[:N//2]; te=decs[N//2:]; full=V*d
    gU=(gain*U)                                                          # gain-weighted unembedding [vocab,d]

    # ---------- Q1: second-stage self-consistency gate (raw readout-aligned basis) ----------
    Vt=basis([(lambda x:x/(np.linalg.norm(x)+1e-30))(gain*(U[p]-U[v])) for p,o,_ in tr for v in o[1:]])
    A=Vt@gU.T                                                            # [maxr, vocab]
    def amax(x,r): return int(np.argmax((Vt[:r]@x)@A[:r]))
    r=92; r2=184; acc=ok=cost=0
    for p,_,x in te:
        a1,a2=amax(x,r),amax(x,r2)
        if a1==a2: acc+=1; ok+=int(a1==p); cost+=V*r2          # agree -> accept core (cost = 2r readout)
        else: ok+=1; cost+=V*r2+V*d                            # disagree -> full fallback (exact)
    n=len(te)
    print(f"== Q1 second-stage gate (SmolLM-135M; r={r} vs 2r={r2}) ==")
    print(f"   agree(r==2r) on {100*acc/n:.0f}% of decisions; among agree, == full: {100*sum(amax(x,r)==p for p,_,x in te if amax(x,r)==amax(x,r2))/max(1,acc):.0f}%")
    print(f"   gated hybrid: overall kept {100*ok/n:.0f}%   avg compression {full/(cost/n):.1f}×  (vs plain core 6.2× @ 67%)")

    # ---------- Q3: whitening by the activation covariance ----------
    X=np.array([x for _,_,x in tr]); C=(X.T@X)/len(X)                    # activation second-moment [d,d]
    w,Q=np.linalg.eigh(C); w=np.clip(w,1e-9,None); reg=0.01*w.mean()
    Whalf=Q@np.diag(np.sqrt(w))@Q.T; Winv=Q@np.diag(1.0/np.sqrt(w+reg))@Q.T  # C^{1/2}, C^{-1/2}
    gUw=gU@Whalf                                                         # whitened readout dirs: C^{1/2}(gain⊙U) [vocab,d]
    Vtw=basis([(lambda x:x/(np.linalg.norm(x)+1e-30))(gUw[p]-gUw[v]) for p,o,_ in tr for v in o[1:]])
    Aw=Vtw@gUw.T
    def amaxw(xw,r): return int(np.argmax((Vtw[:r]@xw)@Aw[:r]))
    def disc(Vb,x,r): P=Vb[:r].T@(Vb[:r]@x); return np.linalg.norm(x-P)/(np.linalg.norm(x)+1e-30)
    keep_raw=sum(amax(x,r)==p for p,_,x in te)/n
    keep_w=sum(amaxw(Winv@x,r)==p for p,_,x in te)/n
    dr=np.mean([disc(Vt,x,r) for _,_,x in te]); dw=np.mean([disc(Vtw,Winv@x,r) for _,_,x in te])
    print(f"\n== Q3 whitening (SmolLM-135M; r={r}) ==")
    print(f"   decode kept:      raw {100*keep_raw:.0f}%   whitened {100*keep_w:.0f}%")
    print(f"   discarded ‖(I-P_r)x‖/‖x‖:  raw {dr:.2f}   whitened {dw:.2f}   (smaller ⇒ tighter δ-bound ⇒ gate can fire)")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),"smollm","smollm"))
