#!/usr/bin/env python3
"""LO1 — the last hatch (Grok): DECODE-OPTIMAL (readout-aligned) fixed basis vs variance-PCA.

lo1_curve fit S = PCA of the runner-up decision direction. Grok: the decode-optimal fixed basis is the
top singular directions of the gain-weighted READOUT diffs gain⊙(U_pred − U_v) over the TOP-K competitors
at each decision — the directions with direct leverage on the logit margin. If the 70% ceiling was basis
misalignment it should rise (Grok: low-mid 80s); if it's a real floor it stays ~PR-bounded and still crashes.
Same held-out decisions, same gain, same decode test. SmolLM-135M.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import bundle_io as bio
from lo1_circuit import forward_capture

HERE=os.path.dirname(os.path.abspath(__file__)); STEM=os.path.join(HERE,"smollm","smollm")
RS=[1,2,4,8,16,24,32,48,64,96,128,192,256,384,576]; KCOMP=8

def main():
    man,W=bio.read_bundle(STEM); cfg,cfg_f=man["config"],man["config_f"]
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64)
    d=int(cfg[4]); eps=float(cfg_f[1]); V=int(cfg[6]); rng=np.random.default_rng(7)
    recs=[]
    for _ in range(800):
        ids=[int(t) for t in rng.integers(0,V,size=int(rng.integers(6,14)))]
        lg,xf,*_=forward_capture(W,cfg,cfg_f,ids); o=np.argsort(lg)[::-1]
        m=float(lg[o[0]]-lg[o[1]]); top=o[:KCOMP+1]
        recs.append((m, int(o[0]), top, xf.astype(np.float64)))
    n=len(recs); tr=recs[:n//2]; te=recs[n//2:]
    # baseline S: runner-up decision direction only (variance-PCA, as in lo1_curve)
    Dr=np.array([gain*(U[pred]-U[top[1]]) for _,pred,top,_ in tr]); Dr/=np.linalg.norm(Dr,axis=1,keepdims=True)+1e-30
    _,_,Vt_run=np.linalg.svd(Dr,full_matrices=False)
    # readout-aligned S: gain⊙(U_pred − U_v) over the top-K competitors, all decisions
    rows=[]
    for _,pred,top,_ in tr:
        for v in top[1:]:
            diff=gain*(U[pred]-U[v]); rows.append(diff/(np.linalg.norm(diff)+1e-30))
    _,_,Vt_read=np.linalg.svd(np.array(rows),full_matrices=False)
    def decode(xp): xn=(xp/np.sqrt((xp**2).mean()+eps))*gain; return int(np.argmax(xn@U.T))
    te.sort(key=lambda r:r[0]); t=len(te)//3
    def curve(Vt,g): return [sum(decode(Vt[:r].T@(Vt[:r]@x))==pred for _,pred,_,x in g)/max(1,len(g)) for r in RS]
    print(f"== readout-aligned vs variance-PCA fixed basis (SmolLM-135M; {len(tr)} train / {len(te)} test; PR≈92) ==")
    print("   basis / subset            " + "".join(f"{r:>5}" for r in RS) + "   peak")
    out={}
    for name,Vt in [("variance-PCA (runner-up)",Vt_run),("readout-aligned (top-8)",Vt_read)]:
        for lbl,g in [("ALL",te),("forge-tax",te[:t]),("retrievable",te[2*t:])]:
            c=curve(Vt,g); out[(name,lbl)]=c
            print(f"   {name[:18]:<18}{lbl:<10}"+"".join(f"{100*v:>5.0f}" for v in c)+f"   {100*max(c):>4.0f}%")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(RS,[100*v for v in out[("variance-PCA (runner-up)","ALL")]],"o-",label="variance-PCA (runner-up) ALL",color="C0")
    ax.plot(RS,[100*v for v in out[("readout-aligned (top-8)","ALL")]],"s-",label="readout-aligned (top-8) ALL",color="C3")
    ax.axvline(92,ls="--",c="gray",alpha=.6); ax.text(95,15,"scalar PR≈92",color="gray",fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("fixed-S valuation rank r"); ax.set_ylabel("% held-out decodes preserved")
    ax.set_title("LO1: decode-optimal (readout) vs variance basis (SmolLM-135M)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); p=os.path.join(HERE,"lo1_readout.png"); fig.savefig(p,dpi=120); print(f"\n   wrote {p}")

if __name__=="__main__": main()
