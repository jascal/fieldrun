#!/usr/bin/env python3
"""Is τ* information-theoretic? Correlate per-token RECOVERABLE RANK against token self-information.

Hypothesis (the entropy framing): the closed-class scaffolding is the low-entropy, predictable part of
language (compressible by a low-rank lens); the open-class lexicon carries the actual Shannon information of
the text (incompressible by a fixed linear lens). If so, the minimal rank at which the PR-core's top-1 recovers
the full-model argmax should TRACK the token's self-information: frequent/low-info tokens recover at low rank,
rare/high-info tokens need high rank — monotonically.

recoverable_rank(decision) = min r in a grid s.t. argmax(core_logits at rank r) == full argmax a*  (=d if never).
self-info proxies: (1) corpus unigram self-info −log2 freq(a*); (2) baseline logit −meanlogit[a*] (model-intrinsic
frequency: frequent tokens have systematically higher logits across contexts). SmolLM-135M, real text.
"""
import os, sys, math
import numpy as np
import bundle_io as bio
from bpe import BPE
from real_recall import forward_all, PASSAGES
from pr_core_gate import LISP_PASSAGES
from grammar_recall import fine_class

HERE=os.path.dirname(os.path.abspath(__file__))
def _norm(v): return v/(np.linalg.norm(v)+1e-30)
def spearman(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    ra=ra-ra.mean(); rb=rb-rb.mean(); return float((ra@rb)/(np.linalg.norm(ra)*np.linalg.norm(rb)+1e-30))

def main(stem):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]
    d=int(cfg[4]); V=int(cfg[6])
    bpe=BPE(os.path.join(os.path.dirname(stem),os.path.basename(stem)+".tokenizer.json"))
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64); gU=gain*U
    enc=[bpe.encode(t) for t in PASSAGES+LISP_PASSAGES]
    cnt=np.zeros(V)                                                       # corpus unigram counts
    for ids in enc:
        for t in ids: cnt[t]+=1
    freq=cnt/max(1,cnt.sum())
    decs=[]; logsum=np.zeros(V); nlog=0
    for ids in enc:
        if len(ids)<4: continue
        xall,lg=forward_all(W,cfg,cfg_f,ids)
        for i in range(2,len(ids)):
            decs.append((int(np.argmax(lg[i])), xall[i])); logsum+=lg[i]; nlog+=1
    meanlogit=logsum/nlog
    n=len(decs); tr=decs[:n//2]; te=decs[n//2:]; nte=len(te)
    rows=[_norm(gU[a]-gU[v]) for a,x in tr for v in np.argsort(gU@x)[::-1][1:9]]
    Vt=np.linalg.svd(np.array(rows),full_matrices=False)[2]; A=Vt@gU.T    # [d, vocab]
    Xte=np.array([x for _,x in te]); a=np.array([a for a,_ in te])
    P=Vt@Xte.T                                                            # [d, nte] full projection coords
    grid=[1,2,4,8,16,24,32,48,64,92,128,192,256,384,512,d]
    rr=np.full(nte,d,dtype=float)                                         # recoverable rank (top-1 exact); default d
    done=np.zeros(nte,bool)
    for r in grid:
        arg=np.argmax((P[:r].T)@A[:r],axis=1)
        hit=(arg==a)&~done; rr[hit]=r; done|=hit
    cls=np.array([fine_class(bpe.decode_token(int(t))) for t in a])
    info_c=-np.log2(np.clip(freq[a],1e-9,None))                          # corpus self-info (bits)
    info_m=-(meanlogit[a]-meanlogit.mean())                             # model baseline-logit rarity (higher=rarer)

    print(f"== recoverable rank vs token self-information (SmolLM-135M; {nte} real test decisions; d={d}) ==")
    print(f"   recoverable_rank = min r where the core top-1 == full argmax (=d if never).")
    print(f"   Spearman(recoverable_rank, corpus self-info)  = {spearman(rr,info_c):+.2f}")
    print(f"   Spearman(recoverable_rank, baseline-logit rarity) = {spearman(rr,info_m):+.2f}")
    print(f"\n   median recoverable rank by lexical class:")
    for c in ["space","punct","digit","function","content"]:
        s=cls==c
        if s.sum()==0: continue
        print(f"     {c:<10}{s.sum():>5}  med self-info {np.median(info_c[s]):>5.1f} bits   med recoverable-rank {np.median(rr[s]):>5.0f}   %recovered≤92 {100*np.mean(rr[s]<=92):>3.0f}%")
    print(f"\n   recoverable rank by corpus-frequency band (rarer ⇒ more information):")
    bands=[("very frequent (≤4 bits)",info_c<4),("frequent (4–7)",(info_c>=4)&(info_c<7)),
           ("rare (7–9)",(info_c>=7)&(info_c<9)),("very rare (≥9 bits)",info_c>=9)]
    print(f"     {'band':<26}{'n':>5}{'med-rank':>10}{'%≤92':>7}{'%open-class':>13}")
    for lbl,sel in bands:
        if sel.sum()==0: continue
        print(f"     {lbl:<26}{sel.sum():>5}{np.median(rr[sel]):>10.0f}{100*np.mean(rr[sel]<=92):>6.0f}%{100*np.mean(cls[sel]=='content'):>12.0f}%")
    print(f"\n   reading: if recoverable-rank rises monotonically with self-information, τ* is the text's ENTROPY")
    print(f"   (closed-class = low-entropy scaffolding, compressible; open-class lexis = Shannon information,")
    print(f"   incompressible by a fixed linear lens) — not an architectural artifact.")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(11,4.2))
        opn=cls=="content"
        ax[0].scatter(info_c[~opn],rr[~opn],s=8,alpha=.4,c="#2a7",label="closed-class")
        ax[0].scatter(info_c[opn],rr[opn],s=8,alpha=.4,c="#c33",label="open-class (content)")
        # median curve
        xs=np.linspace(info_c.min(),info_c.max(),9); mx=[];my=[]
        for k in range(len(xs)-1):
            s=(info_c>=xs[k])&(info_c<xs[k+1])
            if s.sum()>=5: mx.append((xs[k]+xs[k+1])/2); my.append(np.median(rr[s]))
        ax[0].plot(mx,my,"k-o",lw=2,ms=4,label="median")
        ax[0].set_xlabel("token self-information (bits, −log2 unigram freq)"); ax[0].set_ylabel("recoverable rank (top-1 exact)")
        ax[0].set_title(f"recoverable rank vs self-info  (Spearman {spearman(rr,info_c):+.2f})"); ax[0].legend(fontsize=8)
        # bar: %recovered ≤92 by class
        cs=["space","punct","digit","function","content"]; vals=[100*np.mean(rr[cls==c]<=92) for c in cs]
        ax[1].bar(cs,vals,color=["#2a7","#2a7","#2a7","#7b3","#c33"]); ax[1].set_ylabel("% recovered at rank ≤ 92")
        ax[1].set_title("cheap-recoverability by lexical class"); ax[1].tick_params(axis="x",rotation=30)
        fig.tight_layout(); out=os.path.join(HERE,"info_rank.png"); fig.savefig(out,dpi=110); print(f"\n   plot: {out}")
    except Exception as e:
        print(f"   (plot skipped: {e})")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"smollm","smollm"))
