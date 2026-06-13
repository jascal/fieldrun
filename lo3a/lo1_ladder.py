#!/usr/bin/env python3
"""Does the recoverable-rank-vs-entropy law hold across the model ladder? (SmolLM 135M → 360M → 1.7B)

If τ* is a property of LANGUAGE + a linear readout (not model capacity), the recoverable-rank↔self-information
relationship should be SCALE-INVARIANT once rank is normalized by d (each model has a different residual width).
If bigger models flatten the slope, capacity partially mitigates the tax. Same tokenizer across the ladder ⇒
identical tokens & self-information for the same text; only the model geometry differs. Reports, per model:
Spearman(recoverable_rank, self-info), median normalized rank (ρ/d) by frequency band and lexical class, and the
information rate (bits per rank-fraction). Overlays the normalized-rank-vs-info curves in lo1_ladder.png.
"""
import os, sys
import numpy as np
import bundle_io as bio
from bpe import BPE
from real_recall import forward_all, PASSAGES
from pr_core_gate import LISP_PASSAGES
from grammar_recall import fine_class
from info_rank import spearman

HERE=os.path.dirname(os.path.abspath(__file__))
def _norm(v): return v/(np.linalg.norm(v)+1e-30)
MODELS=[("135M",os.path.join(HERE,"smollm","smollm")),
        ("360M",os.path.join(HERE,"smollm360","smollm360")),
        ("1.7B",os.path.join(HERE,"smollm17","smollm17"))]
BANDS=[("freq ≤4b",lambda I:I<4),("4–7b",lambda I:(I>=4)&(I<7)),("7–9b",lambda I:(I>=9)*0+((I>=7)&(I<9))),("≥9b",lambda I:I>=9)]

def analyze(stem):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]
    d=int(cfg[4]); V=int(cfg[6])
    bpe=BPE(os.path.join(os.path.dirname(stem),os.path.basename(stem)+".tokenizer.json"))
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64); gU=gain*U
    enc=[bpe.encode(t) for t in PASSAGES+LISP_PASSAGES]
    cnt=np.zeros(V)
    for ids in enc:
        for t in ids: cnt[t]+=1
    freq=cnt/max(1,cnt.sum())
    decs=[]
    for ids in enc:
        if len(ids)<4: continue
        xall,lg=forward_all(W,cfg,cfg_f,ids)
        for i in range(2,len(ids)): decs.append((int(np.argmax(lg[i])),xall[i]))
    n=len(decs); tr=decs[:n//2]; te=decs[n//2:]; nte=len(te)
    rows=[_norm(gU[a]-gU[v]) for a,x in tr for v in np.argsort(gU@x)[::-1][1:9]]
    Vt=np.linalg.svd(np.array(rows),full_matrices=False)[2]; A=Vt@gU.T
    Xte=np.array([x for _,x in te]); a=np.array([a for a,_ in te]); P=Vt@Xte.T
    grid=sorted({r for r in [1,2,4,8,16,24,32,48,64,92,128,192,256,384,512,640,768,1024,1280,1536,d] if r<=d}|{d})
    rr=np.full(nte,d,float); done=np.zeros(nte,bool)
    for r in grid:
        arg=np.argmax((P[:r].T)@A[:r],axis=1); hit=(arg==a)&~done; rr[hit]=r; done|=hit
    cls=np.array([fine_class(bpe.decode_token(int(t))) for t in a]); info=-np.log2(np.clip(freq[a],1e-9,None))
    return dict(d=d,rr=rr,rn=rr/d,info=info,cls=cls,
                sp=spearman(rr,info),
                band={lbl:(np.median((rr/d)[f(info)]) if f(info).sum() else np.nan) for lbl,f in BANDS},
                cls_rn={c:(np.median((rr/d)[cls==c]) if (cls==c).sum() else np.nan) for c in ["function","content"]},
                bpr=info.sum()/ (rr/d).sum())                            # bits per rank-fraction (info rate)

def main():
    res=[]
    for name,stem in MODELS:
        if not os.path.exists(stem+".fieldrun.json"): print(f"   [skip {name}: no bundle]"); continue
        print(f"   analyzing {name} ...",flush=True); r=analyze(stem); r["name"]=name; res.append(r)
    print(f"\n== recoverable-rank vs entropy across the SmolLM ladder ==")
    print(f"   {'model':<7}{'d':>6}{'Spearman':>10}{'med ρ/d: func':>15}{'content':>9}"+"".join(f"{b:>9}" for b,_ in BANDS)+f"{'bits/rank-frac':>16}")
    for r in res:
        print(f"   {r['name']:<7}{r['d']:>6}{r['sp']:>+10.2f}{r['cls_rn']['function']:>15.2f}{r['cls_rn']['content']:>9.2f}"
              +"".join(f"{r['band'][b]:>9.2f}" for b,_ in BANDS)+f"{r['bpr']:>16.1f}")
    print(f"\n   reading: if Spearman stays high AND the normalized-rank (ρ/d) band profile is stable across")
    print(f"   scale, the entropy law is a property of LANGUAGE+linear-readout, not model capacity. A flattening")
    print(f"   profile at 1.7B would mean bigger models partially condition away the tax.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4.6)); cols={"135M":"#39c","360M":"#7b3","1.7B":"#c33"}
        xs=np.linspace(2,14,13)
        for r in res:
            mx=[];my=[]
            for k in range(len(xs)-1):
                s=(r["info"]>=xs[k])&(r["info"]<xs[k+1])
                if s.sum()>=4: mx.append((xs[k]+xs[k+1])/2); my.append(np.median(r["rn"][s]))
            ax.plot(mx,my,"-o",lw=2,ms=4,color=cols.get(r["name"],"#888"),label=f"{r['name']} (d={r['d']}, ρ={r['sp']:+.2f})")
        ax.set_xlabel("token self-information (bits, −log2 unigram freq)"); ax.set_ylabel("normalized recoverable rank  ρ/d")
        ax.set_title("recoverable-rank↔entropy law across the SmolLM ladder"); ax.legend(); ax.set_ylim(0,1.05)
        fig.tight_layout(); out=os.path.join(HERE,"lo1_ladder.png"); fig.savefig(out,dpi=110); print(f"\n   plot: {out}")
    except Exception as e: print(f"   (plot skipped: {e})")

if __name__=="__main__": main()
