#!/usr/bin/env python3
"""Q1 — does the "content-incoming" probe direction localize to specific heads / MLP neurons?

pr_core_gate.py found a cheap linear probe ŵ (in residual space) that predicts "about to emit CONTENT" at
~83% bal-acc. Here we ask the mechanistic question: which circuits BUILD that signal? The residual at the
decision position decomposes exactly as
      ŵ·x  =  ŵ·embed  +  Σ_{l,h} (head_{l,h} write)·ŵ  +  Σ_{l,j} (neuron_{l,j} write)·ŵ
so each head and neuron has an EXACT additive contribution to the probe coordinate. We accumulate the mean
ABSOLUTE contribution of every head (270) and neuron (~46k) over real decisions, and ask: is it concentrated
(a localized "content detector" circuit) or diffuse? Reports participation ratios, top circuits, the per-layer
build-up profile, and a random-direction control. SmolLM-135M, real text. Reuses bpe + real_recall passages.
"""
import os, sys, re
import numpy as np
import bundle_io as bio
from bpe import BPE
from real_recall import PASSAGES, classify
from pr_core_gate import LISP_PASSAGES

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = {"punct","space","digit"}
def pr(v):                                   # participation ratio of non-negative weights: (Σ)²/Σ² (in [1,N])
    v=np.asarray(v,float); s=v.sum(); return float(s*s/(np.square(v).sum()+1e-30))

def logreg_dir(X, y, l2=1.0, steps=500, lr=0.5):
    """logistic regression; return the RAW residual-space direction (un-standardized), unit-normalized."""
    mu=X.mean(0); sd=X.std(0)+1e-6; Z=np.c_[(X-mu)/sd, np.ones(len(X))]; w=np.zeros(Z.shape[1])
    for _ in range(steps):
        p=1/(1+np.exp(-(Z@w))); w-=lr*(Z.T@(p-y)/len(Z)+l2*np.r_[w[:-1],0]/len(Z))
    wr=w[:-1]/sd; return wr/(np.linalg.norm(wr)+1e-30)

def fwd_collect(W,cfg,cfg_f,ids):
    """return per-position raw residual [seq,d] and full logits [seq,vocab] (for labels)."""
    nl,H,NKV,HD,D,FFN,V,TIED=[int(c) for c in cfg]; theta,eps=map(float,cfg_f); HALF,REP=HD//2,H//NKV
    inv=(1.0/(theta**(2.0*np.arange(HALF,dtype=np.float32)/HD))).astype(np.float32); seq=len(ids)
    ang=np.arange(seq,dtype=np.float32)[:,None]*inv[None,:]; COS=np.cos(ang)[:,None,:].astype(np.float32); SIN=np.sin(ang)[:,None,:].astype(np.float32)
    causal=np.triu(np.ones((seq,seq),bool),1)
    rope=lambda x,nh:(lambda xr:(np.concatenate([xr[...,:HALF]*COS-xr[...,HALF:]*SIN,xr[...,HALF:]*COS+xr[...,:HALF]*SIN],-1)).reshape(seq,nh*HD).astype(np.float32))(x.reshape(seq,nh,HD))
    silu=lambda x:(x/(1.0+np.exp(-x))).astype(np.float32); x=W["embed"][ids].astype(np.float32)
    for l in range(nl):
        p=f"l{l}.";a=bio._rmsnorm(x,W[p+"in_ln"],eps)
        q=rope((a@W[p+"self_attn.q_proj"]).astype(np.float32),H);k=rope((a@W[p+"self_attn.k_proj"]).astype(np.float32),NKV);v=(a@W[p+"self_attn.v_proj"]).astype(np.float32)
        ao=np.zeros((seq,H*HD),np.float32)
        for h in range(H):
            kv=h//REP;sc=(q[:,h*HD:(h+1)*HD]@k[:,kv*HD:(kv+1)*HD].T)/np.float32(np.sqrt(HD));sc[causal]=-1e30
            sc=np.exp(sc-sc.max(1,keepdims=True));sc/=sc.sum(1,keepdims=True);ao[:,h*HD:(h+1)*HD]=(sc@v[:,kv*HD:(kv+1)*HD]).astype(np.float32)
        x=(x+ao@W[p+"self_attn.o_proj"]).astype(np.float32)
        a2=bio._rmsnorm(x,W[p+"post_ln"],eps);hid=(silu(a2@W[p+"mlp.gate_proj"])*(a2@W[p+"mlp.up_proj"])).astype(np.float32)
        x=(x+hid@W[p+"mlp.down_proj"]).astype(np.float32)
    U=W["embed"] if TIED else W["lm_head"]; return x.astype(np.float64),(bio._rmsnorm(x,W["norm"],eps)@U.T).astype(np.float32)

def fwd_attrib(W,cfg,cfg_f,ids,lab,what,Hc,Nh,Ec,cnt):
    """re-run; accumulate SIGNED contribution to ŵ·x, split by class (lab[pos]∈{0 syntax,1 content} for pos≥2).
       Hc[l,h,c]=Σ head write·ŵ over class-c positions; Nh[l,j,c]=Σ activation over class-c (×|down·ŵ| later);
       Ec[c], cnt[c]=position counts. The class-mean difference per circuit = its DISCRIMINATIVE contribution."""
    nl,H,NKV,HD,D,FFN,V,TIED=[int(c) for c in cfg]; theta,eps=map(float,cfg_f); HALF,REP=HD//2,H//NKV
    inv=(1.0/(theta**(2.0*np.arange(HALF,dtype=np.float32)/HD))).astype(np.float32); seq=len(ids)
    ang=np.arange(seq,dtype=np.float32)[:,None]*inv[None,:]; COS=np.cos(ang)[:,None,:].astype(np.float32); SIN=np.sin(ang)[:,None,:].astype(np.float32)
    causal=np.triu(np.ones((seq,seq),bool),1)
    rope=lambda x,nh:(lambda xr:(np.concatenate([xr[...,:HALF]*COS-xr[...,HALF:]*SIN,xr[...,HALF:]*COS+xr[...,:HALF]*SIN],-1)).reshape(seq,nh*HD).astype(np.float32))(x.reshape(seq,nh,HD))
    silu=lambda x:(x/(1.0+np.exp(-x))).astype(np.float32); x=W["embed"][ids].astype(np.float32)
    lab=np.asarray(lab); m=[lab==0,lab==1]                                # boolean masks over positions 2..seq
    for c in (0,1): cnt[c]+=int(m[c].sum()); Ec[c]+=float((x[2:][m[c]]@what).sum())
    for l in range(nl):
        p=f"l{l}.";a=bio._rmsnorm(x,W[p+"in_ln"],eps)
        q=rope((a@W[p+"self_attn.q_proj"]).astype(np.float32),H);k=rope((a@W[p+"self_attn.k_proj"]).astype(np.float32),NKV);v=(a@W[p+"self_attn.v_proj"]).astype(np.float32)
        ao=np.zeros((seq,H*HD),np.float32)
        for h in range(H):
            kv=h//REP;sc=(q[:,h*HD:(h+1)*HD]@k[:,kv*HD:(kv+1)*HD].T)/np.float32(np.sqrt(HD));sc[causal]=-1e30
            sc=np.exp(sc-sc.max(1,keepdims=True));sc/=sc.sum(1,keepdims=True);ao[:,h*HD:(h+1)*HD]=(sc@v[:,kv*HD:(kv+1)*HD]).astype(np.float32)
        op=W[p+"self_attn.o_proj"]
        for h in range(H):
            cw=ao[2:,h*HD:(h+1)*HD]@(op[h*HD:(h+1)*HD,:]@what)            # head write·ŵ per pos
            for c in (0,1): Hc[l,h,c]+=float(cw[m[c]].sum())
        x=(x+ao@op).astype(np.float32)
        a2=bio._rmsnorm(x,W[p+"post_ln"],eps);hid=(silu(a2@W[p+"mlp.gate_proj"])*(a2@W[p+"mlp.up_proj"])).astype(np.float32)
        h2=hid[2:]
        for c in (0,1): Nh[l,:,c]+=h2[m[c]].sum(0)                       # Σ activation per neuron per class
        x=(x+hid@W[p+"mlp.down_proj"]).astype(np.float32)

def main(stem):
    man,W=bio.read_bundle(stem);cfg,cfg_f=man["config"],man["config_f"]
    nl,H,NKV,HD,D,FFN,V,TIED=[int(c) for c in cfg]
    bpe=BPE(os.path.join(os.path.dirname(stem),os.path.basename(stem)+".tokenizer.json"))
    enc=[bpe.encode(t) for t in PASSAGES+LISP_PASSAGES]
    X=[];y=[];labs=[]
    for ids in enc:
        if len(ids)<4: labs.append(np.array([])); continue
        x,lg=fwd_collect(W,cfg,cfg_f,ids)
        lab=np.array([0.0 if classify(bpe.decode_token(int(np.argmax(lg[i])))) in SYN else 1.0 for i in range(2,len(ids))])
        labs.append(lab); X.append(x[2:]); y.append(lab)
    X=np.vstack(X);y=np.concatenate(y)
    what=logreg_dir(X,y)                                                  # ŵ : content-incoming direction [d]

    Hc=np.zeros((nl,H,2));Nh=np.zeros((nl,FFN,2));Ec=[0.0,0.0];cnt=[0,0]
    for ids,lab in zip(enc,labs):
        if len(ids)>=4: fwd_attrib(W,cfg,cfg_f,ids,lab,what,Hc,Nh,Ec,cnt)
    nc,ns=cnt[1],cnt[0]                                                   # content / syntax position counts
    gj=np.array([W[f"l{l}.mlp.down_proj"]@what for l in range(nl)])       # signed down_j·ŵ [nl,FFN]
    # DISCRIMINATIVE contribution = class-mean(content) − class-mean(syntax), per circuit (signs into the probe gap)
    dhead=Hc[:,:,1]/nc - Hc[:,:,0]/ns                                     # [nl,H]
    dneur=gj*(Nh[:,:,1]/nc - Nh[:,:,0]/ns)                                # [nl,FFN]
    demb=Ec[1]/nc - Ec[0]/ns
    gap=dhead.sum()+dneur.sum()+demb                                      # = mean_content(ŵ·x) − mean_syntax(ŵ·x)
    hf=dhead.ravel(); nf=dneur.ravel()
    print(f"== Q1 localization: which circuits DISCRIMINATE content vs syntax? (SmolLM-135M; "
          f"{len(X)} decisions, {int(y.sum())} content / {int((1-y).sum())} syntax) ==")
    print(f"   probe class-gap in ŵ·x = {gap:.2f}  (embed {100*demb/gap:+.0f}%, heads {100*hf.sum()/gap:+.0f}%, "
          f"neurons {100*nf.sum()/gap:+.0f}% of the separation)")
    print(f"   heads  ({nl*H} total): PR {pr(np.abs(hf)):.0f}  top-8 share {100*np.abs(hf)[np.argsort(np.abs(hf))[-8:]].sum()/np.abs(hf).sum():.0f}% of |head| signal")
    print(f"   neurons({nl*FFN} total): PR {pr(np.abs(nf)):.0f}  top-32 share {100*np.abs(nf)[np.argsort(np.abs(nf))[-32:]].sum()/np.abs(nf).sum():.0f}%  "
          f"top-1% share {100*np.abs(nf)[np.argsort(np.abs(nf))[-(nl*FFN//100):]].sum()/np.abs(nf).sum():.0f}% of |neuron| signal")
    th=np.dstack(np.unravel_index(np.argsort(np.abs(hf))[::-1][:6],dhead.shape))[0]
    print(f"   top discriminative heads (L,H, share% of gap): "+", ".join(f"L{l}H{h}={100*dhead[l,h]/gap:+.1f}" for l,h in th))
    tn=np.dstack(np.unravel_index(np.argsort(np.abs(nf))[::-1][:8],dneur.shape))[0]
    print(f"   top discriminative neurons (L#idx, share% of gap): "+", ".join(f"L{l}#{j}={100*dneur[l,j]/gap:+.1f}" for l,j in tn))
    lp=np.abs(dhead).sum(1)+np.abs(dneur).sum(1)
    band=[("early L0-9",lp[:10].sum()),("mid L10-19",lp[10:20].sum()),("late L20-29",lp[20:].sum())]
    print(f"   |discriminative| by depth: "+"  ".join(f"{b}={100*s/lp.sum():.0f}%" for b,s in band))
    # how many neurons to reconstruct most of the gap?
    order=np.argsort(np.abs(nf))[::-1]; cum=np.cumsum(nf[order])/nf.sum()
    for frac in (0.5,0.8,0.9):
        k=int(np.searchsorted(cum,frac))+1; print(f"   neurons for {int(100*frac)}% of the (signed) neuron gap: {k}  ({100*k/(nl*FFN):.1f}% of all neurons)")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"smollm","smollm"))
