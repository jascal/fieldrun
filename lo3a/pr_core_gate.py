#!/usr/bin/env python3
"""The position-adaptive readout codec: does the SEMANTIC split give a cheap gate the correctness signal can't?

real_recall.py showed PR-core rank-r is ~exact on SYNTACTIC positions (punct/space/digit: R@1 71-88%, R@32
94-100%) but lossy on CONTENT words. pr_core_v2.py showed no cheap *correctness* gate works. The new idea:
route by token CLASS, which may be cheaply detectable even though per-decision correctness isn't. The gate is
nearly FREE — the PR-core top-1 token is already computed, so classifying it (syntax vs content) costs nothing:

  gate:   route to the cheap rank-r core  iff  class(core_argmax) ∈ {punct, space, digit}     (else full readout)
  codec:  exact on content (full readout); on syntax, accept the core (exact iff core_argmax == full_argmax).

Measures: (1) confusion of core-argmax class vs full-argmax class (is class cheaply detectable?);
(2) gate quality P(core==full | routed); (3) blended exactness + compute compression, OVERALL and per-domain
(prose vs code — code is format-heavy, so the syntactic-majority win is larger). SmolLM-135M, real text.
"""
import os, sys, re
import numpy as np
import bundle_io as bio
from bpe import BPE
from real_recall import forward_all, PASSAGES, classify

HERE = os.path.dirname(os.path.abspath(__file__))
def _norm(v): return v / (np.linalg.norm(v) + 1e-30)
def is_code(t): return bool(re.search(r"\bdef \b|\bfunction \b|\bimport \b|=>|\);|\{\n|self\.", t))
SYN = {"punct", "space", "digit"}

def logreg(Xtr, ytr, Xte, l2=1.0, steps=400, lr=0.5):
    """tiny full-batch logistic regression (no sklearn). standardize on train; return test P(y=1)."""
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Z=(Xtr-mu)/sd; Zt=(Xte-mu)/sd
    Z=np.c_[Z,np.ones(len(Z))]; Zt=np.c_[Zt,np.ones(len(Zt))]; w=np.zeros(Z.shape[1])
    for _ in range(steps):
        p=1/(1+np.exp(-(Z@w))); g=Z.T@(p-ytr)/len(Z)+l2*np.r_[w[:-1],0]/len(Z); w-=lr*g
    return 1/(1+np.exp(-(Zt@w)))

def main(stem, r=92):
    man,W = bio.read_bundle(stem); cfg,cfg_f = man["config"],man["config_f"]
    V=int(cfg[6]); d=int(cfg[4])
    bpe=BPE(os.path.join(os.path.dirname(stem), os.path.basename(stem)+".tokenizer.json"))
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64); gU=gain*U
    decs=[]                                                               # (a*, x, domain)
    for txt in PASSAGES:
        ids=bpe.encode(txt)
        if len(ids)<4: continue
        xall,lg=forward_all(W,cfg,cfg_f,ids); dom="code" if is_code(txt) else "prose"
        for i in range(2,len(ids)):
            decs.append((int(np.argmax(lg[i])), xall[i], dom))
    n=len(decs); tr=decs[:n//2]; te=decs[n//2:]; nte=len(te)
    rows=[_norm(gU[a]-gU[v]) for a,x,_ in tr for v in np.argsort(gU@x)[::-1][1:9]]
    Vt=np.linalg.svd(np.array(rows),full_matrices=False)[2]; A=(Vt[:r]@gU.T)
    Xtr=np.array([x for _,x,_ in tr]); Xte=np.array([x for _,x,_ in te])
    Str=(Vt[:r]@Xtr.T).T; Ste=(Vt[:r]@Xte.T).T                            # cheap r-dim core coords (free)
    Q=Ste@A                                                                # core logits [nte,vocab]
    c=np.argmax(Q,axis=1); a=np.array([d0 for d0,_,_ in te]); dom=np.array([d2 for _,_,d2 in te])
    syn_a_tr=np.array([classify(bpe.decode_token(int(ai)))in SYN for ai,_,_ in tr]).astype(float)
    syn_a=np.array([classify(bpe.decode_token(int(ai)))in SYN for ai in a])
    syn_c=np.array([classify(bpe.decode_token(int(ci)))in SYN for ci in c])

    print(f"== position-adaptive readout codec (SmolLM-135M; r={r}; {nte} real test decisions; "
          f"{100*syn_a.mean():.0f}% truly syntactic) ==")
    # (1) FREE core-argmax-class gate is BIASED (core defaults to format tokens):
    print(f"   (1) FREE core-class gate: P(true syn | core says syn)={100*np.mean(syn_a[syn_c]):.0f}% (precision), "
          f"routed {100*syn_c.mean():.0f}%, exact-on-routed P(core==full)={100*np.mean((c==a)[syn_c]):.0f}% — "
          f"the lossy core OVER-predicts format tokens, so its own class is a poor gate.")
    # (1b) IS the class cheaply decodable from the residual?  linear probe → syntax(a*)
    def report(name, pte):
        pred=pte>=0.5; ba=0.5*(np.mean(pred[syn_a])+np.mean(~pred[~syn_a]))
        prec_c=np.mean(~syn_a[~pred]) if (~pred).any() else 0              # P(content | probe says content)
        print(f"       probe on {name:<14} bal-acc {100*ba:>3.0f}%   P(content|says content) {100*prec_c:>3.0f}%   "
              f"says-syntax {100*pred.mean():>3.0f}%")
        return pred
    print(f"   (1b) is content-vs-syntax cheaply DECODABLE from the residual? linear probe → class(a*):")
    pred_S=report("Sx (free, r-dim)", logreg(Str,syn_a_tr,Ste))
    pred_x=report("x (full resid)",   logreg(Xtr,syn_a_tr,Xte))
    # (2) codec with the PROBE gate, full-scoring the core top-32 on routed (R@32 high on syntax):
    tk=np.argpartition(-Q,31,axis=1)[:,:32]; in32=np.array([a[j] in tk[j] for j in range(nte)])
    def codec(route,mask):
        rt=route[mask]
        kept=np.mean(np.where(rt, in32[mask], True))                      # routed→shortlist-exact iff a*∈top32; else full
        cost=r*V + 32*d + np.mean(~rt)*(V*d)                              # core+shortlist always; full only on content
        return kept,(V*d)/cost,rt.mean()
    print(f"   (2) codec = route to core+top32-verify iff probe(Sx) says syntax, else full readout:")
    print(f"       {'domain':<7}{'n':>5}{'exact':>8}{'compute':>9}{'core-routed':>13}")
    for lbl,mask in [("ALL",np.ones(nte,bool)),("prose",dom=="prose"),("code",dom=="code")]:
        if mask.sum()==0: continue
        kept,compr,cov=codec(pred_S,mask)
        print(f"       {lbl:<7}{mask.sum():>5}{100*kept:>7.0f}%{compr:>8.1f}×{100*cov:>12.0f}%")
    print(f"\n   reading: class IS partly decodable from the residual, but (a) the cheap Sx probe is imperfect,")
    print(f"   (b) format share is only ~{100*syn_a.mean():.0f}% on this mix, and (c) even perfect routing caps the")
    print(f"   win at the format fraction. A real, honest COMPUTE lever (bigger on code), not an exactness escape.")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"smollm","smollm"))
