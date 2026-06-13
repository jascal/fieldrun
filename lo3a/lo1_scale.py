#!/usr/bin/env python3
"""LO1 scale test — does the PR floor / the ~70%-at-PR knee hold up a model-size ladder?

Same-family rope ladder (SmolLM 135M → 360M → 1.7B). Per model, on random held-out decisions, report the
numbers that decide whether a "PR-core mode" is worth wiring into the toolchain (Grok's practical lever):
  PR        = median circuit participation ratio (the floor location)
  PR/d      = floor as a fraction of width (does the knee scale sub/linearly with the model?)
  span90    = 90%-energy rank of the decision directions
  at_PR     = readout-aligned fixed-S decode preservation at rank = round(PR)   (the recoverable bulk)
  peak      = best fixed-S preservation over a rank sweep                       (the ceiling)
"""
import os, sys, gc
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import bundle_io as bio
from lo1_circuit import forward_capture

HERE = os.path.dirname(os.path.abspath(__file__))
def pr_eig(v): v=np.clip(np.asarray(v,float),0,None); s=v.sum(); return float(s*s/(np.square(v).sum()+1e-30))
def energy_rank(S,frac=.9):
    sv=np.linalg.svd(np.asarray(S,float),compute_uv=False); e=np.cumsum(sv**2); e/=e[-1]; return int(np.searchsorted(e,frac)+1)

def run_model(name, stem, N=250, KC=8):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]
    nl,H,NKV,HD,D,FFN,Vv,TIED=[int(c) for c in cfg]; eps=float(cfg_f[1])
    U=(W["embed"] if TIED else W["lm_head"]); gain=W["norm"].astype(np.float64); rng=np.random.default_rng(11)
    decs=[]; prs=[]
    for _ in range(N):
        ids=[int(t) for t in rng.integers(0,Vv,size=int(rng.integers(6,14)))]
        lg,xf,attn,hid=forward_capture(W,cfg,cfg_f,ids); o=np.argsort(lg)[::-1]
        pred=int(o[0]); m=float(lg[o[0]]-lg[o[1]]); invn=float(1.0/np.sqrt((xf**2).mean()+eps))
        g_up=gain*U[pred].astype(np.float64); dl=[]
        for l in range(nl):
            op=W[f"l{l}.self_attn.o_proj"].astype(np.float64)
            for h in range(H):
                w=attn[l][h*HD:(h+1)*HD].astype(np.float64)@op[h*HD:(h+1)*HD,:]; dl.append(invn*float(w@g_up))
            dp=W[f"l{l}.mlp.down_proj"].astype(np.float64); dl.extend((invn*hid[l].astype(np.float64)*(dp@g_up)).tolist())
        a=np.abs(np.array(dl)); prs.append(pr_eig(a[np.argsort(a)[::-1][:256]]))
        decs.append((m,pred,o[:KC+1],xf.astype(np.float64)))
    PR=float(np.median(prs))
    dU=np.array([gain*(U[p].astype(np.float64)-U[o[1]].astype(np.float64)) for _,p,o,_ in decs])
    dU/=np.linalg.norm(dU,axis=1,keepdims=True)+1e-30; span90=energy_rank(dU)
    decs.sort(key=lambda r:r[0]); tr=decs[:N//2]; te=decs[N//2:]
    rows=[(lambda diff: diff/(np.linalg.norm(diff)+1e-30))(gain*(U[p].astype(np.float64)-U[v].astype(np.float64)))
          for _,p,o,_ in tr for v in o[1:]]
    _,_,Vt=np.linalg.svd(np.array(rows),full_matrices=False)
    Ud=U.astype(np.float64)
    def decode(xp): xn=(xp/np.sqrt((xp**2).mean()+eps))*gain; return int(np.argmax(xn@Ud.T))
    def keep(r): r=min(r,Vt.shape[0]); return sum(decode(Vt[:r].T@(Vt[:r]@x))==p for _,p,_,x in te)/len(te)
    at_pr=keep(int(round(PR))); peak=max(keep(r) for r in [16,32,64,96,128,192,256] if r<=D)
    del W; gc.collect()
    return dict(name=name,d=D,nl=nl,PR=PR,PRd=PR/D,span90=span90,at_pr=at_pr,peak=peak)

if __name__=="__main__":
    models=[("SmolLM-135M",os.path.join(HERE,"smollm","smollm")),
            ("SmolLM-360M",os.path.join(HERE,"smollm360","smollm360")),
            ("SmolLM-1.7B",os.path.join(HERE,"smollm17","smollm17"))]
    res=[]
    for nm,st in models:
        if not os.path.exists(st+".fieldrun.json"): print(f"  skip {nm} (no bundle)"); continue
        print(f"  running {nm}…", flush=True); res.append(run_model(nm,st)); print("   ",res[-1],flush=True)
    print("\n== LO1 scale test ==")
    print(f"   {'model':<14}{'d':>6}{'PR':>8}{'PR/d':>8}{'span90':>9}{'at_PR':>8}{'peak':>8}")
    for r in res:
        print(f"   {r['name']:<14}{r['d']:>6}{r['PR']:>8.1f}{r['PRd']:>8.3f}{r['span90']:>9}{100*r['at_pr']:>7.0f}%{100*r['peak']:>7.0f}%")
    if len(res)>=2:
        ds=[r['d'] for r in res]
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        ax[0].plot(ds,[r['PR'] for r in res],"o-",label="PR (floor)"); ax[0].plot(ds,[r['d'] for r in res],"--",c="gray",label="d")
        ax[0].set_xlabel("hidden dim d"); ax[0].set_ylabel("rank"); ax[0].set_title("PR floor vs model width"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(ds,[100*r['at_pr'] for r in res],"o-",label="decode preserved @ r=PR"); ax[1].plot(ds,[100*r['PRd'] for r in res],"s-",label="PR/d (%)")
        ax[1].set_xlabel("hidden dim d"); ax[1].set_ylabel("%"); ax[1].set_title("PR-core: recoverable bulk + compression"); ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout(); p=os.path.join(HERE,"lo1_scale.png"); fig.savefig(p,dpi=110); print(f"\n   wrote {p}")
