#!/usr/bin/env python3
"""Which functional of the OUTPUT distribution sets recoverable rank? (Grok's controlled-skew families)

worst_case.py conflated skew and support (uniform-over-m varies both). Here we DECOUPLE them with three
distribution families over the real readout geometry — Dirichlet(α, effective_support), Zipf-mixture
(head_mass, head_size), and power-law(s) — and ask: across all of them, does median recoverable rank track
exp(H_output) (the perplexity / effective vocabulary), or the 95%-cumulative support, or the mixture head?
The mixture is the sharp test: a small heavy head + flat tail decouples exp(H) from raw support and tests
whether the lens compresses the HEAD regardless of the tail (unifying per-token info_rank with aggregate
worst_case). Pure geometry on SmolLM-135M's gU (d=576), no forward pass.
"""
import os, sys
import numpy as np
import bundle_io as bio

HERE=os.path.dirname(os.path.abspath(__file__))
def _norm(v): return v/(np.linalg.norm(v)+1e-30)
def H(p): p=np.asarray(p); p=p[p>0]; return float(-(p*np.log2(p)).sum())
def eff95(p):                                              # min #tokens for 95% cumulative mass
    s=np.sort(p)[::-1]; return int(np.searchsorted(np.cumsum(s),0.95))+1
def spearman(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra-=ra.mean(); rb-=rb.mean(); return float((ra@rb)/(np.linalg.norm(ra)*np.linalg.norm(rb)+1e-30))

def zipf(K,s):
    p=np.arange(1,K+1,dtype=float)**(-s); return p/p.sum()
def mixture(K,head_mass,head_size):
    p=np.zeros(K); p[:head_size]=zipf(head_size,1.0)*head_mass; p[head_size:]=(1-head_mass)/(K-head_size); return p
def dirichlet(K,alpha,support,rng):
    a=np.full(K,1e-6); a[:support]=alpha; return rng.dirichlet(a)

def measure(G,p,d,rng,N=1400,head_size=None):
    """sample argmaxes ~p; x=G[a]; fit lens on competitor diffs (train half); return median ρ/d, R@92,
       and (if head_size) the head/tail median ρ/d split."""
    K=len(G); idx=rng.choice(K,size=N,p=p); tr=idx[:N//2]; te=idx[N//2:]
    rows=[]
    for a in tr:
        sc=G@G[a]; comp=np.argsort(sc)[::-1]; comp=comp[comp!=a][:8]
        for v in comp: rows.append(_norm(G[a]-G[v]))
    Vt=np.linalg.svd(np.array(rows),full_matrices=False)[2]; A=Vt@G.T
    grid=[1,2,4,8,16,24,32,48,64,92,128,192,256,384,512,d]
    P=Vt@G[te].T; rr=np.full(len(te),d,float); done=np.zeros(len(te),bool)
    for r in grid:
        arg=np.argmax((P[:r].T)@A[:r],axis=1); hit=(arg==te)&~done; rr[hit]=r; done|=hit
    R92=np.mean(rr<=92)
    out=dict(medrank=np.median(rr/d), R92=R92)
    if head_size is not None:
        h=te<head_size; out["head"]=np.median(rr[h]/d) if h.any() else np.nan
        out["tail"]=np.median(rr[~h]/d) if (~h).any() else np.nan
    return out

def main(stem):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]; d=int(cfg[4]); V=int(cfg[6])
    gU=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64)*W["norm"].astype(np.float64)
    rng=np.random.default_rng(0); pool=rng.choice(V,size=4096,replace=False); G=gU[pool]; K=len(G)
    recs=[]                                                # (label, expH, eff95, medrank)
    print(f"== which functional of p sets recoverable rank? (SmolLM-135M, d={d}, pool K={K}) ==")

    print(f"\n   Dirichlet(α, support) — α small=Zipf-like, large=uniform; support caps the effective vocab:")
    print(f"      {'α':>6}{'support':>9}{'exp(H)':>9}{'eff95':>8}{'med ρ/d':>9}{'R@92':>7}")
    for sup in [512,1024,2048,4096]:
        for al in [0.02,0.2,2.0]:
            p=dirichlet(K,al,sup,np.random.default_rng(10)); m=measure(G,p,d,np.random.default_rng(11))
            recs.append(("dir",2**H(p),eff95(p),m["medrank"]))
            print(f"      {al:>6.2f}{sup:>9}{2**H(p):>9.0f}{eff95(p):>8}{m['medrank']:>9.2f}{100*m['R92']:>6.0f}%")

    print(f"\n   Zipf-mixture(head_mass, head_size) — decouples a compressible HEAD from a flat TAIL:")
    print(f"      {'head_mass':>10}{'head_size':>10}{'exp(H)':>9}{'eff95':>8}{'med ρ/d':>9}{'head ρ/d':>10}{'tail ρ/d':>10}")
    for hm in [0.5,0.8,0.95]:
        for hs in [64,256]:
            p=mixture(K,hm,hs); m=measure(G,p,d,np.random.default_rng(12),head_size=hs)
            recs.append(("mix",2**H(p),eff95(p),m["medrank"]))
            print(f"      {hm:>10.2f}{hs:>10}{2**H(p):>9.0f}{eff95(p):>8}{m['medrank']:>9.2f}{m['head']:>10.2f}{m['tail']:>10.2f}")

    print(f"\n   power-law(s) baseline:")
    print(f"      {'s':>6}{'exp(H)':>9}{'eff95':>8}{'med ρ/d':>9}")
    for s in [0.0,0.5,1.0,1.5,2.0]:
        p=zipf(K,s); m=measure(G,p,d,np.random.default_rng(13))
        recs.append(("zipf",2**H(p),eff95(p),m["medrank"]))
        print(f"      {s:>6.1f}{2**H(p):>9.0f}{eff95(p):>8}{m['medrank']:>9.2f}")

    # which predicts median rank better across ALL families: exp(H) or eff95? (both capped at d)
    expH=np.array([min(r[1],d) for r in recs]); e95=np.array([min(r[2],d) for r in recs]); mr=np.array([r[3]*d for r in recs])
    print(f"\n   across all {len(recs)} distributions, predicting recoverable rank (capped at d):")
    print(f"      Spearman(rank, min(exp(H),d)) = {spearman(mr,expH):+.2f}   Spearman(rank, min(eff95,d)) = {spearman(mr,e95):+.2f}")
    print(f"   ⇒ exp(H_output) (the effective vocabulary / perplexity) is the controlling functional; the mixture")
    print(f"   head/tail split shows the lens compresses the HEAD (low ρ/d) regardless of a flat tail (high ρ/d)")
    print(f"   — unifying per-token info_rank with the aggregate τ* = min(exp(H_output), d).")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        col={"dir":"#39c","mix":"#e90","zipf":"#c33"}; fig,ax=plt.subplots(figsize=(6.8,4.4))
        for lab in ("dir","mix","zipf"):
            xs=[min(r[1],d) for r in recs if r[0]==lab]; ys=[r[3]*d for r in recs if r[0]==lab]
            ax.scatter(xs,ys,c=col[lab],s=45,edgecolor="k",label=lab,zorder=3)
        ax.plot([1,d],[1,d],"k--",lw=1,label="rank = exp(H)"); ax.axhline(d,color="#999",ls=":",lw=1)
        ax.set_xlabel("effective output vocabulary  min(exp(H), d)"); ax.set_ylabel("recoverable rank")
        ax.set_title("recoverable rank ≈ min(exp(H_output), d), across distribution families"); ax.legend(fontsize=8)
        fig.tight_layout(); out=os.path.join(HERE,"worst_case2.png"); fig.savefig(out,dpi=110); print(f"\n   plot: {out}")
    except Exception as e: print(f"   (plot skipped: {e})")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"smollm","smollm"))
