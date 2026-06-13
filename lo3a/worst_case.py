#!/usr/bin/env python3
"""The theoretical worst-case 'conlang' for a low-rank readout — and why it isn't '64 cases'.

The cross-linguistic result showed recoverable rank tracks OUTPUT DIVERSITY = exp(H_output), the effective number
of distinct tokens the model emits, capped at d. So we can compute the worst case directly on the real readout
geometry (gU), without designing or training a conlang: construct synthetic output distributions over a vocabulary
of real token directions and measure recoverable rank as we vary (a) the frequency SKEW (Zipf exponent s: s=0
uniform = worst, s↑ = Zipfian = natural) and (b) the effective vocabulary m. The claim: recoverable rank ≈
min(exp(H_output), d). The worst case is a FLAT (anti-Zipf) distribution over ≥ d equiprobable, decision-distinct
word-forms — NOT merely many cases (which only hurt if usage stays uniform AND a fluent model emits the full
diversity). SmolLM-135M readout (d=576). Pure geometry, no forward pass.
"""
import os, sys
import numpy as np
import bundle_io as bio

HERE=os.path.dirname(os.path.abspath(__file__))
def _norm(v): return v/(np.linalg.norm(v)+1e-30)
def H(p): p=p[p>0]; return float(-(p*np.log2(p)).sum())          # entropy in bits

def measure(G, probs, d, N=1500, rng=None):
    """G[K,d] conlang word directions; probs[K] usage. Sample argmaxes ~ probs, x=G[a]; fit the readout-aligned
       basis from competitor diffs, return median normalized recoverable rank, R@32, and exp(H_output)."""
    K=len(G); rng=rng or np.random.default_rng(0)
    idx=rng.choice(K,size=N,p=probs)
    Gn=G                                                          # competitors scored over the conlang vocab
    # basis from top-competitor diffs (decode-optimal), fit on first half
    tr=idx[:N//2]; te=idx[N//2:]
    rows=[]
    for a in tr:
        sc=Gn@G[a]; comp=np.argsort(sc)[::-1]; comp=comp[comp!=a][:8]
        for v in comp: rows.append(_norm(G[a]-G[v]))
    Vt=np.linalg.svd(np.array(rows),full_matrices=False)[2]; A=Vt@Gn.T
    grid=[1,2,4,8,16,24,32,48,64,92,128,192,256,384,512,d]
    P=Vt@G[te].T; rr=np.full(len(te),d,float); done=np.zeros(len(te),bool)
    for r in grid:
        arg=np.argmax((P[:r].T)@A[:r],axis=1); hit=(arg==te)&~done; rr[hit]=r; done|=hit
    Q92=(P[:min(92,d)].T)@A[:min(92,d)]; tk=np.argpartition(-Q92,min(31,K-1),axis=1)[:,:32]
    R32=np.mean([te[j] in tk[j] for j in range(len(te))])
    return np.median(rr/d), R32, 2**H(probs)

def main(stem):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]; d=int(cfg[4]); V=int(cfg[6])
    gU=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64)*W["norm"].astype(np.float64)
    rng=np.random.default_rng(0); pool=rng.choice(V,size=2048,replace=False); G=gU[pool]   # the conlang's word directions
    print(f"== the theoretical worst-case conlang for a low-rank readout (SmolLM-135M; d={d}) ==")

    print(f"\n   (1) FREQUENCY SKEW sweep — fixed vocab K=2048, Zipf exponent s (s=0 uniform=worst, s↑=natural):")
    print(f"       {'s (skew)':>9}{'exp(H) eff-vocab':>18}{'med ρ/d':>10}{'R@32':>8}")
    for s in [0.0,0.5,0.8,1.0,1.3,1.7]:
        ranks=np.arange(1,2049,dtype=float); p=ranks**(-s); p/=p.sum()
        mr,r32,eh=measure(G,p,d,rng=np.random.default_rng(1))
        print(f"       {s:>9.1f}{eh:>18.0f}{mr:>10.2f}{100*r32:>7.0f}%")

    print(f"\n   (2) EFFECTIVE-VOCAB sweep — UNIFORM over m word-forms (the worst-case shape; m = 'cases'×roots):")
    print(f"       {'m forms':>9}{'med ρ/d':>10}{'recoverable rank':>18}{'R@32':>8}")
    for m in [16,32,64,128,256,384,512,768,1024]:
        p=np.zeros(2048); p[:m]=1.0/m
        mr,r32,eh=measure(G,p,d,rng=np.random.default_rng(2))
        print(f"       {m:>9}{mr:>10.2f}{mr*d:>18.0f}{100*r32:>7.0f}%")
    print(f"\n   recoverable rank ≈ min(exp(H_output), d): the tax is the EFFECTIVE output vocabulary, capped at d.")
    print(f"   Zipf (s≈1, natural) keeps eff-vocab ≪ raw vocab ⇒ bounded tax; a UNIFORM (s=0) language over ≥d")
    print(f"   forms saturates the tax (ρ/d→1, R@32→floor). The worst case is FLAT FREQUENCY, not many cases:")
    print(f"   64 cases hurt only if every form is used equally AND a fluent model emits the full diversity")
    print(f"   (an English-only model fed the conlang collapses to few outputs — artificially cheap, not worst).")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ms=[16,32,64,128,256,384,512,768,1024,1536]; rk=[measure(G,(lambda p:(p.__setitem__(slice(0,m),1/m),p)[1])(np.zeros(2048)),d,rng=np.random.default_rng(3))[0]*d for m in ms]
        fig,ax=plt.subplots(figsize=(6.6,4.2))
        ax.plot(ms,rk,"o-",color="#c33",lw=2,label="uniform conlang (worst-case shape)")
        ax.plot([0,d,2048],[0,d,d],"k--",lw=1,label="min(m, d)")
        ax.axhline(d,color="#999",ls=":",lw=1); ax.text(1300,d+15,f"d={d} (full tax)",fontsize=8,color="#666")
        ax.set_xlabel("effective output vocabulary  m  (= 'cases' × roots, uniformly used)")
        ax.set_ylabel("recoverable rank"); ax.set_title("worst-case conlang: tax = min(eff-vocab, d)")
        ax.legend(fontsize=8); fig.tight_layout(); out=os.path.join(HERE,"worst_case.png"); fig.savefig(out,dpi=110)
        print(f"\n   plot: {out}")
    except Exception as e: print(f"   (plot skipped: {e})")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"smollm","smollm"))
