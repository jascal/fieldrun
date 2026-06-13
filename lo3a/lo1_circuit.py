#!/usr/bin/env python3
"""LO1 — the DECISIVE obstruction test on the CIRCUIT-coupling axis (where lo1_matrix.py localized the forge tax).

The forge tax is a dense sum over ~PR effective circuits (heads + neurons). Grok's matrix/operator-semiring
valuation escapes it IFF those circuits' write vectors are LOW-RANK (factor through few dims); the obstruction
is whether the required width just tracks the PR. So, per decision:
  * PR_dla   = participation ratio of the per-circuit direct-logit-attribution magnitudes (the SCALAR forge tax).
  * effrank  = effective rank of the circuits' WRITE GEOMETRY (the operator-valuation width).
  effrank ≪ PR_dla  => circuits write into few dims => matrix valuation ESCAPES (forge tax = scalar-lens artifact).
  effrank ≈ PR_dla  => writes span as many dims as circuits => OBSTRUCTION holds (intrinsic / Δ_repr on this axis).
Stratified by margin (forge-tax thin vs retrievable thick).  SmolLM-135M (rope).
"""
import os
import numpy as np
import bundle_io as bio

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = os.path.join(HERE, "smollm", "smollm")

def pr_eig(vals):                      # participation ratio of a non-negative spectrum: (Σλ)²/Σλ²
    v = np.clip(np.asarray(vals, float), 0, None); s = v.sum()
    return float(s * s / (np.square(v).sum() + 1e-30))

def _rms(x, w, eps):
    ms = (x ** 2).mean(); return (x * (1.0/np.sqrt(ms+eps)) * w), float(1.0/np.sqrt(ms+eps))

def forward_capture(W, cfg, cfg_f, ids):
    """forward; return logits, and for the LAST position the per-layer attn_out and post-SwiGLU hid."""
    nl,H,NKV,HD,D,FFN,V,TIED = [int(c) for c in cfg]; theta,eps = map(float,cfg_f)
    HALF,REP = HD//2, H//NKV
    inv=(1.0/(theta**(2.0*np.arange(HALF,dtype=np.float32)/HD))).astype(np.float32)
    ids=list(ids); seq=len(ids)
    ang=np.arange(seq,dtype=np.float32)[:,None]*inv[None,:]; COS=np.cos(ang)[:,None,:].astype(np.float32); SIN=np.sin(ang)[:,None,:].astype(np.float32)
    causal=np.triu(np.ones((seq,seq),bool),1)
    rope=lambda x,nh:(lambda xr:(np.concatenate([xr[...,:HALF]*COS-xr[...,HALF:]*SIN, xr[...,HALF:]*COS+xr[...,:HALF]*SIN],-1)).reshape(seq,nh*HD).astype(np.float32))(x.reshape(seq,nh,HD))
    silu=lambda x:(x/(1.0+np.exp(-x))).astype(np.float32)
    x=W["embed"][ids].astype(np.float32); attn_last=[]; hid_last=[]
    for l in range(nl):
        p=f"l{l}."; a=bio._rmsnorm(x,W[p+"in_ln"],eps)
        q=rope((a@W[p+"self_attn.q_proj"]).astype(np.float32),H); k=rope((a@W[p+"self_attn.k_proj"]).astype(np.float32),NKV)
        v=(a@W[p+"self_attn.v_proj"]).astype(np.float32)
        ao=np.zeros((seq,H*HD),np.float32)
        for h in range(H):
            kv=h//REP; sc=(q[:,h*HD:(h+1)*HD]@k[:,kv*HD:(kv+1)*HD].T)/np.float32(np.sqrt(HD)); sc[causal]=-1e30
            sc=np.exp(sc-sc.max(1,keepdims=True)); sc/=sc.sum(1,keepdims=True)
            ao[:,h*HD:(h+1)*HD]=(sc@v[:,kv*HD:(kv+1)*HD]).astype(np.float32)
        attn_last.append(ao[-1].copy())
        x=(x+ao@W[p+"self_attn.o_proj"]).astype(np.float32)
        a2=bio._rmsnorm(x,W[p+"post_ln"],eps); hid=(silu(a2@W[p+"mlp.gate_proj"])*(a2@W[p+"mlp.up_proj"])).astype(np.float32)
        hid_last.append(hid[-1].copy()); x=(x+hid@W[p+"mlp.down_proj"]).astype(np.float32)
    xf=x[-1]; logits=(bio._rmsnorm(x,W["norm"],eps)[-1]@ (W["embed"] if TIED else W["lm_head"]).T).astype(np.float32)
    return logits, xf, attn_last, hid_last

def analyze(W, cfg, cfg_f, ids, topN=256):
    nl,H,NKV,HD,D,FFN,V,TIED = [int(c) for c in cfg]; eps=float(cfg_f[1])
    U = W["embed"] if TIED else W["lm_head"]; gain=W["norm"]
    lg,xf,attn_last,hid_last = forward_capture(W,cfg,cfg_f,ids)
    order=np.argsort(lg)[::-1]; pred=int(order[0]); margin=float(lg[order[0]]-lg[order[1]])
    _,invn=_rms(xf,gain,eps); g_up=(gain*U[pred]).astype(np.float32)   # DLA_c = invn * <w_c, gain⊙U_pred>
    writes=[]; dlas=[]
    for l in range(nl):
        op=W[f"l{l}.self_attn.o_proj"]                                  # [H*hd, d]
        for h in range(H):
            w=attn_last[l][h*HD:(h+1)*HD]@op[h*HD:(h+1)*HD,:]           # head write vector [d]
            writes.append(w*gain); dlas.append(invn*float(w@g_up))
        dp=W[f"l{l}.mlp.down_proj"]                                     # [ffn, d]
        nd=invn*hid_last[l]*(dp@g_up)                                   # all neuron DLAs (vectorized) [ffn]
        # keep only the strongest neurons in this layer to bound cost; the tail is negligible for PR
        keep=np.argsort(np.abs(nd))[::-1][:topN]
        for j in keep:
            writes.append((hid_last[l][j]*dp[j])*gain); dlas.append(float(nd[j]))
    dlas=np.array(dlas); Wm=np.array(writes,dtype=np.float64)           # [n_components, d]
    # take the top-N components by |DLA| (covers the bulk; N ≫ PR)
    sel=np.argsort(np.abs(dlas))[::-1][:topN]
    pr_dla=pr_eig(np.abs(dlas[sel]))                                    # scalar circuit PR (the forge tax)
    A=Wm[sel]                                                           # write geometry of the significant circuits
    Usv,sv,Vt=np.linalg.svd(A,full_matrices=False)                     # V rows = the write subspace directions [d]
    effr=pr_eig(sv**2)                                                  # ENERGY rank = operator-valuation width (bulk)
    # DECODE-FAITHFUL rank: minimal r s.t. projecting the residual onto the top-r write subspace keeps the argmax.
    xfd=xf.astype(np.float64); pred_dir=U[pred].astype(np.float64)
    faithful_r=Vt.shape[0]
    for r in [1,2,4,8,16,32,64,128,256]:
        if r>Vt.shape[0]: faithful_r=Vt.shape[0]; break
        Vr=Vt[:r]; xp=Vr.T@(Vr@xfd)                                    # residual projected onto the rank-r write subspace
        xpn,_=_rms(xp.astype(np.float32),gain,eps)
        if int(np.argmax(xpn@U.T))==pred: faithful_r=r; break
    return margin, pr_dla, effr, faithful_r

if __name__=="__main__":
    man,W=bio.read_bundle(STEM); cfg,cfg_f=man["config"],man["config_f"]
    rng=np.random.default_rng(3); V=int(cfg[6]); recs=[]
    print("== LO1 circuit-axis obstruction test (SmolLM-135M): effrank(write geometry) vs PR(DLA) ==")
    for _ in range(120):
        ids=[int(t) for t in rng.integers(0,V,size=int(rng.integers(6,14)))]
        recs.append(analyze(W,cfg,cfg_f,ids))
    recs.sort(key=lambda r:r[0]); n=len(recs); t=n//3
    strata=[("forge-tax  (thin margin)",recs[:t]),("middle",recs[t:2*t]),("retrievable (thick margin)",recs[2*t:])]
    print(f"   {n} decisions. PR_dla = scalar circuit PR; effrank = write-ENERGY rank; faithful_r = minimal write")
    print(f"   subspace that preserves the ARGMAX (the decision-faithful operator-valuation width).")
    print(f"   {'stratum':<28}{'n':>4}{'margin':>9}{'PR_dla':>9}{'effrank':>9}{'eff/PR':>8}{'faithful_r':>12}")
    for lbl,g in strata:
        mm=np.mean([r[0] for r in g]); pr=np.mean([r[1] for r in g]); er=np.mean([r[2] for r in g]); fr=np.mean([r[3] for r in g])
        print(f"   {lbl:<28}{len(g):>4}{mm:>9.3f}{pr:>9.1f}{er:>9.2f}{er/pr:>8.3f}{fr:>12.1f}")
    allpr=np.mean([r[1] for r in recs]); aller=np.mean([r[2] for r in recs]); allfr=np.mean([r[3] for r in recs])
    print(f"\n   OVERALL: PR_dla≈{allpr:.0f}  write-energy effrank≈{aller:.1f} (ratio {aller/allpr:.2f})  decode-faithful_r≈{allfr:.1f}")
    print("   energy effrank ≪ PR        => the write GEOMETRY is low-rank (operator valuation compresses the bulk).")
    print("   faithful_r small & flat    => the DECISION is low-rank too => matrix valuation ESCAPES (scalar-lens artifact).")
    print("   faithful_r rises as margin↓ => the thin-margin decision lives in the tail => OBSTRUCTION, margin-bounded.")
