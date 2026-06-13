#!/usr/bin/env python3
"""Two-knob PR-core decode head (PROVABLE_OPT §7) — the in-repo realization of the LO1 lever.

Factor the unembedding readout into the readout-aligned rank-r decision basis S:
    logit_v = ⟨x_normed, U_v⟩  ≈  ⟨S·x_normed, S·U_v⟩      (rank r: vocab×r + r×d, vs vocab×d)
Two operating points (knobs) on the SAME basis: knob-1 r≈PR (smallest core), knob-2 r≈span90 (coverage).
A MARGIN GATE routes thin-margin (forge-tax) decisions to the full readout, so the hybrid is decode-EXACT
at reduced AVERAGE cost. Reports the operating table: rank -> unembed size, compression, decode preservation,
and the gated hybrid (routed-to-core fraction, overall preservation, average compression).
"""
import os, sys
import numpy as np
import bundle_io as bio
from lo1_circuit import forward_capture

def main(stem, N=900):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64)
    d=int(cfg[4]); eps=float(cfg_f[1]); V=int(cfg[6]); rng=np.random.default_rng(7)
    decs=[]
    for _ in range(N):
        ids=[int(t) for t in rng.integers(0,V,size=int(rng.integers(6,14)))]
        lg,xf,*_=forward_capture(W,cfg,cfg_f,ids); o=np.argsort(lg)[::-1]
        decs.append((int(o[0]), float(lg[o[0]]-lg[o[1]]), o[:9], xf.astype(np.float64)))  # RAW residual (rms is a scalar)
    tr=decs[:N//2]; te=decs[N//2:]
    # readout-aligned basis S from top-competitor diffs (decode-optimal), fit on TRAIN
    rows=[(lambda df: df/(np.linalg.norm(df)+1e-30))(gain*(U[p]-U[v])) for p,_,o,_ in tr for v in o[1:]]
    _,_,Vt=np.linalg.svd(np.array(rows),full_matrices=False)
    A=Vt@(gain*U).T                                                    # [maxr, vocab] = S·(gain⊙U)ᵀ — gain stays on U
    def core_logits(x,r): return (Vt[:r]@x) @ A[:r]                    # logit_v ∝ ⟨S·x_f, S·(gain⊙U_v)⟩ (rms scalar drops out)
    full=vocab_d=V*d
    print(f"== two-knob PR-core decode head ({os.path.basename(stem)}; {len(tr)} cal / {len(te)} test) ==")
    print(f"   full unembedding: vocab×d = {V}×{d} = {full/1e6:.1f}M.  decode reference = the model's argmax.")
    print(f"   {'knob':<12}{'rank':>5}{'unembed(vocab×r)':>18}{'compression':>13}{'decode kept':>13}")
    preserve={}
    for lbl,r in [("PR-core",92),("span90",128),("wide",256)]:
        keep=sum(int(np.argmax(core_logits(x,r)))==p for p,_,_,x in te)/len(te); preserve[r]=keep
        sz=V*r+r*d
        print(f"   {lbl:<12}{r:>5}{sz/1e6:>16.1f}M{full/sz:>12.1f}×{100*keep:>12.0f}%")
    # margin-gated hybrid at r=PR: route thin-CORE-margin decisions to the full readout (decode-exact there)
    r=92
    print(f"\n   margin-gated hybrid (core r={r} + full fallback when core-margin < τ):")
    print(f"   {'τ':>6}{'→core %':>9}{'overall kept':>14}{'avg unembed':>13}{'avg compression':>16}")
    for tau in [0.0,0.5,1.0,2.0]:
        routed_core=ok=0; cost=0.0
        for p,_,_,x in te:
            cl=core_logits(x,r); s=np.sort(cl)[::-1]; cm=s[0]-s[1]
            if cm>=tau:                                                # accept core
                routed_core+=1; ok+= int(np.argmax(cl)==p); cost+=V*r
            else:                                                       # fall back to full (exact == p)
                ok+=1; cost+=V*d
        nte=len(te)
        print(f"   {tau:>6.1f}{100*routed_core/nte:>8.0f}%{100*ok/nte:>13.0f}%{cost/nte/1e6:>11.1f}M{full/(cost/nte):>15.1f}×")
    print("   reading: pick τ where overall-kept is ~100% and →core is high => decode-exact at avg compression.")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),"smollm","smollm"))
