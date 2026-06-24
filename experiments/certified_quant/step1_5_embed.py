#!/usr/bin/env python3
"""Step 1.5: certify the EMBED / unembed read-out via the frame-quant bound.
The per-block write proxy (step1_allocate.py) cannot certify `embed` (272 MB, 43% of the bundle):
quantizing it shifts EVERY logit via the tied U_v. The frame-quant certificate handles it directly:
quantizing row U_v -> Ut_v perturbs logit v by  ΔL(v) = <r, Ut_v - U_v>  (<= ||r||*||ΔU_v||).
Decode preserved at position x iff  2*max_v |ΔL(v)| < margin(x)  (PIC_Quant.quant_decode_preserved).

Inputs: a --source-dump (r = Σ_b d̃_b, cands, margin) + the bundle .bin/.json (raw f16 embed rows).
Quantizes the candidate embed rows to int8 (per-row) and int4 (group-32) exactly as fieldrun's convert,
and measures the certified embed bit-width.  Usage: step1_5_embed.py source.jsonl bundle_stem"""
import json, sys, numpy as np

def q_int8_row(U):                       # per-row symmetric int8 (fieldrun put_i8)
    s = np.abs(U).max(axis=1, keepdims=True)/127.0; s[s==0]=1
    return np.round(U/s)*s
def q_int4_group(U, g=32):               # group-wise symmetric int4 [-7,7] (fieldrun put_i4, I4_GROUP=32)
    out=np.empty_like(U)
    for j in range(0, U.shape[1], g):
        blk=U[:, j:j+g]; s=np.abs(blk).max(axis=1, keepdims=True)/7.0; s[s==0]=1
        out[:, j:j+g]=np.round(blk/s)*s
    return out

def main():
    src, stem = sys.argv[1], sys.argv[2]
    man=json.load(open(stem+".fieldrun.json")); emb=[a for a in man["arrays"] if a["name"]=="embed"][0]
    assert emb["dtype"]=="f16"; off=emb["offset"]; dim=emb["shape"][1]
    recs=[json.loads(l) for l in open(src) if l.strip()]
    toks=sorted({t for r in recs for t in r["cands"]})
    # read the needed embed rows (f16) straight from the .bin
    U={}
    with open(stem+".fieldrun.bin","rb") as f:
        for t in toks:
            f.seek(off + t*dim*2); U[t]=np.frombuffer(f.read(dim*2), dtype=np.float16).astype(np.float64)
    Um=np.stack([U[t] for t in toks]); idx={t:i for i,t in enumerate(toks)}
    dU8 = q_int8_row(Um)-Um; dU4 = q_int4_group(Um)-Um          # quant error per row, both bit-widths

    # per position: r = sum_b d_b; ΔL(v)=<r,ΔU_v> over cands; certified iff 2*max|ΔL| < margin
    ok8=ok4=0; margins=[]; worst8_pos=[]; worst4_pos=[]
    if "--slim" in sys.argv:                                         # emit r=Σ_b d̃_b + cands + margin (committable, ~250KB)
        sl=sys.argv[sys.argv.index("--slim")+1]
        with open(sl,"w") as g:
            for r in recs:
                rv=np.sum(np.array(r["d"],dtype=np.float64),axis=0)
                g.write(json.dumps({"pos":r["pos"],"margin":r["margin"],"cands":r["cands"],
                                    "r":[round(float(x),5) for x in rv]})+"\n")
        print(f"  wrote slim calib -> {sl}")
    for r in recs:
        rv=np.array(r["r"],dtype=np.float64) if "r" in r else np.sum(np.array(r["d"],dtype=np.float64),axis=0)  # (dim,)
        ci=[idx[t] for t in r["cands"]]
        dl8=dU8[ci]@rv; dl4=dU4[ci]@rv                              # (ncand,)
        w8=np.abs(dl8).max(); w4=np.abs(dl4).max(); mg=r["margin"]; margins.append(mg)
        worst8_pos.append((mg, w8)); worst4_pos.append((mg, w4))
        ok8 += 2*w8 < mg; ok4 += 2*w4 < mg
    N=len(recs)
    print(f"=== embed frame-quant certificate (Qwen2.5-0.5B, N={N} positions, dim={dim}) ===")
    print(f"  ADAPTIVE (per-position) certified to drop embed f16 -> :")
    print(f"     int8 : {ok8}/{N} = {100*ok8/N:.0f}%      int4 : {ok4}/{N} = {100*ok4/N:.0f}%")
    # STATIC: a single embed dtype valid for all kept positions (drop residue% smallest-margin)
    def static_ok(worst, resid):
        kept=sorted(worst)[int(resid*len(worst)):]               # drop smallest-margin
        return all(2*w < mg for mg,w in kept)
    print(f"  STATIC (one embed dtype for all kept positions), residue sweep:")
    for resid in (0.0, 0.05, 0.10, 0.20):
        s8=static_ok(worst8_pos, resid); s4=static_ok(worst4_pos, resid)
        print(f"     residue {int(resid*100):>2}%:  int8 {'OK' if s8 else 'no':>3}   int4 {'OK' if s4 else 'no':>3}")
    # predicted bundle MB for embed at f16 / int8 / int4
    n=emb['shape'][0]*dim
    print(f"  embed bytes: f16 {n*2/1e6:.0f} MB  int8 {n*1.0/1e6:.0f} MB  int4 {n*0.5/1e6:.0f} MB  (of ~630 MB bundle)")
    print(f"  -> certified embed int8 saves ~{n*1.0/1e6:.0f} MB; int4 saves ~{n*1.5/1e6:.0f} MB (if certified)")

    # ALL-VOCAB worst-case (fully rigorous, no cand-set assumption): 2*max_x||r|| * max_v||ΔU_v|| < min margin
    if "--allvocab" in sys.argv:
        rr=lambda r: np.array(r["r"]) if "r" in r else np.sum(np.array(r["d"]),axis=0)
        V,D=emb["shape"]; rho=max(np.linalg.norm(rr(r)) for r in recs)
        mgmin=min(margins); mgmin5=sorted(margins)[int(0.05*len(margins))]
        with open(stem+".fieldrun.bin","rb") as f:
            f.seek(off); A=np.frombuffer(f.read(V*D*2),dtype=np.float16).astype(np.float32).reshape(V,D)
        for name,q in (("int8",q_int8_row),("int4",lambda x:q_int4_group(x))):
            e=np.linalg.norm(q(A.astype(np.float64))-A,axis=1).max()
            print(f"  ALL-VOCAB worst-case (Cauchy-Schwarz) {name}: max||ΔU_v||={e:.3f}  ρ_max={rho:.2f}  2ρε={2*rho*e:.3f}  "
                  f"vs min-margin {mgmin:.3f} ({'OK' if 2*rho*e<mgmin else 'no'})  [loose: TurboQuant √d gap]")

    # EXACT full-vocab decode check (ground truth, no cand-set / no Cauchy-Schwarz): argmax(Aq·r) vs argmax(A·r)
    if "--exact" in sys.argv:
        V,D=emb["shape"]
        with open(stem+".fieldrun.bin","rb") as f:
            f.seek(off); A=np.frombuffer(f.read(V*D*2),dtype=np.float16).astype(np.float32).reshape(V,D)
        Aq8=q_int8_row(A.astype(np.float64)).astype(np.float32); Aq4=q_int4_group(A.astype(np.float64)).astype(np.float32)
        f8=f4=0; mg_flip8=[]
        for r in recs:
            rv=(np.array(r["r"],dtype=np.float32) if "r" in r else np.sum(np.array(r["d"],dtype=np.float32),axis=0))
            t=int(np.argmax(A@rv))                                    # f16 decode (== model pred; recon=1.00)
            if int(np.argmax(Aq8@rv))!=t: f8+=1; mg_flip8.append(r["margin"])
            if int(np.argmax(Aq4@rv))!=t: f4+=1
        print(f"  EXACT full-vocab decode flips vs f16-embed:  int8 {f8}/{N} ({100*f8/N:.0f}%)   int4 {f4}/{N} ({100*f4/N:.0f}%)")
        if mg_flip8: print(f"     int8 flips occur at margins: {sorted(round(m,3) for m in mg_flip8)}  (forge-tax tail)")

if __name__=="__main__": main()
