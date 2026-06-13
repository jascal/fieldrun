#!/usr/bin/env python3
"""PR-core as a shortlist proposer — the speculative-decoding idea, corrected to what actually pays off.

Why not multi-token speculative decoding: the PR-core "draft" shares the WHOLE transformer stack with the
target (it only cheapens the final vocab×d projection). Drafting token k+1 needs the residual at k+1, i.e.
a full forward pass on the drafted token k — so there is NO cheap autoregressive draft and no multi-token
speedup from PR-core alone. The surviving, real win is the SINGLE-POSITION shortlist:

  draft:  q_v = ⟨P_r x, gain⊙U_v⟩  over all v   (cheap: r·vocab)         → take top-K candidates
  verify: full-score ONLY those K rows: p_v = ⟨x, gain⊙U_v⟩  (K·d)       → argmax of the K
  EXACT iff the true argmax a* ∈ shortlist  (the shortlist leader's full logit is then the global max).

So the deciding number is TOP-K RECALL (is a* in the PR-core top-K?), not the top-1 we already had (67%).
Two operating modes, both reported:
  (A) trust-recall  : never check completeness; quality = recall_K (silent misses = 1-recall), cost = cheap.
  (B) certified     : prove no out-of-shortlist token can overtake the leader via the residual bound
                      |p_v-q_v| ≤ ‖(I-P_r)x‖·‖gain⊙U_v‖; on certify → cheap+exact, else full fallback → exact.
                      Coverage = certifiable fraction (expected low: ‖(I-P_r)x‖/‖x‖≈0.99 ⇒ loose bound = τ*).
SmolLM-135M. Reuses lo1_circuit.forward_capture (the verified rope forward).
"""
import os, sys
import numpy as np
import bundle_io as bio
from lo1_circuit import forward_capture

HERE = os.path.dirname(os.path.abspath(__file__))
def _norm(v): return v / (np.linalg.norm(v) + 1e-30)

def main(stem, N=900):
    man, W = bio.read_bundle(stem); cfg, cfg_f = man["config"], man["config_f"]
    U = (W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain = W["norm"].astype(np.float64)
    d = int(cfg[4]); V = int(cfg[6]); rng = np.random.default_rng(7)
    gU = gain * U; gUn = np.linalg.norm(gU, axis=1)                       # ‖gain⊙U_v‖ per token (for the bound)
    decs = []
    for _ in range(N):
        ids = [int(t) for t in rng.integers(0, V, size=int(rng.integers(6, 14)))]
        lg, xf, *_ = forward_capture(W, cfg, cfg_f, ids); o = np.argsort(lg)[::-1]
        decs.append((int(o[0]), o[1:9].astype(int), xf.astype(np.float64), float(lg[o[0]]-lg[o[1]])))
    tr = decs[:N//2]; te = decs[N//2:]; nte = len(te)
    # readout-aligned basis (fit on TRAIN), nested over rank
    Vt = np.linalg.svd(np.array([_norm(gU[p]-gU[v]) for p,comp,_,_ in tr for v in comp]), full_matrices=False)[2]
    A = Vt @ gU.T                                                         # [maxr, vocab]
    full = V * d
    Ks = [1, 2, 4, 8, 16, 32, 64]; Rs = [92, 128, 256]
    Xte = np.array([x for _,_,x,_ in te])

    print(f"== PR-core shortlist decode (SmolLM-135M; {len(tr)} cal / {nte} test) ==")
    print(f"   full unembed vocab×d = {V}×{d} = {full/1e6:.1f}M floats.  a* = the model's full argmax.")
    print("   (A) trust-recall: quality = top-K recall (a*∈shortlist); exact-when-in-shortlist, else SILENT miss.")
    print(f"   {'rank r':>7}{'top1':>8}" + "".join(f"{'R@'+str(k):>8}" for k in Ks[1:]) + f"{'cost(unembed)':>15}{'compr':>7}")
    recall = {}
    Qr = {}
    for r in Rs:
        Q = (Vt[:r] @ Xte.T).T @ A[:r]; Qr[r] = Q                        # [nte, vocab] core logits at rank r
        row = []
        for k in Ks:
            topk = np.argpartition(-Q, kth=min(k, V-1)-1 if k>1 else 0, axis=1)[:, :k]
            hit = np.mean([te[i][0] in topk[i] for i in range(nte)])
            recall[(r, k)] = hit; row.append(hit)
        cost = r*V + (r+64)*d                                            # draft r·vocab + verify shortlist (K=64)·d + S·x
        print(f"   {r:>7}" + "".join(f"{100*x:>7.0f}%" for x in row) + f"{cost/1e6:>13.1f}M{full/cost:>6.1f}×")

    # ---- the deciding caveat: random prompts = worst case. Stratify recall by the model's OWN margin ----
    # confident (thick-margin) decodes dominate real text; thin-margin = the model is itself uncertain (τ*).
    mar = np.array([m for _,_,_,m in te]); idx = np.argsort(mar); t = nte // 3
    strata = [("thin  (model unsure)", idx[:t]), ("mid", idx[t:2*t]), ("thick (confident)", idx[2*t:])]
    print(f"\n   recall stratified by the model's OWN margin (r=92) — thick ≈ real-text confident decodes:")
    print(f"   {'stratum':<22}{'n':>4}{'med margin':>12}" + "".join(f"{'R@'+str(k):>8}" for k in [1,8,32,64]))
    Q92 = Qr[92]
    for lbl, sel in strata:
        cells = []
        for k in [1,8,32,64]:
            tk = np.argpartition(-Q92[sel], kth=min(k,V-1)-1 if k>1 else 0, axis=1)[:, :k]
            cells.append(np.mean([te[sel[j]][0] in tk[j] for j in range(len(sel))]))
        print(f"   {lbl:<22}{len(sel):>4}{np.median(mar[sel]):>12.2f}" + "".join(f"{100*c:>7.0f}%" for c in cells))

    # ---- reach the genuinely-high-margin regime (random prompts cap ~1): greedy self-rollout ----
    # the model's own confident continuations = the retrievable fragment (LE-T2). Does recall → 100% there?
    roll = []
    for s in range(40):
        ids = [int(t) for t in rng.integers(0, V, size=int(rng.integers(3, 7)))]
        for _ in range(40):
            lg, xf, *_ = forward_capture(W, cfg, cfg_f, ids); o = np.argsort(lg)[::-1]
            roll.append((int(o[0]), float(lg[o[0]]-lg[o[1]]), xf.astype(np.float64)))
            ids.append(int(o[0])); ids = ids[-64:]
    Xr = np.array([x for _,_,x in roll]); mr = np.array([m for _,m,_ in roll]); pr_ = [p for p,_,_ in roll]
    Qroll = (Vt[:92] @ Xr.T).T @ A[:92]
    print(f"\n   recall vs margin over {len(roll)} greedy-rollout decisions (r=92) — does the retrievable")
    print(f"   (high-margin) fragment become shortlist-cheap?  bands of the model's full margin:")
    print(f"   {'margin band':<14}{'n':>5}" + "".join(f"{'R@'+str(k):>8}" for k in [1,8,32]))
    bands = [("[0,0.5)",0,0.5),("[0.5,1)",0.5,1),("[1,2)",1,2),("[2,5)",2,5),("[5,15)",5,15),("[15,∞)",15,1e9)]
    for lbl,lo,hi in bands:
        sel = np.where((mr>=lo)&(mr<hi))[0]
        if len(sel)==0: print(f"   {lbl:<14}{0:>5}"); continue
        cells=[]
        for k in [1,8,32]:
            tk = np.argpartition(-Qroll[sel], kth=min(k,V-1)-1 if k>1 else 0, axis=1)[:, :k]
            cells.append(np.mean([pr_[sel[j]] in tk[j] for j in range(len(sel))]))
        print(f"   {lbl:<14}{len(sel):>5}" + "".join(f"{100*c:>7.0f}%" for c in cells))

    # (B) certified-exact coverage at r=92, K=32: can the residual bound rule out every out-of-shortlist token?
    r, K = 92, 32
    Pr = Vt[:r].T @ (Vt[:r] @ Xte.T)                                     # P_r x   [d, nte]
    resid = np.linalg.norm(Xte.T - Pr, axis=0)                          # ‖(I-P_r)x‖ per decision  [nte]
    ncert = 0
    for i in range(nte):
        x = Xte[i]; q = (Vt[:r] @ x) @ A[:r]                            # core logits
        sl = np.argpartition(-q, K-1)[:K]                               # shortlist
        pL = np.max(gU[sl] @ x)                                         # leader's FULL logit (exact)
        # upper bound on any out-of-shortlist full logit: q_v + ‖(I-P_r)x‖·‖gU_v‖
        ub = q + resid[i] * gUn; ub[sl] = -np.inf
        if np.max(ub) < pL: ncert += 1                                  # certified: leader is the global argmax
    cert = ncert / nte
    print(f"\n   (B) certified-exact shortlist (r={r}, K={K}): residual bound certifies completeness on "
          f"{100*cert:.0f}% of decisions")
    print(f"       ‖(I-P_r)x‖/‖x‖ median {np.median(resid/np.linalg.norm(Xte,axis=1)):.2f}  ⇒ bound loose ⇒ low cert "
          f"(the τ* floor: exact-and-cheap is NOT certifiable; trust-recall is the usable mode).")
    print(f"\n   takeaway: shortlist full-scoring lifts decode quality 67% (top-1) → {100*recall[(92,32)]:.0f}% "
          f"(top-32) at ~{full/(92*V+(92+32)*d):.0f}× fewer unembed FLOPs — a COMPUTE-mode win (needs full U in")
    print(f"   memory for the K-row verify); it does NOT shrink storage and is not certified-exact.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "smollm", "smollm"))
