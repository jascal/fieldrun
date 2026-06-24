#!/usr/bin/env python3
"""Step 0 probe: certified-prune ore on real fieldrun --pil-dump data.
Computes per-position (adaptive), static (corpus-intersection), and cross-corpus
certified prune ratios from contrib[block][cand] + margins. No engine change."""
import json, sys, statistics as st

def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in recs:
        C = r["contrib"]                      # nb x K
        nb, K = len(C), len(C[0])
        L = [sum(C[b][k] for b in range(nb)) for k in range(K)]   # reconstructed logit per cand
        w = max(range(K), key=lambda k: L[k])                      # within-cands winner
        srt = sorted(L, reverse=True)
        m = srt[0] - srt[1]                                        # within-cands decode margin
        beta = [max(abs(C[b][k]) for k in range(K)) for b in range(nb)]   # per-block cost
        out.append(dict(C=C, nb=nb, K=K, w=w, m=m, beta=beta, dumped_margin=r["margin"]))
    return out

def worst_signed(C, P, K):
    return max(abs(sum(C[b][k] for b in P)) for k in range(K)) if P else 0.0

def greedy_budget(beta, m):
    order = sorted(range(len(beta)), key=lambda b: beta[b])
    P, run = [], 0.0
    for b in order:
        if 2*(run+beta[b]) < m: P.append(b); run += beta[b]
        else: break
    return P

def greedy_signed(C, beta, m, K):
    order = sorted(range(len(beta)), key=lambda b: beta[b]); P=[]
    for b in order:
        if 2*worst_signed(C, P+[b], K) < m: P.append(b)
    return P

def flips(rec, P):
    """does dropping block-set P change the within-cands argmax?"""
    C, K, nb = rec["C"], rec["K"], rec["nb"]
    keep = [b for b in range(nb) if b not in set(P)]
    Lp = [sum(C[b][k] for b in keep) for k in range(K)]
    return max(range(K), key=lambda k: Lp[k]) != rec["w"]

def pct(xs, p): xs=sorted(xs); i=min(len(xs)-1, max(0,int(round(p/100*(len(xs)-1))))); return xs[i]

def report(name, D):
    nb = D[0]["nb"]
    # recon sanity: within-cands winner should be cand[0] (pred); margins close to dumped
    rb = [len(greedy_budget(d["beta"], d["m"]))/nb for d in D]
    rs = [len(greedy_signed(d["C"], d["beta"], d["m"], d["K"]))/nb for d in D]
    print(f"\n=== {name}  (N={len(D)} positions, nb={nb} blocks) ===")
    print(f"  margin (within-cands): med {st.median(d['m'] for d in D):.3f}"
          f"  min {min(d['m'] for d in D):.3f}  max {max(d['m'] for d in D):.3f}")
    print(f"  ADAPTIVE prune ratio (budget β):  med {st.median(rb):.2f}  p10 {pct(rb,10):.2f}  p90 {pct(rb,90):.2f}  mean {st.mean(rb):.2f}")
    print(f"  ADAPTIVE prune ratio (signed):    med {st.median(rs):.2f}  p10 {pct(rs,10):.2f}  p90 {pct(rs,90):.2f}  mean {st.mean(rs):.2f}")
    # sanity: certified per-position sets never flip the decode
    bad = sum(flips(d, greedy_budget(d["beta"], d["m"])) for d in D)
    print(f"  sanity: per-position certified sets that flip the decode: {bad}/{len(D)} (must be 0)")
    # structure of the adaptive signed-droppable set: early blocks (not compute-skippable) vs late (early-exit)?
    freq=[0]*nb
    for d in D:
        for b in greedy_signed(d["C"], d["beta"], d["m"], d["K"]): freq[b]+=1
    early=sum(freq[:nb//2]); late=sum(freq[nb//2:]); tot=max(1,early+late)
    print(f"  signed-droppable structure: early-half {early} drops, late-half {late} drops"
          f"  -> late share {late/tot:.2f} (high=early-exit/compute-skippable, low=decode-attribution only)")
    # STATIC (corpus-intersection) with residue sweep
    print(f"  STATIC prune ratio vs residue dropped (smallest-margin positions excluded):")
    Dm = sorted(D, key=lambda d: d["m"])
    for q in (0.0, 0.05, 0.10, 0.20, 0.40):
        kept = Dm[int(q*len(Dm)):]
        m_stat = min(d["m"] for d in kept)
        beta_g = [max(d["beta"][b] for d in kept) for b in range(nb)]
        P = greedy_budget(beta_g, m_stat)
        # verify on kept set
        fl = sum(flips(d, P) for d in kept)
        # structure: are dropped blocks late-contiguous? mean block index dropped vs all
        mi = (st.mean(P)/nb) if P else float('nan')
        print(f"     residue {int(q*100):>2}%:  static ratio {len(P)/nb:.2f}  ({len(P)}/{nb} blocks)"
              f"  m_floor {m_stat:.3f}  flips-on-kept {fl}  mean-dropped-block-frac {mi:.2f}")
    return rb, rs

def main():
    sci = load(sys.argv[1]); code = load(sys.argv[2])
    report("SCIENCE prose (Qwen2.5-0.5B)", sci)
    report("CODE (Qwen2.5-0.5B)", code)
    # control: unchecked heuristic prune at a fixed fraction -> decode-flip rate
    nb = sci[0]["nb"]
    print("\n=== CONTROL: unchecked magnitude prune (smallest-β, NO margin gate), decode-flip rate ===")
    for frac in (0.3, 0.5, 0.7):
        cnt = int(frac*nb)
        for nm, D in (("science", sci), ("code", code)):
            fr = st.mean(flips(d, sorted(range(nb), key=lambda b: d["beta"][b])[:cnt]) for d in D)
            print(f"  drop {int(frac*100)}% ({cnt}/{nb}) blocks  {nm:8s}: flip rate {fr:.2f}")
    # cross-corpus: static set certified on science, evaluated on code (and vice versa)
    print("\n=== CROSS-CORPUS: static set certified on A, decode-flip rate on B (corpus-relativity) ===")
    def static_set(D, q=0.10):
        Dm = sorted(D, key=lambda d: d["m"]); kept = Dm[int(q*len(Dm)):]
        m_stat = min(d["m"] for d in kept); beta_g=[max(d["beta"][b] for d in kept) for b in range(D[0]["nb"])]
        return greedy_budget(beta_g, m_stat)
    Psci, Pcode = static_set(sci), static_set(code)
    print(f"  P_science ({len(Psci)} blocks) on CODE  : flip rate {st.mean(flips(d,Psci) for d in code):.2f}")
    print(f"  P_code    ({len(Pcode)} blocks) on SCIENCE: flip rate {st.mean(flips(d,Pcode) for d in sci):.2f}")

if __name__ == "__main__": main()
