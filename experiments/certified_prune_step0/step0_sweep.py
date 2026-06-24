#!/usr/bin/env python3
"""Step 0-i: certified-prune ladder sweep across models. Prints a compact comparison table.
Usage: step0_sweep.py LABEL1=dump1.jsonl LABEL2=dump2.jsonl ..."""
import json, sys, statistics as st

def load(path):
    out=[]
    for l in open(path):
        if not l.strip(): continue
        r=json.loads(l); C=r["contrib"]; nb=len(C); K=len(C[0])
        L=[sum(C[b][k] for b in range(nb)) for k in range(K)]
        w=max(range(K), key=lambda k:L[k]); srt=sorted(L, reverse=True)
        out.append(dict(C=C, nb=nb, K=K, w=w, m=srt[0]-srt[1],
                        beta=[max(abs(C[b][k]) for k in range(K)) for b in range(nb)]))
    return out

def ws(C,P,K): return max(abs(sum(C[b][k] for b in P)) for k in range(K)) if P else 0.0
def gbudget(beta,m):
    P,run=[],0.0
    for b in sorted(range(len(beta)),key=lambda b:beta[b]):
        if 2*(run+beta[b])<m: P.append(b); run+=beta[b]
        else: break
    return P
def gsigned(C,beta,m,K):
    P=[]
    for b in sorted(range(len(beta)),key=lambda b:beta[b]):
        if 2*ws(C,P+[b],K)<m: P.append(b)
    return P
def flips(d,P):
    keep=[b for b in range(d["nb"]) if b not in set(P)]
    Lp=[sum(d["C"][b][k] for b in keep) for k in range(d["K"])]
    return max(range(d["K"]),key=lambda k:Lp[k])!=d["w"]
def med(xs): return st.median(xs)
def p90(xs): xs=sorted(xs); return xs[min(len(xs)-1,int(round(0.9*(len(xs)-1))))]

def static_ratio(D, q):
    nb=D[0]["nb"]; Dm=sorted(D,key=lambda d:d["m"]); kept=Dm[int(q*len(Dm)):]
    m=min(d["m"] for d in kept); bg=[max(d["beta"][b] for d in kept) for b in range(nb)]
    return len(gbudget(bg,m))/nb

def row(label, D):
    nb=D[0]["nb"]; N=len(D)
    rb=[len(gbudget(d["beta"],d["m"]))/nb for d in D]
    rs=[len(gsigned(d["C"],d["beta"],d["m"],d["K"]))/nb for d in D]
    # heuristic 50% flip rate
    h=st.mean(flips(d, sorted(range(nb),key=lambda b:d["beta"][b])[:nb//2]) for d in D)
    # late share of signed-droppable
    freq=[0]*nb
    for d in D:
        for b in gsigned(d["C"],d["beta"],d["m"],d["K"]): freq[b]+=1
    early=sum(freq[:nb//2]); late=sum(freq[nb//2:]); tot=max(1,early+late)
    return (f"| {label} | {nb} | {N} | {med(d['m'] for d in D):.2f} | {med(rb):.2f} | "
            f"{med(rs):.2f} ({p90(rs):.2f}) | {static_ratio(D,0.10):.2f} | {static_ratio(D,0.40):.2f} | "
            f"{h:.2f} | {late/tot:.2f} |")

def main():
    print("| model (corpus) | nb | N | margin med | adapt budget med | adapt signed med (p90) | "
          "static@10%res | static@40%res | heur-50% flip | signed late-share |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for a in sys.argv[1:]:
        label, path = a.split("=",1)
        try: D=load(path)
        except Exception as e: print(f"| {label} | (load failed: {e}) |"); continue
        if D: print(row(label, D))

if __name__=="__main__": main()
