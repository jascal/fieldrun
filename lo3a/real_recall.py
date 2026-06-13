#!/usr/bin/env python3
"""Grok's experiment, the clean form: PR-core top-K RECALL vs decision MARGIN on REAL in-distribution text.

The earlier synthetic regimes (random prompts, greedy self-rollout) both plateaued at ~80% recall, FLAT in
margin — but neither reaches a real tokenized corpus, and random prompts cap at margin ~1. Here we teacher-force
diverse real passages (English prose, encyclopedic, code, dialogue, technical) through the verified rope forward,
capturing EVERY position in one pass, and measure recall of the model's own argmax a* in the PR-core top-K,
binned across the full margin range that real text actually produces. The decisive question (Grok):
  does recall rise to 90%+ once margin is large (5–15) — i.e. is the high-margin RETRIEVABLE fragment (LE-T2)
  shortlist-cheap — or does it stay ~80% even at very high margin (heavy tail = structural, τ*)?
SmolLM-135M. No torch/tokenizers deps: bpe.py + an all-positions numpy forward.
"""
import os, sys, re
import numpy as np
import bundle_io as bio
from bpe import BPE

HERE = os.path.dirname(os.path.abspath(__file__))
def _norm(v): return v / (np.linalg.norm(v) + 1e-30)

def forward_all(W, cfg, cfg_f, ids):
    """Causal forward; return per-position RAW final residual [seq,d] and full logits [seq,vocab]."""
    nl,H,NKV,HD,D,FFN,V,TIED = [int(c) for c in cfg]; theta,eps = map(float,cfg_f)
    HALF,REP = HD//2, H//NKV
    inv=(1.0/(theta**(2.0*np.arange(HALF,dtype=np.float32)/HD))).astype(np.float32)
    ids=list(ids); seq=len(ids)
    ang=np.arange(seq,dtype=np.float32)[:,None]*inv[None,:]; COS=np.cos(ang)[:,None,:].astype(np.float32); SIN=np.sin(ang)[:,None,:].astype(np.float32)
    causal=np.triu(np.ones((seq,seq),bool),1)
    rope=lambda x,nh:(lambda xr:(np.concatenate([xr[...,:HALF]*COS-xr[...,HALF:]*SIN, xr[...,HALF:]*COS+xr[...,:HALF]*SIN],-1)).reshape(seq,nh*HD).astype(np.float32))(x.reshape(seq,nh,HD))
    silu=lambda x:(x/(1.0+np.exp(-x))).astype(np.float32)
    x=W["embed"][ids].astype(np.float32)
    for l in range(nl):
        p=f"l{l}."; a=bio._rmsnorm(x,W[p+"in_ln"],eps)
        q=rope((a@W[p+"self_attn.q_proj"]).astype(np.float32),H); k=rope((a@W[p+"self_attn.k_proj"]).astype(np.float32),NKV)
        v=(a@W[p+"self_attn.v_proj"]).astype(np.float32); ao=np.zeros((seq,H*HD),np.float32)
        for h in range(H):
            kv=h//REP; sc=(q[:,h*HD:(h+1)*HD]@k[:,kv*HD:(kv+1)*HD].T)/np.float32(np.sqrt(HD)); sc[causal]=-1e30
            sc=np.exp(sc-sc.max(1,keepdims=True)); sc/=sc.sum(1,keepdims=True)
            ao[:,h*HD:(h+1)*HD]=(sc@v[:,kv*HD:(kv+1)*HD]).astype(np.float32)
        x=(x+ao@W[p+"self_attn.o_proj"]).astype(np.float32)
        a2=bio._rmsnorm(x,W[p+"post_ln"],eps); hid=(silu(a2@W[p+"mlp.gate_proj"])*(a2@W[p+"mlp.up_proj"])).astype(np.float32)
        x=(x+hid@W[p+"mlp.down_proj"]).astype(np.float32)
    U = W["embed"] if TIED else W["lm_head"]
    logits=(bio._rmsnorm(x,W["norm"],eps)@U.T).astype(np.float32)         # [seq, vocab]
    return x.astype(np.float64), logits                                  # raw residual, full logits

PASSAGES = [
  "The history of natural language processing began in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" which proposed what is now called the Turing test as a criterion of intelligence.",
  "She walked into the room and noticed that everything had been moved. The chairs were stacked in one corner, the table was pushed against the wall, and the curtains had been taken down. \"What happened here?\" she asked, but no one answered.",
  "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
  "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy. During this process, carbon dioxide and water are converted into glucose and oxygen. The overall reaction can be summarized as six carbon dioxide plus six water yields glucose plus six oxygen.",
  "The meeting is scheduled for Tuesday at 3 PM. Please bring the quarterly report and the updated budget figures. We will discuss the new marketing strategy and review the performance of the last campaign. If you cannot attend, let me know as soon as possible.",
  "import numpy as np\nimport pandas as pd\n\ndata = pd.read_csv('input.csv')\ndata['total'] = data['price'] * data['quantity']\nresult = data.groupby('category')['total'].sum()\nprint(result.sort_values(ascending=False))",
  "To make a basic tomato sauce, start by heating two tablespoons of olive oil in a large pan over medium heat. Add two cloves of minced garlic and cook until fragrant, about thirty seconds. Then add a can of crushed tomatoes, a pinch of salt, and a teaspoon of dried basil.",
  "The theory of general relativity, published by Albert Einstein in 1915, describes gravity not as a force but as a curvature of spacetime caused by mass and energy. This was a radical departure from the Newtonian view, which had treated gravity as an instantaneous force acting at a distance.",
  "Quarterly revenue increased by 12 percent compared to the same period last year, driven primarily by strong sales in the cloud services division. Operating margin improved to 23 percent, and the company raised its full year guidance. Analysts had expected a more modest increase.",
  "Once upon a time, in a small village at the edge of a great forest, there lived an old woodcutter and his wife. They had no children, and they were very poor. Every day the woodcutter went into the forest to cut wood, and every evening he returned home tired and hungry.",
  "The mitochondrion is often described as the powerhouse of the cell. It generates most of the cell's supply of adenosine triphosphate, which is used as a source of chemical energy. Mitochondria are found in nearly all eukaryotic organisms and vary in number depending on the cell type.",
  "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        if not self.items:\n            raise IndexError('pop from empty stack')\n        return self.items.pop()",
  "The Roman Empire reached its greatest territorial extent under the emperor Trajan in the second century. At its height it controlled the entire Mediterranean basin, stretching from Britain in the north to Egypt in the south, and from Spain in the west to Mesopotamia in the east.",
  "Interest rates were left unchanged at the central bank's latest meeting, but officials signaled that further increases were likely if inflation did not continue to decline. The decision was widely anticipated by markets, and stocks closed slightly higher following the announcement.",
  "Water is a chemical compound consisting of two hydrogen atoms and one oxygen atom. At room temperature it is a clear, tasteless liquid. It is essential for all known forms of life and covers about seventy one percent of the Earth's surface, mostly in the form of oceans and seas.",
  "\"I don't think we should go that way,\" he said quietly, glancing at the map. \"The bridge was washed out last spring, and no one has repaired it since.\" She frowned and looked back the way they had come, wondering whether there was still time to find another route before dark.",
  "function fibonacci(n) {\n  if (n <= 1) return n;\n  let a = 0, b = 1;\n  for (let i = 2; i <= n; i++) {\n    const next = a + b;\n    a = b;\n    b = next;\n  }\n  return b;\n}",
  "The printing press, invented by Johannes Gutenberg around 1440, revolutionized the spread of information in Europe. By making books cheaper and faster to produce, it contributed to rising literacy rates and played a significant role in the Reformation and the Scientific Revolution.",
  "To reset your password, navigate to the account settings page and click on the security tab. Enter your current password, then choose a new password that is at least eight characters long and contains a mix of letters, numbers, and symbols. Click save to apply the changes.",
  "Photovoltaic cells convert sunlight directly into electricity using semiconducting materials, typically silicon. When photons strike the cell, they knock electrons loose, generating a flow of current. The efficiency of commercial solar panels has improved steadily over the past two decades.",
  "He had always wanted to visit the mountains in winter, when the peaks were covered in snow and the valleys were silent. Now, standing at the trailhead with his pack on his back, he felt a mixture of excitement and apprehension about the days of hiking that lay ahead.",
]

def classify(tok_str):
    s = tok_str
    if s.strip() == "": return "space"
    body = s.lstrip()
    if re.fullmatch(r"[^\w\s]+", body): return "punct"
    if re.fullmatch(r"\d+", body): return "digit"
    return "word"

def main(stem):
    man,W = bio.read_bundle(stem); cfg,cfg_f = man["config"],man["config_f"]
    d=int(cfg[4]); V=int(cfg[6]); bpe=BPE(os.path.join(os.path.dirname(stem), os.path.basename(stem)+".tokenizer.json"))
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64); gU=gain*U
    decs=[]                                                               # (a*, margin, x_raw, a*_token_str)
    for txt in PASSAGES:
        ids=bpe.encode(txt)
        if len(ids)<4: continue
        xall,lg = forward_all(W,cfg,cfg_f,ids)
        for i in range(2,len(ids)):                                       # predict from a real growing context
            row=lg[i]; o=np.argsort(row)[::-1]; a=int(o[0]); mar=float(row[o[0]]-row[o[1]])
            decs.append((a, mar, xall[i], bpe.decode_token(a)))
    n=len(decs); tr=decs[:n//2]; te=decs[n//2:]
    # readout-aligned basis from in-distribution top-competitor diffs (fit on TRAIN half of real decisions)
    rows=[]
    for a,_,x,_ in tr:
        comp=np.argsort((gU@x))[::-1][1:9]                              # top competitors at this real decision
        for v in comp: rows.append(_norm(gU[a]-gU[v]))
    Vt=np.linalg.svd(np.array(rows),full_matrices=False)[2]; A=Vt@gU.T
    r=92; Xte=np.array([x for _,_,x,_ in te]); Q=(Vt[:r]@Xte.T).T@A[:r]   # core logits [nte,vocab]
    mar=np.array([m for _,m,_,_ in te]); at=[a for a,_,_,_ in te]; nte=len(te)

    print(f"== PR-core recall vs margin on REAL text (SmolLM-135M; r={r}; {len(PASSAGES)} passages, "
          f"{n} decisions, {nte} test) ==")
    print(f"   teacher-forced real contexts; a* = the model's full argmax; recall = a* ∈ PR-core top-K.")
    # full-range margin distribution + recall per band
    bands=[("[0,0.5)",0,.5),("[0.5,1)",.5,1),("[1,2)",1,2),("[2,4)",2,4),("[4,8)",4,8),("[8,15)",8,15),("[15,∞)",15,1e18)]
    print(f"   {'margin band':<12}{'n':>5}{'%tot':>6}" + "".join(f"{'R@'+str(k):>8}" for k in [1,8,16,32]))
    for lbl,lo,hi in bands:
        sel=np.where((mar>=lo)&(mar<hi))[0]
        if len(sel)==0: print(f"   {lbl:<12}{0:>5}"); continue
        cells=[]
        for k in [1,8,16,32]:
            tk=np.argpartition(-Q[sel],kth=min(k,V-1)-1 if k>1 else 0,axis=1)[:,:k]
            cells.append(np.mean([at[sel[j]] in tk[j] for j in range(len(sel))]))
        print(f"   {lbl:<12}{len(sel):>5}{100*len(sel)/nte:>5.0f}%"+"".join(f"{100*c:>7.0f}%" for c in cells))
    # overall + token-type breakdown
    tk32=np.argpartition(-Q,31,axis=1)[:,:32]; in32=np.array([at[j] in tk32[j] for j in range(nte)])
    tk1=np.argmax(Q,axis=1); in1=np.array([at[j]==tk1[j] for j in range(nte)])
    print(f"   {'OVERALL':<12}{nte:>5}{100:>5.0f}%{100*in1.mean():>7.0f}%{'':>8}{'':>8}{100*in32.mean():>7.0f}%")
    print(f"\n   by token type (R@1 / R@32):")
    types={}
    for j in range(nte): types.setdefault(classify(te[j][3]),[]).append(j)
    for tp,js in sorted(types.items(),key=lambda kv:-len(kv[1])):
        print(f"   {tp:<8}{len(js):>5} ({100*len(js)/nte:>2.0f}%)   R@1 {100*in1[js].mean():>3.0f}%   R@32 {100*in32[js].mean():>3.0f}%")
    hi=np.where(mar>=4)[0]
    print(f"\n   VERDICT: high-margin (≥4) recall  R@1 {100*in1[hi].mean():.0f}%  R@32 {100*in32[hi].mean():.0f}%  "
          f"(n={len(hi)}, {100*len(hi)/nte:.0f}% of real decisions)")

    # rank sweep on the content-word vs format split — is the content-word floor rank-robust?
    wjs=np.array(types.get("word",[])); fjs=np.array([j for tp in ("punct","space","digit") for j in types.get(tp,[])])
    print(f"\n   rank sweep — does more rank rescue content words? (R@32 by token class)")
    print(f"   {'r':>5}{'word R@32':>12}{'format R@32':>14}")
    for r2 in [92,128,256]:
        Q2=(Vt[:r2]@Xte.T).T@A[:r2]; tk=np.argpartition(-Q2,31,axis=1)[:,:32]
        win=np.mean([at[j] in tk[j] for j in wjs]); fin=np.mean([at[j] in tk[j] for j in fjs])
        print(f"   {r2:>5}{100*win:>11.0f}%{100*fin:>13.0f}%")
    print(f"   ⇒ content-word prediction = the forge-tax fragment (τ*); format/structural tokens = retrievable (LE-T2).")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"smollm","smollm"))
