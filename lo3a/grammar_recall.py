#!/usr/bin/env python3
"""Is the forge/retrievable split really about GRAMMAR ROLE? Split the "word" class into closed-class FUNCTION
words (determiners, prepositions, conjunctions, pronouns, auxiliaries — grammatical glue) vs open-class CONTENT
words (nouns, verbs, adjectives, adverbs), and re-measure PR-core recall. Hypothesis: function words behave like
syntax (recoverable by the rank-r head); only content words are the forge tax (τ*). No POS tagger — a closed-class
stoplist suffices to separate open vs closed class. SmolLM-135M, real teacher-forced text (real_recall corpus)."""
import os, sys
import numpy as np
import bundle_io as bio
from bpe import BPE
from real_recall import forward_all, PASSAGES
from pr_core_gate import LISP_PASSAGES

HERE=os.path.dirname(os.path.abspath(__file__))
def _norm(v): return v/(np.linalg.norm(v)+1e-30)

# English closed-class (function) words: determiners, pronouns, prepositions, conjunctions, auxiliaries, particles.
FUNCTION=set("""the a an this that these those my your his her its our their some any no every each all both few many
much most more less i you he she it we they me him them us who whom whose which what of in on at to for with by from
as into onto upon about over under above below between among through during before after since until against within
without toward towards across behind beyond near and or but nor so yet if then else because although though while
whereas unless whether is are was were be been being am do does did have has had will would shall should can could may
might must ought not very too also just only even still again here there when where why how than such out up down off
again once""".split())

def fine_class(s):
    b=s.strip()
    if b=="": return "space"
    if not b[0].isalnum() and all(not ch.isalnum() for ch in b): return "punct"
    if b.isdigit(): return "digit"
    return "function" if b.lower() in FUNCTION else "content"

def main(stem, r=92):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]
    d=int(cfg[4]); V=int(cfg[6])
    bpe=BPE(os.path.join(os.path.dirname(stem),os.path.basename(stem)+".tokenizer.json"))
    U=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64); gain=W["norm"].astype(np.float64); gU=gain*U
    decs=[]                                                               # (a*, margin, x, is_lisp, after_open_paren)
    for txt in PASSAGES+LISP_PASSAGES:
        ids=bpe.encode(txt)
        if len(ids)<4: continue
        lisp = txt.lstrip().startswith("(")
        xall,lg=forward_all(W,cfg,cfg_f,ids)
        for i in range(2,len(ids)):
            o=np.argsort(lg[i])[::-1]
            ao=bpe.decode_token(int(ids[i])).rstrip().endswith("(")       # context ends in "(" ⇒ operator/verb slot
            decs.append((int(o[0]), float(lg[i][o[0]]-lg[i][o[1]]), xall[i], lisp, ao))
    n=len(decs); tr=decs[:n//2]; te=decs[n//2:]; nte=len(te)
    rows=[_norm(gU[a]-gU[v]) for a,_,x,_,_ in tr for v in np.argsort(gU@x)[::-1][1:9]]
    Vt=np.linalg.svd(np.array(rows),full_matrices=False)[2]; A=(Vt[:r]@gU.T)
    Xte=np.array([x for _,_,x,_,_ in te]); Q=(Vt[:r]@Xte.T).T@A
    cls=np.array([fine_class(bpe.decode_token(int(a))) for a,_,_,_,_ in te]); mar=np.array([m for _,m,_,_,_ in te])
    is_lisp=np.array([l for _,_,_,l,_ in te]); after_open=np.array([ao for _,_,_,_,ao in te])
    tk32=np.argpartition(-Q,31,axis=1)[:,:32]; tk8=np.argpartition(-Q,7,axis=1)[:,:8]; tk1=np.argmax(Q,axis=1)
    a=np.array([a for a,_,_,_,_ in te])
    R1=a==tk1; R8=np.array([a[j] in tk8[j] for j in range(nte)]); R32=np.array([a[j] in tk32[j] for j in range(nte)])

    print(f"== grammar-role recall split (SmolLM-135M; r={r}; {nte} real test decisions) ==")
    print(f"   does the retrievable/forge split track OPEN-class (content) vs CLOSED-class (function/format)?")
    print(f"   {'class':<10}{'n':>5}{'%':>5}{'medMargin':>11}{'R@1':>7}{'R@8':>7}{'R@32':>7}")
    for c in ["space","punct","digit","function","content"]:
        s=cls==c
        if s.sum()==0: continue
        print(f"   {c:<10}{s.sum():>5}{100*s.mean():>4.0f}%{np.median(mar[s]):>11.2f}"
              f"{100*R1[s].mean():>6.0f}%{100*R8[s].mean():>6.0f}%{100*R32[s].mean():>6.0f}%")
    closed=np.isin(cls,["space","punct","digit","function"]); openc=cls=="content"
    print(f"   {'-'*46}")
    print(f"   CLOSED-class (format+function) n={closed.sum():>4}  R@1 {100*R1[closed].mean():.0f}%  R@32 {100*R32[closed].mean():.0f}%")
    print(f"   OPEN-class   (content words)   n={openc.sum():>4}  R@1 {100*R1[openc].mean():.0f}%  R@32 {100*R32[openc].mean():.0f}%")
    print(f"\n   verdict: if FUNCTION ≈ format (high recall) and CONTENT is the floor, the axis is GRAMMATICAL")
    print(f"   (open- vs closed-class), i.e. the forge tax is the cost of predicting open-class lexical content.")

    # Lisp positional grammar: the token after "(" is ALWAYS the operator (verb-first) — positionally pinned.
    print(f"\n   == Lisp positional grammar (verb-first: token after '(' is the operator slot) ==")
    L=is_lisp
    for lbl,sel in [("operator slot (after '(')", L & after_open),
                    ("argument slot (else)",      L & ~after_open),
                    ("  argument & open-class",   L & ~after_open & (cls=="content")),
                    ("non-Lisp content (compare)",(~L) & (cls=="content"))]:
        if sel.sum()==0: continue
        print(f"   {lbl:<28}{sel.sum():>5}  medMargin {np.median(mar[sel]):>5.2f}  R@1 {100*R1[sel].mean():>3.0f}%  R@8 {100*R8[sel].mean():>3.0f}%  R@32 {100*R32[sel].mean():>3.0f}%")
    print(f"   ⇒ if the operator slot is recoverable but Lisp arguments are the floor, RECOVERABILITY tracks")
    print(f"   grammatical PREDICTABILITY (closed-class OR positionally-pinned), not token identity.")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"smollm","smollm"))
