#!/usr/bin/env python3
"""Toki Pona on the entropy spectrum — does a MINIMAL-vocabulary real language compress under the readout?

Toki Pona has ~120 root words: intrinsically the lowest-entropy natural language there is. But those words are
RARE in SmolLM's English-centric training distribution. So this tests the sharp question the natural/Lisp data
left open: is recoverable rank set by the LANGUAGE's intrinsic vocabulary, or by the MODEL's training prior?
We compare three corpora (English prose / Lisp / Toki Pona) under identical analysis — each fit to its OWN
readout-aligned basis — and report, per corpus: the model's cross-entropy (bits/token; "does it know it?"),
overall R@32, closed-vs-open recall, median normalized recoverable rank, and Spearman(rank, self-info).
SmolLM-135M, teacher-forced. (Caveat: 135M barely models Toki Pona; perplexity column quantifies that.)
"""
import os, sys
import numpy as np
import bundle_io as bio
from bpe import BPE
from real_recall import forward_all, PASSAGES
from pr_core_gate import LISP_PASSAGES
from grammar_recall import fine_class, FUNCTION
from info_rank import spearman

HERE=os.path.dirname(os.path.abspath(__file__))
def _norm(v): return v/(np.linalg.norm(v)+1e-30)

# authentic Toki Pona (particles li/e/la/pi/o/en, prepositions lon/tawa/tan/kepeken/sama, pronouns mi/sina/ona/ni)
TOKIPONA=[
 "mi moku e kili. kili li pona. mi wile e telo. telo li lon poki. mi moku e telo kepeken poki.",
 "jan li tawa tomo. tomo li suli. jan li lukin e tomo. ona li pilin pona. jan li lape lon tomo.",
 "soweli li lon ma. soweli li moku e kasi. kasi li laso. soweli li lape lon tenpo pimeja. waso li kalama.",
 "mi sona e toki pona. toki ni li lili. jan mute li toki e ona. mi pana e sona tawa sina. sina ken kama sona.",
 "tenpo suno la mi pana e moku tawa jan pona mi. ona li moku e ona. mi pilin pona. mi toki tawa ona.",
 "waso li tawa sewi. sewi li laso. waso mute li lon sewi. mi lukin e waso. waso li kalama musi.",
 "jan lili li musi lon ma kasi. ona li tawa e telo. telo li tawa anpa. jan lili li pilin pona mute.",
 "mi wile lukin e ma ante. ma ni li jo e kasi mute. mi tawa kepeken noka. tenpo li suli. mi pilin monsuta lili.",
 "ona li pana e lipu tawa mi. lipu li jo e sitelen mute. mi lukin e sitelen. sitelen li pona. mi sona e ona.",
 "kili li lon ma. mi kama jo e kili. mi moku e ona. ona li suwi. mi wile e kili mute.",
 "jan utala li kama. ona li jo e ilo. mi weka tan ona. mi tawa weka. mi pilin monsuta.",
 "tenpo pini la mi lon ma ante. mi kama tawa ma ni. mi sona e jan mute. ona li pona tawa mi. mi awen lon ni.",
]
TP_CLOSED=set("li e la pi o en taso lon tawa tan kepeken sama mi sina ona ni".split())

# Finnish — agglutinative, 15 cases + productive compounding ⇒ a long tail of rare inflected word-forms (the
# real high-entropy extreme that replaces a synthetic conlang). Authentic grammatical prose.
FINNISH=[
 "Aurinko paistaa kirkkaalla taivaalla. Linnut laulavat puiden oksilla. Lapset leikkivät puistossa ja nauravat iloisesti. Vanha mies kävelee hitaasti pitkin katua ja katselee ympärilleen.",
 "Talvella sataa paljon lunta ja järvet jäätyvät. Ihmiset hiihtävät metsässä ja luistelevat jäällä. Illalla perhe istuu takan ääressä ja juo kuumaa kaakaota yhdessä.",
 "Suomen kieli on vaikea oppia, koska siinä on viisitoista sijamuotoa. Sanat taipuvat monella eri tavalla. Kuitenkin kielen rakenne on hyvin looginen ja säännöllinen.",
 "Hän osti kaupasta leipää, maitoa ja juustoa. Sitten hän käveli kotiin ja valmisti ruokaa koko perheelle. Illallinen oli valmis kello kuusi illalla.",
 "Metsässä kasvaa korkeita kuusia ja mäntyjä. Sieniä ja marjoja löytyy syksyllä runsaasti. Karhut ja sudet elävät syvällä erämaassa kaukana ihmisistä.",
 "Opiskelija lukee kirjaa kirjastossa. Hän kirjoittaa muistiinpanoja ja valmistautuu kokeeseen. Huomenna hänellä on tärkeä tentti yliopistolla aamulla.",
 "Meri on tyyni ja sininen kesäaamuna. Veneet keinuvat satamassa hiljaa. Kalastajat lähtevät aikaisin merelle ja palaavat illalla saaliin kanssa.",
 "Kaupunki herää aikaisin aamulla. Autot ajavat kaduilla ja ihmiset kiiruhtavat töihin. Kahvilat avaavat ovensa ja kahvin tuoksu leviää kadulle.",
 "Lapsi piirtää kuvan perheestään. Kuvassa on äiti, isä ja pieni ruskea koira. Hän käyttää monia värejä ja hymyilee tyytyväisenä työhönsä.",
 "Vuoristossa ilma on raikasta ja kylmää. Polku nousee jyrkästi ylöspäin kohti huippua. Vaeltajat lepäävät hetken ja ihailevat upeaa maisemaa ympärillään.",
 "Kirjailija istuu työhuoneessaan ja miettii tarinaa. Sanat tulevat hitaasti mutta varmasti paperille. Hän haluaa kertoa tarinan, joka koskettaa lukijoita syvästi.",
 "Keväällä lumi sulaa ja luonto herää eloon. Ensimmäiset kukat puhkeavat kukkaan ja muuttolinnut palaavat etelästä. Päivät pitenevät ja ilma lämpenee vähitellen.",
]
FI_CLOSED=set("ja mutta tai koska kun että on oli ei en et emme ette eivät ole se ne tämä nämä hän minä sinä me te he joka jossa josta niin myös vielä jo vain kuin kanssa mukaan jälkeen ennen sekä eli jos vaikka kunnes kuten".split())

def cls_for(lang):
    def f(s):
        c=fine_class(s);
        if c in ("space","punct","digit"): return "closed"
        b=s.strip().lower()
        if lang=="English": return "closed" if b in FUNCTION else "open"
        if lang=="TokiPona": return "closed" if b in TP_CLOSED else "open"
        if lang=="Finnish": return "closed" if b in FI_CLOSED else "open"
        return "open"                                   # Lisp: operators/symbols are open; parens already "closed"
    return f

def collect(W,cfg,cfg_f,bpe,texts):
    V=int(cfg[6]); enc=[bpe.encode(t) for t in texts]; cnt=np.zeros(V)
    for ids in enc:
        for t in ids: cnt[t]+=1
    freq=cnt/max(1,cnt.sum()); decs=[]; bits=[]
    for ids in enc:
        if len(ids)<4: continue
        xall,lg=forward_all(W,cfg,cfg_f,ids)
        for i in range(2,len(ids)-1):
            row=lg[i]; m=row.max(); lse=m+np.log(np.exp((row-m).astype(np.float64)).sum())
            bits.append(-(row[ids[i+1]]-lse)/np.log(2)); decs.append((int(np.argmax(row)), xall[i]))
    return decs, freq, float(np.mean(bits))

def fit_basis(decs, gU):
    rows=[_norm(gU[a]-gU[v]) for a,x in decs for v in np.argsort(gU@x)[::-1][1:9]]
    return np.linalg.svd(np.array(rows),full_matrices=False)[2]

def evaluate(name, te, freq, Vt, gU, d, bpe):
    A=Vt@gU.T; Xte=np.array([x for _,x in te]); a=np.array([a for a,_ in te]); P=Vt@Xte.T; nte=len(te)
    grid=[1,2,4,8,16,24,32,48,64,92,128,192,256,384,512,d]; rr=np.full(nte,d,float); done=np.zeros(nte,bool)
    for r in grid:
        arg=np.argmax((P[:r].T)@A[:r],axis=1); hit=(arg==a)&~done; rr[hit]=r; done|=hit
    Q92=(P[:92].T)@A[:92]; tk=np.argpartition(-Q92,31,axis=1)[:,:32]; R32=np.array([a[j] in tk[j] for j in range(nte)])
    cf=cls_for(name); cl=np.array([cf(bpe.decode_token(int(t))) for t in a]); info=-np.log2(np.clip(freq[a],1e-9,None))
    distinct=len(set(int(x) for x in a))                                # output-vocabulary diversity (distinct argmaxes)
    return dict(R32=R32.mean(), R32_open=(R32[cl=="open"].mean() if (cl=="open").any() else np.nan),
                medrank=np.median(rr/d), sp=spearman(rr,info), distinct=distinct, nte=nte)

def main(stem):
    man,W=bio.read_bundle(stem); cfg,cfg_f=man["config"],man["config_f"]; d=int(cfg[4])
    bpe=BPE(os.path.join(os.path.dirname(stem),os.path.basename(stem)+".tokenizer.json"))
    gU=(W["embed"] if cfg[7] else W["lm_head"]).astype(np.float64)*W["norm"].astype(np.float64)
    corp={}
    for nm,tx in [("TokiPona",TOKIPONA),("Lisp",LISP_PASSAGES),("English",PASSAGES),("Finnish",FINNISH)]:
        decs,freq,ppl=collect(W,cfg,cfg_f,bpe,tx); corp[nm]=dict(decs=decs,freq=freq,ppl=ppl)
    rows=[]
    for nm in ("TokiPona","Lisp","English","Finnish"):                   # the journey: minimal → morphologically explosive
        c=corp[nm]; n=len(c["decs"]); tr,te=c["decs"][:n//2],c["decs"][n//2:]
        Vt=fit_basis(tr,gU); m=evaluate(nm,te,c["freq"],Vt,gU,d,bpe); m.update(name=nm,ppl=c["ppl"]); rows.append(m)
    # cross-lens: each extreme's decisions scored against the ENGLISH-fit basis (the model's natural geometry)
    eng=corp["English"]; Vt_eng=fit_basis(eng["decs"][:len(eng["decs"])//2],gU)
    for nm in ("TokiPona","Finnish"):
        c=corp[nm]; m=evaluate(nm,c["decs"][len(c["decs"])//2:],c["freq"],Vt_eng,gU,d,bpe)
        m.update(name=f"{nm[:4]}@Eng-lens",ppl=c["ppl"]); rows.append(m)

    print(f"== an entropy-per-token journey: Toki Pona → Lisp → English → Finnish (SmolLM-135M) ==")
    print(f"   {'corpus':<13}{'n':>5}{'bits/tok':>10}{'distinct a*':>12}{'R@32 all':>10}{'R@32 open':>11}{'med ρ/d':>9}{'Spear':>7}")
    for r in rows:
        print(f"   {r['name']:<13}{r['nte']:>5}{r['ppl']:>10.1f}{r['distinct']:>12}{100*r['R32']:>9.0f}%{100*r['R32_open']:>10.0f}%{r['medrank']:>9.2f}{r['sp']:>+7.2f}")
    print(f"\n   bits/tok = model cross-entropy on the language; distinct a* = number of distinct argmax tokens")
    print(f"   (output-vocabulary diversity); med ρ/d = normalized recoverable rank.")
    print(f"   recoverable rank tracks OUTPUT DIVERSITY, not the model's competence/surprisal: Toki Pona is the")
    print(f"   model's HARDEST (highest bits/tok) yet MOST compressible (few distinct outputs); Finnish, with its")
    print(f"   agglutinative long tail, is the open-class extreme. cross-lens (@Eng): scored in the model's natural")
    print(f"   geometry instead of its own — separates the language's intrinsic diversity from the lens it's fit to.")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"smollm","smollm"))
