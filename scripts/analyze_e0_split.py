#!/usr/bin/env python3
"""Split an --ablate-rows per-position dump by TARGET-token class (function-word/punctuation vs
content word) to test the e0 "competitor vs scaffold" hypothesis: a function-word *competitor*
should HURT function-word targets (+Δloss) and SPARE/HELP content-word targets (≤0), whereas a
load-bearing syntactic core would hurt both.

Usage: python scripts/analyze_e0_split.py ablate/e0_rows.tsv
"""
import csv, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_tree as ct
from tokenizers import Tokenizer

TOK = "bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.tokenizer.json"
tok = Tokenizer.from_file(TOK)

# Multilingual closed-class function words (articles / prepositions / conjunctions / pronouns /
# auxiliaries), incl. the short Romance ones classify_tree deliberately leaves ambiguous. Used ONLY
# to label target tokens here; punctuation/number/whitespace are function by token_function.
FUNC = set("""
the a an of and or to in on at by for from with as is are was were be been being this that these those it
its he she they we you i his her their our your my not no do does did have has had will would can could
should may might must but if then than so such where when which who whom whose
der die das den dem des ein eine einen einer eines und oder aber nicht ist sind war waren sein ich du er sie
es wir ihr in an auf fur fuer mit von zu aus bei nach uber unter durch um als auch noch nur schon wie wenn dass
le la les un une des du et ou mais ne pas est sont etait suis il elle nous vous dans sur pour avec par au aux
ce cette ces qui que dont ou se son sa ses leur leurs
el los las uno unos unas y o pero no son era yo ella en por para con sin sobre su sus lo del al como cuando
donde porque de
""".split())

def tclass(tid):
    txt = tok.decode([int(tid)])
    fn = ct.token_function(txt)
    if fn in ('punct', 'num', 'space'):
        return 'function'
    if fn in ('word', 'affix'):
        return 'function' if txt.strip().lower() in FUNC else 'content'
    return 'other'

dump = sys.argv[1] if len(sys.argv) > 1 else 'ablate/e0_rows.tsv'
rows = list(csv.DictReader(open(dump), delimiter='\t'))
agg = defaultdict(lambda: {'dl': [], 'dlo': [], 'fl': []})
cache = {}
for r in rows:
    tid = r['target_id']
    c = cache.get(tid)
    if c is None:
        c = cache[tid] = tclass(tid)
    dl = float(r['abl_loss']) - float(r['base_loss'])
    dlo = float(r['abl_tlogit']) - float(r['base_tlogit'])
    for ev in (r['eval'], 'ALL'):
        a = agg[(r['spec'], ev, c)]
        a['dl'].append(dl); a['dlo'].append(dlo); a['fl'].append(int(r['flip']))

mean = lambda x: sum(x) / len(x) if x else float('nan')
specs = sorted(set(r['spec'] for r in rows))
print(f"{'spec':16} {'eval':4} {'class':8} {'n':>4} {'mean_Δloss':>11} {'mean_Δlogit':>12} {'flip%':>6}")
print("-" * 64)
for s in specs:
    for ev in ['de', 'fr', 'es', 'en', 'ALL']:
        for c in ['function', 'content', 'other']:
            a = agg.get((s, ev, c))
            if a and a['dl']:
                print(f"{s:16} {ev:4} {c:8} {len(a['dl']):>4} {mean(a['dl']):>+11.4f} {mean(a['dlo']):>+12.3f} {100*mean(a['fl']):>6.1f}")
    print()
