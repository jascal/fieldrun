#!/usr/bin/env python3
"""classify_tree.py — group a fieldrun `--tree` dump by (language, function) and render Mermaid.

Reads the `=== Expert tree — recursive sub-bucketing ===` section of a tree dump, classifies every
leaf by the LANGUAGE and FUNCTION of the tokens routed to it, groups by that pair, counts nodes, and
emits a Mermaid flowchart (+ an audit table). Reusable on any future tree dump.

  language : a specific lang (en/de/es/fr/it/ru/zh/ja/ar/hi) by Unicode script + curated markers;
             X   = cross-language  (>=2 *determined* languages routed to the leaf)
             nil = no language     (punctuation / numbers / math / whitespace only)
             lat = Latin script, language-undetermined (ambiguous ASCII subword/closed-class)
  function : dominant role of the routed tokens —
             punct | word | affix | num | math | space | byte   (byte = broken multibyte / U+FFFD)

Usage:
  python scripts/classify_tree.py [TREE.txt] [-o OUT.mmd]
  python scripts/classify_tree.py tree.txt --no-mermaid     # audit table only
"""
import re, sys, argparse, collections, os, shutil, subprocess, tempfile

# --------------------------------------------------------------------------- parse
# leaf line, e.g.:  "    r.e212   37%  (   1 circ)  "\",\" [11]"·268  ..."   (indent = 2*(depth+1))
LEAF = re.compile(r'^(?P<ind> +)(?P<label>(?:[a-z]+\.)*e\d+)\s+\d+%\s+\(\s*\d+ circ\)\s+(?P<toks>.*)$')
# Two token-unit formats are supported:
#   NEW (post-fix):  "<text>" [id]·count    (label printed with {} — clean)
#   OLD (pre-fix) :  "\"<text>\" [id]"·count (label re-quoted with {:?} — double-escaped)
# They are mutually exclusive per unit, so we try NEW first and fall back to OLD.
TOK_NEW = re.compile(r'("(?:[^"\\]|\\.)*")?\s*\[(\d+)\]·(\d+)')
TOK_OLD = re.compile(r'"((?:[^"\\]|\\.)*)"·(\d+)')

def unescape(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == '\\' and i + 1 < len(s):
            n = s[i+1]
            if n == 'u' and i + 2 < len(s) and s[i+2] == '{':
                j = s.index('}', i); out.append(chr(int(s[i+3:j], 16))); i = j + 1; continue
            out.append({'n':'\n','t':'\t','r':'\r','"':'"','\\':'\\',"'":"'",'0':'\0'}.get(n, n)); i += 2; continue
        out.append(c); i += 1
    return ''.join(out)

def _strip_quote_layer(txt):
    return txt[1:-1] if len(txt) >= 2 and txt[0] == '"' and txt[-1] == '"' else txt

def parse_tokens(region):
    """-> list of (token_text, count), handling both the NEW and OLD token-unit formats."""
    new = list(TOK_NEW.finditer(region))
    if new:                                                    # NEW: "<text>" [id]·count
        out = []
        for m in new:
            txt = _strip_quote_layer(unescape(m.group(1))) if m.group(1) else ''  # '' = id-only (no vocab)
            out.append((txt, int(m.group(3))))
        return out
    out = []                                                   # OLD: "\"<text>\" [id]"·count
    for m in TOK_OLD.finditer(region):
        txt = _strip_quote_layer(unescape(re.sub(r'\s*\[\d+\]$', '', m.group(1))))
        out.append((txt, int(m.group(2))))
    return out

# --------------------------------------------------------------------------- language
def char_script(ch):
    o = ord(ch)
    if 0x4E00<=o<=0x9FFF or 0x3400<=o<=0x4DBF or 0xF900<=o<=0xFAFF or 0x20000<=o<=0x2A6DF: return 'han'
    if 0x3040<=o<=0x309F or 0x30A0<=o<=0x30FF or 0x31F0<=o<=0x31FF: return 'kana'
    if 0x0600<=o<=0x06FF or 0x0750<=o<=0x077F or 0x08A0<=o<=0x08FF or 0xFB50<=o<=0xFDFF or 0xFE70<=o<=0xFEFF: return 'arabic'
    if 0x0900<=o<=0x097F or 0xA8E0<=o<=0xA8FF: return 'deva'
    if 0x0400<=o<=0x04FF or 0x0500<=o<=0x052F: return 'cyr'
    if (0x41<=o<=0x5A) or (0x61<=o<=0x7A) or (0xC0<=o<=0x24F): return 'latin'
    return None
LETTER = ('latin','han','kana','arabic','deva','cyr')

# Curated, *confident* markers per Latin language (closed-class words + distinctive content seen in
# this corpus family). Ambiguous short Romance words (de/la/le/el/di/...) are intentionally NOT here —
# they fall through to 'lat'. Extend these sets as new corpora introduce new confident markers.
DE = set("und der die das den dem des ein eine einen einer nicht sie sich dass war als auch noch würde "
         "großen möchte alle über einem zwischen sind werden wird haben deutsch".split())
ES = set("que el la las los una uno unos unas con del esta este cómo cuando libros personas corazón "
         "derecho señor pero porque sobre entre cada también región interés".split())
FR = set("et les une des où été être dans pour qui histoire avec cette nous vous elles".split())
IT = set("che cui gli della nel attravers perché città questo quella sono anche".split())
EN = set("the of and to is are that this with for as at by be have has who into over share task look "
         "points believed principles evolutionary civil function return def else random derivative "
         "calculus typing class import not from will can one two more most some other time used".split())

def latin_lang(text):
    t = text.strip(); low = t.lower()
    if any(c in t for c in 'ñ¿¡íóú'): return 'es'     # Spanish: ñ, inverted marks, acute vowels
    if any(c in t for c in 'äöüß'):  return 'de'      # German umlauts / ß
    hits = [n for n, s in (('de',DE),('es',ES),('fr',FR),('it',IT),('en',EN)) if low in s]
    if len(hits) == 1: return hits[0]
    if any(c in t for c in 'ìò'):  return 'it'        # Italian grave í/ò (distinctive vs fr)
    if any(c in t for c in 'çœê'): return 'fr'        # French cedilla / ligature / circumflex
    if 'é' in t: return hits[0] if hits else 'fr'     # é shared → wordlist else French
    if hits: return hits[0]                           # rare overlap → priority de>es>fr>it>en
    return 'lat'                                      # undetermined Latin (honest; not forced to en)

def token_lang(text, kana_in_leaf):
    scr = set(s for ch in text if (s := char_script(ch)))
    if not scr: return None
    if 'kana' in scr: return 'ja'
    if 'han' in scr: return 'ja' if kana_in_leaf else 'zh'
    if 'arabic' in scr: return 'ar'
    if 'deva' in scr: return 'hi'
    if 'cyr' in scr: return 'ru'
    if 'latin' in scr: return latin_lang(text)
    return None

# --------------------------------------------------------------------------- function
def token_function(t):
    s = t.strip()
    if s == '': return 'space'
    if '�' in t: return 'byte'                                  # U+FFFD replacement → broken multibyte
    if ('$' in t) or ('frac' in t) or ('\\\\' in t) or s in {'\\','=','\\\\'}: return 'math'
    if re.fullmatch(r'[\d.,]+', s): return 'num'
    if not any(char_script(c) in LETTER for c in s): return 'punct'  # brackets/quotes/marks
    if t.startswith(' ') or (s[:1].isupper() and s.isalpha()): return 'word'  # word-initial
    return 'affix'                                                   # continuation / suffix

# --------------------------------------------------------------------------- classify a leaf
def classify(toks):
    kana = any(char_script(c) == 'kana' for tx, _ in toks for c in tx)
    langw, fnw = collections.Counter(), collections.Counter()
    for tx, c in toks:
        fn = token_function(tx)
        fnw[fn] += c
        if fn in ('word', 'affix', 'byte'):          # only language-bearing tokens vote on language;
            lg = token_lang(tx, kana)                # punct/num/math/space are language-agnostic
            if lg: langw[lg] += c
    specific = [l for l in langw if l != 'lat']
    if not langw:                 lang = 'nil'
    elif len(specific) >= 2:      lang = 'X'          # >=2 determined languages → cross-language
    elif len(specific) == 1:      lang = specific[0]  # one determined language (absorbs any 'lat')
    else:                         lang = 'lat'        # only undetermined Latin
    return lang, fnw.most_common(1)[0][0]

# --------------------------------------------------------------------------- header stats (for reuse)
def header_stats(text):
    g = lambda pat: (m.group(1) if (m := re.search(pat, text)) else None)
    return {
        'C':      g(r'\|C\| distinct circuits\s+(\d+)'),
        'ntok':   g(r'N tokens\s+(\d+)'),
        'depth':  g(r'Expert tree — recursive sub-bucketing \(depth (\d+)'),
        'leaves': g(r'Expert tree — recursive sub-bucketing \(depth \d+, (\d+) leaves'),
    }

LANG_ORDER = ['nil','X','lat','en','de','es','fr','it','ru','zh','ja','ar','hi']
FUNC_ORDER = ['punct','word','affix','num','math','space','byte']
LBL = {'nil':'nil · no-lang','X':'X · cross-lang','lat':'lat · undet.','en':'en','de':'de','es':'es',
       'fr':'fr','it':'it','ru':'ru','zh':'zh','ja':'ja','ar':'ar','hi':'hi'}

def find_chrome():
    """Locate a Chrome/Chromium binary for mermaid-cli's headless render (puppeteer)."""
    if os.environ.get('PUPPETEER_EXECUTABLE_PATH'):
        return os.environ['PUPPETEER_EXECUTABLE_PATH']
    for c in ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
              '/Applications/Chromium.app/Contents/MacOS/Chromium'):
        if os.path.exists(c): return c
    for c in ('google-chrome', 'chromium', 'chromium-browser', 'chrome'):
        if (p := shutil.which(c)): return p
    return None

def render(mmd_path, fmts=('svg', 'png'), scale=3):
    """Render a (fence-free) .mmd file to the given formats via mermaid-cli (mmdc)."""
    mmdc = shutil.which('mmdc')
    if not mmdc:
        print("  [render] mmdc not found — install with: npm i -g @mermaid-js/mermaid-cli", file=sys.stderr)
        return []
    env = dict(os.environ)
    if (chrome := find_chrome()): env['PUPPETEER_EXECUTABLE_PATH'] = chrome
    cfg = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    cfg.write('{"args":["--no-sandbox"]}'); cfg.close()                # headless sandbox is unavailable here
    done = []
    stem = mmd_path.rsplit('.', 1)[0]
    for fmt in fmts:
        out = f'{stem}.{fmt}'
        cmd = [mmdc, '-i', mmd_path, '-o', out, '-p', cfg.name, '-b', 'white']
        if fmt == 'png': cmd += ['-s', str(scale)]                    # supersample raster for crispness
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            done.append(out)
        else:
            print(f"  [render] {fmt} failed: {(r.stderr or r.stdout).strip().splitlines()[-1:]}", file=sys.stderr)
    os.unlink(cfg.name)
    return done

def main():
    ap = argparse.ArgumentParser(description="Group a fieldrun --tree dump by (language, function) → Mermaid.")
    ap.add_argument('tree', nargs='?', default='tree.txt', help='tree dump (default: tree.txt)')
    ap.add_argument('-o', '--out', help='write Mermaid here (default: <tree>.mmd)')
    ap.add_argument('--no-mermaid', action='store_true', help='audit table only')
    ap.add_argument('--render', action='store_true', help='also render SVG+PNG via mermaid-cli (mmdc)')
    ap.add_argument('--formats', default='svg,png', help='formats for --render (default: svg,png)')
    ap.add_argument('--scale', type=int, default=3, help='PNG supersample factor (default: 3)')
    a = ap.parse_args()

    text = open(a.tree, encoding='utf-8').read()
    stats = header_stats(text)
    leaves = []
    for line in text.splitlines():
        m = LEAF.match(line)
        if not m: continue
        toks = parse_tokens(m.group('toks'))
        if not toks: continue
        depth = len(m.group('ind')) // 2 - 1
        lang, func = classify(toks)
        leaves.append((lang, func, depth, m.group('label'), toks))
    if not leaves:
        sys.exit(f"no leaves parsed from {a.tree} (is it a --tree dump?)")

    cell    = collections.Counter((lg, fn) for lg, fn, *_ in leaves)
    langtot = collections.Counter(lg for lg, *_ in leaves)
    functot = collections.Counter(fn for _, fn, *_ in leaves)
    maxd    = max(l[2] for l in leaves)

    # ---- audit table
    print(f"{a.tree}: {len(leaves)} leaves  (|C|={stats['C']}  N={stats['ntok']}  "
          f"depth-cap={stats['depth']}  header-leaves={stats['leaves']}  realized-depth={maxd})\n")
    print("LANGUAGE:", {k: langtot[k] for k in LANG_ORDER if langtot[k]})
    print("FUNCTION:", {k: functot[k] for k in FUNC_ORDER if functot[k]})
    print("\n(lang · func) [count]   e.g. tokens")
    ex = collections.defaultdict(list)
    for lg, fn, dpth, lab, toks in leaves:
        if len(ex[(lg,fn)]) < 5:
            ex[(lg,fn)].append((toks[0][0] or '∅').replace('\n','\\n'))
    for lg in LANG_ORDER:
        for fn in FUNC_ORDER:
            if cell[(lg,fn)]:
                print(f"  {lg:>3} · {fn:<5} [{cell[(lg,fn)]:>3}]   {', '.join(repr(x) for x in ex[(lg,fn)])}")

    if a.no_mermaid: return
    # ---- mermaid
    title = (f"monster expert tree<br/>{len(leaves)} leaves" +
             (f" · |C|={stats['C']}" if stats['C'] else "") +
             (f" · {stats['ntok']} decisions" if stats['ntok'] else ""))
    out = ['flowchart LR', f'  R["{title}"]']
    for lg in LANG_ORDER:
        if not langtot[lg]: continue
        out.append(f'  R --> L_{lg}["{LBL[lg]}<br/>({langtot[lg]})"]')
        for fn in FUNC_ORDER:
            if cell[(lg,fn)]:
                out.append(f'  L_{lg} --> L_{lg}_{fn}["{fn} · {cell[(lg,fn)]}"]')
    out += ['  classDef nilC fill:#eef,stroke:#88a;',
            '  classDef xC   fill:#fde,stroke:#c49;',
            '  classDef latC fill:#efe,stroke:#7a7;']
    for lg, cl in (('nil','nilC'), ('X','xC'), ('lat','latC')):
        if langtot[lg]: out.append(f'  class L_{lg} {cl};')
    body = '\n'.join(out)                                  # raw mermaid (no code fence) — what .mmd / mmdc want
    dst = a.out or (a.tree.rsplit('.', 1)[0] + '.mmd')
    open(dst, 'w').write(body + '\n')
    print(f"\nwrote {dst}\n\n```mermaid\n{body}\n```")     # fenced for chat/markdown display
    if a.render:
        made = render(dst, tuple(f.strip() for f in a.formats.split(',') if f.strip()), a.scale)
        for f in made: print(f"rendered {f}")

if __name__ == '__main__':
    main()
