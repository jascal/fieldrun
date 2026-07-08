#!/usr/bin/env python3
"""Harvest clean English sentences from workspace .md prose → a J-lens fit corpus (one prompt per line)."""
import re, sys, glob, os, hashlib

roots = [
    "/home/allans/code/fieldrun",
    "/home/allans/code/rosetta",
    "/home/allans/code/pic",
    "/home/allans/code/pil",
    "/home/allans/code/polygram",
    "/home/allans/code/sae-forge",
    "/home/allans/code/lm-sae",
]
files = []
for r in roots:
    files += glob.glob(os.path.join(r, "*.md"))
    files += glob.glob(os.path.join(r, "docs", "*.md"))
    files += glob.glob(os.path.join(r, "paper", "*.md"))

def clean(txt):
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)        # fenced code
    txt = re.sub(r"`[^`]*`", " ", txt)                       # inline code
    txt = re.sub(r"\$[^$]*\$", " ", txt)                     # math
    txt = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", txt)       # md links → text
    txt = re.sub(r"https?://\S+", " ", txt)
    return txt

sentences = {}
for f in files:
    try:
        raw = open(f, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    txt = clean(raw)
    # keep only prose paragraph lines (drop headings, lists, tables, quotes)
    lines = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not s or s[0] in "#|->*+=" or s.startswith("![") or "\t" in ln:
            continue
        lines.append(s)
    para = " ".join(lines)
    para = re.sub(r"\s+", " ", para)
    # naive sentence split on . ! ? followed by space+capital
    for m in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", para):
        s = m.strip()
        w = s.split()
        if not (6 <= len(w) <= 32):
            continue
        if not re.match(r"^[A-Z]", s) or s[-1] not in ".!?":
            continue
        # mostly-alphabetic, low symbol density (skip pathy / codey residue)
        alpha = sum(c.isalpha() or c.isspace() for c in s)
        if alpha / len(s) < 0.82:
            continue
        if any(sym in s for sym in ("_", "/", "\\", "{", "}", "|", "~", "^", "@")):
            continue
        if len(s) < 40 or len(s) > 200:
            continue
        key = s.lower()
        sentences.setdefault(key, s)

uniq = list(sentences.values())
# deterministic shuffle (hash-sorted) so the corpus is reproducible without random seeds
uniq.sort(key=lambda s: hashlib.md5(s.encode()).hexdigest())
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300
out = uniq[:cap]
with open(sys.argv[1], "w") as fh:
    fh.write("\n".join(out) + "\n")
print(f"harvested {len(uniq)} unique sentences from {len(files)} files; wrote {len(out)} to {sys.argv[1]}")
print("--- sample ---")
for s in out[:6]:
    print("  ", s[:100])
