#!/usr/bin/env python3
"""Build the largest mixed corpus we can: streamed multilingual Wikipedia + code + math + English holdout,
interleaved so any window spans many domains → a large distinct-circuit budget |C| for a monstrous expert tree.
Run from repo root: ../lm-sae/.venv/bin/python scripts/make_monster_corpus.py"""
import json, sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file("bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.tokenizer.json")
ids = {}

# 1) multilingual Wikipedia via streaming (a few articles per language).
try:
    from datasets import load_dataset
    for lang in ["en", "de", "fr", "es", "it", "ru", "zh", "ja", "ar", "hi"]:
        try:
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
            buf = []
            for ex in ds:
                buf.append(ex["text"])
                if sum(len(t) for t in buf) > 40000:
                    break
            txt = "\n".join(buf)[:40000]
            ids[f"wiki_{lang}"] = tok.encode(txt).ids
            print(f"wiki {lang}: {len(ids[f'wiki_{lang}'])} tokens", file=sys.stderr)
        except Exception as e:
            print(f"wiki {lang} skipped: {e}", file=sys.stderr)
except Exception as e:
    print(f"datasets unavailable: {e}", file=sys.stderr)

# 2) code + math (reuse the tokenized samples, scaled up).
def load_ids(n):
    return json.load(open(f"sweeps/corpora/{n}.json"))["holdout_ids"]
for n, mult in [("code", 6), ("math", 6), ("german", 3), ("spanish", 3)]:
    try:
        ids[n] = load_ids(n) * mult
    except Exception:
        pass

# 3) English holdout (natural text).
try:
    en = json.load(open("../lm-sae/pylm/holdout_Qwen2.5-1.5B.json"))["holdout_ids"]
    ids["english"] = en[:12000]
except Exception:
    pass

# interleave all sources in 100-token chunks so a sliding window spans many domains.
chunk, out, idx = 100, [], {k: 0 for k in ids}
keys = list(ids)
while any(idx[k] < len(ids[k]) for k in keys):
    for k in keys:
        out += ids[k][idx[k]:idx[k] + chunk]
        idx[k] += chunk
json.dump({"model": "Qwen2.5", "holdout_ids": out}, open("sweeps/corpora/monster.json", "w"))
print(f"monster: {len(out)} tokens from {len(ids)} sources: {keys}")
