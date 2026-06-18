#!/usr/bin/env python3
"""Tokenize natural prose with the Qwen2.5 bundle tokenizer into the {"holdout_ids": [...]} corpus format that
fieldrun's --ids flag expects. Two modes:

    make_holdout_qwen.py                 -> small built-in passage  -> holdout_qwen.json
    make_holdout_qwen.py <file> <n> <out>-> first n tokens of <file> body -> <out>

The file mode strips Project-Gutenberg boilerplate so the corpus is flowing narrative English (diverse vocabulary,
natural token distribution) — used for the working-set-growth probe (Stage-0 question (a))."""
import json
import sys
from tokenizers import Tokenizer

TOK = "bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.tokenizer.json"

SMALL = """The history of computing is a story of abstraction. Each generation of engineers built machines that hid the
details of the one beneath it, so that a programmer could think about problems instead of wires. The earliest
electronic computers were rooms full of vacuum tubes, programmed by physically rewiring them. The invention of the
stored-program architecture changed everything: instructions and data lived in the same memory, and a machine could be
reprogrammed by loading new symbols rather than rebuilding its circuits.

Photosynthesis converts sunlight into chemical energy. In the leaves of a plant, pigments absorb light and use its
energy to split water molecules, releasing oxygen as a byproduct. The hydrogen that remains is combined with carbon
dioxide drawn from the air to build sugars. These sugars store the captured energy in their chemical bonds, and the
plant later breaks them down to power its growth."""


def strip_gutenberg(text: str) -> str:
    """Drop the *** START/END OF ... *** boilerplate, keeping only the book body."""
    start = text.find("*** START")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.find("*** END")
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def main():
    tok = Tokenizer.from_file(TOK)
    if len(sys.argv) >= 4:
        path, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
        body = strip_gutenberg(open(path, encoding="utf-8").read())
        ids = tok.encode(body, add_special_tokens=False).ids[:n]
    else:
        ids, out = tok.encode(SMALL, add_special_tokens=False).ids, "holdout_qwen.json"
    json.dump({"holdout_ids": ids}, open(out, "w"))
    print(f"wrote {out}: {len(ids)} tokens (vocab range {min(ids)}..{max(ids)})")


if __name__ == "__main__":
    main()
