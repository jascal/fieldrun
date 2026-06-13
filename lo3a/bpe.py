#!/usr/bin/env python3
"""Minimal byte-level BPE encoder for the SmolLM tokenizer.json (GPT2-style: Digits + ByteLevel BPE).
No `tokenizers`/`regex` dependency — ASCII-focused pretokenizer (sufficient for clean English/code prose,
which is what we want: real in-distribution contexts). Encoding need not be byte-perfect; it must produce
valid, in-distribution real-text token sequences for the recall-vs-margin measurement."""
import json, re, functools

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

# Digits(individual_digits) → single digit; ByteLevel use_regex (GPT2 pattern, ASCII approximation).
_PAT = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

class BPE:
    def __init__(self, tok_json):
        t = json.load(open(tok_json)); m = t["model"]
        self.vocab = m["vocab"]                                           # token-string -> id
        mg = m["merges"]
        mg = [tuple(p.split(" ")) if isinstance(p, str) else tuple(p) for p in mg]
        self.ranks = {p: i for i, p in enumerate(mg)}
        self.b2u = bytes_to_unicode()
        self.inv = {i: s for s, i in self.vocab.items()}
        u2b = {u: b for b, u in self.b2u.items()}
        self._u2b = u2b

    @functools.lru_cache(maxsize=200000)
    def _bpe(self, piece):
        word = list(piece)
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            cand = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if cand not in self.ranks: break
            i = 0; new = []
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == cand:
                    new.append(word[i]+word[i+1]); i += 2
                else:
                    new.append(word[i]); i += 1
            word = new
        return word

    def encode(self, text):
        ids = []
        for chunk in _PAT.findall(text):
            piece = "".join(self.b2u[b] for b in chunk.encode("utf-8"))
            for tok in self._bpe(piece):
                if tok in self.vocab: ids.append(self.vocab[tok])
                else:                                                     # fall back to byte tokens
                    for ch in tok:
                        if ch in self.vocab: ids.append(self.vocab[ch])
        return ids

    def decode_token(self, tid):
        s = self.inv.get(int(tid), "")
        try:    return bytes(self._u2b.get(c, 0) for c in s).decode("utf-8", "replace")
        except Exception: return s

if __name__ == "__main__":
    import os
    b = BPE(os.path.join(os.path.dirname(os.path.abspath(__file__)), "smollm", "smollm.tokenizer.json"))
    for s in ["The quick brown fox jumps over the lazy dog.",
              "import numpy as np\nx = np.zeros((3, 4))",
              "In 2020, the population was 1234 people."]:
        ids = b.encode(s); back = "".join(b.decode_token(i) for i in ids)
        print(f"{len(ids):3d} ids | roundtrip {'OK' if back==s else 'DIFF'} | {back!r}")
