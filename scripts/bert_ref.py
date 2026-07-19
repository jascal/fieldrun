#!/usr/bin/env python3
"""Tiny-random BERT reference for the fieldrun `bert` encoder arch.

build:   make a tiny BertModel with EVERY parameter randomized (incl. LayerNorm
         weights/biases and token-type embeddings — identity/zero inits would
         hide unapplied-tensor bugs, the gemma4 layer_scalar lesson), save it as
         a HF checkpoint for `fieldrun convert`, write a random id holdout, and
         dump the torch hidden states (output_hidden_states convention).
compare: read fieldrun's --encode-dump binary and gate max|diff| / MSE per
         snapshot against the torch reference. f32 gate: max|diff| < 1e-4.

Usage: bert_ref.py build | bert_ref.py compare <rust_hiddens.bin>
"""
import json
import struct
import sys

import torch
from transformers import BertConfig, BertModel

TAG = "bert"
N_IDS = 60
CFG = dict(vocab_size=64, hidden_size=32, num_hidden_layers=5, num_attention_heads=4,
           intermediate_size=64, max_position_embeddings=64, type_vocab_size=2,
           layer_norm_eps=1e-12, hidden_act="gelu")


def build():
    torch.manual_seed(0)
    model = BertModel(BertConfig(**CFG))
    with torch.no_grad():
        for p in model.parameters():  # randomize EVERYTHING, norms and biases included
            p.uniform_(-0.5, 0.5)
    model.eval()
    model.save_pretrained(f"/tmp/{TAG}tiny", safe_serialization=True)
    ids = torch.randint(0, CFG["vocab_size"], (N_IDS,))
    json.dump({"holdout_ids": ids.tolist()}, open(f"/tmp/{TAG}_holdout.json", "w"))
    with torch.no_grad():
        hs = model(ids.unsqueeze(0), output_hidden_states=True).hidden_states
    with open(f"/tmp/{TAG}_torch_hiddens.bin", "wb") as f:
        for h in hs:
            f.write(h.squeeze(0).float().numpy().tobytes())
    print(f"built /tmp/{TAG}tiny ({len(hs)} snapshots x {N_IDS} tokens x {CFG['hidden_size']})")


def compare(path):
    d, nl = CFG["hidden_size"], CFG["num_hidden_layers"]
    n = N_IDS * d
    ref = open(f"/tmp/{TAG}_torch_hiddens.bin", "rb").read()
    got = open(path, "rb").read()
    if len(got) != len(ref):
        print(f"FAIL size {len(got)} != {len(ref)}")
        sys.exit(1)
    worst = 0.0
    for s in range(nl + 1):
        r = struct.unpack(f"<{n}f", ref[s * n * 4:(s + 1) * n * 4])
        g = struct.unpack(f"<{n}f", got[s * n * 4:(s + 1) * n * 4])
        mad = max(abs(a - b) for a, b in zip(r, g))
        mse = sum((a - b) ** 2 for a, b in zip(r, g)) / n
        worst = max(worst, mad)
        print(f"snapshot {s}: max|diff| {mad:.3e}  mse {mse:.3e}")
    ok = worst < 1e-4
    print(f"{'PASS' if ok else 'FAIL'} (worst max|diff| {worst:.3e}, gate 1e-4)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        compare(sys.argv[2])
