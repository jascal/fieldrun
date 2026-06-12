#!/usr/bin/env python3
"""Pure-numpy GPT-NeoX (Pythia) reference forward — the faithfulness gate for fieldrun's `neox` kernel.

No torch (none installs on macOS x86_64 + py3.13): safetensors is parsed with the stdlib (8-byte LE header length +
JSON header + raw little-endian tensors), the forward is plain numpy in f32 mirroring the architecture spec —
LayerNorm(+bias), fused qkv split PER HEAD ([q,k,v] stacked per head), partial rotary (first rotary_ndims of each
head, split-half), causal attention (1/sqrt(head_size)), exact erf GELU (`hidden_act: "gelu"`), parallel residual
x + attn(ln1(x)) + mlp(ln2(x)), final LayerNorm, untied embed_out.

Usage:
  python3 scripts/neox_ref.py <model_dir> <holdout.json> [--ctx 64] [--n 60] [--dump preds.txt]

Prints (and optionally dumps) the top-1 prediction per position i in [ctx, ctx+n): argmax over the logits at the
last token of holdout_ids[i-ctx:i] — the same positions fieldrun's scoring mode evaluates, so:
  fieldrun --bundle <stem> --ids holdout.json --ctx 64 --n-eval 60 --dump /tmp/fr.txt
  python3 scripts/neox_ref.py <model_dir> holdout.json --ctx 64 --n 60 --dump /tmp/np.txt
  diff /tmp/fr.txt /tmp/np.txt    # the gate: must be empty at f32
"""

import json
import math
import struct
import sys

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        data = f.read()
    dt = {"F32": np.float32, "F16": np.float16, "BF16": None, "I64": np.int64}
    out = {}
    for name, info in header.items():
        if name == "__metadata__" or info["dtype"] not in dt:
            continue  # skip buffer tensors (e.g. the U8/BOOL causal-mask `attention.bias` in NeoX checkpoints)
        lo, hi = info["data_offsets"]
        raw = data[lo:hi]
        if info["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=dt[info["dtype"]])
        out[name] = arr.reshape(info["shape"]).astype(np.float32)
    return out


def layernorm(x, g, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps) * g + b).astype(np.float32)


_erf = np.vectorize(math.erf)


def gelu(x):
    return (0.5 * x * (1.0 + _erf(x / np.float32(math.sqrt(2.0))).astype(np.float32))).astype(np.float32)


def softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


class NeoxRef:
    def __init__(self, model_dir):
        self.c = json.load(open(f"{model_dir}/config.json"))
        idx = f"{model_dir}/model.safetensors.index.json"
        import os
        if os.path.exists(idx):
            wm = json.load(open(idx))["weight_map"]
            self.w = {}
            for f in sorted(set(wm.values())):
                self.w.update(load_safetensors(f"{model_dir}/{f}"))
        else:
            self.w = load_safetensors(f"{model_dir}/model.safetensors")
        c = self.c
        self.nl = c["num_hidden_layers"]
        self.nh = c["num_attention_heads"]
        self.d = c["hidden_size"]
        self.hd = self.d // self.nh
        self.rot = max(2, int(round(self.hd * c.get("rotary_pct", 1.0)))) & ~1
        self.theta = c.get("rotary_emb_base", 10000.0)
        self.eps = c.get("layer_norm_eps", 1e-5)
        self.parallel = c.get("use_parallel_residual", True)
        self.inv = (1.0 / self.theta ** (2.0 * np.arange(self.rot // 2) / self.rot)).astype(np.float32)

    def rope(self, x, pos0):
        # x: (seq, nh, hd); rotary on dims [0, rot), split-half within the block
        seq = x.shape[0]
        half = self.rot // 2
        pos = (pos0 + np.arange(seq, dtype=np.float32))[:, None]  # (seq, 1)
        ang = (pos * self.inv[None, :]).astype(np.float32)  # (seq, half)
        c, s = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
        a, b = x[..., :half].copy(), x[..., half:self.rot].copy()
        x[..., :half] = a * c - b * s
        x[..., half:self.rot] = b * c + a * s
        return x

    def hidden(self, ids):
        w = self.w
        seq = len(ids)
        x = w["gpt_neox.embed_in.weight"][ids]
        mask = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), 1)
        for l in range(self.nl):
            p = f"gpt_neox.layers.{l}."
            a = layernorm(x, w[p + "input_layernorm.weight"], w[p + "input_layernorm.bias"], self.eps)
            qkv = a @ w[p + "attention.query_key_value.weight"].T + w[p + "attention.query_key_value.bias"]
            qkv = qkv.reshape(seq, self.nh, 3 * self.hd)  # per-head [q,k,v]
            q, k, v = qkv[..., :self.hd].copy(), qkv[..., self.hd:2 * self.hd].copy(), qkv[..., 2 * self.hd:]
            q, k = self.rope(q, 0), self.rope(k, 0)
            attn = np.zeros((seq, self.nh, self.hd), dtype=np.float32)
            for h in range(self.nh):
                sc = (q[:, h] @ k[:, h].T / np.float32(math.sqrt(self.hd)) + mask).astype(np.float32)
                attn[:, h] = softmax(sc) @ v[:, h]
            attn_res = attn.reshape(seq, self.d) @ w[p + "attention.dense.weight"].T + w[p + "attention.dense.bias"]
            mlp_in = x if self.parallel else x + attn_res
            a2 = layernorm(mlp_in, w[p + "post_attention_layernorm.weight"], w[p + "post_attention_layernorm.bias"], self.eps)
            hm = gelu(a2 @ w[p + "mlp.dense_h_to_4h.weight"].T + w[p + "mlp.dense_h_to_4h.bias"])
            mlp = hm @ w[p + "mlp.dense_4h_to_h.weight"].T + w[p + "mlp.dense_4h_to_h.bias"]
            x = (x + attn_res + mlp).astype(np.float32)
        return layernorm(x, w["gpt_neox.final_layer_norm.weight"], w["gpt_neox.final_layer_norm.bias"], self.eps)

    def predict(self, ids):
        xf = self.hidden(ids)
        logits = xf[-1] @ self.w["embed_out.weight"].T
        return int(np.argmax(logits))


def main():
    args = sys.argv[1:]
    model_dir, holdout = args[0], args[1]
    ctx = int(args[args.index("--ctx") + 1]) if "--ctx" in args else 64
    n = int(args[args.index("--n") + 1]) if "--n" in args else 60
    dump = args[args.index("--dump") + 1] if "--dump" in args else None
    j = json.load(open(holdout))
    ids = j["holdout_ids"] if isinstance(j, dict) else j
    m = NeoxRef(model_dir)
    preds = []
    for i in range(ctx, min(ctx + n, len(ids))):
        preds.append(m.predict(ids[i - ctx:i]))
        print(f"\r{len(preds)} positions", end="", file=sys.stderr)
    print("", file=sys.stderr)
    out = "".join(f"{p}\n" for p in preds)
    if dump:
        open(dump, "w").write(out)
        print(f"wrote {len(preds)} predictions to {dump}", file=sys.stderr)
    else:
        print(out, end="")


if __name__ == "__main__":
    main()
