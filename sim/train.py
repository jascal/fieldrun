#!/usr/bin/env python3
"""Train an unrealistically small RoPE (Llama-style) language model on the Threx
corpus, then export it three ways:

  1. sim/data/hf/{config.json,model.safetensors}  — HF layout, so
     `fieldrun convert --arch rope` reads it (the faithfulness gate runs here).
  2. sim/data/weights.json  — every weight as plain arrays, for the in-browser
     JavaScript forward pass (the live runnable model on the site).
  3. sim/data/probes.json   — the three example contexts + a validation batch of
     (context -> logits) so we can later check JS == torch == fieldrun.

The architecture is exactly HF Llama, minimal: RMSNorm, rotary (rotate_half),
multi-head attention (no GQA), SwiGLU MLP, tied embed/unembed, no biases — the
conventions fieldrun's `rope` kernel mirrors. Pure PyTorch (no transformers).
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threx

# ---- tiny config (everything legible on screen) ---------------------------
D = 32          # hidden size
NL = 2          # layers (>=2 so an induction/copy circuit can form)
NH = 4          # heads
HD = D // NH    # head_dim = 8
FFN = 64        # SwiGLU intermediate
CTX = 24        # context window
VOCAB = threx.VOCAB
THETA = 10000.0
EPS = 1e-6
torch.manual_seed(0)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
        return x * self.weight


def rope_cossin(L):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))   # [HD/2]
    m = torch.arange(L).float()
    f = torch.outer(m, inv)                                        # [L, HD/2]
    emb = torch.cat([f, f], -1)                                    # [L, HD]
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat([-x2, x1], -1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ln = RMSNorm(D)
        self.q = nn.Linear(D, NH * HD, bias=False)
        self.k = nn.Linear(D, NH * HD, bias=False)
        self.v = nn.Linear(D, NH * HD, bias=False)
        self.o = nn.Linear(NH * HD, D, bias=False)
        self.post_ln = RMSNorm(D)
        self.gate = nn.Linear(D, FFN, bias=False)
        self.up = nn.Linear(D, FFN, bias=False)
        self.down = nn.Linear(FFN, D, bias=False)

    def forward(self, x, cos, sin, mask):
        B, L, _ = x.shape
        h = self.in_ln(x)
        q = self.q(h).view(B, L, NH, HD).transpose(1, 2)   # [B,NH,L,HD]
        k = self.k(h).view(B, L, NH, HD).transpose(1, 2)
        v = self.v(h).view(B, L, NH, HD).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HD)
        att = att + mask[:, :, :L, :L]
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, L, NH * HD)
        x = x + self.o(o)
        h = self.post_ln(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class TinyLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.norm = RMSNorm(D)
        cos, sin = rope_cossin(CTX)
        self.register_buffer("cos", cos[None, None])   # [1,1,L,HD]
        self.register_buffer("sin", sin[None, None])
        m = torch.full((CTX, CTX), float("-inf")).triu(1)
        self.register_buffer("mask", m[None, None])

    def forward(self, ids):
        L = ids.shape[1]
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x, self.cos[:, :, :L], self.sin[:, :, :L], self.mask)
        x = self.norm(x)
        return x @ self.embed.weight.T            # tied unembed


def batches(ids, bs, L, steps, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids)
    hi = len(ids) - L - 1
    for _ in range(steps):
        s = rng.integers(0, hi, size=bs)
        xb = np.stack([ids[i:i + L] for i in s])
        yb = np.stack([ids[i + 1:i + 1 + L] for i in s])
        yield torch.tensor(xb), torch.tensor(yb)


def main():
    global NL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=NL)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--wd", type=float, default=0.01)   # raise for grokking
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    NL = a.layers
    torch.manual_seed(a.seed)

    data = json.load(open("sim/data/corpus.json"))["ids"]
    lex = json.load(open("sim/data/lexicon.json"))
    model = TinyLlama()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    STEPS, BS = a.steps, a.bs
    print(f"config: layers={NL} d={D} heads={NH} ffn={FFN} ctx={CTX} "
          f"steps={STEPS} lr={a.lr}")
    K = len(threx.BEARINGS)
    CELLS = [(i, j) for i in range(K) for j in range(K)]

    def tri_acc():   # accuracy over the WHOLE far-operand triangulation table
        n = 0
        for (i, j) in CELLS:
            pre = [threx.BOS, threx.TRI, threx.BEARINGS[i], threx.BEARINGS[j],
                   threx.HUSH, threx.HUSH, threx.GI]
            with torch.no_grad():
                n += int(model(torch.tensor([pre]))[0, -1].argmax()) == threx.THINGS[i + j]
        return n

    model.train()
    for i, (xb, yb) in enumerate(batches(data, BS, CTX, STEPS, seed=a.seed)):
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if i % 1000 == 0 or i == STEPS - 1:
            model.eval(); h = tri_acc(); model.train()
            print(f"step {i:5d}  loss {loss.item():.4f}  tri-table {h}/{len(CELLS)}", flush=True)
    model.eval()

    # ---- sanity: does it produce the in-world-correct token on each example?
    print("\nexample predictions (greedy):")
    for e in lex["examples"]:
        ctx = torch.tensor([e["prefix"]])
        with torch.no_grad():
            lg = model(ctx)[0, -1]
        pred = int(lg.argmax())
        ok = "✓" if pred == e["expect"] else "✗"
        print(f"  [{e['key']:9}] {threx.render(e['prefix'])} → {threx.GLYPH[pred]} "
              f"(want {threx.GLYPH[e['expect']]}) {ok}")

    # ---- export 1: HF safetensors + config (for fieldrun convert) ----------
    os.makedirs("sim/data/hf", exist_ok=True)
    sd = model.state_dict()
    hf = {}
    hf["model.embed_tokens.weight"] = sd["embed.weight"]
    hf["model.norm.weight"] = sd["norm.weight"]
    for l in range(NL):
        p, q = f"model.layers.{l}.", f"blocks.{l}."
        hf[p + "input_layernorm.weight"] = sd[q + "in_ln.weight"]
        hf[p + "post_attention_layernorm.weight"] = sd[q + "post_ln.weight"]
        hf[p + "self_attn.q_proj.weight"] = sd[q + "q.weight"]
        hf[p + "self_attn.k_proj.weight"] = sd[q + "k.weight"]
        hf[p + "self_attn.v_proj.weight"] = sd[q + "v.weight"]
        hf[p + "self_attn.o_proj.weight"] = sd[q + "o.weight"]
        hf[p + "mlp.gate_proj.weight"] = sd[q + "gate.weight"]
        hf[p + "mlp.up_proj.weight"] = sd[q + "up.weight"]
        hf[p + "mlp.down_proj.weight"] = sd[q + "down.weight"]
    save_safetensors({k: v.float().cpu().numpy() for k, v in hf.items()},
                     "sim/data/hf/model.safetensors")
    json.dump({"model_type": "llama", "architectures": ["LlamaForCausalLM"],
               "hidden_size": D, "num_hidden_layers": NL, "num_attention_heads": NH,
               "num_key_value_heads": NH, "head_dim": HD, "intermediate_size": FFN,
               "vocab_size": VOCAB, "max_position_embeddings": CTX,
               "rope_theta": THETA, "rms_norm_eps": EPS, "tie_word_embeddings": True,
               "hidden_act": "silu"}, open("sim/data/hf/config.json", "w"))

    # ---- export 2: weights.json (for the in-browser JS engine) -------------
    def arr(t):
        return np.asarray(t.detach().cpu().float().numpy()).tolist()
    wj = {"config": {"d": D, "n_layer": NL, "n_head": NH, "head_dim": HD,
                     "ffn": FFN, "ctx": CTX, "vocab": VOCAB, "theta": THETA,
                     "eps": EPS},
          "embed": arr(sd["embed.weight"]), "norm": arr(sd["norm.weight"]),
          "layers": []}
    for l in range(NL):
        q = f"blocks.{l}."
        wj["layers"].append({k: arr(sd[q + n]) for k, n in [
            ("in_ln", "in_ln.weight"), ("post_ln", "post_ln.weight"),
            ("q", "q.weight"), ("k", "k.weight"), ("v", "v.weight"),
            ("o", "o.weight"), ("gate", "gate.weight"), ("up", "up.weight"),
            ("down", "down.weight")]})
    json.dump(wj, open("sim/data/weights.json", "w"))

    # ---- export 3: probes.json (validation contexts -> torch logits) -------
    val = []
    rng = np.random.default_rng(7)
    hi = len(data) - CTX - 1
    for s in rng.integers(0, hi, size=40):
        ctx = data[s:s + CTX]
        with torch.no_grad():
            lg = model(torch.tensor([ctx]))[0, -1]
        val.append({"ctx": [int(x) for x in ctx],
                    "logits": [float(x) for x in lg]})
    ex = []
    for e in lex["examples"]:
        with torch.no_grad():
            lg = model(torch.tensor([e["prefix"]]))[0, -1]
        ex.append({"key": e["key"], "prefix": e["prefix"],
                   "logits": [float(x) for x in lg],
                   "argmax": int(lg.argmax())})
    json.dump({"val": val, "examples": ex}, open("sim/data/probes.json", "w"))
    print("\nexported sim/data/hf, weights.json, probes.json")


def save_safetensors(tensors, path):
    """Write the minimal safetensors format (no torch/safetensors dep)."""
    import struct
    header, blob, off = {}, bytearray(), 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b; off += len(b)
    hj = json.dumps(header).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj); f.write(blob)


if __name__ == "__main__":
    main()
