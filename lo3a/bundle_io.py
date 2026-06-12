"""Read/write fieldrun rope bundles + a parametric numpy forward (mirrors src/rope.rs, f32).
Shared by reduce.py. Orientation per FORMAT.md / bundle.rs: 2D weights stored [in, out] row-major;
embed/lm_head stored [vocab, d] row-per-token. config = [n_layer,H,nkv,hd,d,ffn,vocab,tied]."""
import json, os
import numpy as np

def read_bundle(stem):
    with open(stem + ".fieldrun.json") as f: man = json.load(f)
    blob = open(stem + ".fieldrun.bin", "rb").read()
    W = {}
    for a in man["arrays"]:
        assert a["dtype"] == "f32", f"reduce.py handles f32 bundles only (got {a['dtype']} for {a['name']})"
        n = int(np.prod(a["shape"]))
        v = np.frombuffer(blob, dtype="<f4", count=n, offset=a["offset"]).reshape(a["shape"]).astype(np.float32)
        W[a["name"]] = v.copy()
    return man, W

def write_bundle(stem, arch, config, config_f, W, order):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    blob = bytearray(); arrays = []; off = 0
    for name in order:
        a = np.ascontiguousarray(W[name], dtype="<f4"); b = a.tobytes(); blob += b
        arrays.append({"name": name, "dtype": "f32", "shape": list(a.shape), "offset": off, "bytes": len(b)})
        off += len(b)
    man = {"format": "fieldrun-bundle", "version": 1, "arch": arch,
           "config": list(config), "config_f": list(config_f), "arrays": arrays}
    json.dump(man, open(stem + ".fieldrun.json", "w"), indent=2)
    open(stem + ".fieldrun.bin", "wb").write(bytes(blob))
    return off

def layer_order(n_layer, tied, bias):
    order = ["embed"] + ([] if tied else ["lm_head"])
    for l in range(n_layer):
        p = f"l{l}."
        order += [p+"in_ln", p+"self_attn.q_proj", p+"self_attn.k_proj", p+"self_attn.v_proj"]
        if bias: order += [p+"self_attn.q_proj.bias", p+"self_attn.k_proj.bias", p+"self_attn.v_proj.bias"]
        order += [p+"self_attn.o_proj", p+"post_ln", p+"mlp.gate_proj", p+"mlp.up_proj", p+"mlp.down_proj"]
    return order + ["norm"]

# ---- parametric forward (f32), returns final-position logits and per-layer SwiGLU activations ----
def _rmsnorm(x, w, eps):
    ms = (x.astype(np.float32) ** 2).mean(axis=1, keepdims=True)
    return (x * (1.0 / np.sqrt(ms + np.float32(eps))).astype(np.float32) * w).astype(np.float32)

def _silu(x): return (x / (1.0 + np.exp(-x))).astype(np.float32)

def forward(W, cfg, cfg_f, ids, want_acts=False):
    n_layer, H, NKV, HD, D, FFN, VOCAB, TIED = [int(c) for c in cfg]
    theta, eps = float(cfg_f[0]), float(cfg_f[1])
    HALF, REP = HD // 2, H // NKV
    inv = (1.0 / (theta ** (2.0 * np.arange(HALF, dtype=np.float32) / HD))).astype(np.float32)
    bias = (f"l0.self_attn.q_proj.bias" in W)
    def rope(x, nh):
        x = x.copy(); seq = x.shape[0]
        for i in range(seq):
            for h in range(nh):
                b = h * HD
                for j in range(HALF):
                    ang = np.float32(i) * inv[j]; c, s = np.float32(np.cos(ang)), np.float32(np.sin(ang))
                    a, bb = x[i, b+j], x[i, b+j+HALF]
                    x[i, b+j] = a*c - bb*s; x[i, b+j+HALF] = bb*c + a*s
        return x.astype(np.float32)
    ids = list(ids); seq = len(ids)
    x = W["embed"][ids].astype(np.float32)
    acts = []
    for l in range(n_layer):
        p = f"l{l}."
        a = _rmsnorm(x, W[p+"in_ln"], eps)
        q = (a @ W[p+"self_attn.q_proj"]).astype(np.float32)
        k = (a @ W[p+"self_attn.k_proj"]).astype(np.float32)
        v = (a @ W[p+"self_attn.v_proj"]).astype(np.float32)
        if bias:
            q += W[p+"self_attn.q_proj.bias"]; k += W[p+"self_attn.k_proj.bias"]; v += W[p+"self_attn.v_proj.bias"]
        q, k = rope(q, H), rope(k, NKV)
        ao = np.zeros((seq, H*HD), dtype=np.float32)
        for h in range(H):
            kv = h // REP
            qh, kh, vh = q[:, h*HD:(h+1)*HD], k[:, kv*HD:(kv+1)*HD], v[:, kv*HD:(kv+1)*HD]
            sc = (qh @ kh.T).astype(np.float32) / np.float32(np.sqrt(HD))
            for i in range(seq):
                for j in range(i+1, seq): sc[i, j] = np.float32(-1e30)
            sc = np.exp(sc - sc.max(axis=1, keepdims=True)).astype(np.float32)
            sc = (sc / sc.sum(axis=1, keepdims=True)).astype(np.float32)
            ao[:, h*HD:(h+1)*HD] = (sc @ vh).astype(np.float32)
        x = (x + ao @ W[p+"self_attn.o_proj"]).astype(np.float32)
        a2 = _rmsnorm(x, W[p+"post_ln"], eps)
        hid = (_silu(a2 @ W[p+"mlp.gate_proj"]) * (a2 @ W[p+"mlp.up_proj"])).astype(np.float32)
        if want_acts: acts.append(hid[-1].copy())   # last-position SwiGLU activation per neuron
        x = (x + hid @ W[p+"mlp.down_proj"]).astype(np.float32)
    xf = _rmsnorm(x, W["norm"], eps)
    unemb = W["lm_head"] if TIED == 0 else W["embed"]
    logits = (xf[-1] @ unemb.T).astype(np.float32)
    return (logits, acts) if want_acts else logits

def predict(W, cfg, cfg_f, ids):
    return int(np.argmax(forward(W, cfg, cfg_f, ids)))
