#!/usr/bin/env python3
"""Export a fieldrun rope bundle -> a Hugging-Face-publishable model (safetensors + config.json),
and close the COMPLETE ROUND TRIP: bundle -> HF -> `fieldrun convert` -> bundle' -> decode-compare.

HF Llama/Qwen2 layout: nn.Linear weights are [out, in]; the fieldrun bundle stores [in, out], so the
projections are transposed on the way out. embed/lm_head are [vocab, d] in both (no transpose).
No safetensors dependency — the format is a u64 header length + JSON header + concatenated f32 blobs.
"""
import os, sys, json, struct, subprocess, re
import numpy as np
import bundle_io as bio

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(HERE, "..", "target", "release", "fieldrun")

def write_safetensors(path, tensors):
    """tensors: dict name -> np.ndarray (will be stored F32, row-major little-endian)."""
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype="<f4")
        b = a.tobytes(); n = len(b)
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + n]}
        blob += b; off += n
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hjson)) % 8                      # header must be 8-byte aligned
    hjson += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson))); f.write(hjson); f.write(bytes(blob))

def export_hf(stem, out_dir):
    man, W = bio.read_bundle(stem)
    c = man["config"]; n_layer, H, NKV, HD, D, FFN, VOCAB, TIED = [int(x) for x in c]
    theta, eps = man["config_f"]
    tied = TIED != 0
    bias = (f"l0.self_attn.q_proj.bias" in W)
    os.makedirs(out_dir, exist_ok=True)
    T = {}
    T["model.embed_tokens.weight"] = W["embed"]                 # [vocab, d]
    T["model.norm.weight"] = W["norm"]                          # [d]
    if not tied: T["lm_head.weight"] = W["lm_head"]             # [vocab, d]
    for l in range(n_layer):
        p, hp = f"l{l}.", f"model.layers.{l}."
        T[hp+"input_layernorm.weight"] = W[p+"in_ln"]
        T[hp+"post_attention_layernorm.weight"] = W[p+"post_ln"]
        for proj in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                     "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            T[hp+proj+".weight"] = W[p+proj].T                  # [in,out] -> HF [out,in]
        if bias:
            for proj in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]:
                T[hp+proj+".bias"] = W[p+proj+".bias"]
    write_safetensors(os.path.join(out_dir, "model.safetensors"), T)
    config = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_act": "silu",
        "hidden_size": D, "intermediate_size": FFN, "num_hidden_layers": n_layer,
        "num_attention_heads": H, "num_key_value_heads": NKV, "head_dim": HD,
        "vocab_size": VOCAB, "rope_theta": theta, "rms_norm_eps": eps,
        "tie_word_embeddings": tied, "torch_dtype": "float32", "max_position_embeddings": 4096,
    }
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    return config, sum(np.prod(t.shape) for t in T.values())

def fr_decode(stem, ids):
    qp = os.path.join(HERE, "_st.json"); json.dump({"holdout_ids": list(ids)+[0]}, open(qp,"w"))
    out = subprocess.run([FR,"--bundle",stem,"--ids",qp,"--ctx",str(len(ids)),"export","--logic"],
                         capture_output=True, text=True)
    m = re.search(r"model predicts: \[(\d+)\]", out.stderr+out.stdout)
    return int(m.group(1)) if m else None

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "red_small", "red_small")
    if not os.path.exists(src + ".fieldrun.json"):
        print(f"no bundle at {src}; run reduce.py first"); sys.exit(1)
    hf_dir = os.path.join(HERE, "hf_export")
    print(f"== export {os.path.basename(src)} -> HF safetensors ({hf_dir}) ==")
    cfg, nparam = export_hf(src, hf_dir)
    sz = os.path.getsize(os.path.join(hf_dir, "model.safetensors"))
    print(f"   wrote model.safetensors ({sz:,} bytes, {int(nparam):,} params) + config.json")
    print(f"   config: {cfg['architectures'][0]}  d={cfg['hidden_size']} ffn={cfg['intermediate_size']} "
          f"layers={cfg['num_hidden_layers']} heads={cfg['num_attention_heads']}/{cfg['num_key_value_heads']} vocab={cfg['vocab_size']}")

    print("== ROUND TRIP: fieldrun convert (HF safetensors -> bundle') ==")
    rt = os.path.join(HERE, "hf_roundtrip", "hf_roundtrip")
    os.makedirs(os.path.dirname(rt), exist_ok=True)
    r = subprocess.run([FR,"convert","--model",hf_dir,"--arch","rope","--dtype","f32","--out",rt], capture_output=True, text=True)
    if not os.path.exists(rt + ".fieldrun.json"):
        print("   convert FAILED:\n  " + (r.stderr or r.stdout).strip()[-800:]); sys.exit(1)
    print("   convert OK ->", os.path.basename(rt) + ".fieldrun.{json,bin}")

    print("== verify: decode(reduced bundle) == decode(round-tripped HF model) ==")
    rng = np.random.default_rng(99); ok = 0; N = 12
    for _ in range(N):
        ids = [int(t) for t in rng.integers(0, cfg["vocab_size"], size=int(rng.integers(1,12)))]
        a, b = fr_decode(src, ids), fr_decode(rt, ids)
        ok += (a == b)
        if a != b: print(f"   MISMATCH ctx={ids} reduced={a} roundtrip={b}")
    print(f"   {ok}/{N} decodes identical -> COMPLETE ROUND TRIP {'VERIFIED ✓' if ok==N else 'FAILED'}")
