#!/usr/bin/env python3
"""Faithfulness gate for the fieldrun `gemma3` arch.

Builds a *tiny* random-init Gemma3ForCausalLM (transformers), sized to exercise every code path the real models use —
both sliding-window (local) and full (global) layers, GQA, QK-norm, dual-base RoPE, window masking — saves it as
safetensors+config (so `fieldrun convert --arch gemma3` reads it like any HF checkpoint), and dumps the torch
reference's top-1 next-token prediction for each position over a random id stream, using the *same* fixed-window
context the fieldrun scoring loop uses. A second invocation (`compare`) checks fieldrun's --dump against it.

No gated download needed: the architecture math is what we validate, and a tiny instance exercises it identically to
the full model. Run with the torch venv:  lm-sae/.venv/bin/python scripts/gemma3_ref.py build|compare ...
"""
import json
import sys

ARCH = sys.argv[2] if len(sys.argv) >= 3 and sys.argv[1] == "build" else (
    sys.argv[3] if len(sys.argv) >= 4 else "gemma3")
OUT_DIR = f"/tmp/{ARCH}tiny"
IDS_PATH = f"/tmp/{ARCH}_holdout.json"
REF_PATH = f"/tmp/{ARCH}_torch_preds.json"
CTX = 16
N_EVAL = 60
SEQ_LEN = 96


def make_config():
    """A tiny config that exercises every path the real model uses."""
    import torch

    if ARCH == "gemma4":
        from transformers import Gemma4TextConfig
        return Gemma4TextConfig(
            vocab_size=64,
            vocab_size_per_layer_input=64,
            hidden_size=32,
            hidden_size_per_layer_input=8,   # PLE dim
            intermediate_size=64,
            num_hidden_layers=7,             # 0-4 sliding, 5 full, 6 forced full -> both paths + per-layer-type head_dim
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,                      # sliding head_dim
            global_head_dim=16,              # global layers use a DIFFERENT head_dim
            sliding_window=4,
            rms_norm_eps=1e-6,
            max_position_embeddings=256,
            tie_word_embeddings=True,
            enable_moe_block=False,
            attention_k_eq_v=False,
            num_kv_shared_layers=0,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
    from transformers import Gemma3TextConfig
    return Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=7,      # layer 5 -> full ((5+1)%6==0); 0-4,6 sliding -> both paths exercised
        num_attention_heads=4,
        num_key_value_heads=2,    # GQA, rep=2
        head_dim=8,
        sliding_window=4,         # small so seq>window actually masks in the sliding layers
        query_pre_attn_scalar=8,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        attn_implementation="eager",  # plain matmul+softmax: exactly what the Rust kernel implements
        torch_dtype=torch.float32,
    )


def build():
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = make_config()
    Cls = __import__("transformers").Gemma4ForCausalLM if ARCH == "gemma4" else __import__("transformers").Gemma3ForCausalLM
    model = Cls(cfg).eval()
    # randomise the norm weights too (they init to 0 -> (1+w)=1, which would hide a QK-norm/4-norm bug)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(torch.randn_like(p) * 0.1)
    model.save_pretrained(OUT_DIR, safe_serialization=True)

    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (SEQ_LEN,), generator=g).tolist()
    json.dump({"holdout_ids": ids}, open(IDS_PATH, "w"))

    end = min(CTX + N_EVAL, len(ids))
    preds = []
    with torch.no_grad():
        for i in range(CTX, end):
            ctx = ids[max(0, i - CTX):i]              # fieldrun's fixed-window context (positions 0..len)
            inp = torch.tensor([ctx], dtype=torch.long)
            logits = model(input_ids=inp).logits[0, -1]
            preds.append(int(logits.argmax()))
    json.dump({"preds": preds, "ctx": CTX, "n": len(preds)}, open(REF_PATH, "w"))
    print(f"[gemma3_ref] saved tiny model to {OUT_DIR}; {len(preds)} torch reference predictions -> {REF_PATH}")
    print(f"[gemma3_ref] layer_types = {cfg.layer_types}")


def compare(dump_path):
    ref = json.load(open(REF_PATH))["preds"]
    rust = [int(x) for x in open(dump_path).read().split()]
    n = min(len(ref), len(rust))
    agree = sum(1 for a, b in zip(ref[:n], rust[:n]) if a == b)
    cls = "Gemma4ForCausalLM" if ARCH == "gemma4" else "Gemma3ForCausalLM"
    print(f"[gemma3_ref] fieldrun vs torch {cls}: {agree}/{n} top-1 agree ({100*agree/n:.1f}%)")
    if agree != n:
        mism = [(i, ref[i], rust[i]) for i in range(n) if ref[i] != rust[i]][:10]
        print(f"[gemma3_ref] first mismatches (pos, torch, rust): {mism}")
    sys.exit(0 if agree == n else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "compare":
        compare(sys.argv[2])
    else:
        build()
