#!/usr/bin/env python3
"""Build a TINY but architecturally-faithful Qwen3.6 (model_type `qwen3_5_moe`) text model + dump a reference.

Verifying a new arch off the big machine: keep the *ops* identical, shrink the *dims*. We start from the
REAL config (so layer_types pattern, the short conv, gating, expert layout, partial-RoPE, MTP are faithful),
shrink hidden/layers/experts/vocab to toy sizes, random-init, and dump (input_ids → logits) + per-layer
hidden states. A 4-layer hidden-64 toy exercises the exact code path the 35B uses.

Confirmed arch (from transformers, see README): Qwen3.6-35B-A3B is `qwen3_5_moe` with a NESTED text_config;
text path is a hybrid — `layer_types` = 3×linear_attention → 1×full_attention; the linear path has a short
causal conv (`linear_conv_kernel_dim=4`) then Gated DeltaNet; MoE 256 experts (8 routed + shared); 1 MTP.

Requires transformers with the Qwen3_5Moe class (>=5.12). Usage:
  python make_tiny_qwen3next.py [--repo Qwen/Qwen3.6-35B-A3B] [--layers 4] [--out experiments/qwen3next/tiny]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def shrink_text_config(tc, layers, hidden, vocab, experts):
    """Shrink the SIZE fields of the nested text config, preserving the hybrid arch (layer_types pattern,
    short conv, MoE routing, MTP). Field names are the real ones confirmed from the HF config."""
    # rebuild layer_types to `layers` entries following the model's own 3:1 (linear:full) period
    if getattr(tc, "layer_types", None):
        period = tc.layer_types[: (tc.layer_types.index("full_attention") + 1)] or tc.layer_types[:4]
        tc.layer_types = [period[i % len(period)] for i in range(layers)]
    size = {
        "num_hidden_layers": layers, "hidden_size": hidden, "intermediate_size": 2 * hidden,
        "vocab_size": vocab, "head_dim": 16, "num_attention_heads": 4, "num_key_value_heads": 2,
        "num_experts": experts, "num_experts_per_tok": 2, "moe_intermediate_size": 2 * hidden,
        "shared_expert_intermediate_size": 2 * hidden, "decoder_sparse_step": 1,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 16, "linear_value_head_dim": 16,   # linear_conv_kernel_dim kept as-is (faithful)
        "mtp_num_hidden_layers": 1,
    }
    for k, v in size.items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    return tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--layers", type=int, default=4)           # ≥4 so both linear- and full-attention layers appear
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)            # long-ish: exercises DeltaNet state over many steps
    ap.add_argument("--out", default="experiments/qwen3next/tiny")
    a = ap.parse_args()

    import torch
    import transformers as T

    cfg = T.AutoConfig.from_pretrained(a.repo, trust_remote_code=True)
    tc = getattr(cfg, "text_config", None) or getattr(cfg, "thinker_config", None) or cfg
    print(f"[tiny] base model_type={cfg.model_type} | text model_type={getattr(tc,'model_type','?')} "
          f"layers={getattr(tc,'num_hidden_layers','?')} experts={getattr(tc,'num_experts','?')}")
    shrink_text_config(tc, a.layers, a.hidden, a.vocab, a.experts)
    print(f"[tiny] toy layer_types={getattr(tc,'layer_types','n/a')} conv_kernel={getattr(tc,'linear_conv_kernel_dim','n/a')}")

    # instantiate the TEXT causal-LM at toy size (random, fixed seed)
    cls = getattr(T, "Qwen3_5MoeForCausalLM", None) or T.AutoModelForCausalLM
    torch.manual_seed(0)
    model = (cls(tc) if cls is not T.AutoModelForCausalLM
             else T.AutoModelForCausalLM.from_config(tc, trust_remote_code=True)).eval().float()
    os.makedirs(a.out, exist_ok=True)
    json.dump(tc.to_dict(), open(f"{a.out}/config.json", "w"), indent=1)
    model.save_pretrained(a.out, safe_serialization=True)
    print(f"[tiny] built {sum(p.numel() for p in model.parameters())/1e6:.2f}M-param toy at {a.out}")

    rng = np.random.default_rng(0)
    ids = torch.tensor(rng.integers(0, a.vocab, size=(1, a.seq)), dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    hs = np.stack([h[0].float().numpy() for h in out.hidden_states])
    np.savez(f"{a.out}/ref.npz", input_ids=ids[0].numpy(),
             logits=out.logits[0].float().numpy(), hidden_states=hs)
    json.dump({"holdout_ids": ids[0].tolist()}, open(f"{a.out}/ref_ids.json", "w"))
    print(f"[tiny] wrote ref.npz: logits {tuple(out.logits[0].shape)}, hidden_states {hs.shape}")
    print("[tiny] next: implement the arch in Rust, `fieldrun convert` this dir, dump per-layer, compare.py")


if __name__ == "__main__":
    main()
