#!/usr/bin/env python3
"""Build a TINY but architecturally-faithful Qwen3.6 / Qwen3-Next model + dump a reference forward.

The trick for verifying a new arch off the big machine: keep the *ops* identical, shrink the *dims*. We
start from the REAL config (so the layer-type pattern, gating, expert layout, norms, partial-RoPE, MTP head
are all faithful), shrink hidden/layers/experts/vocab to toy sizes, random-init, and dump (input_ids →
logits) + per-layer hidden states. The fieldrun Rust port must reproduce these (see compare.py); a 4-layer
hidden-64 toy exercises the exact code path the 35B uses — correctness is size-independent.

Requires: transformers recent enough to include the Qwen3-Next/Qwen3.6 model class (model_type 'qwen3_next'
or 'qwen3.6'). This box doesn't have it yet — `pip install -U transformers` (or the Qwen-provided build),
then run.  Usage: python make_tiny_qwen3next.py [--repo Qwen/Qwen3.6-35B-A3B] [--layers 4] [--out tiny/]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--layers", type=int, default=4)        # keep ≥4 so both DeltaNet and full-attn layers appear
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)         # long-ish: exercises DeltaNet state over many steps
    ap.add_argument("--out", default="experiments/qwen3next/tiny")
    a = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    # 1) start from the REAL config so every arch-defining field is faithful, then shrink only the SIZES.
    cfg = AutoConfig.from_pretrained(a.repo, trust_remote_code=True)
    print(f"[tiny] base config model_type={getattr(cfg, 'model_type', '?')}")
    shrink = {  # size fields — names vary by version; set if present (leaves the arch pattern intact)
        "num_hidden_layers": a.layers, "hidden_size": a.hidden, "intermediate_size": 2 * a.hidden,
        "vocab_size": a.vocab, "num_experts": a.experts, "n_routed_experts": a.experts,
        "moe_intermediate_size": 2 * a.hidden, "head_dim": 16,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 16,
        "linear_value_head_dim": 16, "max_position_embeddings": max(a.seq, 1024),
    }
    for k, val in shrink.items():
        if hasattr(cfg, k):
            setattr(cfg, k, val)
    # keep MoE actually routing at toy scale (a couple of experts active + the shared one if present)
    for k, val in (("num_experts_per_tok", 2), ("moe_topk", 2), ("decoder_sparse_step", 1)):
        if hasattr(cfg, k):
            setattr(cfg, k, val)
    os.makedirs(a.out, exist_ok=True)
    json.dump(cfg.to_dict(), open(f"{a.out}/config.json", "w"), indent=1)

    # 2) random-init the real architecture at toy size (fixed seed -> reproducible reference)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval().float()
    model.save_pretrained(a.out, safe_serialization=True)        # safetensors for `fieldrun convert`
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[tiny] built {nparam/1e6:.2f}M-param toy at {a.out} (layers={a.layers} hidden={a.hidden})")

    # 3) reference forward in f32 (tight tolerance) — dump logits + every layer's hidden state
    rng = np.random.default_rng(0)
    ids = torch.tensor(rng.integers(0, a.vocab, size=(1, a.seq)), dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    hs = np.stack([h[0].float().numpy() for h in out.hidden_states])   # (L+1, seq, hidden)
    np.savez(f"{a.out}/ref.npz", input_ids=ids[0].numpy(),
             logits=out.logits[0].float().numpy(), hidden_states=hs)
    json.dump({"holdout_ids": ids[0].tolist()}, open(f"{a.out}/ref_ids.json", "w"))  # for fieldrun --ids
    print(f"[tiny] wrote ref.npz: logits {tuple(out.logits[0].shape)}, hidden_states {hs.shape}")
    print("[tiny] next: `fieldrun convert` this dir, run with the per-layer debug dump, then compare.py")


if __name__ == "__main__":
    main()
