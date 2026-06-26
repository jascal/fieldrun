#!/usr/bin/env python
"""Build a *tiny* random-init Gemma4ForCausalLM (the real `gemma4_text` arch, shrunk dims), dump a faithful
whole-model reference (logits + per-layer hidden states), and save the checkpoint so `fieldrun convert --arch gemma4`
can read it. The Python side of the verify-before-trust harness for fieldrun's gemma4 port (mirrors the qwen3next one).

Variants (CLI flags) exercise the Gemma-4-specific paths incrementally, so a parity break localizes to one feature:
  (base)            dense FFN, k_eq_v off, no KV-sharing  — the backbone (PLE, value-norm, per-type head_dim, proportional RoPE)
  --k-eq-v          attention_k_eq_v: global layers drop v_proj, V = value-normed k_proj output, nkv_g KV heads
  --kv-shared N     last N layers reuse an earlier same-type layer's K/V (no own k/v/k_norm)
  --moe             enable_moe_block: dense MLP + summed sigmoid-free top-k expert branch (router scale + per-expert scale)

Re-inits ALL params (incl. RMSNorm weights, to NON-1.0 values) + the per-layer `layer_scalar` buffers from a fixed seed,
so the (1+w)-vs-w RMSNorm distinction and the layer_scalar multiply are both actually exercised (degenerate ones would hide them).
Needs transformers>=5.12 (has Gemma4). Run in the harness venv.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import Gemma4TextConfig, Gemma4ForCausalLM


def build_config(args):
    nl = args.layers
    # explicit layer_types: alternate sliding, force the LAST to full (Gemma4 requires it) + at least one *non-last* full
    # so the global-head_dim / proportional-RoPE path runs on more than the final layer.
    lt = ["sliding_attention"] * nl
    for i in range(nl):
        if (i + 1) % 2 == 0:        # every 2nd layer full → exercises per-type head_dim + window interleaving
            lt[i] = "full_attention"
    lt[-1] = "full_attention"
    cfg = Gemma4TextConfig(
        vocab_size=args.vocab,
        vocab_size_per_layer_input=args.vocab,
        hidden_size=args.d,
        intermediate_size=args.ffn,
        num_hidden_layers=nl,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
        global_head_dim=args.global_head_dim,
        num_global_key_value_heads=args.global_kv_heads,
        hidden_size_per_layer_input=args.ple,
        sliding_window=args.window,
        layer_types=lt,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        tie_word_embeddings=True,
        attention_k_eq_v=args.k_eq_v,
        num_kv_shared_layers=args.kv_shared,
        enable_moe_block=args.moe,
        num_experts=args.experts if args.moe else None,
        top_k_experts=args.topk if args.moe else None,
        moe_intermediate_size=args.moe_inter if args.moe else None,
        max_position_embeddings=512,
    )
    return cfg


def reinit(model, seed=0):
    """Fill every parameter + the layer_scalar buffers with deterministic non-degenerate values."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and name.endswith("norm.weight"):
                # RMSNorm weights: spread around ~1 but clearly != 1 and != (1+x) ambiguous — catches the (1+w) vs w bug
                p.copy_(0.5 + 0.6 * torch.rand(p.shape, generator=g))
            elif "per_expert_scale" in name or name.endswith("router.scale"):
                p.copy_(0.5 + 0.6 * torch.rand(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
        # layer_scalar: a per-layer buffer (default 1.0) applied as the LAST op of each decoder layer — make it != 1.0
        for i, layer in enumerate(model.model.layers):
            layer.layer_scalar.copy_(torch.tensor([0.85 + 0.03 * i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/tiny_g4")
    ap.add_argument("--seq", type=int, default=12)
    ap.add_argument("--vocab", type=int, default=100)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--ffn", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--global-head-dim", type=int, default=32)
    ap.add_argument("--global-kv-heads", type=int, default=2)
    ap.add_argument("--ple", type=int, default=16)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--k-eq-v", action="store_true")
    ap.add_argument("--kv-shared", type=int, default=0)
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--moe-inter", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = build_config(args)
    torch.manual_seed(args.seed)
    model = Gemma4ForCausalLM(cfg).eval()
    reinit(model, args.seed)

    rng = np.random.default_rng(args.seed)
    ids = rng.integers(0, args.vocab, size=args.seq).tolist()
    input_ids = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    logits = out.logits[0].float().numpy()                          # (seq, vocab)
    hs = np.stack([h[0].float().numpy() for h in out.hidden_states])  # (n_hs, seq, d)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    np.savez(os.path.join(args.out, "ref.npz"), logits=logits, hidden_states=hs, ids=np.array(ids))
    with open(os.path.join(args.out, "ref_ids.json"), "w") as f:
        json.dump({"holdout_ids": ids}, f)  # fieldrun --ids expects {"holdout_ids": [...]}

    print(f"saved tiny gemma4 -> {args.out}")
    print(f"  layer_types = {cfg.layer_types}")
    print(f"  k_eq_v={cfg.attention_k_eq_v} kv_shared={cfg.num_kv_shared_layers} moe={cfg.enable_moe_block}")
    print(f"  logits {logits.shape}  hidden_states {hs.shape} (n_hs={hs.shape[0]} = embed + {args.layers} layers)")
    print(f"  ids = {ids}")


if __name__ == "__main__":
    main()
