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
import os
import sys

SEED = int(os.environ.get("SEED", "0"))        # vary for the quality sweep (scripts/bench.sh)
N_EVAL_ENV = int(os.environ.get("N_EVAL", "0"))  # 0 -> default
ARCH = sys.argv[2] if len(sys.argv) >= 3 and sys.argv[1] == "build" else (
    sys.argv[3] if len(sys.argv) >= 4 else "gemma3")
OUT_DIR = f"/tmp/{ARCH}tiny"
IDS_PATH = f"/tmp/{ARCH}_holdout.json"
REF_PATH = f"/tmp/{ARCH}_torch_preds.json"
CTX = 16
N_EVAL = N_EVAL_ENV or 60
SEQ_LEN = CTX + N_EVAL + 4


def make_config():
    """A tiny config that exercises every path the real model uses."""
    import torch

    if ARCH == "qwen3":
        from transformers import Qwen3Config
        # Qwen3 DENSE (Qwen3-4B/8B): the RoPE family + QK-norm (per-head RMSNorm on q/k) — exercises the optional
        # q_norm/k_norm path added to the `rope` arch.
        return Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
            tie_word_embeddings=False, max_position_embeddings=256,
            attn_implementation="eager", torch_dtype=torch.float32,
        )
    if ARCH == "minimax":
        from transformers import MiniMaxM2Config
        return MiniMaxM2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=16,          # expert width
            num_hidden_layers=3,           # all-MoE
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=8,
            num_experts_per_tok=2,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            max_position_embeddings=256,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
    if ARCH.startswith("mla"):
        from transformers import DeepseekV3Config
        yarn = ARCH == "mlayarn"
        extra = {}
        if yarn:
            # YaRN + a deliberately LARGE init: with the default 0.02 init the interleave-vs-split-half rotary
            # difference is ~1e-2 in the logits and rarely flips top-1 — a big init makes this column a
            # discriminating gate for the rotary details (yarn ramp, mscale factors, de-interleave), not a lucky pass.
            extra = dict(
                initializer_range=0.2,
                max_position_embeddings=128,   # = 4 x original_max -> implicit factor matches the explicit one
                rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                                 "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 0.8, "mscale_all_dim": 0.5,
                                 "original_max_position_embeddings": 32},
            )
        return DeepseekV3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,          # dense layers
            moe_intermediate_size=16,      # expert + shared-expert width
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,            # qk_head_dim = 12
            v_head_dim=8,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            n_group=4,                     # 2 experts/group
            topk_group=2,                  # -> 4 eligible, then top-2 (exercises group limiting)
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
            first_k_dense_replace=1,       # layer 0 dense, 1-3 MoE -> both paths
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            attn_implementation="eager",
            torch_dtype=torch.float32,
            **({"max_position_embeddings": 256} if not yarn else {}),
            **extra,
        )
    if ARCH.startswith("qwen3moe"):
        from transformers import Qwen3MoeConfig
        swa = ARCH == "qwen3moeswa"
        return Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,          # dense (mlp_only) layers
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            attention_bias=False,
            use_sliding_window=swa,
            sliding_window=4 if swa else None,  # small so seq>window actually masks (all layers in Qwen3)
            decoder_sparse_step=2,         # MoE on layers 1,3; dense on 0,2 -> both paths
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            norm_topk_prob=True,           # exercise the top-k renorm
            max_position_embeddings=256,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
    if ARCH.startswith("gemma4"):
        from transformers import Gemma4TextConfig
        is_moe = ARCH == "gemma4moe"
        is_keqv = ARCH == "gemma4keqv"  # global layers: V = K (no v_proj), num_global_key_value_heads KV heads
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
            enable_moe_block=is_moe,
            num_experts=4 if is_moe else None,
            top_k_experts=2 if is_moe else None,
            moe_intermediate_size=16 if is_moe else None,
            attention_k_eq_v=is_keqv,
            num_global_key_value_heads=1 if is_keqv else None,  # 1 < num_key_value_heads(2) → discriminates nkv_g
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

    torch.manual_seed(SEED)
    cfg = make_config()
    tf = __import__("transformers")
    Cls = (tf.Qwen3ForCausalLM if ARCH == "qwen3"
           else tf.MiniMaxM2ForCausalLM if ARCH == "minimax"
           else tf.DeepseekV3ForCausalLM if ARCH.startswith("mla")
           else tf.Qwen3MoeForCausalLM if ARCH.startswith("qwen3moe")
           else tf.Gemma4ForCausalLM if ARCH.startswith("gemma4")
           else tf.Gemma3ForCausalLM)
    model = Cls(cfg).eval()
    # randomise the norm weights too (they init to 0 -> (1+w)=1, which would hide a QK-norm/4-norm bug).
    # mlayarn uses mean-1 norms: mean-0 norms shrink q/k so attention is near-uniform and rotary details can't
    # flip top-1 — mean-1 keeps attention sharp, making the column a discriminating gate for the YaRN ramp,
    # the mscale factors, and the rope de-interleave (measured: a wrong interleave agrees only ~11/60).
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if ARCH in ("mlayarn", "qwen3", "gemma4keqv") else 0.0))
            # Identity-init router scales (Gemma-4 MoE): mean-1 so the `*scale`/`*per_expert_scale` terms become
            # discriminating — at the ones() default a bug in either is invisible.
            elif name.endswith(("router.scale", "router.per_expert_scale")):
                p.copy_(torch.randn_like(p) * 0.1 + 1.0)
        # Identity-init BUFFERS that the kernels read from the checkpoint but that init to a no-op value, so the
        # 60/60 gate can't otherwise discriminate the code paths that consume them:
        #   layer_scalar (Gemma-4, per-layer ×scalar, inits 1.0) and e_score_correction_bias (DeepSeek/MiniMax
        #   sigmoid-router CHOICE bias, inits 0.0). NB: do NOT touch rotary inv_freq buffers — the kernels recompute
        #   those from theta, so randomising them would (correctly) diverge.
        for name, buf in model.named_buffers():
            if name.endswith("layer_scalar"):
                buf.copy_(torch.randn_like(buf) * 0.1 + 1.0)          # mean-1 ×scalar per layer
            elif name.endswith("e_score_correction_bias"):
                buf.copy_(torch.randn_like(buf) * 0.2)               # mean-0 bias — shifts the top-k CHOICE
    model.save_pretrained(OUT_DIR, safe_serialization=True)

    g = torch.Generator().manual_seed(1000 + SEED)
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
    print(f"[gemma3_ref] layer_types = {getattr(cfg, 'layer_types', '(n/a)')}")


def compare(dump_path):
    ref = json.load(open(REF_PATH))["preds"]
    rust = [int(x) for x in open(dump_path).read().split()]
    n = min(len(ref), len(rust))
    agree = sum(1 for a, b in zip(ref[:n], rust[:n]) if a == b)
    cls = ("Qwen3ForCausalLM" if ARCH == "qwen3"
           else "MiniMaxM2ForCausalLM" if ARCH == "minimax"
           else "DeepseekV3ForCausalLM" if ARCH.startswith("mla")
           else "Qwen3MoeForCausalLM" if ARCH.startswith("qwen3moe")
           else "Gemma4ForCausalLM" if ARCH.startswith("gemma4") else "Gemma3ForCausalLM")
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
