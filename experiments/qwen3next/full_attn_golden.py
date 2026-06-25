#!/usr/bin/env python3
"""Isolate the fieldrun full_attn bug: diff each sub-step against transformers Qwen3_5MoeAttention.

Loads the tiny model's full-attention layer (layer 3) + its weights, runs a fixed input through both the
real transformers attention and a numpy replica of fieldrun's `full_attn`, and reports where they first
diverge — rope being the prime suspect (Qwen3.5 partial RoPE). Usage: python full_attn_golden.py <tiny-dir>
"""
import sys
import numpy as np
import torch


def main():
    tiny = sys.argv[1]
    import transformers as T
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_rotary_pos_emb
    m = T.AutoModelForCausalLM.from_pretrained(tiny, trust_remote_code=True).eval().float()
    cfg = m.config.text_config if hasattr(m.config, "text_config") else m.config
    L = next(i for i, t in enumerate(cfg.layer_types) if t == "full_attention")
    attn = m.model.layers[L].self_attn
    rot = m.model.rotary_emb
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, attn.head_dim
    seq, hidden = 8, cfg.hidden_size
    g = torch.Generator().manual_seed(1)
    h = torch.randn(1, seq, hidden, generator=g)
    pos = torch.arange(seq).unsqueeze(0)
    cos, sin = rot(h, pos)
    print(f"[golden] full layer L{L}: nh={nh} nkv={nkv} hd={hd} | cos shape {tuple(cos.shape)} "
          f"(rotary_dim≈{cos.shape[-1]}) | partial_rotary_factor={getattr(cfg,'partial_rotary_factor',1.0)}")

    # ---- transformers reference (authoritative) ----
    mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)[None, None]
    with torch.no_grad():
        ref_out, _ = attn(h, position_embeddings=(cos, sin), attention_mask=mask)
        # also grab transformers' query AFTER rope, to test rope in isolation
        qg = attn.q_proj(h).view(1, seq, -1, hd * 2)
        q_ts, _gate = torch.chunk(qg, 2, dim=-1)
        q_ts = attn.q_norm(q_ts.view(1, seq, nh, hd)).transpose(1, 2)
        k_ts = attn.k_norm(attn.k_proj(h).view(1, seq, nkv, hd)).transpose(1, 2)
        q_rot_ref, k_rot_ref = apply_rotary_pos_emb(q_ts, k_ts, cos, sin)
    ref_out = ref_out[0].numpy()

    # ---- fieldrun full_attn replica (numpy, EXACTLY the Rust logic) ----
    Wq, Wk, Wv, Wo = (attn.q_proj.weight.detach().numpy(), attn.k_proj.weight.detach().numpy(),
                      attn.v_proj.weight.detach().numpy(), attn.o_proj.weight.detach().numpy())
    qn_w, kn_w = attn.q_norm.weight.detach().numpy(), attn.k_norm.weight.detach().numpy()
    eps = cfg.rms_norm_eps
    inv = rot.inv_freq.detach().numpy().astype(np.float64) # the model's actual rope frequencies
    half = len(inv)
    rotary_dim = 2 * half
    print(f"[golden] rot.inv_freq len={half} -> rotary_dim={rotary_dim} (hd={hd})")
    hh = h[0].numpy()
    qg = hh @ Wq.T  # (seq, nh*2*hd)
    q = np.zeros((seq, nh * hd)); gate = np.zeros((seq, nh * hd))
    for t in range(seq):
        for head in range(nh):
            s = head * 2 * hd
            q[t, head * hd:head * hd + hd] = qg[t, s:s + hd]
            gate[t, head * hd:head * hd + hd] = qg[t, s + hd:s + 2 * hd]
    k = hh @ Wk.T
    v = hh @ Wv.T
    # isolate split: my q POST-split PRE-qnorm vs transformers query_states (pre-qnorm)
    with torch.no_grad():
        q_pre_ref = torch.chunk(attn.q_proj(h).view(1, seq, nh, hd * 2), 2, dim=-1)[0]
    q_pre_ref_np = q_pre_ref[0].detach().numpy().reshape(seq, nh * hd)
    print(f"[golden] SPLIT q diff (mine vs transformers, pre-qnorm): {np.abs(q - q_pre_ref_np).max():.3e}")

    def headnorm(x, w, nheads):  # Qwen3.5 RMSNorm is (1+weight), Gemma-style
        x = x.copy().reshape(seq, nheads, hd)
        x = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * (1.0 + w)
        return x.reshape(seq, nheads * hd)
    q = headnorm(q, qn_w, nh); k = headnorm(k, kn_w, nkv)

    # isolate: my q PRE-rope (post-qnorm) vs transformers q_ts (post-qnorm, pre-rope)
    q_ts_np = q_ts[0].detach().numpy().transpose(1, 0, 2).reshape(seq, nh * hd)
    print(f"[golden] PRE-rope q diff (mine vs transformers, post-qnorm): {np.abs(q - q_ts_np).max():.3e}")

    def myrope(x, nheads):
        x = x.copy()
        for t in range(seq):
            for head in range(nheads):
                base = head * hd
                for j in range(half):
                    ang = t * inv[j]
                    c, s = np.cos(ang), np.sin(ang)
                    a, b = x[t, base + j], x[t, base + j + half]
                    x[t, base + j] = a * c - b * s
                    x[t, base + j + half] = b * c + a * s
        return x
    q_mine = myrope(q, nh); k_mine = myrope(k, nkv)

    # ROPE isolation: my rope vs transformers apply_rotary_pos_emb (reshape ref to (seq, nh*hd))
    q_rot_ref_np = q_rot_ref[0].detach().numpy().transpose(1, 0, 2).reshape(seq, nh * hd)  # (nh,seq,hd)->(seq,nh*hd)
    print(f"[golden] ROPE q diff (mine vs transformers): {np.abs(q_mine - q_rot_ref_np).max():.3e}")

    # full attention with MY post-rope q/k
    rep = nh // nkv
    scale = hd ** -0.5
    attn_out = np.zeros((seq, nh * hd))
    for head in range(nh):
        kv = head // rep
        qh = q_mine[:, head * hd:head * hd + hd]
        kh = k_mine[:, kv * hd:kv * hd + hd]
        vh = v[:, kv * hd:kv * hd + hd]
        sc = (qh @ kh.T) * scale
        for i in range(seq):
            for j in range(seq):
                if j > i:
                    sc[i, j] = -1e30
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        attn_out[:, head * hd:head * hd + hd] = sc @ vh
    attn_out *= 1.0 / (1.0 + np.exp(-gate))  # sigmoid gate
    mine_out = attn_out @ Wo.T

    print(f"[golden] FULL ATTN out diff (mine vs transformers): {np.abs(mine_out - ref_out).max():.3e}")
    print("VERDICT:", "full_attn matches" if np.abs(mine_out - ref_out).max() < 1e-4
          else "DIVERGES — see the ROPE diff above to tell if it's the rope or the attention/gate")


if __name__ == "__main__":
    main()
