#!/usr/bin/env python3
"""LFM3 parity references.

  lfm3_ref.py build <tag>            tiny random-weight fixture + its torch reference (offline; the gate)
  lfm3_ref.py compare <dump> <tag>   top-1 agreement of a fieldrun --dump against that reference
  lfm3_ref.py build-bias             conv_bias checkpoints the converter must REFUSE (negative gate)
  lfm3_ref.py <model> <ids> <out> [--ctx N --n-eval M --hidden-out F --logits-out F]
                                     top-1 sweep of a real checkpoint (its own remote code, f32)

Tags: `lfm3` — tied, flat `model.*` names; `lfm3vl` — untied, wrapped as an LFM3-VL composite
(`model.language_model.*` + `text_config` + a vision tensor the converter must skip).

The tiny model is built from the vendored reference code in `scripts/lfm3_remote/` (transformers 5.22 has no lfm3).
Its shape is chosen to exercise every LFM3-specific bit: all three layer types, `sliding_window` (4) SMALLER than the
eval context (16) so the window mask actually binds, GQA (nh 4 / nkv 2), and a full-attention RoPE width (0.5) that
differs from the sliding one (1.0), with distinct per-type thetas. `build` also re-runs the reference with each of those bits ablated and FAILS if
any ablation leaves all 60 predictions unchanged — a fixture that can't see a bit can't gate it.
"""
import argparse
import json
import math
import sys
from pathlib import Path

CTX, N_EVAL, N_IDS = 16, 60, 76
TAGS = ("lfm3", "lfm3vl")


def paths(tag):
    return Path(f"/tmp/{tag}tiny"), Path(f"/tmp/{tag}_holdout.json"), Path(f"/tmp/{tag}_ref.json")


def tiny_config(tag, **over):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lfm3_remote.configuration_lfm3 import Lfm3Config

    kw = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
        layer_types=["conv", "sliding_attention", "conv", "full_attention", "sliding_attention", "full_attention"],
        sliding_window=4, conv_L_cache=3, conv_bias=False, norm_eps=1e-5,
        # Small thetas (real: 1e6 / 1e4): with head_dim 8 and a 4-token window, the real bases leave the
        # dims that only full-head sliding RoPE rotates at ~1e-3 rad — invisible, so the width bit went ungated.
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 1_000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 100.0},
        },
        tie_word_embeddings=(tag == "lfm3"), pad_token_id=0, bos_token_id=1, eos_token_id=2,
        attn_implementation="eager", dtype="float32",
    )
    kw.update(over)
    return Lfm3Config(**kw)


def make_model(cfg, state=None):
    from lfm3_remote.modeling_lfm3 import Lfm3ForCausalLM

    model = Lfm3ForCausalLM(cfg).eval()
    if state is not None:
        model.load_state_dict(state)
    return model


def predict(model, ids):
    import torch

    ctxs = torch.tensor([ids[i - CTX:i] for i in range(CTX, CTX + N_EVAL)])
    with torch.no_grad():
        return model(input_ids=ctxs, use_cache=False).logits[:, -1].argmax(-1).tolist()


def randomize(model, seed):
    """Default init (std 0.02, unit norms) gives near-tied logits and makes the norm gains invisible; use
    fan-in-scaled weights and non-unit gains so every term can move a prediction."""
    import torch

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:  # RMSNorm gains (operator/ffn sandwich, q/k norm, embedding_norm)
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[-1]))


class _OpenGate(__import__("torch").nn.Module):
    """Replaces `gate_proj`: sigmoid(30) == 1 in f32, i.e. the per-head output gate removed."""

    def __init__(self, nh):
        super().__init__()
        self.nh = nh

    def forward(self, x):
        return x.new_full((*x.shape[:-1], self.nh), 30.0)


def build(tag):
    import torch
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(1)
    torch.manual_seed(3001)
    out, ids_path, ref_path = paths(tag)
    cfg = tiny_config(tag)
    model = make_model(cfg)
    randomize(model, 3002 if tag == "lfm3" else 3003)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ids = torch.randint(0, cfg.vocab_size, (N_IDS,), generator=torch.Generator().manual_seed(3004)).tolist()
    preds = predict(model, ids)

    # Discrimination: each LFM3-specific bit must be load-bearing on this fixture.
    ablations = {
        "no-sliding-window": predict(make_model(tiny_config(tag, sliding_window=10**6), state), ids),
        "shared-rope-width": predict(make_model(tiny_config(tag, rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 1_000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 100.0, "partial_rotary_factor": 0.5},
        }), state), ids),
    }
    gated = make_model(cfg, state)
    for layer in gated.model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.gate_proj = _OpenGate(cfg.num_attention_heads)
    ablations["no-output-gate"] = predict(gated, ids)
    moved = {k: sum(a != b for a, b in zip(preds, v)) for k, v in ablations.items()}
    print(f"{tag}: ablation moves (of {N_EVAL}): " + ", ".join(f"{k} {n}" for k, n in moved.items()))
    if not all(moved.values()):
        raise SystemExit(f"{tag}: fixture does not discriminate {[k for k, n in moved.items() if not n]}")

    out.mkdir(exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    if tag == "lfm3vl":
        wpath = out / "model.safetensors"
        w = load_file(wpath)
        w = {k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k: v for k, v in w.items()}
        w["model.vision_tower.dummy.weight"] = torch.ones(1)  # text-only conversion must skip vision tensors
        save_file(w, wpath)
        top = {"model_type": "lfm3_vl", "architectures": ["Lfm3VlForConditionalGeneration"],
               "text_config": cfg.to_dict(), "vision_config": {"model_type": "siglip2_vision_model"},
               "tie_word_embeddings": False}
        (out / "config.json").write_text(json.dumps(top))
    ids_path.write_text(json.dumps({"holdout_ids": ids}))
    ref_path.write_text(json.dumps(preds))
    print(f"{tag}: reference {len(preds)} positions -> {out}")


def build_bias():
    """Two checkpoints fieldrun does not support and must refuse rather than run with the conv biases dropped:
    /tmp/lfm3biastiny (conv_bias=true) and /tmp/lfm3biashiddentiny (the same bias tensors under a config that
    says false — the converter must look at the tensors, not just trust the flag)."""
    import shutil

    cfg = tiny_config("lfm3", conv_bias=True)
    model = make_model(cfg)
    randomize(model, 3005)
    on, hidden = Path("/tmp/lfm3biastiny"), Path("/tmp/lfm3biashiddentiny")
    for d in (on, hidden):
        shutil.rmtree(d, ignore_errors=True)
    model.save_pretrained(on, safe_serialization=True)
    shutil.copytree(on, hidden)
    c = json.loads((hidden / "config.json").read_text())
    c["conv_bias"] = False
    (hidden / "config.json").write_text(json.dumps(c))
    print(f"conv_bias checkpoints -> {on}, {hidden}")


def compare(dump, tag):
    """Prints `N/60 top-1`; exit status is left to the caller (f32 is gated, lower precisions are informational)."""
    expected = json.loads(paths(tag)[2].read_text())
    actual = [int(s) for s in Path(dump).read_text().split()]
    matched = sum(a == b for a, b in zip(expected, actual)) if len(actual) == len(expected) else 0
    print(f"{matched}/{len(expected)} top-1")
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
    if bad or len(actual) != len(expected):
        print(f"  len ref {len(expected)} fieldrun {len(actual)}; first mismatches (pos, ref, fieldrun): {bad[:8]}")


def sweep(argv):
    import torch
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ids")
    ap.add_argument("out")
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--hidden-out", help="write post-decoder hidden states as f32 .npy")
    ap.add_argument("--logits-out", help="write final-position logits as f32 .npy")
    args = ap.parse_args(argv)
    ids = json.load(open(args.ids))
    ids = ids.get("holdout_ids", ids) if isinstance(ids, dict) else ids
    stop = min(len(ids), args.ctx + args.n_eval)
    if stop <= args.ctx:
        raise SystemExit("need more ids than --ctx")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()
    positions = list(range(args.ctx, stop))
    # Every row is an independent fixed-width causal context, exactly matching fieldrun's
    # scoring loop while evaluating the reference in one batched forward.
    contexts = [ids[i - args.ctx:i] for i in positions]
    captures = []
    hooks = []
    if args.hidden_out:
        # `capture_outputs` has changed tuple conventions across transformers releases;
        # direct block hooks unambiguously capture post-decoder residuals.
        for layer in model.model.layers:
            hooks.append(layer.register_forward_hook(lambda _m, _a, out: captures.append(out.detach())))
    with torch.inference_mode():
        output = model(input_ids=torch.tensor(contexts))
        preds = output.logits[:, -1].argmax(dim=-1).tolist()
        if args.logits_out:
            import numpy as np
            np.save(args.logits_out, output.logits[:, -1].float().cpu().numpy())
        if args.hidden_out:
            import numpy as np
            np.save(args.hidden_out, torch.stack(captures).float().cpu().numpy())
    for hook in hooks:
        hook.remove()
    json.dump({"ctx": args.ctx, "positions": positions, "preds": preds}, open(args.out, "w"))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "build" and sys.argv[2] in TAGS:
        build(sys.argv[2])
    elif sys.argv[1:] == ["build-bias"]:
        build_bias()
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare" and sys.argv[3] in TAGS:
        compare(sys.argv[2], sys.argv[3])
    else:
        sweep(sys.argv[1:])
