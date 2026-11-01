#!/usr/bin/env python3
"""Top-1 reference sweep for an official LFM3 checkpoint.

Uses the checkpoint's own remote-code implementation in float32, matching an f32 fieldrun bundle.
The independent fixed-width contexts match fieldrun's `--ctx N --n-eval M` scoring surface.
"""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ids")
    ap.add_argument("out")
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--hidden-out", help="write post-decoder hidden states as f32 .npy")
    ap.add_argument("--logits-out", help="write final-position logits as f32 .npy")
    args = ap.parse_args()
    ids = json.load(open(args.ids))
    ids = ids.get("holdout_ids", ids)
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
        x = torch.tensor(contexts)
        output = model(input_ids=x)
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
    main()
