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
    with torch.inference_mode():
        x = torch.tensor(contexts)
        preds = model(input_ids=x).logits[:, -1].argmax(dim=-1).tolist()
    json.dump({"ctx": args.ctx, "positions": positions, "preds": preds}, open(args.out, "w"))


if __name__ == "__main__":
    main()
