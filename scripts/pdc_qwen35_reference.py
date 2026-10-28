"""Independent transformers engineering reference; no PDC atom selection."""

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "float16"], required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    torch.set_num_threads(4)
    top = json.loads((args.model / "config.json").read_text())
    config = Qwen3_5TextConfig.from_dict(top.get("text_config", top))
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
        output_loading_info=True,
        key_mapping={r"^model\.language_model\.": "model."},
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert all(
        "visual" in k or k.startswith("mtp.") or ".mtp." in k
        for k in loading["unexpected_keys"]
    ), loading["unexpected_keys"]
    model.eval()
    selection = json.loads(args.selection.read_text())
    rows = []
    with torch.no_grad():
        for ids in selection["contexts"]:
            output = model(
                input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=1
            )
            scores = output.logits[0, -1].float()
            rows.append(
                dict(
                    ids=ids,
                    prediction=int(scores.argmax()),
                    candidate_scores=scores[selection["candidate_tokens"]].tolist(),
                )
            )
            print("reference completed", len(rows), flush=True)
    result = dict(
        schema="pdc.qwen35-transformers-reference.v1",
        dtype=args.dtype,
        config=top,
        cases=rows,
        unexpected_nontext_keys=len(loading["unexpected_keys"]),
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
