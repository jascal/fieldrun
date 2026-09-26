"""Tiny MiMo-shaped Qwen2 parity fixture: head_dim=128, 4:1 GQA, high theta, q/k/v bias.

Run `build` with a Python environment containing torch, transformers, safetensors. The
checkpoint carries a dummy MTP tensor so conversion also exercises the strip path.
"""
import json
import sys
from pathlib import Path

OUT = Path("/tmp/fieldrun_mimo_tiny")
IDS = Path("/tmp/fieldrun_mimo_holdout.json")
REF = Path("/tmp/fieldrun_mimo_ref.json")


def build():
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(919)
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=512, intermediate_size=320, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=1, head_dim=128,
        rope_theta=640000.0, rms_norm_eps=1e-5, attention_bias=True,
        tie_word_embeddings=False, max_position_embeddings=256,
        attn_implementation="eager", torch_dtype=torch.float32,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    OUT.mkdir(exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text())
    config.update(model_type="mimo", architectures=["MiMoForCausalLM"],
                  use_sliding_window=False, sliding_window=32,
                  max_window_layers=2, use_mrope=False, num_nextn_predict_layers=1)
    config_path.write_text(json.dumps(config))
    weights_path = OUT / "model.safetensors"
    weights = load_file(weights_path)
    weights["model.mtp_layers.0.input_proj.weight"] = torch.ones(1, 1)
    save_file(weights, weights_path)
    ids = torch.randint(0, cfg.vocab_size, (76,), generator=torch.Generator().manual_seed(991)).tolist()
    IDS.write_text(json.dumps({"holdout_ids": ids}))
    preds = []
    with torch.no_grad():
        for i in range(16, 76):
            logits = model(torch.tensor([ids[i-16:i]])).logits[0, -1]
            preds.append(int(logits.argmax()))
    REF.write_text(json.dumps(preds))
    print(f"MiMo-shaped torch reference: {len(preds)} positions")


def compare(path):
    expected = json.loads(REF.read_text())
    actual = [int(s) for s in Path(path).read_text().split()]
    matched = sum(a == b for a, b in zip(expected, actual))
    print(f"MiMo backbone: {matched}/{len(expected)} top-1 agree")
    if len(actual) != len(expected) or matched != len(expected):
        print([(i, a, b) for i, (a, b) in enumerate(zip(expected, actual)) if a != b][:10])
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        compare(sys.argv[2])
