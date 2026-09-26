"""Tiny MiMo-V2.6/Qwen3.5 dense text parity fixture (torch/transformers)."""

import json
import sys
from pathlib import Path

OUT = Path("/tmp/fieldrun_qwen35_dense_tiny")
IDS = Path("/tmp/fieldrun_qwen35_dense_ids.json")
REF = Path("/tmp/fieldrun_qwen35_dense_ref.json")


def build():
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.set_num_threads(1)
    torch.manual_seed(2609)
    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000.0,
                         "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                         "mrope_section": [1, 1, 0]},
        partial_rotary_factor=0.25, attention_bias=False, tie_word_embeddings=False,
        attn_implementation="eager", dtype="float32",
    )
    model = Qwen3_5ForCausalLM(cfg).eval()
    OUT.mkdir(exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    weight_path = OUT / "model.safetensors"
    weights = load_file(weight_path)
    weights = {name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name: value
               for name, value in weights.items()}
    # A composite checkpoint includes vision weights, which text-only conversion must skip.
    weights["model.visual.dummy.weight"] = torch.ones(1)
    save_file(weights, weight_path)
    top = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
           "text_config": cfg.to_dict(), "vision_config": {"model_type": "qwen3_5_vision"},
           "tie_word_embeddings": False}
    (OUT / "config.json").write_text(json.dumps(top))
    ids = torch.randint(0, cfg.vocab_size, (76,), generator=torch.Generator().manual_seed(2610)).tolist()
    IDS.write_text(json.dumps({"holdout_ids": ids}))
    preds = []
    with torch.no_grad():
        for i in range(16, 76):
            logits = model(torch.tensor([ids[i-16:i]]), use_cache=False).logits[0, -1]
            preds.append(int(logits.argmax()))
    REF.write_text(json.dumps(preds))
    print(f"Qwen3.5 dense reference: {len(preds)} positions")


def compare(path):
    expected = json.loads(REF.read_text())
    actual = [int(s) for s in Path(path).read_text().split()]
    matched = sum(a == b for a, b in zip(expected, actual))
    print(f"Qwen3.5 dense: {matched}/{len(expected)} top-1 agree")
    if len(actual) != len(expected) or matched != len(expected):
        print([(i, a, b) for i, (a, b) in enumerate(zip(expected, actual)) if a != b][:10])
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        compare(sys.argv[2])
