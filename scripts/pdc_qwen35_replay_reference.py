"""Independent Transformers reference for explicit final-query embedding replay."""

import argparse
import json
from pathlib import Path
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["model", "selection", "out"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--dtype", choices=["float32", "float16"], required=True)
    a = p.parse_args()
    assert not a.out.exists()
    torch.set_num_threads(4)
    top = json.loads((a.model / "config.json").read_text())
    cfg = Qwen3_5TextConfig.from_dict(top.get("text_config", top))
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        a.model,
        config=cfg,
        dtype=getattr(torch, a.dtype),
        attn_implementation="eager",
        output_loading_info=True,
        key_mapping={r"^model\.language_model\.": "model."},
    )
    assert not loading["missing_keys"]
    assert all(
        "visual" in k or k.startswith("mtp.") or ".mtp." in k
        for k in loading["unexpected_keys"]
    )
    model.eval()
    selection = json.loads(a.selection.read_text())
    contexts = selection["contexts"]
    memories = selection.get("memory_writes") or [None] * len(contexts)
    assert len(memories) == len(contexts)
    records = []
    for ids, memory in zip(contexts, memories, strict=True):
        calls = []
        if memory is not None:
            assert memory["site"] == "embed" and len(memory["write"]) == cfg.hidden_size
            write = torch.tensor(memory["write"], dtype=torch.float32)
            assert bool(torch.isfinite(write).all())

            def hook(module, args, output):
                altered = output.clone()
                altered[0, -1] = (output[0, -1].float() + write).to(output.dtype)
                assert torch.equal(altered[0, :-1], output[0, :-1])
                calls.append(1)
                return altered

            handle = model.get_input_embeddings().register_forward_hook(hook)
        else:
            handle = None
        try:
            with torch.no_grad():
                output = model(
                    input_ids=torch.tensor([ids]),
                    use_cache=False,
                    logits_to_keep=1,
                    output_hidden_states=True,
                )
        finally:
            if handle is not None:
                handle.remove()
        assert len(calls) == int(memory is not None)
        logits = output.logits[0, -1].float()
        records.append(
            dict(
                ids=ids,
                prediction=int(logits.argmax()),
                candidate_scores=logits[selection["candidate_tokens"]].tolist(),
                final_hidden=output.hidden_states[-1][0, -1].float().tolist(),
                memory_replay=memory,
                embedding_hook_calls=len(calls),
            )
        )
        print("replay reference completed", len(records), flush=True)
    a.out.write_text(
        json.dumps(
            dict(schema="pdc.qwen35-replay-reference.v1", dtype=a.dtype, cases=records),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
