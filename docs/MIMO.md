# MiMo-7B text inference

MiMo-7B-RL and MiMo-7B-Base use a Qwen2 text backbone. `convert` detects
`model_type: "mimo"` and writes a `rope` bundle. The ordinary next-token path
uses `model.layers.*` and omits `model.mtp_layers.*`; speculative decoding with
the MTP head is not implemented. The bundle records `source_model_type` and
`mtp_layers_omitted` so this choice is visible.

```bash
fieldrun convert --model XiaomiMiMo/MiMo-7B-RL --dtype int8
fieldrun --bundle MiMo-7B-RL --ids prompt.json --ctx 16 --generate 32
```

The converter copies `tokenizer.json` into the bundle. The default build's API
feature checks that MiMo's tokenizer JSON loads and encodes text before converting
the weights. It then uses the tokenizer for text endpoints and chat. Use the
checkpoint's chat template when comparing prompts with another engine.

The released 7B config has head dimension 128, 32 query heads, 8 KV heads,
RoPE theta 640000, Q/K/V biases, and `use_sliding_window: false`. The rope
kernel supports those dimensions and full attention. Conversion rejects a MiMo
config with sliding attention or multimodal RoPE enabled, since neither is
implemented for this path.

To run the small torch parity fixture (requires torch, transformers and
safetensors in the Python environment):

```bash
python scripts/mimo_ref.py build
fieldrun convert --model /tmp/fieldrun_mimo_tiny --dtype f32 -o /tmp/fieldrun_mimo_f32 --force
fieldrun --bundle /tmp/fieldrun_mimo_f32 --ids /tmp/fieldrun_mimo_holdout.json --ctx 16 --n-eval 60 --dump /tmp/fieldrun_mimo_f32.txt
python scripts/mimo_ref.py compare /tmp/fieldrun_mimo_f32.txt
```

The fixture exercises the same Qwen2 backbone math with MiMo's head dimension,
4:1 GQA, theta, biases, and an extra MTP tensor. It is an architecture check;
the full MiMo-7B weights still need a separate top-1 comparison with the
official checkpoint. Quantized bundles can flip greedy tokens, so use f32 for
the faithfulness gate. An int8 or smaller bundle is needed for constrained RAM;
the actual footprint depends on the chosen tensor dtypes and runtime buffers.

The optional GPU rope kernel accepts head dimension 128 and the 4:1 GQA layout,
and has a tiled int8 matmul. It currently expands the embedding and unembedding
tables to f32 on upload, so fitting the full 7B model on an 8 GB GPU needs a
measured budget and likely further memory work.
