# MiMo-V2.6-Distill-Qwen-9B text path

The published checkpoint is a Qwen3.5 dense hybrid inside a vision-language
wrapper. `convert` detects `model_type: "qwen3_5"`, reads the nested
`text_config` and `model.language_model.*` weights, and writes a `qwen35`
bundle. It omits `model.visual.*` weights. The text path uses a 3:1 pattern of
Gated DeltaNet and gated full attention, followed by dense SwiGLU FFNs.
Conversion streams the large embedding and output tables in row blocks.

```bash
fieldrun convert --model XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --dtype int4
fieldrun --bundle MiMo-V2.6-Distill-Qwen-9B --chat
```

The converter copies `tokenizer.json` and records the EOS id from the nested
text config. The default build checks that the published tokenizer JSON loads
and encodes text before converting the weights. Text-only prompts use the same position for all three MRoPE axes,
so the Qwen3.5 partial RoPE path applies. Image and video inputs need the
vision encoder and multimodal position handling; those are not supported by
this bundle. The MTP head is not used for generation. Plain ChatML text turns
work with fieldrun's chat renderer; exact replay of the checkpoint's richer
reasoning and tool-call template is not yet implemented.

The tiny parity fixture builds a four-layer torch Qwen3.5 dense model with the
same attention pattern, causal conv, partial RoPE, dense MLP, and composite
weight names, then checks 60 held-out next-token predictions:

```bash
python scripts/qwen35_dense_ref.py build
fieldrun convert --model /tmp/fieldrun_qwen35_dense_tiny --dtype f32 -o /tmp/fieldrun_qwen35_dense_f32 --force
fieldrun --bundle /tmp/fieldrun_qwen35_dense_f32 --ids /tmp/fieldrun_qwen35_dense_ids.json --ctx 16 --n-eval 60 --dump /tmp/fieldrun_qwen35_dense_f32.txt
python scripts/qwen35_dense_ref.py compare /tmp/fieldrun_qwen35_dense_f32.txt
```

The f32, f16, and int8 tiny bundles all reached 60/60 top-1 against torch in this
fixture; int4 reached 56/60. This is an architecture gate, not a full-checkpoint
parity result.
Generation currently recomputes the full context on each token;
an incremental DeltaNet state and full-attention KV cache are still needed for
usable long-context speed. `--kv-int8` has no effect on this path and the
GPU-resident forward is not wired for Qwen3.5. The full checkpoint's memory fit has not been
measured on an 8 GB device; int4 is the smaller starting point and needs its
own parity check.
