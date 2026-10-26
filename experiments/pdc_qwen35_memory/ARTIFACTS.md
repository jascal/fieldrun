# Supplemental Qwen memory/learning evidence

Release: https://github.com/jascal/pdc/releases/tag/pdc-qwen35-memory-learning-v1

Archive `pdc-qwen35-memory-learning-v1.tar.gz`: 561684314 bytes,
143 payload files plus manifest, SHA-256 `633fa531742d5fbda36d239e2b7cf532d8b4e84ebfc8f28caf4b5acf99ef463b`. Every member was verified
against its manifest before publication. This archive includes raw memory
captures/reference/bundle, the full learning input NPZ, both initial/final
frames and endpoint oracle arrays, protocols, scripts, results and checks.
It depends on the original `pdc-qwen35-4b-v1` release for first-hop captures,
checkpoint metadata and native binary; model weights/environments are external.
The manifest lists exact paths and limitations. Existing releases are unchanged.

Extract the two Qwen archives into a common artifact root. Use the
`fieldrun/scripts/pdc_qwen35_memory_check.py` and
`pdc_qwen35_learning_data_check.py` command-line options to map checkpoint
and bundle locations; the learned endpoint checker is
`pil/experiments/pdc_qwen35_learning_check.py`. Each script documents its
required arguments through `--help`. Fieldrun and PIL exchange NPZ/JSON
artifacts and do not import each other's local source paths.

A certificate preserves covered score decisions, not truth. No admitted
replay, fresh confirmation or useful-service result is in this release.
