# Raw memory-control evidence

[Supplemental release](https://github.com/jascal/pdc/releases/tag/pdc-qwen35-memory-controls-v1), `pdc-qwen35-memory-controls-v1.tar.gz`,
93344761 bytes,39 payload files plus manifest; SHA-256 `b4196b098f5d673f4418ed5f2b9dd0bc78d762f8f75b9bbb61db29c0ff5d10ad`.
Every member verified before upload. Includes all54 native captures,
independent reference, exact selections, scripts, baseline tables, results
and verification, plus the v0.15 paper/spec snapshot. Dependencies are the
original Qwen and Qwen memory-learning releases; weights/environments external.

After extraction run from the fieldrun artifact root:

```sh
python3 scripts/pdc_qwen35_memory_controls_check.py --artifacts experiments/pdc_qwen35_memory_controls --baseline experiments/pdc_qwen35_memory --domain experiments/pdc_qwen35_4b/domain.json --out /tmp/control-verification.json
```

This independently audits input assembly, reference predictions and counts.
`--verify-original-paths` additionally verifies all23 frozen input/code/model/
runtime hashes when the original recorded paths are present. Without that
option the report explicitly records zero original-path hashes verified.
No source-score, general causal mechanism or admitted-replay claim follows.
