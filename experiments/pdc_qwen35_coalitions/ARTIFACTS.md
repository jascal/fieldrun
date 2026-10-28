# Full coalition metrics

[Supplemental release](https://github.com/jascal/pdc/releases/tag/pdc-qwen35-coalitions-v1),4942886 bytes,16 payload files plus manifest;
SHA-256 `4ec2b1e5faa6b559145c3dddad602572a98fb600e5027d57312cdd66270b5ada`. All payload hashes verified before publication.
Includes complete body_metrics.npz with209100 margins, slacks, drift ranges
and strict midpoint gamma for centered passes, plus protocols/body indices,
results, checks, source scripts and v0.16 paper/spec snapshot.

Original Qwen release supplies100 raw native captures. The memory-learning
release supplies learning_data.npz for primary scoring. Independent raw
scoring also requires the original converted model frame and NumPy/Torch;
model weights and environments remain external. Run each script with
`--help` for artifact arguments. `--verify-original-paths` on the independent
checker verifies recorded original locations when present; otherwise its
report explicitly records zero original-path hashes verified. No local-path
imports from other repos are used. Previous releases are unchanged.
