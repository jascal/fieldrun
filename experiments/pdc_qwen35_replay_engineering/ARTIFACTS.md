# Replay engineering reconstruction

[Verified supplemental release](https://github.com/jascal/pdc/releases/tag/pdc-qwen35-replay-engineering-v1):78 payload files plus manifest,
21665828 bytes, SHA-256 `42233206bee1cf7cdd6eb745558c9e02d2b68364d3d00993d6f47abdee9e0a1e`. All payload hashes verified.
Contains complete Rust source/Cargo files, official-reference regression
fixture and generator, replay reference/checker, failed and corrected tiny
captures, real captures/reference, old/corrected experimental executables,
tiny converted and HF checkpoints, protocol/results and v0.17 paper snapshot.
Real Qwen model weights/converted bundle and environments remain external.

The archived old scientific runtime is not replaced. The package's legacy
executable is the experimental replay runtime BEFORE the normalization fix;
legacy/src/deltanet.rs preserves its old kernel source. Corrected executable
is a Linux CPU build. Rebuild source for another platform.

After extracting, run from the fieldrun directory:

```sh
python3 scripts/pdc_qwen35_replay_engineering_check.py --capture experiments/pdc_qwen35_replay_engineering/tiny_corrected_capture.json --reference experiments/pdc_qwen35_replay_engineering/tiny_reference.json --selection experiments/pdc_qwen35_replay_engineering/tiny_selection.json --out /tmp/tiny-replay-audit.json --tiny
```

The original tiny_capture.json intentionally FAILS its absolute hidden gate;
initial_engineering_record/metrics and the red low-norm regression retain
that failure. Real verification uses the analogous real_* files without
--tiny. Source builds include the independent low-norm fixture under the
recorded experiments path. These are engineering checks, not admitted memory
transfer or a scientific arithmetic study. Previous releases unchanged.
