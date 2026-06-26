#!/usr/bin/env python
"""Compare fieldrun's gemma4 forward against the transformers reference dumped by make_tiny_gemma4.py.

  python compare.py <tiny_dir> <hidden_dump.bin> <logits_dump.bin>

<tiny_dir>/ref.npz holds logits (seq,vocab) + hidden_states (nl+1,seq,d). The hidden dump is nl+1 snapshots of seq*d
f32 LE in HF output_hidden_states order [inputs_embeds, out-L0, …, out-L{n-2}, post-final-norm]; the logits dump is
seq positions x vocab f32 LE (each = fieldrun's last-position logits over the prefix ids[:p+1], which by causality
equals the reference's position-p logits). Reports per-snapshot max-abs-diff (localizes any break to one layer) and the
final logits max-abs-diff + argmax agreement (decode parity)."""
import sys
import numpy as np

tiny, hidden_bin, logits_bin = sys.argv[1], sys.argv[2], sys.argv[3]
ref = np.load(f"{tiny}/ref.npz")
ref_hs = ref["hidden_states"]          # (n_hs, seq, d)
ref_lg = ref["logits"]                 # (seq, vocab)
n_hs, seq, d = ref_hs.shape
vocab = ref_lg.shape[1]

fr_hs = np.fromfile(hidden_bin, dtype="<f4").reshape(n_hs, seq, d)
fr_lg = np.fromfile(logits_bin, dtype="<f4").reshape(seq, vocab)

labels = ["inputs_embeds"] + [f"out-L{l}" for l in range(n_hs - 2)] + ["POST-final-norm"]
print(f"=== per-snapshot residual parity ({n_hs} snaps, seq={seq}, d={d}) ===")
worst = 0.0
for i in range(n_hs):
    diff = float(np.abs(fr_hs[i] - ref_hs[i]).max())
    worst = max(worst, diff)
    print(f"  {labels[i]:>16s} : {diff:.3e}")

amax_fr = fr_lg.argmax(1)
amax_ref = ref_lg.argmax(1)
agree = float((amax_fr == amax_ref).mean()) * 100.0
lg_diff = float(np.abs(fr_lg - ref_lg).max())
print(f"=== logits ===")
print(f"  max_abs_diff = {lg_diff:.3e}   argmax agreement = {agree:.1f}%")

ok = worst < 5e-3 and lg_diff < 5e-3 and agree == 100.0
print("RESULT:", "PARITY ✓" if ok else "MISMATCH ✗")
sys.exit(0 if ok else 1)
