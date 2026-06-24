#!/usr/bin/env python3
"""Compare a fieldrun forward against the transformers reference (the whole-model parity test).

Consumes the reference `ref.npz` from make_tiny_qwen3next.py and a fieldrun dump `fr.npz` (logits +
per-layer hidden states, emitted by the fieldrun debug path once the Qwen3-Next arch is implemented), and
reports max-abs-diff per layer + on the logits. Per-LAYER is the point: it localizes WHICH op diverges
(DeltaNet state? gating? partial-RoPE? router? norm? MTP) instead of just "logits are wrong".

Tolerance bands (set by the dtype fieldrun ran in — develop in f32 FIRST to separate arch bugs from quant):
  f32:  ~1e-4   (arch correct)        f16:  ~1e-2        int8: looser — don't debug arch in int8.
Usage:  python compare.py experiments/qwen3next/tiny/ref.npz fr.npz [--tol 1e-4]

fieldrun side (to produce fr.npz) — the debug dump should write, for the same input_ids as ref.npz:
  logits: (seq, vocab)   hidden_states: (L+1, seq, hidden)   in load order matching transformers' blocks.
"""

from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("fr")
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    R = np.load(a.ref)
    F = np.load(a.fr)
    assert (R["input_ids"] == F["input_ids"]).all(), "input_ids differ — compare the SAME input"

    print(f"=== fieldrun vs transformers reference (tol {a.tol:g}) ===")
    ok = True
    if "hidden_states" in F:
        hr, hf = R["hidden_states"], F["hidden_states"]
        assert hr.shape == hf.shape, f"layer-state shape {hr.shape} vs {hf.shape}"
        print(f"  {'layer':<8}{'max_abs_diff':>14}{'  status':>10}")
        for li in range(hr.shape[0]):
            d = float(np.abs(hr[li] - hf[li]).max())
            tag = "ok" if d < a.tol else "DIVERGES <-- first bad op is in/after this block"
            print(f"  {li:<8}{d:>14.2e}   {tag}")
            if d >= a.tol:
                ok = False
                print(f"  -> localize: the bug is in block {li} (the op that produced hidden_state[{li}]).")
                break
    dl = float(np.abs(R["logits"] - F["logits"]).max())
    print(f"  logits max_abs_diff = {dl:.2e}  ->  {'PARITY OK' if dl < a.tol else 'MISMATCH'}")
    # argmax agreement is the decode-relevant check (what the certificate ultimately cares about)
    am = float((R['logits'].argmax(-1) == F['logits'].argmax(-1)).mean())
    print(f"  argmax agreement = {am*100:.1f}%  (decode parity)")
    print("VERDICT:", "arch matches the reference — same code runs the big model" if (ok and dl < a.tol)
          else "diverges — fix the localized block, re-run (develop in f32 before f16/int8)")


if __name__ == "__main__":
    main()
